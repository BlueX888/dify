"""Application service for dataset segment and child-chunk HTTP use cases."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Literal, Protocol
from uuid import uuid4

from libs.datetime_utils import naive_utc_now
from machinery.context import RequestContext
from services.errors.dataset import DatasetAccessDeniedError
from services.knowledge.dataset_access import DatasetAccess, DatasetAccessSnapshot
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef, SegmentRef
from services.knowledge.entities.segments import (
    ChildChunkRecord,
    ChildChunkUpdateArgs,
    ChildVectorGuard,
    ChildVectorPublication,
    SegmentCreateData,
    SegmentReadSnapshot,
    SegmentRecord,
    SegmentStateIntent,
    SegmentUpdateArgs,
)
from services.knowledge.indexing.errors import DatasetIndexingGenerationLostError

_HIGH_QUALITY = "high_quality"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SegmentDatasetRecord:
    id: str
    workspace_id: str
    indexing_technique: str
    embedding_model_provider: str | None
    embedding_model: str | None


@dataclass(frozen=True, slots=True)
class SegmentDocumentRecord:
    id: str
    dataset_id: str
    workspace_id: str
    doc_form: str

    @property
    def ref(self) -> DocumentRef:
        return DatasetRef(self.workspace_id, self.dataset_id).document(self.id)


@dataclass(frozen=True, slots=True)
class SegmentScope:
    dataset: SegmentDatasetRecord | None
    document: SegmentDocumentRecord | None
    access_snapshot: DatasetAccessSnapshot | None = None


@dataclass(frozen=True, slots=True)
class SegmentListFilter:
    page: int = 1
    limit: int = 20
    statuses: tuple[str, ...] = ()
    hit_count_gte: int | None = None
    enabled: str = "all"
    keyword: str | None = None


@dataclass(frozen=True, slots=True)
class ChildChunkListFilter:
    page: int = 1
    limit: int = 20
    keyword: str | None = None


@dataclass(frozen=True, slots=True)
class SegmentPage[T = SegmentRecord]:
    items: tuple[T, ...]
    total: int
    total_pages: int
    page: int
    limit: int


@dataclass(frozen=True, slots=True)
class SegmentDetail[T = SegmentRecord]:
    data: T
    doc_form: str


@dataclass(frozen=True, slots=True)
class ChildChunkPage:
    items: tuple[ChildChunkRecord, ...]
    total: int
    total_pages: int
    page: int
    limit: int


@dataclass(frozen=True, slots=True)
class SegmentBatchImport:
    job_id: str
    job_status: str


class SegmentScopeReader(Protocol):
    def get_segment_scope(
        self,
        *,
        dataset_ref: DatasetRef,
        document_id: str,
        actor_id: str | None,
    ) -> SegmentScope: ...


@dataclass(frozen=True, slots=True)
class ChildChunkState:
    data: ChildChunkRecord
    index_node_id: str | None
    index_node_hash: str | None
    created_by: str
    updated_by: str | None = None
    source_index_node_id: str | None = None


@dataclass(frozen=True, slots=True)
class SegmentIndexTarget:
    id: str
    index_node_id: str | None
    enabled: bool


@dataclass(frozen=True, slots=True)
class SegmentUpdateState:
    """Persisted fields needed to decide a segment update, without presentation data."""

    content: str
    enabled: bool
    keywords: tuple[str, ...] | None
    answer: str | None = None


class SegmentChildMutation(Protocol):
    @property
    def enabled(self) -> bool: ...

    @property
    def vector_guard(self) -> ChildVectorGuard: ...

    def get_children(self) -> tuple[ChildChunkState, ...]: ...

    def is_current_regeneration(
        self,
        *,
        index_node_hash: str | None,
        process_rule_id: str,
        process_rule_mode: str,
        process_rule_rules: str | None,
    ) -> bool: ...

    def require_current(
        self,
        *,
        expected: Sequence[ChildChunkState],
        expected_vector_guard: ChildVectorGuard,
    ) -> None: ...

    def save(
        self,
        *,
        expected: Sequence[ChildChunkState],
        expected_vector_guard: ChildVectorGuard,
        publication: ChildVectorPublication,
        added: Sequence[ChildChunkState] = (),
        updated: Sequence[ChildChunkState] = (),
        deleted: Sequence[ChildChunkState] = (),
    ) -> SegmentStateIntent: ...


class SegmentStore(Protocol):
    """Owner-scoped persistence; each operation owns a short session."""

    def list_segments(
        self, document_ref: DocumentRef, query: SegmentListFilter
    ) -> SegmentPage[SegmentReadSnapshot]: ...
    def get_segments(self, document_ref: DocumentRef, segment_ids: Sequence[str]) -> tuple[SegmentIndexTarget, ...]: ...
    def get_segment(self, segment_ref: SegmentRef) -> SegmentDetail[SegmentReadSnapshot] | None: ...
    def get_segment_update_state(self, segment_ref: SegmentRef) -> SegmentUpdateState | None: ...
    def lock_index_write(
        self, segment_ref: SegmentRef, *, lock_segment: bool = True
    ) -> AbstractContextManager[bool]: ...
    def create_segments(
        self,
        document_ref: DocumentRef,
        segments: Sequence[SegmentCreateData],
    ) -> None: ...
    def save_segment(self, segment_ref: SegmentRef, values: Mapping[str, object]) -> None: ...
    def delete_segments(
        self, document_ref: DocumentRef, segment_ids: Sequence[str]
    ) -> tuple[SegmentStateIntent, ...]: ...
    def list_child_chunks(self, segment_ref: SegmentRef, query: ChildChunkListFilter) -> ChildChunkPage | None: ...
    def get_children(self, segment_ref: SegmentRef) -> tuple[ChildChunkState, ...] | None: ...
    def child_mutation(self, segment_ref: SegmentRef) -> AbstractContextManager[SegmentChildMutation]: ...


class SegmentIndex(Protocol):
    def count_tokens(self, dataset: SegmentDatasetRecord, text: str) -> int: ...
    def create_many(
        self,
        document_ref: DocumentRef,
        *,
        segment_ids: Sequence[str],
        keywords_list: Sequence[Sequence[str] | None] | None,
    ) -> None: ...
    def create(
        self, segment_ref: SegmentRef, *, keywords: Sequence[str] | None, attachment_ids: Sequence[str]
    ) -> None: ...
    def update(self, segment_ref: SegmentRef, *, keywords: Sequence[str] | None, regenerate_children: bool) -> None: ...
    def update_attachments(self, segment_ref: SegmentRef, attachment_ids: Sequence[str]) -> None: ...
    def update_summary(self, segment_ref: SegmentRef, *, summary: str | None, content_changed: bool) -> None: ...
    def prepare_child_publication(
        self,
        segment_ref: SegmentRef,
        *,
        guard: ChildVectorGuard,
        added: Sequence[ChildChunkState] = (),
        updated: Sequence[ChildChunkState] = (),
        deleted: Sequence[ChildChunkState] = (),
    ) -> PreparedChildPublication: ...
    def abandon_child_publication(self, publication: ChildVectorPublication) -> None: ...


class PreparedChildPublication(Protocol):
    """A durable child-vector writer registered before its mutation boundary."""

    def submit(self) -> AbstractContextManager[ChildVectorPublication]: ...
    def discard(self) -> None: ...


class SegmentLimits(Protocol):
    SINGLE_CHUNK_ATTACHMENT_LIMIT: int


class SegmentModelGuard(Protocol):
    def check(self, dataset: SegmentDatasetRecord) -> None: ...


class SegmentUploadCatalog(Protocol):
    def get_file_name(self, *, workspace_id: str, upload_file_id: str) -> str | None: ...
    def get_existing_file_ids(self, *, workspace_id: str, file_ids: Sequence[str]) -> frozenset[str]: ...


class SegmentIndexingState(Protocol):
    def lock(self, name: str, *, timeout: int) -> AbstractContextManager[object]: ...
    def is_segment_indexing(self, segment_id: str, *, deleting: bool = False) -> bool: ...

    def is_document_indexing(self, document_id: str) -> bool: ...

    def set_batch_waiting(self, job_id: str) -> None: ...

    def get_batch_status(self, job_id: str) -> str | None: ...


class SegmentBatchImportDispatcher(Protocol):
    def dispatch(
        self,
        *,
        job_id: str,
        upload_file_id: str,
        dataset_id: str,
        document_id: str,
        workspace_id: str,
        actor_id: str,
    ) -> None: ...


class SegmentStateTransitions(Protocol):
    def request(
        self,
        *,
        document_ref: DocumentRef,
        segment_ids: Sequence[str],
        action: Literal["enable", "disable"],
        actor_id: str,
    ) -> tuple[SegmentStateIntent, ...]: ...


class SegmentStateDispatcher(Protocol):
    def dispatch(self, intent_id: str) -> None: ...
    def dispatch_many(self, intent_ids: Sequence[str]) -> None: ...


class SegmentApplicationError(Exception):
    """Base class for framework-neutral segment use-case errors."""


class SegmentDatasetNotFoundError(SegmentApplicationError):
    pass


class SegmentDocumentNotFoundError(SegmentApplicationError):
    pass


class SegmentNotFoundError(SegmentApplicationError):
    pass


class ChildChunkNotFoundError(SegmentApplicationError):
    pass


class SegmentUploadFileNotFoundError(SegmentApplicationError):
    pass


class SegmentPermissionDeniedError(SegmentApplicationError):
    pass


class SegmentInvalidFileTypeError(SegmentApplicationError):
    pass


class SegmentDocumentIndexingError(SegmentApplicationError):
    pass


class SegmentStatusUpdateError(SegmentApplicationError):
    pass


class SegmentBatchImportDispatchError(SegmentApplicationError):
    pass


class SegmentBatchImportNotFoundError(SegmentApplicationError):
    pass


class SegmentBatchIndexingFailureError(Exception):
    """The adapter already CAS-persisted failure for its exact attempted members."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


class SegmentModelProviderError(Exception):
    """Adapter error raised when the configured embedding model cannot be loaded."""

    def __init__(self, *, kind: Literal["bad_request", "token"], description: str) -> None:
        super().__init__(description)
        self.kind = kind
        self.description = description


class SegmentDatasetModelUnavailableError(SegmentApplicationError):
    pass


class SegmentEmbeddingModelUnavailableError(SegmentApplicationError):
    pass


class ChildChunkIndexingApplicationError(SegmentApplicationError):
    pass


class ChildChunkDeleteIndexApplicationError(SegmentApplicationError):
    pass


class SegmentPresenter(Protocol):
    def present(self, snapshots: Sequence[SegmentReadSnapshot]) -> tuple[SegmentRecord, ...]: ...


class DatasetSegmentApplicationService:
    def __init__(
        self,
        *,
        dataset_access: DatasetAccess,
        scopes: SegmentScopeReader,
        store: SegmentStore,
        presenter: SegmentPresenter,
        index: SegmentIndex,
        limits: SegmentLimits,
        text_hash: Callable[[str], str],
        uploads: SegmentUploadCatalog,
        model_guard: SegmentModelGuard,
        indexing_state: SegmentIndexingState,
        batch_dispatcher: SegmentBatchImportDispatcher,
        state_transitions: SegmentStateTransitions,
        state_dispatcher: SegmentStateDispatcher,
        job_id_factory: Callable[[], str],
    ) -> None:
        self._dataset_access = dataset_access
        self._scopes = scopes
        self._store = store
        self._presenter = presenter
        self._index = index
        self._limits = limits
        self._uploads = uploads
        self._model_guard = model_guard
        self._indexing_state = indexing_state
        self._batch_dispatcher = batch_dispatcher
        self._state_transitions = state_transitions
        self._state_dispatcher = state_dispatcher
        self._job_id_factory = job_id_factory
        self.mutations = SegmentMutationService(
            store=store,
            index=index,
            uploads=uploads,
            text_hash=text_hash,
            indexing_state=indexing_state,
            state_transitions=state_transitions,
            state_dispatcher=state_dispatcher,
        )

    def list_segments(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        query: SegmentListFilter,
    ) -> SegmentPage:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        page = self._store.list_segments(scope.document.ref, query)
        return SegmentPage(
            items=self._presenter.present(page.items),
            total=page.total,
            total_pages=page.total_pages,
            page=page.page,
            limit=page.limit,
        )

    def delete_segments(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_ids: Sequence[str],
    ) -> None:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        segments = self._store.get_segments(scope.document.ref, segment_ids)
        if segments:
            intents = self._store.delete_segments(scope.document.ref, [segment.id for segment in segments])
            self._dispatch_state_intents(intents)

    def change_segment_status(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_ids: Sequence[str],
        action: Literal["enable", "disable"],
    ) -> None:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        if action == "enable":
            self._check_dataset_model(scope.dataset)
            self._check_embedding_model(scope.dataset)
        if self._indexing_state.is_document_indexing(scope.document.id):
            raise SegmentDocumentIndexingError("Document is being indexed, please try again later")
        try:
            segments = self._store.get_segments(scope.document.ref, segment_ids)
            intents = self._state_transitions.request(
                document_ref=scope.document.ref,
                segment_ids=[segment.id for segment in segments],
                action=action,
                actor_id=context.account_id,
            )
        except Exception as error:
            raise SegmentStatusUpdateError(str(error)) from error
        self._dispatch_state_intents(intents)

    def _dispatch_state_intents(self, intents: Sequence[SegmentStateIntent]) -> None:
        if not intents:
            return
        try:
            self._state_dispatcher.dispatch_many([intent.id for intent in intents])
        except Exception:
            # The state and durable intents are already committed. Periodic
            # recovery will retry publication after a broker outage.
            logger.exception("Failed to publish %s segment state intents", len(intents))

    def create_segment(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        values: Mapping[str, object],
    ) -> SegmentDetail:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        self._check_embedding_model(scope.dataset)
        args = self._validate_values(values, scope.document.doc_form)
        segment_ref = self.mutations.create_segment(scope, actor_id=context.account_id, args=args)
        return self._detail(segment_ref)

    def update_segment(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_id: str,
        values: Mapping[str, object],
    ) -> SegmentDetail:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        args = self._validate_values(values, scope.document.doc_form)
        if args.enabled is not False:
            self._check_dataset_model(scope.dataset)
            self._check_embedding_model(scope.dataset)
        self.mutations.update_segment(scope, actor_id=context.account_id, segment_id=segment_id, args=args)
        return self._detail(scope.document.ref.segment(segment_id))

    def _detail(self, segment_ref: SegmentRef) -> SegmentDetail:
        result = self._store.get_segment(segment_ref)
        if result is None:
            raise SegmentNotFoundError("Segment not found")
        return SegmentDetail(
            data=self._presenter.present((result.data,))[0],
            doc_form=result.doc_form,
        )

    def delete_segment(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_id: str,
    ) -> None:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        return self.mutations.delete_segment(scope.document.ref.segment(segment_id))

    def start_batch_import(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        upload_file_id: str,
    ) -> SegmentBatchImport:
        scope = self._require_scope(
            context,
            dataset_id=dataset_id,
            document_id=document_id,
        )
        filename = self._uploads.get_file_name(
            workspace_id=scope.dataset.workspace_id,
            upload_file_id=upload_file_id,
        )
        if filename is None:
            raise SegmentUploadFileNotFoundError("UploadFile not found")
        if not filename.lower().endswith(".csv"):
            raise SegmentInvalidFileTypeError("Invalid file type. Only CSV files are allowed")

        job_id = self._job_id_factory()
        try:
            self._indexing_state.set_batch_waiting(job_id)
            self._batch_dispatcher.dispatch(
                job_id=job_id,
                upload_file_id=upload_file_id,
                dataset_id=scope.dataset.id,
                document_id=scope.document.id,
                workspace_id=scope.dataset.workspace_id,
                actor_id=context.account_id,
            )
        except Exception as error:
            raise SegmentBatchImportDispatchError(str(error)) from error
        return SegmentBatchImport(job_id=job_id, job_status="waiting")

    def get_batch_import_status(self, job_id: str) -> SegmentBatchImport:
        status = self._indexing_state.get_batch_status(job_id)
        if status is None:
            raise SegmentBatchImportNotFoundError("The job does not exist.")
        return SegmentBatchImport(job_id=job_id, job_status=status)

    def create_child_chunk(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_id: str,
        content: str,
    ) -> ChildChunkRecord:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        self._check_embedding_model(scope.dataset)
        return self.mutations.create_child_chunk(
            scope.document.ref.segment(segment_id), content=content, actor_id=context.account_id
        )

    def list_child_chunks(
        self,
        *,
        workspace_id: str,
        dataset_id: str,
        document_id: str,
        segment_id: str,
        query: ChildChunkListFilter,
    ) -> ChildChunkPage:
        scope = self._require_scope_without_actor(
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            document_id=document_id,
        )
        self._check_dataset_model(scope.dataset)
        result = self._store.list_child_chunks(scope.document.ref.segment(segment_id), query)
        if result is None:
            raise SegmentNotFoundError("Segment not found")
        return result

    def update_child_chunks(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_id: str,
        chunks: Sequence[ChildChunkUpdateArgs],
    ) -> tuple[ChildChunkRecord, ...]:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        self._check_dataset_model(scope.dataset)
        return self.mutations.update_child_chunks(
            scope.document.ref.segment(segment_id), chunks=chunks, actor_id=context.account_id
        )

    def delete_child_chunk(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_id: str,
        child_chunk_id: str,
    ) -> None:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        return self.mutations.delete_child_chunk(scope.document.ref.segment(segment_id), child_chunk_id=child_chunk_id)

    def update_child_chunk(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_id: str,
        child_chunk_id: str,
        content: str,
    ) -> ChildChunkRecord:
        scope = self._require_scope(context, dataset_id=dataset_id, document_id=document_id)
        self._check_dataset_model(scope.dataset)
        return self.mutations.update_child_chunk(
            scope.document.ref.segment(segment_id),
            child_chunk_id=child_chunk_id,
            content=content,
            actor_id=context.account_id,
        )

    def _validate_values(self, values: Mapping[str, object], doc_form: str) -> SegmentUpdateArgs:
        return validate_segment_values(
            values, doc_form=doc_form, attachment_limit=self._limits.SINGLE_CHUNK_ATTACHMENT_LIMIT
        )

    def _require_scope(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
    ) -> SegmentMutationScope:
        scope = self._scopes.get_segment_scope(
            dataset_ref=DatasetRef(context.active_workspace_id, dataset_id),
            document_id=document_id,
            actor_id=context.account_id,
        )
        dataset = scope.dataset
        if dataset is None:
            raise SegmentDatasetNotFoundError("Dataset not found")
        if scope.access_snapshot is None:
            raise SegmentPermissionDeniedError("Dataset access state is missing")
        try:
            self._dataset_access.check_access(context, scope.access_snapshot)
        except DatasetAccessDeniedError as error:
            raise SegmentPermissionDeniedError(str(error)) from error
        document = scope.document
        if document is None:
            raise SegmentDocumentNotFoundError("Document not found")
        return SegmentMutationScope(dataset=dataset, document=document)

    def _require_scope_without_actor(
        self,
        *,
        workspace_id: str,
        dataset_id: str,
        document_id: str,
    ) -> SegmentMutationScope:
        scope = self._scopes.get_segment_scope(
            dataset_ref=DatasetRef(workspace_id, dataset_id),
            document_id=document_id,
            actor_id=None,
        )
        if scope.dataset is None:
            raise SegmentDatasetNotFoundError("Dataset not found")
        if scope.document is None:
            raise SegmentDocumentNotFoundError("Document not found")
        return SegmentMutationScope(dataset=scope.dataset, document=scope.document)

    def _check_dataset_model(self, dataset: SegmentDatasetRecord) -> None:
        if dataset.indexing_technique != _HIGH_QUALITY:
            return
        try:
            self._model_guard.check(dataset)
        except SegmentModelProviderError as error:
            if error.kind == "token":
                message = f"The dataset is unavailable, due to: {error.description}"
            else:
                message = (
                    "No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider."
                )
            raise SegmentDatasetModelUnavailableError(message) from error

    def _check_embedding_model(self, dataset: SegmentDatasetRecord) -> None:
        if dataset.indexing_technique != _HIGH_QUALITY:
            return
        try:
            self._model_guard.check(dataset)
        except SegmentModelProviderError as error:
            if error.kind == "bad_request":
                message = (
                    "No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider."
                )
            else:
                message = error.description
            raise SegmentEmbeddingModelUnavailableError(message) from error


class SegmentMutationService:
    """Shared writes for Console and Service API after their own admission checks.

    Callers supply an owner chain reconstructed from authorized database rows.
    Persistence ports recheck that chain and own their transactions. Partial
    update arguments are accepted here; transport validation stays at admission.
    """

    def __init__(
        self,
        *,
        store: SegmentStore,
        index: SegmentIndex,
        uploads: SegmentUploadCatalog,
        text_hash: Callable[[str], str],
        indexing_state: SegmentIndexingState,
        state_transitions: SegmentStateTransitions,
        state_dispatcher: SegmentStateDispatcher,
    ) -> None:
        self._store = store
        self._index = index
        self._uploads = uploads
        self._text_hash = text_hash
        self._indexing_state = indexing_state
        self._state_transitions = state_transitions
        self._state_dispatcher = state_dispatcher

    def index_segments(
        self,
        document_ref: DocumentRef,
        *,
        segment_ids: Sequence[str],
        keywords_list: Sequence[Sequence[str] | None] | None = None,
    ) -> None:
        """Index a committed batch and persist failures through the scoped store.

        Callers must finish their write transaction before invoking this method.
        The index adapter reloads the complete owner chain before external I/O.
        """
        if keywords_list is not None and len(keywords_list) != len(segment_ids):
            raise ValueError("Keywords must correspond to the segment batch")
        if not segment_ids:
            return
        try:
            self._index.create_many(document_ref, segment_ids=segment_ids, keywords_list=keywords_list)
        except SegmentBatchIndexingFailureError as failure:
            logger.exception("Batch indexing failed for current members of document %s", document_ref.document_id)
            raise failure.cause from None
        except DatasetIndexingGenerationLostError:
            logger.info("Segment batch handed over to the current Dataset generation: %s", document_ref.document_id)
            return
        except Exception:
            logger.exception("Batch indexing failed for document %s", document_ref.document_id)
            raise

    def create_segments(
        self,
        scope: SegmentMutationScope,
        *,
        actor_id: str,
        args_list: Sequence[SegmentUpdateArgs],
        token_counts: Sequence[int],
    ) -> tuple[SegmentRef, ...]:
        """Persist an already-tokenized batch before any indexing I/O."""
        if len(args_list) != len(token_counts):
            raise ValueError("Token counts must correspond to the segment batch")
        now = naive_utc_now()
        rows: list[SegmentCreateData] = []
        refs: list[SegmentRef] = []
        is_qa = scope.document.doc_form == "qa_model"
        for args, tokens in zip(args_list, token_counts, strict=True):
            segment_ref = scope.document.ref.segment(str(uuid4()))
            content = args.content or ""
            answer = args.answer if is_qa else None
            rows.append(
                SegmentCreateData(
                    id=segment_ref.segment_id,
                    content=content,
                    answer=answer,
                    word_count=len(content) + len(answer or ""),
                    tokens=tokens,
                    keywords=(
                        tuple(args.keywords)
                        if args.keywords is not None
                        else None
                        if "keywords" in args.model_fields_set
                        else ()
                    ),
                    index_node_id=str(uuid4()),
                    index_node_hash=self._text_hash(content),
                    created_by=actor_id,
                    status="completed",
                    indexing_at=now,
                    completed_at=now,
                    attachment_ids=tuple(dict.fromkeys(args.attachment_ids or ())),
                )
            )
            refs.append(segment_ref)
        self._store.create_segments(scope.document.ref, rows)
        return tuple(refs)

    def create_segment(
        self,
        scope: SegmentMutationScope,
        *,
        actor_id: str,
        args: SegmentUpdateArgs,
    ) -> SegmentRef:
        now = naive_utc_now()
        segment_ref = scope.document.ref.segment(str(uuid4()))
        self._store.create_segments(
            scope.document.ref,
            (
                SegmentCreateData(
                    id=segment_ref.segment_id,
                    content=args.content or "",
                    answer=args.answer if scope.document.doc_form == "qa_model" else None,
                    word_count=len(args.content or "")
                    + (len(args.answer or "") if scope.document.doc_form == "qa_model" else 0),
                    tokens=self._index.count_tokens(scope.dataset, args.content or ""),
                    index_node_id=str(uuid4()),
                    index_node_hash=self._text_hash(args.content or ""),
                    created_by=actor_id,
                    status="completed",
                    indexing_at=now,
                    completed_at=now,
                ),
            ),
        )
        try:
            self._index.create(segment_ref, keywords=args.keywords, attachment_ids=args.attachment_ids or ())
        except DatasetIndexingGenerationLostError:
            # The committed row is part of the new generation's durable rebuild.
            # A superseded publisher must not mark that current row as failed.
            logger.info("Segment creation handed over to the current Dataset generation: %s", segment_ref.segment_id)
        except Exception as error:
            self._mark_error(segment_ref, error)
        return segment_ref

    def update_segment(
        self,
        scope: SegmentMutationScope,
        *,
        actor_id: str,
        segment_id: str,
        args: SegmentUpdateArgs,
    ) -> None:
        segment_ref = scope.document.ref.segment(segment_id)
        previous = self._store.get_segment_update_state(segment_ref)
        if previous is None:
            raise SegmentNotFoundError("Segment not found")
        if self._indexing_state.is_segment_indexing(segment_id):
            raise ValueError("Segment is indexing, please try again later")
        if args.attachment_ids is not None:
            requested_attachment_ids = tuple(dict.fromkeys(args.attachment_ids))
            existing_attachment_ids = self._uploads.get_existing_file_ids(
                workspace_id=scope.dataset.workspace_id,
                file_ids=requested_attachment_ids,
            )
            missing_attachment_ids = [
                attachment_id
                for attachment_id in requested_attachment_ids
                if attachment_id not in existing_attachment_ids
            ]
            if missing_attachment_ids:
                raise SegmentUploadFileNotFoundError(
                    f"UploadFile not found in the current workspace: {missing_attachment_ids}"
                )
        if args.model_fields_set == {"enabled"} and args.enabled is not None:
            self._request_status_change(
                scope.document.ref,
                segment_id,
                "enable" if args.enabled else "disable",
                actor_id,
            )
            return
        if args.enabled is False and previous.enabled:
            self._request_status_change(scope.document.ref, segment_id, "disable", actor_id)
            return
        if not previous.enabled and args.enabled is not True:
            raise ValueError("Can't update disabled segment")
        content = args.content or previous.content
        content_changed = content != previous.content
        is_qa = scope.document.doc_form == "qa_model"
        answer_provided = "answer" in args.model_fields_set
        answer = args.answer if answer_provided else previous.answer
        answer_changed = is_qa and answer_provided and answer != previous.answer
        text_changed = content_changed or answer_changed
        keywords_provided = "keywords" in args.model_fields_set
        keyword_changed = keywords_provided and (
            (previous.keywords is None) != (args.keywords is None)
            or (
                previous.keywords is not None
                and args.keywords is not None
                and Counter(previous.keywords) != Counter(args.keywords)
            )
        )
        keywords_for_index = args.keywords
        changes: dict[str, object] = {
            "content": content,
            "word_count": len(content) + (len(answer or "") if is_qa else 0),
        }
        if args.enabled is True:
            changes.update(enabled=True, disabled_at=None, disabled_by=None)
        if is_qa and answer_provided:
            changes["answer"] = answer
        if keywords_provided:
            changes["keywords"] = args.keywords
        try:
            if text_changed:
                now = naive_utc_now()
                changes.update(
                    index_node_hash=self._text_hash(content),
                    tokens=self._index.count_tokens(scope.dataset, content + (answer or "") if is_qa else content),
                    status="completed",
                    indexing_at=now,
                    completed_at=now,
                    updated_at=now,
                    updated_by=actor_id,
                )
            self._store.save_segment(segment_ref, changes)
            needs_body_update = scope.document.doc_form != "hierarchical_model" and (
                text_changed or args.enabled or keyword_changed
            )
            try:
                if scope.document.doc_form == "hierarchical_model":
                    if args.regenerate_child_chunks:
                        self._index.update(segment_ref, keywords=keywords_for_index, regenerate_children=True)
                elif needs_body_update:
                    # The indexing adapter owns the generation/document guard for
                    # its detached inputs and the actual vector or keyword write.
                    self._index.update(segment_ref, keywords=keywords_for_index, regenerate_children=False)
            except DatasetIndexingGenerationLostError:
                # The segment body is already committed and belongs to the new
                # Dataset generation. Continue independent business mutations,
                # especially attachment replacement, before returning success.
                logger.info("Discarded superseded segment body publisher: %s", segment_ref.segment_id)
            if scope.dataset.indexing_technique == _HIGH_QUALITY:
                try:
                    self._index.update_summary(segment_ref, summary=args.summary, content_changed=content_changed)
                except Exception:
                    logger.exception("Failed to update summary for segment %s", segment_id)
            if args.attachment_ids is not None:
                self._index.update_attachments(segment_ref, args.attachment_ids)
        except DatasetIndexingGenerationLostError:
            logger.info("Discarded superseded segment publisher: %s", segment_ref.segment_id)
        except Exception as error:
            self._mark_error(segment_ref, error)

    def _request_status_change(
        self,
        document_ref: DocumentRef,
        segment_id: str,
        action: Literal["enable", "disable"],
        actor_id: str,
    ) -> None:
        intents = self._state_transitions.request(
            document_ref=document_ref,
            segment_ids=[segment_id],
            action=action,
            actor_id=actor_id,
        )
        for intent in intents:
            try:
                self._state_dispatcher.dispatch(intent.id)
            except Exception:
                logger.exception("Failed to publish segment state intent %s", intent.id)

    def delete_segment(self, segment_ref: SegmentRef) -> None:
        segments = self._store.get_segments(segment_ref.document, (segment_ref.segment_id,))
        if not segments:
            raise SegmentNotFoundError("Segment not found")
        if self._indexing_state.is_segment_indexing(segment_ref.segment_id, deleting=True):
            raise ValueError("Segment is deleting.")
        intents = self._store.delete_segments(segment_ref.document, (segment_ref.segment_id,))
        for intent in intents:
            try:
                self._state_dispatcher.dispatch(intent.id)
            except Exception:
                logger.exception("Failed to publish segment deletion intent %s", intent.id)

    def create_child_chunk(self, segment_ref: SegmentRef, *, content: str, actor_id: str) -> ChildChunkRecord:
        with self._indexing_state.lock(f"add_child_lock_{segment_ref.segment_id}", timeout=20):
            children = self._children(segment_ref)
            child = self._new_child(
                segment_ref, content, max((c.data.position for c in children), default=0) + 1, actor_id
            )
            self._write_children(segment_ref, added=(child,))
        return child.data

    def update_child_chunks(
        self, segment_ref: SegmentRef, *, chunks: Sequence[ChildChunkUpdateArgs], actor_id: str
    ) -> tuple[ChildChunkRecord, ...]:
        try:
            # Plan from one locked snapshot. Publication happens after this
            # short transaction; the save phase accepts only the same snapshot.
            with self._store.child_mutation(segment_ref) as mutation:
                existing = mutation.get_children()
                vector_guard = mutation.vector_guard
                existing_by_id = {child.data.id: child for child in existing}
                explicit_ids = [chunk.id for chunk in chunks if chunk.id is not None]
                duplicate_ids = sorted(child_id for child_id, count in Counter(explicit_ids).items() if count > 1)
                if duplicate_ids:
                    raise ValueError(f"Duplicate child chunk IDs are not allowed: {duplicate_ids}")
                missing_ids = sorted(set(explicit_ids) - existing_by_id.keys())
                if missing_ids:
                    raise LookupError(f"Child chunks do not belong to the target segment: {missing_ids}")

                remaining = dict(existing_by_id)
                added: list[ChildChunkState] = []
                updated: list[ChildChunkState] = []
                next_position = max((child.data.position for child in existing), default=0) + 1
                for chunk in chunks:
                    if chunk.id is not None:
                        previous = remaining.pop(chunk.id)
                        if previous.data.content != chunk.content:
                            updated.append(self._updated_child(previous, chunk.content, actor_id))
                    else:
                        added.append(self._new_child(segment_ref, chunk.content, next_position + len(added), actor_id))
            intent = self._publish_and_save_children(
                segment_ref,
                expected=existing,
                expected_vector_guard=vector_guard,
                added=added,
                updated=updated,
                deleted=tuple(remaining.values()),
            )
        except Exception as error:
            raise ChildChunkIndexingApplicationError(str(error)) from error
        self._dispatch_child_intent(intent)
        return tuple(child.data for child in sorted(added + updated, key=lambda child: child.data.position))

    def delete_child_chunk(self, segment_ref: SegmentRef, *, child_chunk_id: str) -> None:
        child = self._child(segment_ref, child_chunk_id)
        try:
            self._write_children(segment_ref, deleted=(child,))
        except ChildChunkIndexingApplicationError as error:
            raise ChildChunkDeleteIndexApplicationError(str(error)) from error

    def update_child_chunk(
        self, segment_ref: SegmentRef, *, child_chunk_id: str, content: str, actor_id: str
    ) -> ChildChunkRecord:
        child = self._updated_child(self._child(segment_ref, child_chunk_id), content, actor_id)
        self._write_children(segment_ref, updated=(child,))
        return child.data

    def _mark_error(self, segment_ref: SegmentRef, error: Exception) -> None:
        logger.exception("Segment indexing failed: %s", segment_ref.segment_id)
        self._store.save_segment(
            segment_ref, {"enabled": False, "disabled_at": naive_utc_now(), "status": "error", "error": str(error)}
        )

    def _children(self, segment_ref: SegmentRef) -> tuple[ChildChunkState, ...]:
        children = self._store.get_children(segment_ref)
        if children is None:
            raise SegmentNotFoundError("Segment not found")
        return children

    def _child(self, segment_ref: SegmentRef, child_id: str) -> ChildChunkState:
        for child in self._children(segment_ref):
            if child.data.id == child_id:
                return child
        raise ChildChunkNotFoundError("Child chunk not found")

    def _new_child(self, segment_ref: SegmentRef, content: str, position: int, actor_id: str) -> ChildChunkState:
        now = naive_utc_now()
        return ChildChunkState(
            data=ChildChunkRecord(
                id=str(uuid4()),
                segment_id=segment_ref.segment_id,
                content=content,
                position=position,
                word_count=len(content),
                type="customized",
                created_at=now,
                updated_at=now,
            ),
            index_node_id=str(uuid4()),
            index_node_hash=self._text_hash(content),
            created_by=actor_id,
        )

    def _updated_child(self, child: ChildChunkState, content: str, actor_id: str) -> ChildChunkState:
        return replace(
            child,
            data=child.data.model_copy(
                update={
                    "content": content,
                    "word_count": len(content),
                    "type": "customized",
                    "updated_at": naive_utc_now(),
                }
            ),
            index_node_id=str(uuid4()),
            index_node_hash=self._text_hash(content),
            updated_by=actor_id,
            source_index_node_id=child.index_node_id,
        )

    def _write_children(
        self,
        segment_ref: SegmentRef,
        *,
        added: Sequence[ChildChunkState] = (),
        updated: Sequence[ChildChunkState] = (),
        deleted: Sequence[ChildChunkState] = (),
    ) -> None:
        try:
            with self._store.child_mutation(segment_ref) as mutation:
                expected = mutation.get_children()
                expected_vector_guard = mutation.vector_guard
                self._validate_child_plan(expected, added=added, updated=updated, deleted=deleted)
            intent = self._publish_and_save_children(
                segment_ref,
                expected=expected,
                expected_vector_guard=expected_vector_guard,
                added=added,
                updated=updated,
                deleted=deleted,
            )
        except Exception as error:
            raise ChildChunkIndexingApplicationError(str(error)) from error
        self._dispatch_child_intent(intent)

    def _publish_and_save_children(
        self,
        segment_ref: SegmentRef,
        *,
        expected: Sequence[ChildChunkState],
        expected_vector_guard: ChildVectorGuard,
        added: Sequence[ChildChunkState] = (),
        updated: Sequence[ChildChunkState] = (),
        deleted: Sequence[ChildChunkState] = (),
    ) -> SegmentStateIntent:
        prepared = self._index.prepare_child_publication(
            segment_ref,
            guard=expected_vector_guard,
            added=added,
            updated=updated,
            deleted=deleted,
        )
        publication: ChildVectorPublication | None = None
        try:
            with self._store.child_mutation(segment_ref) as mutation:
                mutation.require_current(
                    expected=expected,
                    expected_vector_guard=expected_vector_guard,
                )
                with prepared.submit() as publication:
                    return mutation.save(
                        expected=expected,
                        expected_vector_guard=expected_vector_guard,
                        publication=publication,
                        added=added,
                        updated=updated,
                        deleted=deleted,
                    )
        except Exception:
            if publication is not None:
                self._index.abandon_child_publication(publication)
            else:
                prepared.discard()
            raise

    @staticmethod
    def _validate_child_plan(
        expected: Sequence[ChildChunkState],
        *,
        added: Sequence[ChildChunkState],
        updated: Sequence[ChildChunkState],
        deleted: Sequence[ChildChunkState],
    ) -> None:
        expected_by_id = {child.data.id: child for child in expected}
        requested_ids = [child.data.id for child in (*added, *updated, *deleted)]
        if len(set(requested_ids)) != len(requested_ids):
            raise RuntimeError("Child mutation contains duplicate IDs")
        if any(child.data.id in expected_by_id for child in added):
            raise RuntimeError("Child mutation reuses an existing ID")
        for child in updated:
            current = expected_by_id.get(child.data.id)
            if current is None or current.index_node_id != child.source_index_node_id:
                raise RuntimeError("Child chunk changed before vector publication")
        for child in deleted:
            if expected_by_id.get(child.data.id) != child:
                raise RuntimeError("Child chunk changed before deletion")

    def _dispatch_child_intent(self, intent: SegmentStateIntent) -> None:
        try:
            self._state_dispatcher.dispatch(intent.id)
        except Exception:
            logger.exception("Failed to publish segment child reconciliation intent %s", intent.id)


@dataclass(frozen=True, slots=True)
class SegmentMutationScope:
    """Authorized dataset and document values for shared mutation policy."""

    dataset: SegmentDatasetRecord
    document: SegmentDocumentRecord

    def __post_init__(self) -> None:
        if self.document.dataset_id != self.dataset.id or self.document.workspace_id != self.dataset.workspace_id:
            raise ValueError("Document does not belong to the dataset")


def validate_segment_values(values: Mapping[str, object], *, doc_form: str, attachment_limit: int) -> SegmentUpdateArgs:
    answer = values.get("answer")
    if doc_form == "qa_model":
        if not answer:
            raise ValueError("Answer is required")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer is empty")
    content = values.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content is empty")
    attachments = values.get("attachment_ids")
    if attachments:
        if not isinstance(attachments, list):
            raise ValueError("Attachment IDs is invalid")
        if len(attachments) > attachment_limit:
            raise ValueError(f"Exceeded maximum attachment limit of {attachment_limit}")
    return SegmentUpdateArgs.model_validate(values)

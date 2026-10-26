"""Indexing adapters that materialize database inputs before external I/O."""

import logging
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from threading import Event, Thread
from typing import Never
from uuid import uuid4

from sqlalchemy.orm import Session

from core.credit_usage import CreditUsageCreatedBy
from core.model_context import with_credit_usage_created_by
from core.model_manager import ModelManager
from core.rag.datasource.keyword.jieba.jieba import Jieba
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.entities import Rule
from core.rag.index_processor.constant.doc_type import DocType
from core.rag.index_processor.processor.paragraph_index_processor import ParagraphIndexProcessor
from core.rag.index_processor.processor.parent_child_index_processor import ParentChildIndexProcessor
from core.rag.models.document import Document
from graphon.model_runtime.entities.model_entities import ModelType
from libs.datetime_utils import naive_utc_now
from models.enums import SegmentType
from repositories.knowledge.attachment_vector_repository import (
    SQLAlchemyAttachmentVectorStore,
    delete_unreferenced_attachment_vectors,
    publish_bound_attachment_vectors,
)
from repositories.knowledge.child_vector_cleanup_repository import SQLAlchemyChildVectorCleanupRepository
from repositories.knowledge.segment_cleanup_repository import lock_segment_attachment_cleanup_dataset
from repositories.knowledge.segment_repository import SegmentIndexingSnapshot, SQLAlchemySegmentRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.entities.resource_scope import DocumentRef, SegmentRef
from services.knowledge.entities.segments import (
    ChildChunkRecord,
    ChildVectorCleanupClaim,
    ChildVectorGuard,
    ChildVectorPublication,
    ChildVectorRoute,
    ChildVectorWriteIntent,
)
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.errors import DatasetIndexingGenerationLostError
from services.knowledge.segments.adapters import RedisSegmentClient
from services.knowledge.segments.application import (
    ChildChunkState,
    SegmentBatchIndexingFailureError,
    SegmentDatasetRecord,
)
from services.knowledge.summaries.adapters import SummaryIndexAdapter

logger = logging.getLogger(__name__)

_CHILD_WRITER_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


@contextmanager
def _maintain_child_writer_claims(
    repository: SQLAlchemyChildVectorCleanupRepository,
    intents: Sequence[ChildVectorWriteIntent],
    *,
    interval_seconds: float = _CHILD_WRITER_HEARTBEAT_SECONDS,
):
    if interval_seconds <= 0:
        raise ValueError("child vector writer heartbeat interval must be positive")
    if not intents:
        yield
        return
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                repository.renew_writes(intents)
            except Exception:
                logger.exception("Child vector writer lost ownership, cleanup_ids=%s", [item.id for item in intents])
                return

    thread = Thread(target=heartbeat, name=f"child-vector-heartbeat-{intents[0].id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning(
                "Child vector writer heartbeat did not stop promptly, cleanup_ids=%s", [i.id for i in intents]
            )


@contextmanager
def _maintain_child_publication_claims(
    repository: SQLAlchemyChildVectorCleanupRepository,
    claims: Sequence[ChildVectorCleanupClaim],
    *,
    interval_seconds: float = _CHILD_WRITER_HEARTBEAT_SECONDS,
) -> Generator[None]:
    if interval_seconds <= 0:
        raise ValueError("child vector publication heartbeat interval must be positive")
    if not claims:
        yield
        return
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                repository.renew_publication(claims)
            except Exception:
                logger.exception(
                    "Child vector publication lost ownership, cleanup_ids=%s", [item.id for item in claims]
                )
                return

    thread = Thread(target=heartbeat, name=f"child-publication-heartbeat-{claims[0].id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning(
                "Child vector publication heartbeat did not stop promptly, cleanup_ids=%s", [item.id for item in claims]
            )


class _PreparedChildPublication:
    """Registered child writers submitted while the caller holds mutation locks."""

    def __init__(
        self,
        *,
        repository: SQLAlchemyChildVectorCleanupRepository,
        intents: Sequence[ChildVectorWriteIntent],
        write: Callable[[Callable[[], None]], None],
    ) -> None:
        self._repository = repository
        self._intents = tuple(intents)
        self._write = write
        self._entered = False
        self._discarded = False
        self._write_started = False

    def discard(self) -> None:
        if self._entered or self._discarded:
            return
        self._repository.discard_unsubmitted_writes(self._intents)
        self._discarded = True

    @contextmanager
    def submit(self) -> Generator[ChildVectorPublication]:
        """Cross the vector boundary under the caller's Dataset/owner locks."""
        if self._entered or self._discarded:
            raise RuntimeError("Child vector publication was already consumed")
        self._entered = True
        if not self._intents:
            yield ChildVectorPublication()
            return

        def mark_write_started() -> None:
            self._write_started = True

        try:
            with _maintain_child_writer_claims(self._repository, self._intents):
                self._write(mark_write_started)
            publication = ChildVectorPublication(self._repository.finish_writes(self._intents))
            with _maintain_child_publication_claims(self._repository, publication.claims):
                yield publication
        except Exception:
            if not self._write_started:
                self._repository.discard_unsubmitted_writes(self._intents)
                self._discarded = True
            raise


class SegmentIndexingGateway:
    def __init__(
        self,
        *,
        segments: SQLAlchemySegmentRepository,
        uploads: SQLAlchemyKnowledgeUploadRepository,
        attachment_vectors: SQLAlchemyAttachmentVectorStore,
        child_vector_cleanups: SQLAlchemyChildVectorCleanupRepository,
        redis: RedisSegmentClient,
        new_session: Callable[[], Session],
    ) -> None:
        self._segments = segments
        self._uploads = uploads
        self._attachment_vectors = attachment_vectors
        self._child_vector_cleanups = child_vector_cleanups
        self._redis = redis
        self._new_session = new_session

    def count_tokens(self, dataset: SegmentDatasetRecord, text: str) -> int:
        if dataset.indexing_technique != "high_quality":
            return 0
        model = ModelManager.for_tenant(tenant_id=dataset.workspace_id).get_model_instance(
            tenant_id=dataset.workspace_id,
            provider=dataset.embedding_model_provider or "",
            model_type=ModelType.TEXT_EMBEDDING,
            model=dataset.embedding_model or "",
        )
        return model.get_text_embedding_num_tokens(texts=[text])[0]

    @staticmethod
    def _vector(snapshot: SegmentIndexingSnapshot) -> Vector:
        return Vector(snapshot.dataset, session=None, vector_type=snapshot.vector_type)

    def _record_vector_route_before_write(self, snapshot: SegmentIndexingSnapshot) -> Vector:
        """Commit an exact segment-snapshot receipt before external vector I/O."""
        vector = self._vector(snapshot)
        if snapshot.dataset.indexing_request_id is None:
            return vector
        route = self._vector_route(snapshot, vector)
        self._segments.record_indexing_route(snapshot, route)
        return vector

    @staticmethod
    def _vector_route(snapshot: SegmentIndexingSnapshot, vector: Vector) -> ChildVectorRoute:
        vector_type, collection_name = vector.get_routing_target()
        return ChildVectorRoute.capture(
            tenant_id=snapshot.dataset.tenant_id,
            dataset_id=snapshot.dataset.id,
            vector_type=vector_type,
            collection_name=collection_name,
            routing_index_struct=snapshot.dataset.index_struct,
            collection_binding_id=snapshot.dataset.collection_binding_id,
            indexing_technique=str(snapshot.dataset.indexing_technique),
            doc_form=str(snapshot.document.doc_form),
        )

    @staticmethod
    def _document(snapshot: SegmentIndexingSnapshot) -> Document:
        segment = snapshot.segment
        return Document(
            page_content=segment.content,
            metadata={
                "doc_id": segment.index_node_id,
                "doc_hash": segment.index_node_hash,
                "document_id": segment.document_id,
                "dataset_id": segment.dataset_id,
                "doc_type": DocType.TEXT,
            },
        )

    @contextmanager
    def _attachment_publication_guard(self, snapshot: SegmentIndexingSnapshot, session: Session):
        self._segments.require_indexing_snapshot(session, snapshot)
        yield

    @contextmanager
    def _attachment_batch_publication_guard(
        self,
        document_ref: DocumentRef,
        snapshots: Sequence[SegmentIndexingSnapshot],
        session: Session,
    ):
        self._segments.require_any_indexing_snapshot(session, document_ref, snapshots)
        yield

    @staticmethod
    def _child_vector_guard(snapshot: SegmentIndexingSnapshot) -> ChildVectorGuard:
        dataset = snapshot.dataset
        return ChildVectorGuard(
            required=snapshot.segment.enabled and dataset.indexing_technique == "high_quality",
            generation_id=dataset.indexing_request_id,
            indexing_technique=dataset.indexing_technique,
            embedding_model_provider=dataset.embedding_model_provider,
            embedding_model=dataset.embedding_model,
            index_struct=dataset.index_struct,
            collection_binding_id=dataset.collection_binding_id,
        )

    def _raise_batch_failure(
        self,
        document_ref: DocumentRef,
        snapshots: Sequence[SegmentIndexingSnapshot],
        error: Exception,
    ) -> Never:
        try:
            self._segments.mark_batch_indexing_failed(document_ref, snapshots, error)
        except Exception:
            logger.exception("Failed to persist exact segment batch failures for document %s", document_ref.document_id)
        raise SegmentBatchIndexingFailureError(error) from error

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def create_many(
        self,
        document_ref: DocumentRef,
        *,
        segment_ids: Sequence[str],
        keywords_list: Sequence[Sequence[str] | None] | None,
    ) -> None:
        """Materialize a scoped batch, then publish only members whose snapshots remain exact."""
        snapshots = self._segments.get_indexing_snapshots(document_ref, segment_ids)
        if not snapshots:
            return
        failure_snapshots = snapshots
        try:
            first = snapshots[0]
            vector = None
            if first.dataset.indexing_technique == "high_quality":
                vector = self._vector(first)
                route = self._vector_route(first, vector)
                snapshots = self._segments.record_batch_indexing_route(document_ref, snapshots, route)
                failure_snapshots = snapshots
                if not snapshots:
                    return
                first = snapshots[0]
            if first.document.doc_form == "hierarchical_model":
                for snapshot in snapshots:
                    failure_snapshots = (snapshot,)
                    self._regenerate_children(
                        document_ref.segment(snapshot.segment.id), snapshot, replace_existing=False
                    )
                failure_snapshots = snapshots
            else:
                with self._segments.lock_index_batch_write(document_ref, expected_snapshots=snapshots) as publication:
                    current, publication_session = publication
                    if not current:
                        return
                    failure_snapshots = current
                    documents = [self._document(snapshot) for snapshot in current]
                    if first.dataset.indexing_technique == "high_quality":
                        assert vector is not None
                        vector.create(documents)
                    elif first.document.doc_form != "qa_model":
                        keywords_by_segment = (
                            dict(zip(segment_ids, keywords_list, strict=True)) if keywords_list is not None else None
                        )
                        current_keywords = (
                            [keywords_by_segment[snapshot.segment.id] for snapshot in current]
                            if keywords_by_segment is not None
                            else None
                        )
                        self._write_keywords(
                            document_ref,
                            current[0],
                            documents,
                            current_keywords,
                            replace_existing=False,
                            persistence_session=publication_session,
                        )
                    snapshots = current
                failure_snapshots = current
            if first.dataset.is_multimodal:
                attachment_ids = tuple(
                    dict.fromkeys(attachment_id for snapshot in snapshots for attachment_id in snapshot.attachment_ids)
                )
                files = self._uploads.get_files(
                    workspace_id=document_ref.dataset.tenant_id,
                    file_ids=attachment_ids,
                )
                attachments = self._attachment_documents(
                    first, [files[file_id] for file_id in attachment_ids if file_id in files]
                )
                if attachments and first.dataset.indexing_technique == "high_quality":
                    assert vector is not None

                    actual_vector_type, collection_name = vector.get_routing_target()
                    attachment_route = VectorRoute.capture(
                        tenant_id=first.dataset.tenant_id,
                        dataset_id=first.dataset.id,
                        vector_type=actual_vector_type,
                        collection_name=collection_name,
                        routing_index_struct=first.dataset.index_struct,
                        collection_binding_id=first.dataset.collection_binding_id,
                        indexing_technique=str(first.dataset.indexing_technique),
                        doc_form="attachment",
                    )

                    def publish(
                        bound_refs: frozenset[tuple[str, str]],
                        before_write: Callable[[Sequence[str]], None],
                        after_write: Callable[[Sequence[str]], None],
                    ) -> None:
                        bound_ids = {attachment_id for _, attachment_id in bound_refs}
                        bound_documents = [item for item in attachments if str(item.metadata["doc_id"]) in bound_ids]
                        if bound_documents:
                            vector.create_multimodal(
                                bound_documents,
                                upload_files={file_id: file for file_id, file in files.items() if file_id in bound_ids},
                                before_write=before_write,
                                after_write=after_write,
                            )

                    with self._new_session() as session, session.begin():
                        publish_bound_attachment_vectors(
                            session,
                            tenant_id=first.dataset.tenant_id,
                            dataset_id=first.dataset.id,
                            attachment_refs=[(first.document.id, attachment_id) for attachment_id in attachment_ids],
                            route=attachment_route,
                            publish=publish,
                            store=self._attachment_vectors,
                            mutation_guard=lambda session: self._attachment_batch_publication_guard(
                                document_ref,
                                snapshots,
                                session,
                            ),
                        )
        except DatasetIndexingGenerationLostError:
            raise
        except SegmentBatchIndexingFailureError:
            raise
        except Exception as error:
            self._raise_batch_failure(document_ref, failure_snapshots, error)

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def create(self, segment_ref: SegmentRef, *, keywords: Sequence[str] | None, attachment_ids: Sequence[str]) -> None:
        files = self._uploads.get_files(workspace_id=segment_ref.document.dataset.tenant_id, file_ids=attachment_ids)
        self._segments.replace_attachments(segment_ref, tuple(files))
        snapshot = self._segments.get_indexing_snapshot(segment_ref)
        vector = (
            self._record_vector_route_before_write(snapshot)
            if snapshot.dataset.indexing_technique == "high_quality" and snapshot.segment.enabled
            else None
        )
        if snapshot.document.doc_form == "hierarchical_model":
            self._regenerate_children(segment_ref, snapshot, replace_existing=False)
        else:
            with self._segments.lock_index_write(
                segment_ref,
                lock_segment=snapshot.dataset.indexing_technique == "high_quality",
                expected_snapshot=snapshot,
            ) as enabled:
                if not enabled:
                    return
                if snapshot.dataset.indexing_technique == "high_quality":
                    assert vector is not None
                    vector.create([self._document(snapshot)])
                else:
                    self._update_keywords(segment_ref, snapshot, keywords, replace_existing=False)
        if files and snapshot.dataset.is_multimodal:
            assert vector is not None
            attachments = self._attachment_documents(snapshot, files.values())

            actual_vector_type, collection_name = vector.get_routing_target()
            attachment_route = VectorRoute.capture(
                tenant_id=snapshot.dataset.tenant_id,
                dataset_id=snapshot.dataset.id,
                vector_type=actual_vector_type,
                collection_name=collection_name,
                routing_index_struct=snapshot.dataset.index_struct,
                collection_binding_id=snapshot.dataset.collection_binding_id,
                indexing_technique=str(snapshot.dataset.indexing_technique),
                doc_form="attachment",
            )

            def publish(
                bound_refs: frozenset[tuple[str, str]],
                before_write: Callable[[Sequence[str]], None],
                after_write: Callable[[Sequence[str]], None],
            ) -> None:
                bound_ids = {attachment_id for _, attachment_id in bound_refs}
                bound_documents = [item for item in attachments if str(item.metadata["doc_id"]) in bound_ids]
                if bound_documents:
                    vector.create_multimodal(
                        bound_documents,
                        upload_files={file_id: file for file_id, file in files.items() if file_id in bound_ids},
                        before_write=before_write,
                        after_write=after_write,
                    )

            with self._new_session() as session, session.begin():
                publish_bound_attachment_vectors(
                    session,
                    tenant_id=snapshot.dataset.tenant_id,
                    dataset_id=snapshot.dataset.id,
                    attachment_refs=[(snapshot.document.id, attachment_id) for attachment_id in files],
                    route=attachment_route,
                    publish=publish,
                    store=self._attachment_vectors,
                    mutation_guard=lambda session: self._attachment_publication_guard(snapshot, session),
                )

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def update(self, segment_ref: SegmentRef, *, keywords: Sequence[str] | None, regenerate_children: bool) -> None:
        snapshot = self._segments.get_indexing_snapshot(segment_ref)
        vector = (
            self._record_vector_route_before_write(snapshot)
            if snapshot.dataset.indexing_technique == "high_quality" and snapshot.segment.enabled
            else None
        )
        if regenerate_children:
            self._regenerate_children(segment_ref, snapshot, replace_existing=True)
        else:
            with self._segments.lock_index_write(
                segment_ref,
                lock_segment=snapshot.dataset.indexing_technique == "high_quality",
                expected_snapshot=snapshot,
            ) as enabled:
                if not enabled:
                    return
                if snapshot.dataset.indexing_technique == "high_quality":
                    assert vector is not None
                    assert snapshot.segment.index_node_id is not None
                    vector.delete_by_ids([snapshot.segment.index_node_id])
                    vector.add_texts([self._document(snapshot)], duplicate_check=True)
                else:
                    self._update_keywords(segment_ref, snapshot, keywords, replace_existing=True)

    def _regenerate_children(
        self, segment_ref: SegmentRef, snapshot: SegmentIndexingSnapshot, *, replace_existing: bool
    ) -> None:
        if snapshot.dataset.indexing_technique != "high_quality":
            raise ValueError("The knowledge base index technique is not high quality!")
        rule = snapshot.process_rule
        if rule is None:
            if replace_existing:
                return
            raise ValueError("No processing rule found.")
        manager = ModelManager.for_tenant(tenant_id=snapshot.dataset.tenant_id)
        model = (
            manager.get_model_instance(
                tenant_id=snapshot.dataset.tenant_id,
                provider=snapshot.dataset.embedding_model_provider,
                model_type=ModelType.TEXT_EMBEDDING,
                model=snapshot.dataset.embedding_model,
            )
            if snapshot.dataset.embedding_model_provider
            else manager.get_default_model_instance(
                tenant_id=snapshot.dataset.tenant_id, model_type=ModelType.TEXT_EMBEDDING
            )
        )
        children = ParentChildIndexProcessor().split_child_nodes(
            self._document(snapshot), Rule.model_validate(rule.to_dict()["rules"]), rule.mode, model
        )
        now = naive_utc_now()
        added = tuple(
            ChildChunkState(
                data=ChildChunkRecord(
                    id=str(uuid4()),
                    segment_id=segment_ref.segment_id,
                    content=child.page_content,
                    position=position,
                    word_count=len(child.page_content),
                    type=SegmentType.AUTOMATIC,
                    created_at=now,
                    updated_at=now,
                ),
                index_node_id=child.metadata["doc_id"],
                index_node_hash=child.metadata["doc_hash"],
                created_by=snapshot.document.created_by,
            )
            for position, child in enumerate(children, start=1)
        )
        with self._segments.child_mutation(segment_ref) as mutation:
            if (
                not mutation.enabled
                or mutation.vector_guard != self._child_vector_guard(snapshot)
                or not mutation.is_current_regeneration(
                    index_node_hash=snapshot.segment.index_node_hash,
                    process_rule_id=rule.id,
                    process_rule_mode=str(rule.mode),
                    process_rule_rules=rule.rules,
                )
            ):
                return
            previous = mutation.get_children()
            vector_guard = mutation.vector_guard
        deleted = previous if replace_existing else ()
        prepared = self.prepare_child_publication(
            segment_ref,
            guard=vector_guard,
            added=added,
            deleted=deleted,
        )
        publication: ChildVectorPublication | None = None
        try:
            stale = False
            with self._segments.child_mutation(segment_ref) as mutation:
                if not mutation.enabled or not mutation.is_current_regeneration(
                    index_node_hash=snapshot.segment.index_node_hash,
                    process_rule_id=rule.id,
                    process_rule_mode=str(rule.mode),
                    process_rule_rules=rule.rules,
                ):
                    stale = True
                else:
                    mutation.require_current(
                        expected=previous,
                        expected_vector_guard=vector_guard,
                    )
                    with prepared.submit() as publication:
                        mutation.save(
                            expected=previous,
                            expected_vector_guard=vector_guard,
                            publication=publication,
                            added=added,
                            deleted=deleted,
                        )
            if stale:
                prepared.discard()
                return
        except Exception:
            if publication is not None:
                self.abandon_child_publication(publication)
            else:
                prepared.discard()
            raise

    def prepare_child_publication(
        self,
        segment_ref: SegmentRef,
        *,
        guard: ChildVectorGuard,
        added: Sequence[ChildChunkState] = (),
        updated: Sequence[ChildChunkState] = (),
        deleted: Sequence[ChildChunkState] = (),
    ) -> _PreparedChildPublication:
        """Register fresh nodes before the caller locks and publishes its mutation."""
        del deleted
        snapshot = self._segments.get_indexing_snapshot(segment_ref)
        if self._child_vector_guard(snapshot) != guard:
            raise RuntimeError("Child vector dataset generation changed before publication")
        if not guard.required:
            return _PreparedChildPublication(
                repository=self._child_vector_cleanups,
                intents=(),
                write=lambda _before_write: None,
            )
        documents = [
            Document(
                page_content=child.data.content,
                metadata={
                    "doc_id": child.index_node_id,
                    "doc_hash": child.index_node_hash,
                    "document_id": segment_ref.document.document_id,
                    "dataset_id": segment_ref.document.dataset.dataset_id,
                },
            )
            for child in (*added, *updated)
        ]
        if not documents:
            return _PreparedChildPublication(
                repository=self._child_vector_cleanups,
                intents=(),
                write=lambda _before_write: None,
            )
        targets = []
        for child in (*added, *updated):
            if child.index_node_id is None:
                raise ValueError("Child vector target has no node ID")
            targets.append((child.data.id, child.index_node_id))
        # Vector construction is before durable registration and therefore a
        # definite no-submit failure. Once registered, retain unknown writes.
        vector = self._vector(snapshot)
        vector_type, collection_name = vector.get_routing_target()
        route = ChildVectorRoute.capture(
            tenant_id=snapshot.dataset.tenant_id,
            dataset_id=snapshot.dataset.id,
            vector_type=vector_type,
            collection_name=collection_name,
            routing_index_struct=snapshot.dataset.index_struct,
            collection_binding_id=snapshot.dataset.collection_binding_id,
            indexing_technique=str(snapshot.dataset.indexing_technique),
            doc_form=str(snapshot.document.doc_form),
        )
        intents = self._child_vector_cleanups.begin_writes(segment_ref, guard, targets, route)
        return _PreparedChildPublication(
            repository=self._child_vector_cleanups,
            intents=intents,
            write=lambda before_write: self._write_child_vectors(vector, documents, before_write),
        )

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def _write_child_vectors(
        self,
        vector: Vector,
        documents: list[Document],
        before_write: Callable[[], None],
    ) -> None:
        vector.add_texts(documents, duplicate_check=True, before_write=before_write)

    def abandon_child_publication(self, publication: ChildVectorPublication) -> None:
        self._child_vector_cleanups.release_publication(publication.claims)

    @staticmethod
    def _attachment_documents(snapshot: SegmentIndexingSnapshot, files) -> list[Document]:
        return [
            Document(
                page_content=file.name,
                metadata={
                    "doc_id": file.id,
                    "doc_hash": "",
                    "dataset_id": snapshot.dataset.id,
                    "document_id": snapshot.document.id,
                    "doc_type": DocType.IMAGE,
                },
            )
            for file in files
        ]

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def update_attachments(self, segment_ref: SegmentRef, attachment_ids: Sequence[str]) -> None:
        snapshot = self._segments.get_indexing_snapshot(segment_ref)
        requested_ids = tuple(dict.fromkeys(attachment_ids))
        files = self._uploads.get_files(workspace_id=snapshot.dataset.tenant_id, file_ids=requested_ids)
        missing_ids = [file_id for file_id in requested_ids if file_id not in files]
        if missing_ids:
            raise LookupError(f"Attachment files do not exist in the current workspace: {missing_ids}")
        # Pass the complete request to the transactional repository. Besides
        # preserving explicit [] replacement, this repeats ownership/liveness
        # validation under the binding locks if a file disappears after this read.
        replacement = self._segments.replace_attachments(segment_ref, requested_ids)
        if not replacement.changed:
            return
        snapshot = self._segments.get_indexing_snapshot(segment_ref)
        # Bindings are business state in every indexing mode. Only the physical
        # attachment-vector phase depends on high-quality indexing.
        if snapshot.dataset.indexing_technique != "high_quality":
            return
        if snapshot.dataset.is_multimodal:
            vector = self._record_vector_route_before_write(snapshot)
            attachments = self._attachment_documents(snapshot, [files[file_id] for file_id in requested_ids])

            actual_vector_type, collection_name = vector.get_routing_target()
            attachment_route = VectorRoute.capture(
                tenant_id=snapshot.dataset.tenant_id,
                dataset_id=snapshot.dataset.id,
                vector_type=actual_vector_type,
                collection_name=collection_name,
                routing_index_struct=snapshot.dataset.index_struct,
                collection_binding_id=snapshot.dataset.collection_binding_id,
                indexing_technique=str(snapshot.dataset.indexing_technique),
                doc_form="attachment",
            )

            def publish(
                bound_refs: frozenset[tuple[str, str]],
                before_write: Callable[[Sequence[str]], None],
                after_write: Callable[[Sequence[str]], None],
            ) -> None:
                bound_ids = {attachment_id for _, attachment_id in bound_refs}
                bound_documents = [item for item in attachments if str(item.metadata["doc_id"]) in bound_ids]
                if bound_documents:
                    vector.create_multimodal(
                        bound_documents,
                        upload_files={file_id: file for file_id, file in files.items() if file_id in bound_ids},
                        before_write=before_write,
                        after_write=after_write,
                    )

            with self._new_session() as session, session.begin():
                # The UPDATE lock covers both sides of this replacement: stale
                # publications are skipped and globally live old vectors remain.
                lock_segment_attachment_cleanup_dataset(
                    session,
                    tenant_id=snapshot.dataset.tenant_id,
                    dataset_id=snapshot.dataset.id,
                )
                publish_bound_attachment_vectors(
                    session,
                    tenant_id=snapshot.dataset.tenant_id,
                    dataset_id=snapshot.dataset.id,
                    attachment_refs=[(snapshot.document.id, attachment_id) for attachment_id in requested_ids],
                    route=attachment_route,
                    publish=publish,
                    store=self._attachment_vectors,
                    mutation_guard=lambda session: self._attachment_publication_guard(snapshot, session),
                )
                delete_unreferenced_attachment_vectors(
                    session,
                    tenant_id=snapshot.dataset.tenant_id,
                    dataset_id=snapshot.dataset.id,
                    candidate_attachment_ids=replacement.removed_ids,
                    route=attachment_route,
                )

    def update_summary(self, segment_ref: SegmentRef, *, summary: str | None, content_changed: bool) -> None:
        if summary is None and not content_changed:
            return
        snapshot = self._segments.get_indexing_snapshot(segment_ref)
        if snapshot.dataset.indexing_technique != "high_quality" or snapshot.document.doc_form == "qa_model":
            return
        setting = snapshot.dataset.summary_index_setting
        previous = snapshot.summary
        expected_summary = (previous.id, previous.version) if previous is not None else None
        expected_hash = snapshot.segment.index_node_hash
        if summary is None or (previous is not None and summary == previous.summary_content):
            if not (content_changed and previous and setting and setting.get("enable") is True):
                return
            try:
                pending = self._segments.save_summary(segment_ref, previous.summary_content or "").summary
                assert pending is not None
                expected_summary = (pending.id, pending.version)
                summary, _ = ParagraphIndexProcessor.generate_summary_from_inputs(
                    snapshot.dataset.tenant_id,
                    snapshot.segment.content,
                    setting,
                    document_language=snapshot.document.doc_language,
                    image_loader=lambda: self._segments.get_summary_images(segment_ref),
                )
                if not summary.strip():
                    raise ValueError("Generated summary is empty")
            except Exception as error:
                with self._segments.lock_index_write(segment_ref, lock_segment=False) as enabled:
                    current = self._segments.get_indexing_snapshot(segment_ref)
                    current_summary = current.summary
                    if (
                        enabled
                        and current.segment.index_node_hash == expected_hash
                        and current_summary is not None
                        and expected_summary == (current_summary.id, current_summary.version)
                    ):
                        self._segments.save_summary(segment_ref, previous.summary_content or "", error=str(error))
                raise
        record = None
        current = None
        with self._segments.lock_index_write(segment_ref, lock_segment=False) as enabled:
            current = self._segments.get_indexing_snapshot(segment_ref)
            current_summary = current.summary
            if not enabled or current.segment.index_node_hash != expected_hash:
                return
            if expected_summary is None:
                if current_summary is not None:
                    return
            elif current_summary is None or expected_summary != (current_summary.id, current_summary.version):
                return
            if not summary.strip():
                self._segments.save_summary(segment_ref, None)
                return
            write = self._segments.save_summary(segment_ref, summary)
            record = write.summary
            assert record is not None
        assert record is not None
        assert current is not None
        # Model resolution, retries, and remote vector I/O must not hold the
        # document lock. Publication remains safe: save_summary advanced the
        # exact summary version while locked, and the summary repository rejects
        # completion if a later edit/disable advances it again.
        if not record.enabled and write.inherited_enable_intent_id is None:
            return
        SummaryIndexAdapter.vectorize_summary(
            record,
            current.segment,
            current.dataset,
            new_session=self._new_session,
            vector_factory=lambda: self._vector(current),
            indexing_request_id=current.dataset.indexing_request_id,
            expected_segment_state_intent_id=write.inherited_enable_intent_id,
        )

    def _update_keywords(
        self,
        segment_ref: SegmentRef,
        snapshot: SegmentIndexingSnapshot,
        keywords: Sequence[str] | None,
        *,
        replace_existing: bool,
    ) -> None:
        assert snapshot.segment.index_node_id is not None
        self._write_keywords(
            segment_ref.document, snapshot, [self._document(snapshot)], [keywords], replace_existing=replace_existing
        )

    def _write_keywords(
        self,
        document_ref: DocumentRef,
        snapshot: SegmentIndexingSnapshot,
        documents: list[Document],
        keywords_list: Sequence[Sequence[str] | None] | None,
        *,
        replace_existing: bool,
        persistence_session: Session | None = None,
    ) -> None:
        dataset_ref = document_ref.dataset

        def persist(storage_type: str, data: str, selected: Mapping[str, Sequence[str]]) -> None:
            if persistence_session is None:
                self._segments.save_keyword_table(dataset_ref, storage_type=storage_type, data=data, keywords=selected)
            else:
                self._segments.save_keyword_table_in_session(
                    persistence_session,
                    dataset_ref,
                    storage_type=storage_type,
                    data=data,
                    keywords=selected,
                )

        Jieba(snapshot.dataset).update_texts(
            documents,
            read=lambda: self._segments.get_keyword_table(dataset_ref),
            write=persist,
            lock=self._redis.lock(f"keyword_indexing_lock_{snapshot.dataset.id}", timeout=600),
            cleanup_session_factory=self._new_session,
            keywords_list=keywords_list,
            replace_existing=replace_existing,
        )

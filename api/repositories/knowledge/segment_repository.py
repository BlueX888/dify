"""Infrastructure adapters for dataset segment use cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from sqlalchemy import String, case, delete, func, literal, or_, select, tuple_, update
from sqlalchemy import cast as sql_cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, sessionmaker

from configs import dify_config
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.docstore.dataset_docstore import DatasetDocumentStore
from core.rag.index_processor.constant.index_type import IndexStructureType
from core.rag.index_processor.processor.paragraph_index_processor import ParagraphIndexProcessor
from core.rag.models.document import ChildDocument
from core.rag.models.document import Document as IndexDocument
from graphon.file import File
from libs.datetime_utils import naive_utc_now
from libs.helper import escape_like_pattern
from libs.pagination import PaginatedResult, paginate_query
from models.dataset import (
    ChildChunk,
    Dataset,
    DatasetProcessRule,
    Document,
    DocumentSegment,
    DocumentSegmentSummary,
    SegmentAttachmentBinding,
)
from models.enums import IndexingStatus, SegmentStatus, SegmentType, SummaryStatus
from models.model import SegmentStateOutbox, UploadFile
from repositories.knowledge.child_vector_cleanup_repository import SQLAlchemyChildVectorCleanupRepository
from repositories.knowledge.child_vector_route_repository import persist_child_vector_routes
from repositories.knowledge.dataset_indexing_outbox_repository import (
    SQLAlchemyDatasetIndexingOutboxRepository,
    lock_dataset_indexing_generation_shared,
)
from repositories.knowledge.dataset_read_repository import (
    get_content_file_ids,
    get_dataset_keyword_table,
    get_segment_attachment_files,
    get_segment_child_chunks,
    retrieved_segment_statement,
    segment_content_file_ids,
)
from repositories.knowledge.dataset_repository import _get_dataset
from repositories.knowledge.document_repository import _get_document, require_indexing_document
from repositories.knowledge.keyword_table_repository import persist_keyword_table
from repositories.knowledge.segment_cleanup_repository import (
    SegmentAttachmentReplacementResult,
    lock_owned_upload_files,
    replace_segment_attachment_bindings,
)
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository
from repositories.upload_file_repository import (
    complete_created_upload_artifact_publication_in_session,
    lock_created_upload_artifact_publication_in_session,
)
from services.knowledge.entities.datasets import DatasetIndexingGenerationLostError
from services.knowledge.entities.hit_testing import (
    RetrievalAttachmentRecord,
    RetrievalChildRecord,
    RetrievalSegmentRecord,
    RetrievalSnapshot,
)
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef, SegmentRef
from services.knowledge.entities.segments import (
    ChildChunkRecord,
    ChildVectorGuard,
    ChildVectorPublication,
    ChildVectorRoute,
    SegmentAttachmentFile,
    SegmentCreateData,
    SegmentFileAccess,
    SegmentReadSnapshot,
    SegmentStateIntent,
)
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.execution import IndexingDocument
from services.knowledge.segments.application import (
    ChildChunkListFilter,
    ChildChunkPage,
    ChildChunkState,
    SegmentDetail,
    SegmentIndexTarget,
    SegmentListFilter,
    SegmentPage,
    SegmentUpdateState,
)


class _SQLAlchemySegmentChildMutation:
    def __init__(
        self,
        repository: SQLAlchemySegmentRepository,
        session: Session,
        segment_ref: SegmentRef,
        segment: DocumentSegment,
        *,
        vector_guard: ChildVectorGuard,
    ) -> None:
        self._repository = repository
        self._session = session
        self._segment_ref = segment_ref
        self._segment = segment
        self._vector_guard = vector_guard

    @property
    def enabled(self) -> bool:
        return self._segment.enabled

    @property
    def vector_guard(self) -> ChildVectorGuard:
        return self._vector_guard

    def get_children(self) -> tuple[ChildChunkState, ...]:
        children = self._session.scalars(
            _children_query(self._segment_ref.document, (self._segment_ref.segment_id,)).order_by(
                ChildChunk.position.asc()
            )
        ).all()
        return tuple(_child_state(child) for child in children)

    def is_current_regeneration(
        self,
        *,
        index_node_hash: str | None,
        process_rule_id: str,
        process_rule_mode: str,
        process_rule_rules: str | None,
    ) -> bool:
        if self._segment.index_node_hash != index_node_hash:
            return False
        rule = self._session.scalar(
            select(DatasetProcessRule).where(
                DatasetProcessRule.id == process_rule_id,
                DatasetProcessRule.dataset_id == self._segment.dataset_id,
            )
        )
        return bool(rule is not None and str(rule.mode) == process_rule_mode and rule.rules == process_rule_rules)

    def save(
        self,
        *,
        expected: Sequence[ChildChunkState],
        expected_vector_guard: ChildVectorGuard,
        publication: ChildVectorPublication,
        added: Sequence[ChildChunkState] = (),
        updated: Sequence[ChildChunkState] = (),
        deleted: Sequence[ChildChunkState] = (),
    ) -> SegmentStateIntent:
        self.require_current(expected=expected, expected_vector_guard=expected_vector_guard)
        return self._repository._save_children_in_session(
            self._session,
            self._segment_ref,
            self._segment,
            vector_required=self._vector_guard.required,
            publication=publication,
            added=added,
            updated=updated,
            deleted=deleted,
        )

    def require_current(
        self,
        *,
        expected: Sequence[ChildChunkState],
        expected_vector_guard: ChildVectorGuard,
    ) -> None:
        if self.get_children() != tuple(expected) or self._vector_guard != expected_vector_guard:
            raise RuntimeError("Child chunks changed while vector publication was in progress")


class SQLAlchemySegmentRepository:
    """Read and write scoped segment rows in bounded database transactions."""

    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_retrieval_snapshot(
        self,
        ref: DatasetRef,
        *,
        node_pairs: Sequence[tuple[str, str]],
        summary_pairs: Sequence[tuple[str, str, str]],
        attachment_pairs: Sequence[tuple[str, str]],
    ) -> RetrievalSnapshot:
        if not (node_pairs or summary_pairs or attachment_pairs):
            return RetrievalSnapshot((), (), (), {})
        owned_segments = retrieved_segment_statement(tenant_id=ref.tenant_id, dataset_ids=[ref.dataset_id]).where(
            DocumentSegment.status == SegmentStatus.COMPLETED,
            Document.indexing_status == "completed",
        )
        owned_ids = owned_segments.with_only_columns(DocumentSegment.id)
        summary_segment_pairs = {(document_id, segment_id) for document_id, segment_id, _ in summary_pairs}
        with self._session_factory() as session:
            children = (
                session.scalars(
                    select(ChildChunk)
                    .join(DocumentSegment, DocumentSegment.id == ChildChunk.segment_id)
                    .where(
                        ChildChunk.segment_id.in_(owned_ids),
                        ChildChunk.tenant_id == ref.tenant_id,
                        ChildChunk.dataset_id == ref.dataset_id,
                        ChildChunk.document_id == DocumentSegment.document_id,
                        tuple_(ChildChunk.document_id, ChildChunk.index_node_id).in_(node_pairs),
                    )
                ).all()
                if node_pairs
                else []
            )
            attachment_rows = (
                session.execute(
                    select(SegmentAttachmentBinding, UploadFile)
                    .join(UploadFile, UploadFile.id == SegmentAttachmentBinding.attachment_id)
                    .join(DocumentSegment, DocumentSegment.id == SegmentAttachmentBinding.segment_id)
                    .where(
                        SegmentAttachmentBinding.segment_id.in_(owned_ids),
                        SegmentAttachmentBinding.tenant_id == ref.tenant_id,
                        SegmentAttachmentBinding.dataset_id == ref.dataset_id,
                        SegmentAttachmentBinding.document_id == DocumentSegment.document_id,
                        UploadFile.tenant_id == ref.tenant_id,
                        tuple_(SegmentAttachmentBinding.document_id, UploadFile.id).in_(attachment_pairs),
                    )
                ).all()
                if attachment_pairs
                else []
            )
            segments = session.scalars(
                owned_segments.where(
                    or_(
                        tuple_(DocumentSegment.document_id, DocumentSegment.index_node_id).in_(node_pairs),
                        tuple_(DocumentSegment.document_id, DocumentSegment.id).in_(summary_segment_pairs),
                        DocumentSegment.id.in_([child.segment_id for child in children]),
                        DocumentSegment.id.in_([binding.segment_id for binding, _ in attachment_rows]),
                    )
                )
            ).all()
            segment_ids = {segment.id for segment in segments}
            summary_segment_ids = {segment_id for _, segment_id in summary_segment_pairs} & segment_ids
            summaries = (
                session.scalars(
                    select(DocumentSegmentSummary)
                    .join(DocumentSegment, DocumentSegment.id == DocumentSegmentSummary.chunk_id)
                    .where(
                        DocumentSegmentSummary.chunk_id.in_(summary_segment_ids),
                        tuple_(
                            DocumentSegmentSummary.document_id,
                            DocumentSegmentSummary.chunk_id,
                            DocumentSegmentSummary.summary_index_node_id,
                        ).in_(summary_pairs),
                        DocumentSegmentSummary.dataset_id == ref.dataset_id,
                        DocumentSegmentSummary.dataset_id == DocumentSegment.dataset_id,
                        DocumentSegmentSummary.document_id == DocumentSegment.document_id,
                        DocumentSegment.tenant_id == ref.tenant_id,
                        DocumentSegment.dataset_id == ref.dataset_id,
                        DocumentSegmentSummary.enabled.is_(True),
                        DocumentSegmentSummary.status == SummaryStatus.COMPLETED,
                    )
                ).all()
                if summary_segment_ids
                else []
            )
            return RetrievalSnapshot(
                segments=tuple(
                    RetrievalSegmentRecord(segment.id, segment.document_id, segment.index_node_id, asdict(segment))
                    for segment in segments
                ),
                children=tuple(
                    RetrievalChildRecord(
                        child.id,
                        child.segment_id,
                        child.document_id,
                        child.index_node_id,
                        child.content,
                        child.position,
                    )
                    for child in children
                    if child.segment_id in segment_ids
                ),
                attachments=tuple(
                    RetrievalAttachmentRecord(
                        binding.segment_id,
                        binding.document_id,
                        file.id,
                        file.name,
                        file.extension,
                        file.mime_type,
                        file.size,
                    )
                    for binding, file in attachment_rows
                    if binding.segment_id in segment_ids
                ),
                summaries={
                    (summary.document_id, summary.chunk_id, summary.summary_index_node_id): summary.summary_content
                    for summary in summaries
                    if summary.summary_content is not None and summary.summary_index_node_id is not None
                },
            )

    def save_for_indexing(
        self,
        document: IndexingDocument,
        chunks: list[IndexDocument],
        token_counts: list[int],
        *,
        created_upload_file_ids: Sequence[str] = (),
    ) -> None:
        ref = document.ref
        with self._session_factory.begin() as session:
            dataset = lock_dataset_indexing_generation_shared(
                session,
                dataset_id=ref.dataset.dataset_id,
                generation_id=document.dataset_generation_id,
            )
            if dataset is None or dataset.tenant_id != ref.dataset.tenant_id:
                raise DatasetIndexingGenerationLostError(ref.dataset.dataset_id)
            row = require_indexing_document(session, ref, claim_id=document.claim_id, lock=True)
            if row.indexing_dataset_generation_id != document.dataset_generation_id:
                raise DatasetIndexingGenerationLostError(ref.dataset.dataset_id)
            lock_created_upload_artifact_publication_in_session(
                session,
                workspace_id=ref.dataset.tenant_id,
                upload_file_ids=created_upload_file_ids,
            )
            DatasetDocumentStore(dataset=dataset, user_id=row.created_by, document_id=row.id).add_documents(
                session=session,
                docs=chunks,
                token_counts=token_counts,
                save_child=document.doc_form == IndexStructureType.PARENT_CHILD_INDEX,
            )
            complete_created_upload_artifact_publication_in_session(
                session,
                workspace_id=ref.dataset.tenant_id,
                upload_file_ids=created_upload_file_ids,
            )
            now = naive_utc_now()
            row.indexing_status = IndexingStatus.INDEXING
            row.cleaning_completed_at = now
            row.splitting_completed_at = now
            row.word_count = sum(len(chunk.page_content) for chunk in chunks)
            session.execute(
                update(DocumentSegment)
                .where(*self._indexing_scope(ref))
                .values(
                    status=SegmentStatus.INDEXING,
                    indexing_at=now,
                )
            )

    def resume_indexing(self, document: IndexingDocument) -> tuple[list[IndexDocument], int]:
        ref = document.ref
        with self._session_factory() as session:
            require_indexing_document(session, ref, claim_id=document.claim_id)
            rows = session.scalars(
                select(DocumentSegment).where(*self._indexing_scope(ref)).order_by(DocumentSegment.position)
            ).all()
            chunks = []
            for row in rows:
                if row.status == SegmentStatus.COMPLETED:
                    continue
                chunk = IndexDocument(
                    page_content=row.content,
                    metadata={
                        "doc_id": row.index_node_id,
                        "doc_hash": row.index_node_hash,
                        "document_id": ref.document_id,
                        "dataset_id": ref.dataset.dataset_id,
                    },
                )
                if document.doc_form == IndexStructureType.PARENT_CHILD_INDEX:
                    chunk.children = [
                        ChildDocument(
                            page_content=child.content,
                            metadata={
                                "doc_id": child.index_node_id,
                                "doc_hash": child.index_node_hash,
                                "document_id": ref.document_id,
                                "dataset_id": ref.dataset.dataset_id,
                            },
                        )
                        for child in get_segment_child_chunks(row, session=session)
                    ]
                chunks.append(chunk)
            return chunks, sum(row.tokens for row in rows)

    def complete_indexing_segments(
        self,
        ref: DocumentRef,
        node_ids: Sequence[str],
        *,
        claim_id: str | None = None,
        dataset_generation_id: str | None = None,
        child_vector_route: ChildVectorRoute | None = None,
    ) -> None:
        with self._session_factory.begin() as session:
            dataset = lock_dataset_indexing_generation_shared(
                session,
                dataset_id=ref.dataset.dataset_id,
                generation_id=dataset_generation_id,
            )
            if dataset is None or dataset.tenant_id != ref.dataset.tenant_id:
                raise DatasetIndexingGenerationLostError(ref.dataset.dataset_id)
            document = require_indexing_document(session, ref, claim_id=claim_id, lock=True)
            if document.indexing_dataset_generation_id != dataset_generation_id:
                raise DatasetIndexingGenerationLostError(ref.dataset.dataset_id)
            segment_ids = tuple(
                session.scalars(
                    select(DocumentSegment.id).where(
                        *self._indexing_scope(ref),
                        DocumentSegment.index_node_id.in_(node_ids),
                        DocumentSegment.status == SegmentStatus.INDEXING,
                    )
                )
            )
            session.execute(
                update(DocumentSegment)
                .where(
                    *self._indexing_scope(ref),
                    DocumentSegment.index_node_id.in_(node_ids),
                    DocumentSegment.status == SegmentStatus.INDEXING,
                )
                .values(status=SegmentStatus.COMPLETED, enabled=True, completed_at=naive_utc_now())
            )
            if child_vector_route is not None:
                if (
                    child_vector_route.tenant_id != ref.dataset.tenant_id
                    or child_vector_route.dataset_id != ref.dataset.dataset_id
                ):
                    raise ValueError("Child vector route does not belong to the indexed document")
                child_node_ids = tuple(
                    node_id
                    for node_id in session.scalars(
                        select(ChildChunk.index_node_id).where(
                            ChildChunk.tenant_id == ref.dataset.tenant_id,
                            ChildChunk.dataset_id == ref.dataset.dataset_id,
                            ChildChunk.document_id == ref.document_id,
                            ChildChunk.segment_id.in_(segment_ids),
                            ChildChunk.index_node_id.is_not(None),
                        )
                    )
                    if node_id is not None
                )
                persist_child_vector_routes(
                    session,
                    tenant_id=ref.dataset.tenant_id,
                    dataset_id=ref.dataset.dataset_id,
                    node_ids=child_node_ids,
                    route_serialized=child_vector_route.serialized,
                    route_hash=child_vector_route.key,
                    route_doc_form=child_vector_route.doc_form,
                )

    @staticmethod
    def _indexing_scope(ref: DocumentRef):
        return (
            DocumentSegment.tenant_id == ref.dataset.tenant_id,
            DocumentSegment.dataset_id == ref.dataset.dataset_id,
            DocumentSegment.document_id == ref.document_id,
        )

    def list_segments(self, document_ref: DocumentRef, query: SegmentListFilter) -> SegmentPage[SegmentReadSnapshot]:
        with self._session_factory() as session:
            statement = (
                select(DocumentSegment)
                .join(Document, Document.id == DocumentSegment.document_id)
                .join(Dataset, Dataset.id == Document.dataset_id)
                .where(
                    Dataset.tenant_id == document_ref.dataset.tenant_id,
                    Dataset.id == DocumentSegment.dataset_id,
                    Document.tenant_id == DocumentSegment.tenant_id,
                    Document.dataset_id == DocumentSegment.dataset_id,
                    DocumentSegment.tenant_id == document_ref.dataset.tenant_id,
                    DocumentSegment.dataset_id == document_ref.dataset.dataset_id,
                    DocumentSegment.document_id == document_ref.document_id,
                )
                .order_by(DocumentSegment.position.asc())
            )
            if query.statuses:
                statement = statement.where(DocumentSegment.status.in_(query.statuses))
            if query.hit_count_gte is not None:
                statement = statement.where(DocumentSegment.hit_count >= query.hit_count_gte)
            if query.keyword:
                escaped_keyword = escape_like_pattern(query.keyword)
                if dify_config.SQLALCHEMY_DATABASE_URI_SCHEME == "postgresql":
                    keywords_jsonb = sql_cast(DocumentSegment.keywords, JSONB)
                    keywords_array = case(
                        (func.jsonb_typeof(keywords_jsonb) == "array", keywords_jsonb),
                        else_=sql_cast(literal("[]"), JSONB),
                    )
                    keywords_condition = func.array_to_string(
                        func.array(
                            select(func.jsonb_array_elements_text(keywords_array))
                            .correlate(DocumentSegment)
                            .scalar_subquery()
                        ),
                        ",",
                    ).ilike(f"%{escaped_keyword}%", escape="\\")
                else:
                    keywords_condition = sql_cast(DocumentSegment.keywords, String).ilike(
                        f"%{escaped_keyword}%", escape="\\"
                    )
                statement = statement.where(
                    or_(
                        DocumentSegment.content.ilike(f"%{escaped_keyword}%", escape="\\"),
                        keywords_condition,
                    )
                )
            if query.enabled.lower() == "true":
                statement = statement.where(DocumentSegment.enabled.is_(True))
            elif query.enabled.lower() == "false":
                statement = statement.where(DocumentSegment.enabled.is_(False))

            limit = min(query.limit, 100)
            segments = paginate_query(statement, session=session, page=query.page, per_page=limit, max_per_page=100)
            items = list(segments.items)
            file_access = _content_file_access(items, session)
            summaries = (
                {
                    summary.chunk_id: summary
                    for summary in session.scalars(
                        select(DocumentSegmentSummary).where(
                            DocumentSegmentSummary.chunk_id.in_([segment.id for segment in items]),
                            DocumentSegmentSummary.dataset_id == document_ref.dataset.dataset_id,
                            DocumentSegmentSummary.document_id == document_ref.document_id,
                        )
                    )
                }
                if items
                else {}
            )
            return SegmentPage(
                items=tuple(
                    _segment_data(
                        segment,
                        summaries[segment.id].summary_content if segment.id in summaries else None,
                        session,
                        file_access[segment.id],
                    )
                    for segment in items
                ),
                total=segments.total,
                total_pages=segments.pages,
                page=query.page,
                limit=limit,
            )

    def list_child_chunks(
        self,
        segment_ref: SegmentRef,
        query: ChildChunkListFilter,
    ) -> ChildChunkPage | None:
        with self._session_factory() as session:
            if _get_segment(session, segment_ref) is None:
                return None
            limit = min(query.limit, 100)
            child_chunks = query_child_chunks(
                session,
                segment_ref,
                page=query.page,
                limit=limit,
                keyword=query.keyword,
            )
            return ChildChunkPage(
                items=tuple(_child_chunk_data(chunk) for chunk in child_chunks.items),
                total=child_chunks.total,
                total_pages=child_chunks.pages,
                page=query.page,
                limit=limit,
            )

    def get_segment(self, segment_ref: SegmentRef) -> SegmentDetail[SegmentReadSnapshot] | None:
        with self._session_factory() as session:
            segment = _get_segment(session, segment_ref)
            if segment is None:
                return None
            _, document = _require_scope_models(session, segment_ref.document)
            return _detail(session, segment, document)

    def get_segment_update_state(self, segment_ref: SegmentRef) -> SegmentUpdateState | None:
        with self._session_factory() as session:
            segment = _get_segment(session, segment_ref)
            if segment is None:
                return None
            # Legacy rows may contain scalar or malformed JSON keywords. Treat
            # those as absent so a write can repair them without building a DTO.
            keywords = segment.keywords
            return SegmentUpdateState(
                content=segment.content,
                enabled=segment.enabled,
                keywords=tuple(keywords)
                if isinstance(keywords, list) and all(isinstance(keyword, str) for keyword in keywords)
                else None,
                answer=segment.answer,
            )

    def get_segments(self, document_ref: DocumentRef, segment_ids: Sequence[str]) -> tuple[SegmentIndexTarget, ...]:
        if not segment_ids:
            return ()
        with self._session_factory() as session:
            segments = session.scalars(
                select(DocumentSegment).where(
                    DocumentSegment.tenant_id == document_ref.dataset.tenant_id,
                    DocumentSegment.dataset_id == document_ref.dataset.dataset_id,
                    DocumentSegment.document_id == document_ref.document_id,
                    DocumentSegment.id.in_(segment_ids),
                )
            ).all()
            return tuple(SegmentIndexTarget(segment.id, segment.index_node_id, segment.enabled) for segment in segments)

    def save_segment(self, segment_ref: SegmentRef, values: Mapping[str, object]) -> None:
        with self._session_factory.begin() as session:
            _, document = _require_scope_models(session, segment_ref.document, lock=True)
            segment = _get_segment(session, segment_ref, lock=True)
            previous_words = segment.word_count if segment is not None else 0
            if segment is None:
                raise LookupError("Segment no longer exists")
            for key, value in values.items():
                setattr(segment, key, value)
            if "word_count" in values:
                document.word_count = max(0, (document.word_count or 0) + segment.word_count - previous_words)

    @staticmethod
    def _dataset_matches_indexing_snapshot(dataset: Dataset, snapshot: SegmentIndexingSnapshot) -> bool:
        expected = snapshot.dataset
        return (
            dataset.id == expected.id
            and dataset.tenant_id == expected.tenant_id
            and dataset.indexing_request_id == expected.indexing_request_id
            and dataset.indexing_technique == expected.indexing_technique
            and dataset.index_struct == expected.index_struct
            and dataset.collection_binding_id == expected.collection_binding_id
            and dataset.embedding_model_provider == expected.embedding_model_provider
            and dataset.embedding_model == expected.embedding_model
            and dataset.is_multimodal == expected.is_multimodal
        )

    @staticmethod
    def _document_matches_indexing_snapshot(document: Document, snapshot: SegmentIndexingSnapshot) -> bool:
        expected = snapshot.document
        return (
            document.id == expected.id
            and document.tenant_id == expected.tenant_id
            and document.dataset_id == expected.dataset_id
            and document.doc_form == expected.doc_form
            and document.dataset_process_rule_id == expected.dataset_process_rule_id
            and document.enabled == expected.enabled
            and document.archived == expected.archived
            and document.indexing_status == expected.indexing_status
        )

    @staticmethod
    def _segment_matches_indexing_snapshot(segment: DocumentSegment, snapshot: SegmentIndexingSnapshot) -> bool:
        expected = snapshot.segment
        return (
            segment.id == expected.id
            and segment.tenant_id == expected.tenant_id
            and segment.dataset_id == expected.dataset_id
            and segment.document_id == expected.document_id
            and segment.enabled
            and segment.enabled == expected.enabled
            and segment.status == expected.status
            and segment.content == expected.content
            and segment.answer == expected.answer
            and segment.word_count == expected.word_count
            and segment.tokens == expected.tokens
            and segment.keywords == expected.keywords
            and segment.index_node_id == expected.index_node_id
            and segment.index_node_hash == expected.index_node_hash
            and segment.indexing_request_id == expected.indexing_request_id
        )

    @classmethod
    def _lock_current_indexing_snapshots(
        cls,
        session: Session,
        document_ref: DocumentRef,
        snapshots: Sequence[SegmentIndexingSnapshot],
    ) -> tuple[Dataset, tuple[SegmentIndexingSnapshot, ...]]:
        """Lock one owner chain and retain each still-exact batch member."""
        if not snapshots:
            raise ValueError("Indexing snapshot batch must not be empty")
        if any(
            snapshot.dataset.id != document_ref.dataset.dataset_id
            or snapshot.dataset.tenant_id != document_ref.dataset.tenant_id
            or snapshot.document.id != document_ref.document_id
            or snapshot.document.dataset_id != document_ref.dataset.dataset_id
            or snapshot.document.tenant_id != document_ref.dataset.tenant_id
            or snapshot.segment.document_id != document_ref.document_id
            or snapshot.segment.dataset_id != document_ref.dataset.dataset_id
            or snapshot.segment.tenant_id != document_ref.dataset.tenant_id
            for snapshot in snapshots
        ):
            raise ValueError("Indexing snapshots do not belong to the requested document")

        dataset = session.scalar(
            select(Dataset)
            .where(
                Dataset.id == document_ref.dataset.dataset_id,
                Dataset.tenant_id == document_ref.dataset.tenant_id,
            )
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        if dataset is None:
            raise LookupError("Dataset no longer exists")
        if any(not cls._dataset_matches_indexing_snapshot(dataset, snapshot) for snapshot in snapshots):
            raise DatasetIndexingGenerationLostError("Segment inputs belong to an older Dataset generation")

        document = session.scalar(
            select(Document)
            .where(
                Document.id == document_ref.document_id,
                Document.dataset_id == dataset.id,
                Document.tenant_id == dataset.tenant_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if document is None:
            raise LookupError("Dataset document scope no longer exists")
        if any(not cls._document_matches_indexing_snapshot(document, snapshot) for snapshot in snapshots):
            raise DatasetIndexingGenerationLostError("Document inputs changed before vector publication")

        segment_ids = tuple(dict.fromkeys(snapshot.segment.id for snapshot in snapshots))
        segments = {
            segment.id: segment
            for segment in session.scalars(
                select(DocumentSegment)
                .where(
                    DocumentSegment.tenant_id == dataset.tenant_id,
                    DocumentSegment.dataset_id == dataset.id,
                    DocumentSegment.document_id == document.id,
                    DocumentSegment.id.in_(segment_ids),
                )
                .order_by(DocumentSegment.id)
                .with_for_update()
                .execution_options(populate_existing=True)
            ).all()
        }
        return dataset, tuple(
            snapshot
            for snapshot in snapshots
            if (segment := segments.get(snapshot.segment.id)) is not None
            and cls._segment_matches_indexing_snapshot(segment, snapshot)
        )

    def record_batch_indexing_route(
        self,
        document_ref: DocumentRef,
        snapshots: Sequence[SegmentIndexingSnapshot],
        route: VectorRoute,
    ) -> tuple[SegmentIndexingSnapshot, ...]:
        """Commit route evidence after filtering every detached batch member."""
        if not snapshots:
            return ()
        with self._session_factory.begin() as session:
            dataset, current = self._lock_current_indexing_snapshots(session, document_ref, snapshots)
            if not current:
                return ()
            SQLAlchemyDatasetIndexingOutboxRepository.record_generation_writer_route_locked(
                session,
                dataset=dataset,
                generation_id=current[0].dataset.indexing_request_id,
                route=route,
            )
            return current

    @contextmanager
    def lock_index_batch_write(
        self,
        document_ref: DocumentRef,
        *,
        expected_snapshots: Sequence[SegmentIndexingSnapshot],
    ):
        """Hold owner locks through I/O and expose their transaction to keyword persistence."""
        with self._session_factory.begin() as session:
            _, current = self._lock_current_indexing_snapshots(session, document_ref, expected_snapshots)
            yield current, session

    def mark_batch_indexing_failed(
        self,
        document_ref: DocumentRef,
        snapshots: Sequence[SegmentIndexingSnapshot],
        error: Exception,
    ) -> tuple[str, ...]:
        """CAS an external failure onto only still-current attempted members."""
        if not snapshots:
            return ()
        with self._session_factory.begin() as session:
            _, current = self._lock_current_indexing_snapshots(session, document_ref, snapshots)
            current_ids = {snapshot.segment.id for snapshot in current}
            if not current_ids:
                return ()
            rows = session.scalars(
                select(DocumentSegment).where(
                    DocumentSegment.tenant_id == document_ref.dataset.tenant_id,
                    DocumentSegment.dataset_id == document_ref.dataset.dataset_id,
                    DocumentSegment.document_id == document_ref.document_id,
                    DocumentSegment.id.in_(current_ids),
                )
            ).all()
            disabled_at = naive_utc_now()
            for segment in rows:
                segment.enabled = False
                segment.disabled_at = disabled_at
                segment.status = SegmentStatus.ERROR
                segment.error = str(error)
            return tuple(snapshot.segment.id for snapshot in current)

    @classmethod
    def require_any_indexing_snapshot(
        cls,
        session: Session,
        document_ref: DocumentRef,
        snapshots: Sequence[SegmentIndexingSnapshot],
    ) -> None:
        """Fence a document-scoped publication when at least one member remains exact."""
        _, current = cls._lock_current_indexing_snapshots(session, document_ref, snapshots)
        if not current:
            raise DatasetIndexingGenerationLostError("No current segment remains for vector publication")

    @contextmanager
    def lock_index_write(
        self,
        segment_ref: SegmentRef,
        *,
        lock_segment: bool = True,
        expected_snapshot: SegmentIndexingSnapshot | None = None,
    ):
        """Fence index I/O with the document lock shared by state workers.

        The guard intentionally does not lock the segment row: summary and
        keyword adapters use their own bounded sessions while this top-level
        document lock prevents state/delete workers from passing them. The
        Dataset SHARE lock also excludes configuration rebuilds throughout I/O.
        A caller with detached inputs must supply their snapshot so a newer
        Dataset generation cannot reinterpret those inputs.
        """

        with self._session_factory.begin() as session:
            dataset = session.scalar(
                select(Dataset)
                .where(
                    Dataset.id == segment_ref.document.dataset.dataset_id,
                    Dataset.tenant_id == segment_ref.document.dataset.tenant_id,
                )
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if dataset is None:
                raise LookupError("Dataset no longer exists")
            if expected_snapshot is not None and (
                dataset.indexing_request_id != expected_snapshot.dataset.indexing_request_id
                or dataset.indexing_technique != expected_snapshot.dataset.indexing_technique
                or dataset.collection_binding_id != expected_snapshot.dataset.collection_binding_id
                or dataset.index_struct != expected_snapshot.dataset.index_struct
            ):
                raise DatasetIndexingGenerationLostError("Segment inputs belong to an older Dataset generation")
            document = session.scalar(
                select(Document)
                .where(
                    Document.id == segment_ref.document.document_id,
                    Document.dataset_id == dataset.id,
                    Document.tenant_id == dataset.tenant_id,
                )
                .with_for_update()
            )
            if document is None:
                raise LookupError("Dataset document scope no longer exists")
            # Locking reads are current reads under MySQL REPEATABLE READ. Body
            # writers lock the segment too; summary publication asks only for
            # the stable owner lock because its nested versioned write must lock
            # that segment from a separate bounded session.
            segment = _get_segment(session, segment_ref, lock=lock_segment)
            if segment is None:
                raise LookupError("Segment no longer exists")
            if lock_segment:
                session.refresh(segment)
            if expected_snapshot is not None and (
                segment.index_node_id != expected_snapshot.segment.index_node_id
                or segment.index_node_hash != expected_snapshot.segment.index_node_hash
            ):
                raise DatasetIndexingGenerationLostError("Segment content changed before vector publication")
            yield segment.enabled

    @staticmethod
    def require_indexing_snapshot(
        session: Session,
        snapshot: SegmentIndexingSnapshot,
    ) -> None:
        """Validate detached publication inputs in the caller's owner-lock transaction.

        Attachment publication already owns the Dataset lock. Reusing that
        session keeps its owner check and remote write in one generation without
        requesting a second shared lock from a separate connection.
        """
        ref = DatasetRef(snapshot.dataset.tenant_id, snapshot.dataset.id).document(snapshot.document.id)
        dataset, document = _require_scope_models(session, ref, lock=True)
        session.refresh(dataset)
        if (
            dataset.indexing_request_id != snapshot.dataset.indexing_request_id
            or dataset.indexing_technique != snapshot.dataset.indexing_technique
            or dataset.collection_binding_id != snapshot.dataset.collection_binding_id
            or dataset.index_struct != snapshot.dataset.index_struct
        ):
            raise DatasetIndexingGenerationLostError("Attachment inputs belong to an older Dataset generation")
        segment = _get_segment(session, ref.segment(snapshot.segment.id), lock=True)
        if segment is None:
            raise LookupError("Segment no longer exists")
        session.refresh(segment)
        if (
            not document.enabled
            or document.archived
            or not segment.enabled
            or segment.index_node_id != snapshot.segment.index_node_id
            or segment.index_node_hash != snapshot.segment.index_node_hash
        ):
            raise DatasetIndexingGenerationLostError("Attachment publication inputs are no longer current")

    def record_indexing_route(self, snapshot: SegmentIndexingSnapshot, route: VectorRoute) -> None:
        """Commit route evidence for one exact manual segment snapshot before I/O."""
        with self._session_factory.begin() as session:
            self.require_indexing_snapshot(session, snapshot)
            dataset = session.scalar(
                select(Dataset).where(
                    Dataset.id == snapshot.dataset.id,
                    Dataset.tenant_id == snapshot.dataset.tenant_id,
                )
            )
            if dataset is None:
                raise DatasetIndexingGenerationLostError(snapshot.dataset.id)
            SQLAlchemyDatasetIndexingOutboxRepository.record_generation_writer_route_locked(
                session,
                dataset=dataset,
                generation_id=snapshot.dataset.indexing_request_id,
                route=route,
            )

    def create_segments(
        self,
        document_ref: DocumentRef,
        segments: Sequence[SegmentCreateData],
    ) -> None:
        """Append a batch under the persisted owner lock.

        The owner rows are locked before the current maximum position is read,
        so every caller shares one allocation boundary regardless of its
        process-local or Redis locking strategy.
        """
        if not segments:
            return
        with self._session_factory.begin() as session:
            _, document = _require_scope_models(session, document_ref, lock=True)
            segment_ids = [segment.id for segment in segments]
            if len(set(segment_ids)) != len(segment_ids):
                raise ValueError("Segment IDs must be unique")
            if session.scalar(
                select(DocumentSegment.id)
                .where(
                    DocumentSegment.tenant_id == document_ref.dataset.tenant_id,
                    DocumentSegment.dataset_id == document_ref.dataset.dataset_id,
                    DocumentSegment.document_id == document_ref.document_id,
                    DocumentSegment.id.in_(segment_ids),
                )
                .limit(1)
            ):
                raise ValueError("Segment already exists")
            requested_attachment_ids = tuple(
                dict.fromkeys(attachment_id for segment in segments for attachment_id in segment.attachment_ids)
            )
            owned_attachment_ids = lock_owned_upload_files(
                session,
                tenant_id=document_ref.dataset.tenant_id,
                file_ids=requested_attachment_ids,
            )
            missing_attachment_ids = set(requested_attachment_ids) - owned_attachment_ids
            if missing_attachment_ids:
                raise LookupError(f"Attachment files no longer exist: {sorted(missing_attachment_ids)}")
            position = (
                session.scalar(
                    select(func.max(DocumentSegment.position)).where(
                        DocumentSegment.tenant_id == document_ref.dataset.tenant_id,
                        DocumentSegment.dataset_id == document_ref.dataset.dataset_id,
                        DocumentSegment.document_id == document_ref.document_id,
                    )
                )
                or 0
            )
            word_count_change = 0
            bindings: list[SegmentAttachmentBinding] = []
            for create_data in segments:
                position += 1
                segment = DocumentSegment(
                    tenant_id=document_ref.dataset.tenant_id,
                    dataset_id=document_ref.dataset.dataset_id,
                    document_id=document_ref.document_id,
                    position=position,
                    content=create_data.content,
                    answer=create_data.answer,
                    word_count=create_data.word_count,
                    tokens=create_data.tokens,
                    keywords=list(create_data.keywords) if create_data.keywords is not None else None,
                    index_node_id=create_data.index_node_id,
                    index_node_hash=create_data.index_node_hash,
                    created_by=create_data.created_by,
                    status=SegmentStatus(create_data.status),
                    indexing_at=create_data.indexing_at,
                    completed_at=create_data.completed_at,
                )
                segment.id = create_data.id
                word_count_change += segment.word_count
                session.add(segment)
                bindings.extend(
                    SegmentAttachmentBinding(
                        tenant_id=document_ref.dataset.tenant_id,
                        dataset_id=document_ref.dataset.dataset_id,
                        document_id=document_ref.document_id,
                        segment_id=create_data.id,
                        attachment_id=attachment_id,
                    )
                    for attachment_id in dict.fromkeys(create_data.attachment_ids)
                )
            session.add_all(bindings)
            document.word_count = max(0, (document.word_count or 0) + word_count_change)

    def delete_segments(self, document_ref: DocumentRef, segment_ids: Sequence[str]) -> tuple[SegmentStateIntent, ...]:
        if not segment_ids:
            return ()
        intents: list[SegmentStateIntent] = []
        with self._session_factory.begin() as session:
            _, document = _require_scope_models(session, document_ref, lock=True)
            rows = session.scalars(
                select(DocumentSegment)
                .where(
                    DocumentSegment.tenant_id == document_ref.dataset.tenant_id,
                    DocumentSegment.dataset_id == document_ref.dataset.dataset_id,
                    DocumentSegment.document_id == document_ref.document_id,
                    DocumentSegment.id.in_(segment_ids),
                )
                .order_by(DocumentSegment.id)
                .with_for_update()
            ).all()
            document.word_count = max(0, (document.word_count or 0) - sum(row.word_count for row in rows))
            children = session.scalars(
                select(ChildChunk)
                .where(
                    ChildChunk.tenant_id == document_ref.dataset.tenant_id,
                    ChildChunk.dataset_id == document_ref.dataset.dataset_id,
                    ChildChunk.document_id == document_ref.document_id,
                    ChildChunk.segment_id.in_([row.id for row in rows]),
                )
                .order_by(ChildChunk.segment_id, ChildChunk.id)
                .with_for_update()
            ).all()
            children_by_segment: dict[str, list[tuple[str, str, ChildVectorRoute | None]]] = {}
            for child in children:
                if child.index_node_id:
                    children_by_segment.setdefault(child.segment_id, []).append(
                        (child.id, child.index_node_id, _stored_child_vector_route(child))
                    )
                session.delete(child)
            for row in rows:
                self._invalidate_state_intent(session, row)
                SQLAlchemyChildVectorCleanupRepository.add_terminal_cleanups(
                    session,
                    ref=document_ref.segment(row.id),
                    obsolete_nodes=children_by_segment.get(row.id, ()),
                )
                intent = SegmentStateOutbox(
                    tenant_id=row.tenant_id,
                    dataset_id=row.dataset_id,
                    document_id=row.document_id,
                    segment_id=row.id,
                    action="delete",
                    index_node_id=row.index_node_id,
                    child_node_ids="[]",
                    doc_form=document.doc_form,
                )
                session.add(intent)
                intents.append(SegmentStateIntent(intent.id))
                session.delete(row)
        return tuple(intents)

    @staticmethod
    def _invalidate_state_intent(session: Session, segment: DocumentSegment) -> None:
        if segment.indexing_request_id is None:
            return
        session.execute(delete(SegmentStateOutbox).where(SegmentStateOutbox.id == segment.indexing_request_id))
        segment.indexing_request_id = None

    def get_children(self, segment_ref: SegmentRef) -> tuple[ChildChunkState, ...] | None:
        with self._session_factory() as session:
            if _get_segment(session, segment_ref) is None:
                return None
            children = session.scalars(
                _children_query(segment_ref.document, (segment_ref.segment_id,)).order_by(ChildChunk.position.asc())
            ).all()
            return tuple(_child_state(child) for child in children)

    @contextmanager
    def child_mutation(self, segment_ref: SegmentRef):
        with self._session_factory.begin() as session:
            dataset, _ = _require_scope_models(session, segment_ref.document, lock=True)
            segment = _get_segment(session, segment_ref, lock=True)
            if segment is None:
                raise LookupError("Segment no longer exists")
            yield _SQLAlchemySegmentChildMutation(
                self,
                session,
                segment_ref,
                segment,
                vector_guard=_child_vector_guard(dataset, enabled=segment.enabled),
            )

    @staticmethod
    def _save_children_in_session(
        session: Session,
        segment_ref: SegmentRef,
        segment: DocumentSegment,
        *,
        vector_required: bool,
        publication: ChildVectorPublication,
        added: Sequence[ChildChunkState] = (),
        updated: Sequence[ChildChunkState] = (),
        deleted: Sequence[ChildChunkState] = (),
    ) -> SegmentStateIntent:
        publication_claims = tuple(publication.claims)
        vector_targets = {
            child.data.id: child.index_node_id for child in (*added, *updated) if child.index_node_id is not None
        }
        published_routes: Mapping[str, ChildVectorRoute] = {}
        if vector_required:
            published_routes = SQLAlchemyChildVectorCleanupRepository.consume_publication(
                session,
                ref=segment_ref,
                targets=vector_targets,
                claims=publication_claims,
            )
        elif publication_claims:
            raise RuntimeError("Disabled child mutation cannot publish vector claims")

        obsolete_nodes: list[tuple[str, str, ChildVectorRoute | None]] = []
        for child in deleted:
            row = _get_child_chunk(session, segment_ref, child.data.id)
            if row is not None:
                if row.index_node_id:
                    obsolete_nodes.append((row.id, row.index_node_id, _stored_child_vector_route(row)))
                session.delete(row)
        for child in updated:
            row = _get_child_chunk(session, segment_ref, child.data.id)
            if row is None:
                raise LookupError("Child chunk no longer exists")
            if row.index_node_id and row.index_node_id != child.index_node_id:
                obsolete_nodes.append((row.id, row.index_node_id, _stored_child_vector_route(row)))
            row.content = child.data.content
            row.word_count = child.data.word_count
            row.type = SegmentType(child.data.type)
            row.index_node_id = child.index_node_id
            row.index_node_hash = child.index_node_hash
            route = published_routes.get(child.data.id)
            row.vector_route = route.serialized if route is not None else None
            row.vector_route_hash = route.key if route is not None else None
            row.updated_by = child.updated_by
            row.updated_at = child.data.updated_at
        for child in added:
            route = published_routes.get(child.data.id)
            row = ChildChunk(
                **child.data.model_dump(exclude={"id", "created_at", "updated_at"}),
                tenant_id=segment_ref.document.dataset.tenant_id,
                dataset_id=segment_ref.document.dataset.dataset_id,
                document_id=segment_ref.document.document_id,
                index_node_id=child.index_node_id,
                index_node_hash=child.index_node_hash,
                vector_route=route.serialized if route is not None else None,
                vector_route_hash=route.key if route is not None else None,
                created_by=child.created_by,
            )
            row.id = child.data.id
            row.created_at = child.data.created_at
            row.updated_at = child.data.updated_at
            row.updated_by = child.updated_by
            session.add(row)
        SQLAlchemyChildVectorCleanupRepository.add_terminal_cleanups(
            session,
            ref=segment_ref,
            obsolete_nodes=obsolete_nodes,
        )
        action = "enable" if segment.enabled else "disable"
        if segment.indexing_request_id:
            existing = session.scalar(
                select(SegmentStateOutbox).where(
                    SegmentStateOutbox.id == segment.indexing_request_id,
                    SegmentStateOutbox.action == action,
                )
            )
            if existing is not None:
                return SegmentStateIntent(existing.id)
            session.execute(delete(SegmentStateOutbox).where(SegmentStateOutbox.id == segment.indexing_request_id))
        actor_id = next(
            (child.updated_by or child.created_by for child in (*added, *updated, *deleted)),
            segment.updated_by or segment.created_by,
        )
        summary_id, summary_version = (
            lock_segment_summary_snapshot(session, dataset_id=segment.dataset_id, segment_id=segment.id)
            if action == "enable"
            else (None, None)
        )
        intent = SegmentStateOutbox(
            tenant_id=segment.tenant_id,
            dataset_id=segment.dataset_id,
            document_id=segment.document_id,
            segment_id=segment.id,
            action=action,
            actor_id=actor_id,
            summary_id=summary_id,
            summary_version=summary_version,
        )
        session.add(intent)
        segment.indexing_request_id = intent.id
        return SegmentStateIntent(intent.id)

    def get_indexing_snapshot(self, segment_ref: SegmentRef) -> SegmentIndexingSnapshot:
        return self.get_indexing_snapshots(segment_ref.document, (segment_ref.segment_id,))[0]

    def get_indexing_snapshots(
        self, document_ref: DocumentRef, segment_ids: Sequence[str]
    ) -> tuple[SegmentIndexingSnapshot, ...]:
        """Load an ordered batch and its owner chain in one bounded read session."""
        if not segment_ids:
            return ()
        with self._session_factory() as session:
            dataset, document = _require_scope_models(session, document_ref)
            segments = {
                segment.id: segment
                for segment in session.scalars(
                    select(DocumentSegment).where(
                        DocumentSegment.tenant_id == dataset.tenant_id,
                        DocumentSegment.dataset_id == dataset.id,
                        DocumentSegment.document_id == document.id,
                        DocumentSegment.id.in_(segment_ids),
                    )
                )
            }
            if set(segments) != set(segment_ids):
                raise LookupError("Segment no longer exists")
            rule = (
                session.scalar(
                    select(DatasetProcessRule).where(
                        DatasetProcessRule.id == document.dataset_process_rule_id,
                        DatasetProcessRule.dataset_id == dataset.id,
                    )
                )
                if document.dataset_process_rule_id
                else None
            )
            vector_type = (
                Vector.resolve_vector_type(dataset, session=session)
                if dataset.indexing_technique == "high_quality"
                else None
            )
            attachments: dict[str, list[str]] = {segment_id: [] for segment_id in segment_ids}
            for binding in session.scalars(
                select(SegmentAttachmentBinding)
                .where(
                    SegmentAttachmentBinding.tenant_id == dataset.tenant_id,
                    SegmentAttachmentBinding.dataset_id == dataset.id,
                    SegmentAttachmentBinding.document_id == document.id,
                    SegmentAttachmentBinding.segment_id.in_(segment_ids),
                )
                .order_by(SegmentAttachmentBinding.segment_id, SegmentAttachmentBinding.attachment_id)
            ):
                attachments[binding.segment_id].append(binding.attachment_id)
            summaries = {
                summary.chunk_id: summary
                for summary in session.scalars(
                    select(DocumentSegmentSummary).where(
                        DocumentSegmentSummary.dataset_id == dataset.id,
                        DocumentSegmentSummary.document_id == document.id,
                        DocumentSegmentSummary.chunk_id.in_(segment_ids),
                    )
                )
            }
            return tuple(
                SegmentIndexingSnapshot(
                    dataset,
                    document,
                    segments[segment_id],
                    rule,
                    vector_type,
                    summaries.get(segment_id),
                    tuple(attachments[segment_id]),
                )
                for segment_id in segment_ids
            )

    def replace_attachments(
        self, segment_ref: SegmentRef, attachment_ids: Sequence[str]
    ) -> SegmentAttachmentReplacementResult:
        with self._session_factory.begin() as session:
            return replace_segment_attachment_bindings(
                session,
                tenant_id=segment_ref.document.dataset.tenant_id,
                dataset_id=segment_ref.document.dataset.dataset_id,
                document_id=segment_ref.document.document_id,
                segment_id=segment_ref.segment_id,
                attachment_ids=attachment_ids,
            )

    def save_summary(
        self, segment_ref: SegmentRef, content: str | None, *, error: str | None = None
    ) -> SummarySaveResult:
        with self._session_factory.begin() as session:
            segment = _get_segment(session, segment_ref, lock=True)
            if segment is None:
                raise LookupError("Segment no longer exists")
            intent = _lock_current_segment_state_intent(session, segment)
            summary = _summary(session, segment, lock=True)
            previous_summary_snapshot = (summary.id, summary.version) if summary is not None else (None, 0)
            if content is None:
                if summary is not None:
                    SQLAlchemySummaryVectorCleanupRepository.add_terminal_cleanups(
                        session,
                        tenant_id=segment_ref.document.dataset.tenant_id,
                        summaries=(summary,),
                    )
                    session.delete(summary)
                inherited_intent_id = _advance_inherited_summary_enable_intent(
                    intent,
                    previous_snapshot=previous_summary_snapshot,
                    current_snapshot=(None, 0),
                )
                return SummarySaveResult(summary=None, inherited_enable_intent_id=inherited_intent_id)
            if summary is None:
                summary = DocumentSegmentSummary(
                    dataset_id=segment.dataset_id, document_id=segment.document_id, chunk_id=segment.id
                )
                session.add(summary)
            else:
                summary.version += 1
            summary.summary_content = content
            summary.status = SummaryStatus.ERROR if error else SummaryStatus.GENERATING
            summary.error = error
            session.flush()
            inherited_intent_id = _advance_inherited_summary_enable_intent(
                intent,
                previous_snapshot=previous_summary_snapshot,
                current_snapshot=(summary.id, summary.version),
            )
            session.expunge(summary)
            return SummarySaveResult(summary=summary, inherited_enable_intent_id=inherited_intent_id)

    def get_summary_images(self, segment_ref: SegmentRef) -> list[File]:
        with self._session_factory() as session:
            segment = _get_segment(session, segment_ref)
            if segment is None:
                raise LookupError("Segment no longer exists")
            images = ParagraphIndexProcessor._extract_images_from_segment_attachments(
                segment.tenant_id, segment.id, session
            )
            return images or ParagraphIndexProcessor._extract_images_from_text(
                segment.tenant_id, segment.content, session
            )

    def get_keyword_table(self, dataset_ref: DatasetRef) -> tuple[str, str | None]:
        with self._session_factory() as session:
            dataset = _get_dataset(session, dataset_ref)
            if dataset is None:
                raise LookupError("Dataset no longer exists")
            row = get_dataset_keyword_table(dataset, session=session)
            return (row.data_source_type, row.keyword_table) if row else (dify_config.KEYWORD_DATA_SOURCE_TYPE, None)

    def save_keyword_table(
        self,
        dataset_ref: DatasetRef,
        *,
        storage_type: str,
        data: str,
        keywords: Mapping[str, Sequence[str]],
        indexing_document: IndexingDocument | None = None,
    ) -> None:
        with self._session_factory.begin() as session:
            self.save_keyword_table_in_session(
                session,
                dataset_ref,
                storage_type=storage_type,
                data=data,
                keywords=keywords,
                indexing_document=indexing_document,
            )

    @staticmethod
    def save_keyword_table_in_session(
        session: Session,
        dataset_ref: DatasetRef,
        *,
        storage_type: str,
        data: str,
        keywords: Mapping[str, Sequence[str]],
        indexing_document: IndexingDocument | None = None,
    ) -> None:
        """Persist keyword output in a caller-owned publication transaction."""
        if indexing_document is not None:
            require_indexing_document(
                session,
                indexing_document.ref,
                claim_id=indexing_document.claim_id,
                lock=True,
            )
        persist_keyword_table(
            session,
            tenant_id=dataset_ref.tenant_id,
            dataset_id=dataset_ref.dataset_id,
            storage_type=storage_type,
            data=data,
            keywords=keywords,
        )


@dataclass(frozen=True)
class SegmentIndexingSnapshot:
    dataset: Dataset
    document: Document
    segment: DocumentSegment
    process_rule: DatasetProcessRule | None
    vector_type: str | None
    summary: DocumentSegmentSummary | None
    attachment_ids: tuple[str, ...]


@dataclass(frozen=True)
class SummarySaveResult:
    summary: DocumentSegmentSummary | None
    inherited_enable_intent_id: str | None


def lock_segment_summary_snapshot(session: Session, *, dataset_id: str, segment_id: str) -> tuple[str | None, int]:
    """Return the exact locked summary revision an enable intent may publish."""

    summary = session.scalar(
        select(DocumentSegmentSummary)
        .where(
            DocumentSegmentSummary.dataset_id == dataset_id,
            DocumentSegmentSummary.chunk_id == segment_id,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if summary is None:
        return None, 0
    return summary.id, summary.version


def _summary(session: Session, segment: DocumentSegment, *, lock: bool = False) -> DocumentSegmentSummary | None:
    statement = (
        select(DocumentSegmentSummary)
        .where(
            DocumentSegmentSummary.dataset_id == segment.dataset_id,
            DocumentSegmentSummary.document_id == segment.document_id,
            DocumentSegmentSummary.chunk_id == segment.id,
        )
        .limit(1)
    )
    if lock:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    return session.scalar(statement)


def _lock_current_segment_state_intent(session: Session, segment: DocumentSegment) -> SegmentStateOutbox | None:
    """Lock the segment's exact state intent before its summary revision.

    State workers use the same Segment -> Intent -> Summary order.  Loading the
    exact owner token here lets a manual summary edit atomically inherit a
    pending enable request without reviving an older request superseded by a
    disable.
    """

    if segment.indexing_request_id is None:
        return None
    return session.scalar(
        select(SegmentStateOutbox)
        .where(
            SegmentStateOutbox.id == segment.indexing_request_id,
            SegmentStateOutbox.tenant_id == segment.tenant_id,
            SegmentStateOutbox.dataset_id == segment.dataset_id,
            SegmentStateOutbox.document_id == segment.document_id,
            SegmentStateOutbox.segment_id == segment.id,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )


def _advance_inherited_summary_enable_intent(
    intent: SegmentStateOutbox | None,
    *,
    previous_snapshot: tuple[str | None, int],
    current_snapshot: tuple[str | None, int],
) -> str | None:
    """Advance only the enable intent that owned the pre-edit summary revision."""

    if intent is None or intent.action != "enable":
        return None
    if (intent.summary_id, intent.summary_version) != previous_snapshot:
        return None
    intent.summary_id, intent.summary_version = current_snapshot
    return intent.id


def _children_query(document_ref: DocumentRef, segment_ids: Sequence[str]):
    return select(ChildChunk).where(
        ChildChunk.tenant_id == document_ref.dataset.tenant_id,
        ChildChunk.dataset_id == document_ref.dataset.dataset_id,
        ChildChunk.document_id == document_ref.document_id,
        ChildChunk.segment_id.in_(segment_ids),
    )


def _child_state(child: ChildChunk) -> ChildChunkState:
    return ChildChunkState(
        data=_child_chunk_data(child),
        index_node_id=child.index_node_id,
        index_node_hash=child.index_node_hash,
        created_by=child.created_by,
        updated_by=child.updated_by,
    )


def _stored_child_vector_route(child: ChildChunk) -> ChildVectorRoute | None:
    if child.vector_route is None or child.vector_route_hash is None:
        return None
    try:
        route = ChildVectorRoute.model_validate_json(child.vector_route)
    except ValueError:
        return None
    if route.key != child.vector_route_hash:
        return None
    if route.tenant_id != child.tenant_id or route.dataset_id != child.dataset_id:
        return None
    return route


def _child_vector_guard(dataset: Dataset, *, enabled: bool) -> ChildVectorGuard:
    return ChildVectorGuard(
        required=enabled and dataset.indexing_technique == "high_quality",
        generation_id=dataset.indexing_request_id,
        indexing_technique=dataset.indexing_technique,
        embedding_model_provider=dataset.embedding_model_provider,
        embedding_model=dataset.embedding_model,
        index_struct=dataset.index_struct,
        collection_binding_id=dataset.collection_binding_id,
    )


def _detail(session: Session, segment: DocumentSegment, document: Document) -> SegmentDetail[SegmentReadSnapshot]:
    summary = _summary(session, segment)
    return SegmentDetail(
        data=_segment_data(
            segment,
            summary.summary_content if summary else None,
            session,
            _content_file_access((segment,), session)[segment.id],
        ),
        doc_form=str(document.doc_form),
    )


def _require_scope_models(
    session: Session, document_ref: DocumentRef, *, lock: bool = False
) -> tuple[Dataset, Document]:
    if lock:
        dataset = session.scalar(
            select(Dataset)
            .where(
                Dataset.id == document_ref.dataset.dataset_id,
                Dataset.tenant_id == document_ref.dataset.tenant_id,
            )
            .with_for_update(read=True)
            .limit(1)
        )
        document = (
            session.scalar(
                select(Document)
                .where(
                    Document.id == document_ref.document_id,
                    Document.dataset_id == document_ref.dataset.dataset_id,
                    Document.tenant_id == document_ref.dataset.tenant_id,
                )
                .with_for_update()
                .limit(1)
            )
            if dataset is not None
            else None
        )
    else:
        dataset = _get_dataset(session, document_ref.dataset)
        document = _get_document(session, document_ref)
    if dataset is None or document is None:
        raise LookupError("Dataset document scope no longer exists")
    return dataset, document


def _get_segment(session: Session, segment_ref: SegmentRef, *, lock: bool = False) -> DocumentSegment | None:
    statement = (
        select(DocumentSegment)
        .where(
            DocumentSegment.id == segment_ref.segment_id,
            DocumentSegment.document_id == segment_ref.document.document_id,
            DocumentSegment.dataset_id == segment_ref.document.dataset.dataset_id,
            DocumentSegment.tenant_id == segment_ref.document.dataset.tenant_id,
        )
        .limit(1)
    )
    if lock:
        statement = statement.with_for_update()
    return session.scalar(statement)


def _get_child_chunk(session: Session, segment_ref: SegmentRef, child_chunk_id: str) -> ChildChunk | None:
    return session.scalar(
        select(ChildChunk)
        .where(
            ChildChunk.id == child_chunk_id,
            ChildChunk.segment_id == segment_ref.segment_id,
            ChildChunk.document_id == segment_ref.document.document_id,
            ChildChunk.dataset_id == segment_ref.document.dataset.dataset_id,
            ChildChunk.tenant_id == segment_ref.document.dataset.tenant_id,
        )
        .limit(1)
    )


def _content_file_access(segments: Sequence[DocumentSegment], session: Session) -> dict[str, SegmentFileAccess]:
    """Materialize file access for already scoped rows within their original read transaction."""
    if not segments:
        return {}
    references = {segment.id: segment_content_file_ids(segment.content) for segment in segments}
    upload_ids = {file_id for uploads, _ in references.values() for file_id in uploads}
    tool_ids = {file_id for _, tools in references.values() for file_id in tools}
    uploads, tools = get_content_file_ids(
        segments[0].tenant_id, upload_ids=upload_ids, tool_ids=tool_ids, session=session
    )
    return {
        segment_id: SegmentFileAccess(frozenset(upload_ids & uploads), frozenset(tool_ids & tools))
        for segment_id, (upload_ids, tool_ids) in references.items()
    }


def _segment_data(
    segment: DocumentSegment, summary: str | None, session: Session, file_access: SegmentFileAccess
) -> SegmentReadSnapshot:
    return SegmentReadSnapshot(
        values=asdict(segment),
        summary=summary,
        file_access=file_access,
        child_chunks=tuple(
            ChildChunkRecord.model_validate(child)
            for child in get_segment_child_chunks(segment, session=session, include_full_doc=False)
        ),
        attachments=tuple(
            SegmentAttachmentFile(file.id, file.name, file.size, file.extension, file.mime_type)
            for file in get_segment_attachment_files(segment, session=session)
        ),
    )


def _child_chunk_data(child_chunk: ChildChunk) -> ChildChunkRecord:
    return ChildChunkRecord.model_validate(child_chunk)


def query_child_chunks(
    session: Session,
    segment_ref: SegmentRef,
    *,
    page: int,
    limit: int,
    keyword: str | None,
) -> PaginatedResult[ChildChunk]:
    """Share the query with callers that already own an explicit session."""
    query = _children_query(segment_ref.document, (segment_ref.segment_id,)).order_by(ChildChunk.position.asc())
    if keyword:
        query = query.where(ChildChunk.content.ilike(f"%{escape_like_pattern(keyword)}%", escape="\\"))
    return paginate_query(query, session=session, page=page, per_page=limit, max_per_page=100)

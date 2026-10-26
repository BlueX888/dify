import concurrent.futures
import datetime
import logging
import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from flask import current_app
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from core.db.session_factory import session_factory
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.index_processor.index_processor_base import (
    AttachmentVectorStore,
    BaseIndexProcessor,
    ChildVectorRouteStore,
    PipelineIndexingGuardFactory,
    PipelineIndexingRouteRecorderFactory,
    SummaryIndexSettingDict,
)
from core.workflow.nodes.knowledge_index.exc import KnowledgeIndexNodeError
from core.workflow.nodes.knowledge_index.protocols import IndexingResultDict, Preview, PreviewItem, QaPreview
from models.dataset import (
    ChildChunk,
    Dataset,
    Document,
    DocumentSegment,
    DocumentSegmentSummary,
)
from models.enums import SegmentStatus
from repositories.knowledge.segment_cleanup_repository import (
    delete_document_segments_and_claim_uploads,
    lock_segment_attachment_cleanup_dataset,
    require_locked_segment_attachment_document_owner,
)
from services.vector_space_admission_service import VectorSpaceAdmissionService

from .processor.paragraph_index_processor import ParagraphIndexProcessor
from .processor.parent_child_index_processor import ParentChildIndexProcessor
from .processor.qa_index_processor import QAIndexProcessor

logger = logging.getLogger(__name__)


class IndexProcessorFactory:
    """Select the processor for a document's index structure."""

    def __init__(
        self,
        index_type: str | None,
        *,
        attachment_vectors: AttachmentVectorStore | None = None,
        child_vector_routes: ChildVectorRouteStore | None = None,
    ):
        self._index_type = index_type
        self._attachment_vectors = attachment_vectors
        self._child_vector_routes = child_vector_routes

    def init_index_processor(self) -> BaseIndexProcessor:
        """Create a processor for a supported index type."""
        if not self._index_type:
            raise ValueError("Index type must be specified.")

        if self._index_type == IndexStructureType.PARAGRAPH_INDEX:
            return ParagraphIndexProcessor(
                attachment_vectors=self._attachment_vectors,
                child_vector_routes=self._child_vector_routes,
            )
        elif self._index_type == IndexStructureType.QA_INDEX:
            return QAIndexProcessor(
                attachment_vectors=self._attachment_vectors,
                child_vector_routes=self._child_vector_routes,
            )
        elif self._index_type == IndexStructureType.PARENT_CHILD_INDEX:
            return ParentChildIndexProcessor(
                attachment_vectors=self._attachment_vectors,
                child_vector_routes=self._child_vector_routes,
            )
        else:
            raise ValueError(f"Index type {self._index_type} is not supported.")


class IndexProcessor:
    def __init__(
        self,
        *,
        attachment_vectors: AttachmentVectorStore | None = None,
        child_vector_routes: ChildVectorRouteStore | None = None,
        indexing_guard_factory: PipelineIndexingGuardFactory | None = None,
        indexing_route_recorder_factory: PipelineIndexingRouteRecorderFactory | None = None,
    ) -> None:
        self._attachment_vectors = attachment_vectors
        self._child_vector_routes = child_vector_routes
        self._indexing_guard_factory = indexing_guard_factory
        self._indexing_route_recorder_factory = indexing_route_recorder_factory

    def format_preview(self, chunk_structure: str, chunks: Any) -> Preview:
        index_processor = IndexProcessorFactory(chunk_structure).init_index_processor()
        preview = index_processor.format_preview(chunks)
        data = Preview(
            chunk_structure=preview["chunk_structure"],
            total_segments=preview["total_segments"],
            preview=[],
            parent_mode=None,
            qa_preview=[],
        )
        if "parent_mode" in preview:
            data.parent_mode = preview["parent_mode"]

        # Different index processors return different preview shapes:
        # - paragraph/parent-child processors: {"preview": [...]}
        # - QA processor: {"qa_preview": [...]} (no "preview" key)
        for item in preview.get("preview", []):
            if "content" in item and "child_chunks" in item:
                data.preview.append(
                    PreviewItem(content=item["content"], child_chunks=item["child_chunks"], summary=None)
                )
            elif "question" in item and "answer" in item:
                data.qa_preview.append(QaPreview(question=item["question"], answer=item["answer"]))
            elif "content" in item:
                data.preview.append(PreviewItem(content=item["content"], child_chunks=None, summary=None))

        for item in preview.get("qa_preview", []):
            if "question" in item and "answer" in item:
                data.qa_preview.append(QaPreview(question=item["question"], answer=item["answer"]))
        return data

    def index_and_clean(
        self,
        dataset_id: str,
        document_id: str,
        original_document_id: str,
        chunks: Mapping[str, Any],
        batch: Any,
        summary_index_setting: SummaryIndexSettingDict | None = None,
        *,
        session: Session,
        enforce_projected_capacity: bool | None = None,
        indexing_request_id: str | None = None,
        claim_id: str | None = None,
    ) -> IndexingResultDict:
        dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
        if not dataset:
            raise KnowledgeIndexNodeError(f"Dataset {dataset_id} not found.")

        document = session.scalar(
            select(Document)
            .where(
                Document.id == document_id,
                Document.dataset_id == dataset.id,
                Document.tenant_id == dataset.tenant_id,
            )
            .limit(1)
        )
        if not document:
            raise KnowledgeIndexNodeError(f"Document {document_id} not found.")

        if indexing_request_id is None and claim_id is None:
            mutation_guard = None
        else:
            if self._indexing_guard_factory is None:
                raise RuntimeError("pipeline indexing guard factory is required for claimed indexing")
            mutation_guard = self._indexing_guard_factory(
                session,
                dataset_id=dataset.id,
                document_id=document.id,
                indexing_request_id=indexing_request_id,
                claim_id=claim_id,
            )
        with mutation_guard(session) if mutation_guard is not None else nullcontext():
            # Attach the validated row to this transaction before gathering the
            # cleanup snapshot. Claimed mutations below reacquire this guard at
            # every transaction and external-index boundary.
            pass

        dataset_name_value = dataset.name
        document_name_value = document.name
        created_at_value = document.created_at
        if summary_index_setting is None:
            summary_index_setting = dataset.summary_index_setting
        index_node_ids: list[str] = []
        child_node_ids: list[str] = []
        summary_node_ids: list[str] = []

        index_processor = IndexProcessorFactory(
            dataset.chunk_structure,
            attachment_vectors=self._attachment_vectors,
            child_vector_routes=self._child_vector_routes,
        ).init_index_processor()
        if original_document_id:
            segments = session.scalars(
                select(DocumentSegment).where(
                    DocumentSegment.document_id == original_document_id,
                    DocumentSegment.dataset_id == dataset.id,
                    DocumentSegment.tenant_id == dataset.tenant_id,
                )
            ).all()
            if segments:
                index_node_ids = [segment.index_node_id for segment in segments if segment.index_node_id]
            child_node_ids = [
                node_id
                for node_id in session.scalars(
                    select(ChildChunk.index_node_id).where(
                        ChildChunk.tenant_id == dataset.tenant_id,
                        ChildChunk.dataset_id == dataset.id,
                        ChildChunk.document_id == original_document_id,
                        ChildChunk.index_node_id.is_not(None),
                    )
                )
                if node_id
            ]
            summary_node_ids = [
                node_id
                for node_id in session.scalars(
                    select(DocumentSegmentSummary.summary_index_node_id).where(
                        DocumentSegmentSummary.dataset_id == dataset.id,
                        DocumentSegmentSummary.document_id == original_document_id,
                        DocumentSegmentSummary.summary_index_node_id.is_not(None),
                    )
                )
                if node_id
            ]
        high_quality_cleanup_node_ids: list[str] = []
        vector_type = None
        if original_document_id and dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            body_node_ids = (
                child_node_ids if dataset.chunk_structure == IndexStructureType.PARENT_CHILD_INDEX else index_node_ids
            )
            high_quality_cleanup_node_ids = list(dict.fromkeys([*body_node_ids, *summary_node_ids]))
            if high_quality_cleanup_node_ids:
                vector_type = Vector.resolve_vector_type(dataset, session=session)
            session.expunge(dataset)

        indexing_start_at = time.perf_counter()
        # The metadata reads above must not keep a transaction open across vector I/O.
        session.commit()

        if enforce_projected_capacity is None:
            enforce_projected_capacity = not bool(original_document_id)
        if enforce_projected_capacity:
            VectorSpaceAdmissionService().ensure_pipeline_can_be_indexed(
                dataset=dataset,
                document_id=document.id,
                chunk_structure=dataset.chunk_structure,
                chunks=chunks,
                include_summaries=bool(summary_index_setting and summary_index_setting.get("enable")),
                session=session,
            )

        if original_document_id:
            vector = Vector(dataset, session=None, vector_type=vector_type) if vector_type is not None else None
            if high_quality_cleanup_node_ids:
                with mutation_guard(session) if mutation_guard is not None else nullcontext():
                    assert vector is not None
                    vector.delete_by_ids(high_quality_cleanup_node_ids)
                session.commit()
            elif index_node_ids and dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
                with mutation_guard(session) if mutation_guard is not None else nullcontext():
                    index_processor.clean(dataset, index_node_ids, with_keywords=True, session=session)
                session.commit()

            # Dataset UPDATE must precede the claim/document lock. It remains
            # held through the attachment reference check, durable cleanup
            # handoff and relational detach so a publisher cannot slip into
            # the gap.
            lock_segment_attachment_cleanup_dataset(
                session,
                tenant_id=dataset.tenant_id,
                dataset_id=dataset.id,
            )
            with mutation_guard(session) if mutation_guard is not None else nullcontext():
                if dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                    if vector is None:
                        attachment_vector_type = Vector.resolve_vector_type(dataset, session=session)
                        vector = Vector(dataset, session=None, vector_type=attachment_vector_type)
                    actual_vector_type, attachment_collection_name = vector.get_routing_target()

                    if self._attachment_vectors is None:
                        raise RuntimeError("attachment vector store is required for document replacement")
                    self._attachment_vectors.delete_unreferenced(
                        session,
                        tenant_id=dataset.tenant_id,
                        dataset_id=dataset.id,
                        collection_binding_id=dataset.collection_binding_id,
                        indexing_technique=str(dataset.indexing_technique),
                        vector_type=actual_vector_type,
                        collection_name=attachment_collection_name,
                        routing_index_struct=dataset.index_struct,
                        document_ids=(original_document_id,),
                    )
                delete_document_segments_and_claim_uploads(
                    session,
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_ids=(original_document_id,),
                )
            session.commit()

        if mutation_guard is not None and dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            if self._indexing_route_recorder_factory is None:
                raise RuntimeError("pipeline indexing route recorder factory is required for claimed indexing")
            route_recorder = self._indexing_route_recorder_factory(
                session,
                dataset_id=dataset.id,
                document_id=document.id,
                indexing_request_id=indexing_request_id,
                claim_id=claim_id,
            )
            assert route_recorder is not None
            route_vector_type = Vector.resolve_vector_type(dataset, session=session)
            route_vector = Vector(dataset, session=None, vector_type=route_vector_type)
            actual_vector_type, collection_name = route_vector.get_routing_target()
            route_recorder(actual_vector_type, collection_name, str(document.doc_form))

        require_locked_segment_attachment_document_owner(
            session,
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
        )
        if mutation_guard is None:
            index_processor.index(dataset, document, chunks, session)
        else:
            index_processor.index(dataset, document, chunks, session, mutation_guard=mutation_guard)
        session.commit()
        indexing_end_at = time.perf_counter()

        with mutation_guard(session) if mutation_guard is not None else nullcontext():
            word_count = (
                session.scalar(
                    select(func.sum(DocumentSegment.word_count)).where(
                        DocumentSegment.document_id == document_id,
                        DocumentSegment.dataset_id == dataset_id,
                        DocumentSegment.tenant_id == dataset.tenant_id,
                    )
                )
            ) or 0
            document.indexing_latency = indexing_end_at - indexing_start_at
            document.word_count = word_count
            document.need_summary = bool(summary_index_setting and summary_index_setting.get("enable") is True)
            document.indexing_status = "completed"
            document.completed_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            session.add(document)
            # update document segment status
            session.execute(
                update(DocumentSegment)
                .where(
                    DocumentSegment.document_id == document_id,
                    DocumentSegment.dataset_id == dataset_id,
                    DocumentSegment.tenant_id == dataset.tenant_id,
                )
                .values(
                    status=SegmentStatus.COMPLETED,
                    enabled=True,
                    completed_at=datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
                )
            )
        session.flush()

        result: IndexingResultDict = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name_value,
            "batch": batch,
            "document_id": document_id,
            "document_name": document_name_value,
            "created_at": created_at_value.timestamp(),
            "display_status": "completed",
        }
        return result

    def get_preview_output(
        self,
        chunks: Any,
        dataset_id: str,
        document_id: str,
        chunk_structure: str,
        summary_index_setting: SummaryIndexSettingDict | None,
        *,
        session: Session,
    ) -> Preview:
        doc_language = None
        dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
        if not dataset:
            raise KnowledgeIndexNodeError(f"Dataset {dataset_id} not found.")

        if document_id:
            document = session.scalar(
                select(Document)
                .where(
                    Document.id == document_id,
                    Document.dataset_id == dataset.id,
                    Document.tenant_id == dataset.tenant_id,
                )
                .limit(1)
            )
        else:
            document = None

        if summary_index_setting is None:
            summary_index_setting = dataset.summary_index_setting

        if document:
            doc_language = document.doc_language
        indexing_technique = dataset.indexing_technique
        tenant_id = dataset.tenant_id
        session.commit()

        preview_output = self.format_preview(chunk_structure, chunks)
        if indexing_technique != IndexTechniqueType.HIGH_QUALITY:
            return preview_output

        if not summary_index_setting or not summary_index_setting.get("enable"):
            return preview_output

        if preview_output.preview is not None:
            chunk_count = len(preview_output.preview)
            logger.info(
                "Generating summaries for %s chunks in preview mode (dataset: %s)",
                chunk_count,
                dataset_id,
            )

            flask_app = None
            try:
                flask_app = current_app._get_current_object()  # type: ignore
            except RuntimeError:
                logger.warning("No Flask application context available, summary generation may fail")

            def generate_summary_for_chunk(preview_item: PreviewItem) -> None:
                """Generate summary for a single chunk."""
                if flask_app:
                    with flask_app.app_context():
                        with session_factory.create_session() as worker_session:
                            summary, _ = ParagraphIndexProcessor.generate_summary(
                                tenant_id=tenant_id,
                                text=preview_item.content if preview_item.content is not None else "",
                                summary_index_setting=summary_index_setting,
                                document_language=doc_language,
                                session=worker_session,
                            )
                            if summary:
                                preview_item.summary = summary

            # Generate summaries concurrently using ThreadPoolExecutor
            # Set a reasonable timeout to prevent hanging (60 seconds per chunk, max 5 minutes total)
            timeout_seconds = min(300, 60 * len(preview_output.preview))
            errors: list[Exception] = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(preview_output.preview))) as executor:
                futures = [
                    executor.submit(generate_summary_for_chunk, preview_item) for preview_item in preview_output.preview
                ]
                # Wait for all tasks to complete with timeout
                done, not_done = concurrent.futures.wait(futures, timeout=timeout_seconds)

                # Cancel tasks that didn't complete in time
                if not_done:
                    timeout_error_msg = (
                        f"Summary generation timeout: {len(not_done)} chunks did not complete within {timeout_seconds}s"
                    )
                    logger.warning("%s. Cancelling remaining tasks...", timeout_error_msg)
                    # In preview mode, timeout is also an error
                    errors.append(TimeoutError(timeout_error_msg))
                    for future in not_done:
                        future.cancel()
                    # Wait a bit for cancellation to take effect
                    concurrent.futures.wait(not_done, timeout=5)

                # Collect exceptions from completed futures
                for future in done:
                    try:
                        future.result()  # This will raise any exception that occurred
                    except Exception as e:
                        logger.exception("Error in summary generation future")
                        errors.append(e)

            # In preview mode, if there are any errors, fail the request
            if errors:
                error_messages = [str(e) for e in errors]
                error_summary = (
                    f"Failed to generate summaries for {len(errors)} chunk(s). "
                    f"Errors: {'; '.join(error_messages[:3])}"  # Show first 3 errors
                )
                if len(errors) > 3:
                    error_summary += f" (and {len(errors) - 3} more)"
                logger.error("Summary generation failed in preview mode: %s", error_summary)
                raise KnowledgeIndexNodeError(error_summary)

            completed_count = sum(1 for item in preview_output.preview if item.summary is not None)
            logger.info(
                "Completed summary generation for preview chunks: %s/%s succeeded",
                completed_count,
                len(preview_output.preview),
            )
        return preview_output

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import click
from celery import shared_task
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from core.db.session_factory import session_factory
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.index_processor import IndexProcessorFactory
from extensions.ext_storage import storage
from models.dataset import Dataset, Document, DocumentSegmentSummary
from repositories.knowledge.attachment_vector_repository import (
    SQLAlchemyAttachmentVectorStore,
    delete_unreferenced_attachment_vectors,
)
from repositories.knowledge.segment_cleanup_repository import detach_owned_segment_attachments
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.upload_file_cleanup import UploadFileCleanupArtifact, cleanup_claimed_upload_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SegmentDeleteCleanupResult:
    files: SQLAlchemyKnowledgeUploadRepository | None
    artifacts: Sequence[UploadFileCleanupArtifact]


def cleanup_segment_delete_blobs(result: SegmentDeleteCleanupResult) -> None:
    if result.files is None:
        return
    cleanup_claimed_upload_files(
        artifacts=result.artifacts,
        files=result.files,
        delete_blob=storage.delete,
        logger=logger,
    )


def cleanup_segment_index_in_session(
    session: Session,
    *,
    tenant_id: str | None,
    dataset_id: str,
    document_id: str,
    segment_ids: Sequence[str],
    index_node_ids: Sequence[str],
    child_node_ids: Sequence[str],
    doc_form: str | None,
    upload_session_factory: sessionmaker[Session],
) -> SegmentDeleteCleanupResult:
    """Delete captured segment artifacts while the caller owns durable intent."""

    dataset_stmt = select(Dataset).where(Dataset.id == dataset_id)
    if tenant_id is not None:
        dataset_stmt = dataset_stmt.where(Dataset.tenant_id == tenant_id)
    dataset = session.scalar(dataset_stmt.limit(1))
    if dataset is None:
        return SegmentDeleteCleanupResult(None, ())

    document_stmt = select(Document).where(
        Document.id == document_id,
        Document.dataset_id == dataset.id,
        Document.tenant_id == dataset.tenant_id,
    )
    dataset_document = session.scalar(document_stmt.limit(1))
    resolved_doc_form = dataset_document.doc_form if dataset_document is not None else doc_form
    if resolved_doc_form is None:
        raise LookupError("Cannot resolve document form for segment index cleanup")

    processor = IndexProcessorFactory(
        resolved_doc_form,
        attachment_vectors=SQLAlchemyAttachmentVectorStore(),
    ).init_index_processor()
    if index_node_ids:
        processor.clean(
            dataset,
            list(index_node_ids),
            with_keywords=True,
            delete_child_chunks=True,
            precomputed_child_node_ids=[],
            delete_summaries=False,
            session=session,
        )
    summaries = session.scalars(
        select(DocumentSegmentSummary)
        .where(
            DocumentSegmentSummary.dataset_id == dataset.id,
            DocumentSegmentSummary.document_id == document_id,
            DocumentSegmentSummary.chunk_id.in_(segment_ids),
        )
        .with_for_update()
    ).all()
    SQLAlchemySummaryVectorCleanupRepository.add_terminal_cleanups(
        session,
        tenant_id=dataset.tenant_id,
        summaries=summaries,
    )
    if summaries:
        session.execute(
            delete(DocumentSegmentSummary).where(DocumentSegmentSummary.id.in_([row.id for row in summaries]))
        )

    uploads = SQLAlchemyKnowledgeUploadRepository(session_factory=upload_session_factory)
    if dataset.is_multimodal:
        attachment_vector_type = Vector.resolve_vector_type(dataset, session=session)
        attachment_vector = Vector(dataset, session=session, vector_type=attachment_vector_type)
        actual_vector_type, collection_name = attachment_vector.get_routing_target()
        attachment_route = VectorRoute.capture(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            vector_type=actual_vector_type,
            collection_name=collection_name,
            routing_index_struct=dataset.index_struct,
            collection_binding_id=dataset.collection_binding_id,
            indexing_technique=str(dataset.indexing_technique),
            doc_form="attachment",
        )
        delete_unreferenced_attachment_vectors(
            session,
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_ids=(document_id,),
            segment_ids=tuple(segment_ids),
            route=attachment_route,
        )
    detached_attachment_ids = detach_owned_segment_attachments(
        session,
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_ids=(document_id,),
        segment_ids=tuple(segment_ids),
    )
    artifacts = uploads.claim_unreferenced_files_for_cleanup_in_session(
        session,
        workspace_id=dataset.tenant_id,
        file_ids=detached_attachment_ids,
    )
    return SegmentDeleteCleanupResult(uploads, artifacts)


@shared_task(queue="dataset")
def delete_segment_from_index_task(
    index_node_ids: list, dataset_id: str, document_id: str, segment_ids: list, child_node_ids: list | None = None
):
    """
    Async Remove segment from index
    :param index_node_ids:
    :param dataset_id:
    :param document_id:

    Usage: delete_segment_from_index_task.delay(index_node_ids, dataset_id, document_id)
    """
    logger.info(click.style("Start delete segment from index", fg="green"))
    start_at = time.perf_counter()
    with session_factory.create_session() as session:
        try:
            result = cleanup_segment_index_in_session(
                session,
                tenant_id=None,
                dataset_id=dataset_id,
                document_id=document_id,
                segment_ids=segment_ids,
                index_node_ids=index_node_ids,
                child_node_ids=child_node_ids or (),
                doc_form=None,
                upload_session_factory=session_factory.get_session_maker(),
            )
            session.commit()
            cleanup_segment_delete_blobs(result)

            end_at = time.perf_counter()
            logger.info(click.style(f"Segment deleted from index latency: {end_at - start_at}", fg="green"))
        except Exception:
            session.rollback()
            logger.exception("delete segment from index failed")

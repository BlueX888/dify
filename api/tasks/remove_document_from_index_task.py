import logging
import time

import click
from celery import shared_task
from sqlalchemy import select

from core.db.session_factory import session_factory
from core.rag.index_processor.index_processor import IndexProcessorFactory
from extensions.ext_redis import redis_client
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import IndexingStatus
from repositories.knowledge.document_repository import has_active_indexing_claim
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository

logger = logging.getLogger(__name__)


def _clear_indexing_hint(document_id: str) -> None:
    try:
        redis_client.delete(f"document_{document_id}_indexing")
    except Exception:
        # Redis is only a dispatch hint; it must not decide whether the durable
        # database/vector transition commits.
        logger.warning("Failed to clear document indexing hint for %s", document_id, exc_info=True)


@shared_task(queue="dataset")
def remove_document_from_index_task(document_id: str) -> bool:
    """Remove only a still-current disabled or archived document generation.

    The queue payload is not ownership. A non-locking lookup supplies only a
    candidate owner scope; Dataset SHARE and Document UPDATE current reads then
    rebuild that scope and serialize this cleanup with enable/index writers.
    Those locks remain held across summary handoff, body-vector cleanup, segment
    state changes, and commit. A stale queued disable therefore cannot clean a
    document that a newer operation has already enabled.
    """
    logger.info(click.style(f"Start remove document segments from index: {document_id}", fg="green"))
    start_at = time.perf_counter()

    with session_factory.create_session() as session:
        candidate_scope = session.execute(
            select(Document.tenant_id, Document.dataset_id).where(Document.id == document_id).limit(1)
        ).one_or_none()
        if candidate_scope is None:
            logger.info(click.style(f"Document not found: {document_id}", fg="red"))
            return False

        tenant_id, dataset_id = candidate_scope
        dataset = session.scalar(
            select(Dataset)
            .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        document = session.scalar(
            select(Document)
            .where(
                Document.id == document_id,
                Document.tenant_id == tenant_id,
                Document.dataset_id == dataset_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        now = naive_utc_now()
        active_claim = bool(
            document is not None
            and document.indexing_claim_id is not None
            and (document.indexing_claim_expires_at is None or has_active_indexing_claim(session, document))
        )
        if (
            dataset is None
            or document is None
            or document.indexing_status != IndexingStatus.COMPLETED
            or (document.enabled and not document.archived)
            or active_claim
        ):
            logger.info("Skipped stale or ineligible document removal, document_id=%s", document_id)
            return False

        try:
            index_processor = IndexProcessorFactory(document.doc_form).init_index_processor()
            segments = session.scalars(
                select(DocumentSegment)
                .where(
                    DocumentSegment.tenant_id == dataset.tenant_id,
                    DocumentSegment.dataset_id == dataset.id,
                    DocumentSegment.document_id == document.id,
                )
                .order_by(DocumentSegment.id)
                .with_for_update()
                .execution_options(populate_existing=True)
            ).all()

            # Disable summary indexes for all segments in this document
            segment_ids_list = [segment.id for segment in segments]
            if segment_ids_list:
                summaries = session.scalars(
                    select(DocumentSegmentSummary)
                    .where(
                        DocumentSegmentSummary.dataset_id == dataset.id,
                        DocumentSegmentSummary.document_id == document.id,
                        DocumentSegmentSummary.chunk_id.in_(segment_ids_list),
                    )
                    .order_by(DocumentSegmentSummary.id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                ).all()
                SQLAlchemySummaryVectorCleanupRepository.add_terminal_cleanups(
                    session,
                    tenant_id=dataset.tenant_id,
                    summaries=summaries,
                )
                for summary in summaries:
                    summary.version += 1
                    summary.enabled = False
                    summary.disabled_at = now
                    summary.disabled_by = document.disabled_by

            index_node_ids = [segment.index_node_id for segment in segments if segment.index_node_id]
            if index_node_ids:
                try:
                    index_processor.clean(
                        dataset,
                        index_node_ids,
                        with_keywords=True,
                        delete_child_chunks=False,
                        session=session,
                    )
                except Exception:
                    logger.exception("clean dataset %s from index failed", dataset.id)
            for segment in segments:
                segment.enabled = False
                segment.disabled_at = now
                segment.disabled_by = document.disabled_by
                segment.updated_at = now
            # Clear the old task's hint before releasing Document ownership, so
            # a newer enable cannot publish its hint and then have it deleted by
            # this worker after commit.
            _clear_indexing_hint(document.id)
            session.commit()

            end_at = time.perf_counter()
            logger.info(
                click.style(
                    f"Document removed from index: {document.id} latency: {end_at - start_at}",
                    fg="green",
                )
            )
            return True
        except Exception:
            logger.exception("remove document from index failed")
            _clear_indexing_hint(document.id)
            session.rollback()
            return False

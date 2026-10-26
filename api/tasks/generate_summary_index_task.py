"""Async task for generating summary indexes."""

import logging
import time

import click
from celery import shared_task
from sqlalchemy import select

from core.db.session_factory import session_factory
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from models.dataset import Document as DatasetDocument
from repositories.knowledge.dataset_indexing_outbox_repository import lock_dataset_indexing_generation_shared
from services.knowledge.summaries.adapters import SummaryIndexAdapter

logger = logging.getLogger(__name__)


@shared_task(queue="dataset_summary")
def generate_summary_index_task(
    dataset_id: str,
    document_id: str,
    segment_ids: list[str] | None = None,
    indexing_request_id: str | None = None,
):
    """Generate document summaries for the generation captured at dispatch.

    A missing token is the legacy ``NULL`` generation, not permission to adopt
    whichever generation is current when the task starts. Superseded tasks exit
    without changing summary versions, content, or errors; same-generation
    failures keep their normal Celery retry/error behavior.

    Args:
        dataset_id: Dataset ID
        document_id: Document ID
        segment_ids: Optional list of specific segment IDs to process. If None, process all segments.
        indexing_request_id: Exact Dataset generation captured by the dispatcher.

    Usage:
        generate_summary_index_task.delay(dataset_id, document_id)
        generate_summary_index_task.delay(dataset_id, document_id, segment_ids)
    """
    logger.info(
        click.style(
            f"Start generating summary index for document {document_id} in dataset {dataset_id}",
            fg="green",
        )
    )
    start_at = time.perf_counter()

    try:
        with session_factory.create_session() as session:
            dataset = lock_dataset_indexing_generation_shared(
                session,
                dataset_id=dataset_id,
                generation_id=indexing_request_id,
            )
            if not dataset:
                logger.info(
                    click.style(
                        f"Skipping superseded summary generation for dataset {dataset_id}: "
                        f"generation={indexing_request_id}",
                        fg="cyan",
                    )
                )
                return

            document = session.scalar(
                select(DatasetDocument).where(
                    DatasetDocument.id == document_id,
                    DatasetDocument.dataset_id == dataset.id,
                    DatasetDocument.tenant_id == dataset.tenant_id,
                )
            )
            if not document:
                logger.error(click.style(f"Document not found: {document_id}", fg="red"))
                return

            # Check if document needs summary
            if not document.need_summary:
                logger.info(
                    click.style(
                        f"Skipping summary generation for document {document_id}: need_summary is False",
                        fg="cyan",
                    )
                )
                return

            # Only generate summary index for high_quality indexing technique
            if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
                logger.info(
                    click.style(
                        f"Skipping summary generation for dataset {dataset_id}: "
                        f"indexing_technique is {dataset.indexing_technique}, not 'high_quality'",
                        fg="cyan",
                    )
                )
                return

            # Check if summary index is enabled
            summary_index_setting = dataset.summary_index_setting
            if not summary_index_setting or not summary_index_setting.get("enable"):
                logger.info(
                    click.style(
                        f"Summary index is disabled for dataset {dataset_id}",
                        fg="cyan",
                    )
                )
                return

            # Determine if only parent chunks should be processed
            only_parent_chunks = dataset.chunk_structure == "parent_child_index"

            # Release the task's read transaction before the adapter invokes models.
            session.commit()
            summary_records = SummaryIndexAdapter.generate_summaries_for_document(
                dataset=dataset,
                document=document,
                summary_index_setting=summary_index_setting,
                segment_ids=segment_ids,
                only_parent_chunks=only_parent_chunks,
                indexing_request_id=indexing_request_id,
            )

            end_at = time.perf_counter()
            logger.info(
                click.style(
                    f"Summary index generation completed for document {document_id}: "
                    f"{len(summary_records)} summaries generated, latency: {end_at - start_at}",
                    fg="green",
                )
            )

    except Exception:
        logger.exception("Failed to generate summary index for document %s", document_id)

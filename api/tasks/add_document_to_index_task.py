import logging

import click
from celery import shared_task
from sqlalchemy import select

from core.db.session_factory import session_factory
from extensions.ext_redis import redis_client
from models.dataset import Dataset, DocumentSegment
from models.dataset import Document as DatasetDocument
from models.enums import IndexingStatus, SegmentStatus
from models.model import SegmentStateOutbox
from tasks.segment_state_outbox_task import publish_segment_state_intents

logger = logging.getLogger(__name__)


@shared_task(queue="dataset")
def add_document_to_index_task(dataset_document_id: str) -> bool:
    """Forward a durable document-enable reconciliation without reauthorizing it.

    The document status transaction owns creation of exact SegmentStateOutbox
    rows. This rolling-deploy bridge publishes only intents still referenced by
    the current enabled segments; a later segment/document disable therefore
    wins without relying on queue ordering.
    """

    logger.info(click.style(f"Forward document enable intents: {dataset_document_id}", fg="green"))
    indexing_cache_key = f"document_{dataset_document_id}_indexing"
    try:
        with session_factory.create_session() as session:
            dataset_document = session.scalar(
                select(DatasetDocument).where(DatasetDocument.id == dataset_document_id).limit(1)
            )
            if dataset_document is None:
                return False
            dataset = session.scalar(
                select(Dataset)
                .where(Dataset.id == dataset_document.dataset_id, Dataset.tenant_id == dataset_document.tenant_id)
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if dataset is None:
                return False
            dataset_document = session.scalar(
                select(DatasetDocument)
                .where(
                    DatasetDocument.id == dataset_document_id,
                    DatasetDocument.dataset_id == dataset.id,
                    DatasetDocument.tenant_id == dataset.tenant_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if (
                dataset_document is None
                or dataset_document.indexing_status != IndexingStatus.COMPLETED
                or not dataset_document.enabled
                or dataset_document.archived
                or dataset_document.indexing_request_id is not None
                or dataset_document.indexing_claim_id is not None
            ):
                return False
            intent_ids = tuple(
                session.scalars(
                    select(SegmentStateOutbox.id)
                    .join(DocumentSegment, DocumentSegment.indexing_request_id == SegmentStateOutbox.id)
                    .where(
                        DocumentSegment.tenant_id == dataset.tenant_id,
                        DocumentSegment.dataset_id == dataset.id,
                        DocumentSegment.document_id == dataset_document.id,
                        DocumentSegment.status == SegmentStatus.COMPLETED,
                        DocumentSegment.enabled.is_(True),
                        SegmentStateOutbox.tenant_id == dataset.tenant_id,
                        SegmentStateOutbox.dataset_id == dataset.id,
                        SegmentStateOutbox.document_id == dataset_document.id,
                        SegmentStateOutbox.action == "enable",
                    )
                    .order_by(SegmentStateOutbox.id)
                ).all()
            )
        if not intent_ids:
            logger.info("No current document enable intents to forward, document_id=%s", dataset_document_id)
            return False
        try:
            publish_segment_state_intents(intent_ids)
        except Exception:
            # The committed outbox rows remain discoverable by the periodic
            # drainer, so a broker outage cannot lose the enable request.
            logger.exception("Failed to publish document enable intents for %s", dataset_document_id)
        logger.info(
            "Document enable remains pending durable reconciliation, document_id=%s, intents=%s",
            dataset_document_id,
            len(intent_ids),
        )
        return False
    finally:
        redis_client.delete(indexing_cache_key)

"""Rolling-deployment bridge for legacy dataset rebuild messages."""

import logging

from celery import shared_task

from core.db.session_factory import session_factory
from repositories.knowledge.dataset_indexing_outbox_repository import SQLAlchemyDatasetIndexingOutboxRepository

logger = logging.getLogger(__name__)


@shared_task(queue="dataset")
def deal_dataset_index_update_task(dataset_id: str, action: str) -> None:
    """Redirect a legacy action-only message to the trusted current durable intent."""
    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=session_factory.get_session_maker())
    intent_id = repository.find_current_intent_id(dataset_id=dataset_id, generation_id=None)
    if intent_id is None:
        logger.info(
            "Ignoring legacy dataset rebuild without a current durable intent, dataset_id=%s, action=%s",
            dataset_id,
            action,
        )
        return
    from tasks.dataset_indexing_outbox_task import enqueue_dataset_indexing_intent

    enqueue_dataset_indexing_intent(intent_id)

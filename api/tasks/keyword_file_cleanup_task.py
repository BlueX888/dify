"""Drain durable cleanup intents for immutable keyword-table files."""

import logging

from celery import shared_task

from core.db.session_factory import session_factory
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from repositories.knowledge.keyword_file_cleanup_repository import (
    SQLAlchemyKeywordFileCleanupRepository,
)
from services.knowledge.keyword_file_cleanup import drain_pending_keyword_file_cleanups

logger = logging.getLogger(__name__)

_CLEANUP_BATCH_SIZE = 100
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 60


@shared_task(queue="dataset", bind=True, max_retries=_MAX_RETRIES, default_retry_delay=_RETRY_DELAY_SECONDS)
def drain_keyword_file_cleanup_outbox_task(self) -> None:
    repository = SQLAlchemyKeywordFileCleanupRepository(session_factory=session_factory.get_session_maker())
    failed = drain_pending_keyword_file_cleanups(
        repository=repository,
        delete_file=storage.delete,
        publication_lock=lambda artifact: redis_client.lock(
            f"keyword_indexing_lock_{artifact.dataset_id}", timeout=600
        ),
        logger=logger,
        limit=_CLEANUP_BATCH_SIZE,
    )
    if failed:
        logger.warning("Keyword file cleanup batch has retryable failures, cleanup_ids=%s", failed)
        raise self.retry(exc=RuntimeError("keyword file cleanup batch did not complete"))

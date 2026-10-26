import logging

from celery import shared_task

from core.db.session_factory import session_factory
from extensions.ext_storage import storage
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.upload_file_cleanup import drain_pending_upload_file_cleanups

logger = logging.getLogger(__name__)

_CLEANUP_BATCH_SIZE = 100
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 60


@shared_task(queue="dataset", bind=True, max_retries=_MAX_RETRIES, default_retry_delay=_RETRY_DELAY_SECONDS)
def drain_upload_file_cleanup_outbox_task(self) -> None:
    """Retry durable blob deletions independently of the resource cleanup that created them."""
    failed_cleanup_ids = drain_pending_upload_file_cleanups(
        files=SQLAlchemyKnowledgeUploadRepository(session_factory=session_factory.get_session_maker()),
        delete_blob=storage.delete,
        logger=logger,
        limit=_CLEANUP_BATCH_SIZE,
    )
    if failed_cleanup_ids:
        logger.warning(
            "Upload blob cleanup batch has retryable failures, cleanup_ids=%s",
            failed_cleanup_ids,
        )
        raise self.retry(exc=RuntimeError("upload blob cleanup batch did not complete"))

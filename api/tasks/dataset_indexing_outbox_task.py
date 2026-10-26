"""Execute durable dataset configuration index rebuild intents."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from threading import Event, Thread
from typing import Protocol

from celery import shared_task

from core.db.session_factory import session_factory
from repositories.knowledge.dataset_indexing_outbox_repository import (
    DatasetIndexingClaimLostError,
    DatasetIndexingIntent,
    SQLAlchemyDatasetIndexingOutboxRepository,
)
from tasks.deal_dataset_vector_index_task import deal_dataset_vector_index_task
from tasks.regenerate_summary_index_task import regenerate_summary_index_task

logger = logging.getLogger(__name__)

_DRAIN_BATCH_SIZE = 100
_CLAIM_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


class DatasetIndexingClaimStore(Protocol):
    def renew_claim(self, intent_id: str, *, claim_id: str) -> None: ...


def _repository() -> SQLAlchemyDatasetIndexingOutboxRepository:
    return SQLAlchemyDatasetIndexingOutboxRepository(session_factory=session_factory.get_session_maker())


def enqueue_dataset_indexing_intent(intent_id: str) -> None:
    """Best-effort fast path; the periodic outbox drain recovers broker failures."""
    try:
        process_dataset_indexing_intent_task.delay(intent_id)
    except Exception:
        logger.exception(
            "Failed to publish dataset indexing intent; periodic drain will retry, intent_id=%s",
            intent_id,
        )


@contextmanager
def _maintain_dataset_indexing_claim(
    repository: DatasetIndexingClaimStore,
    intent: DatasetIndexingIntent,
    *,
    interval_seconds: float = _CLAIM_HEARTBEAT_SECONDS,
) -> Generator[None]:
    if interval_seconds <= 0:
        raise ValueError("dataset indexing claim heartbeat interval must be positive")
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                repository.renew_claim(intent.id, claim_id=intent.claim_id)
            except DatasetIndexingClaimLostError:
                logger.info("Dataset indexing heartbeat lost ownership, intent_id=%s", intent.id)
                return
            except Exception:
                logger.exception("Dataset indexing heartbeat failed, intent_id=%s", intent.id)

    thread = Thread(target=heartbeat, name=f"dataset-indexing-heartbeat-{intent.id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning("Dataset indexing heartbeat did not stop promptly, intent_id=%s", intent.id)


@shared_task(queue="dataset")
def process_dataset_indexing_intent_task(intent_id: str) -> None:
    """Run vector then summary phases and acknowledge only complete work."""
    repository = _repository()
    intent = repository.claim(intent_id)
    if intent is None:
        return
    try:
        with _maintain_dataset_indexing_claim(repository, intent):
            if not intent.vector_completed:
                completed = deal_dataset_vector_index_task.run(
                    intent.dataset_id,
                    intent.action,
                    intent.generation_id,
                    indexing_intent_id=intent.id,
                    indexing_claim_id=intent.claim_id,
                    tenant_id=intent.tenant_id,
                    source_representation=intent.source.model_dump_json(),
                    target_representation=intent.target.model_dump_json(),
                )
                if not completed or not repository.mark_vector_completed(intent):
                    raise DatasetIndexingClaimLostError(intent.id)
            # Automatic generation controls creation of new summary text. It
            # does not invalidate already-published, enabled summaries. Every
            # high-quality representation must restore retained vectors and,
            # when enabled, fill missing/pending text before the Dataset
            # transition can be acknowledged.
            if intent.target.indexing_technique == "high_quality":
                completed = regenerate_summary_index_task.run(
                    intent.dataset_id,
                    regenerate_reason="embedding_model_changed",
                    regenerate_vectors_only=True,
                    indexing_request_id=intent.generation_id,
                    indexing_intent_id=intent.id,
                    indexing_claim_id=intent.claim_id,
                    tenant_id=intent.tenant_id,
                )
                if not completed:
                    raise RuntimeError(f"Summary rebuild did not complete for intent {intent.id}")
            if not repository.complete(intent):
                raise DatasetIndexingClaimLostError(intent.id)
    except DatasetIndexingClaimLostError:
        logger.info("Dataset indexing execution lost ownership, intent_id=%s", intent.id)
        return
    except Exception:
        repository.release(intent)
        raise


@shared_task(queue="dataset")
def drain_dataset_indexing_outbox_task() -> None:
    """Publish a bounded fair batch while retaining execution until success."""
    repository = _repository()
    for intent_id in repository.list_pending_intent_ids(limit=_DRAIN_BATCH_SIZE):
        enqueue_dataset_indexing_intent(intent_id)
        repository.mark_attempted(intent_id)

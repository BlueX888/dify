"""Reconcile durable cleanup intents for unpublished summary vectors."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from threading import Event, Thread

from celery import shared_task

from core.db.session_factory import session_factory
from repositories.knowledge.summary_vector_cleanup_repository import (
    SQLAlchemySummaryVectorCleanupRepository,
    SummaryVectorCleanupClaimLostError,
)
from services.knowledge.entities.summaries import SummaryVectorCleanupClaim
from services.knowledge.indexing.vector_route import vector_for_route

logger = logging.getLogger(__name__)

_DRAIN_BATCH_SIZE = 100
_CLAIM_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


def _repository() -> SQLAlchemySummaryVectorCleanupRepository:
    return SQLAlchemySummaryVectorCleanupRepository(session_factory=session_factory.get_session_maker())


def enqueue_summary_vector_cleanup(cleanup_id: str) -> None:
    try:
        process_summary_vector_cleanup_task.delay(cleanup_id)
    except Exception:
        logger.exception(
            "Failed to publish summary vector cleanup; periodic drain will retry, cleanup_id=%s", cleanup_id
        )


@contextmanager
def _maintain_cleanup_claim(
    repository: SQLAlchemySummaryVectorCleanupRepository,
    claim: SummaryVectorCleanupClaim,
    *,
    interval_seconds: float = _CLAIM_HEARTBEAT_SECONDS,
) -> Generator[None]:
    if interval_seconds <= 0:
        raise ValueError("summary cleanup heartbeat interval must be positive")
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                repository.renew_cleanup(claim)
            except SummaryVectorCleanupClaimLostError:
                logger.info("Summary vector cleanup lost ownership, cleanup_id=%s", claim.id)
                return
            except Exception:
                logger.exception("Summary vector cleanup heartbeat failed, cleanup_id=%s", claim.id)

    thread = Thread(target=heartbeat, name=f"summary-cleanup-heartbeat-{claim.id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning("Summary vector cleanup heartbeat did not stop promptly, cleanup_id=%s", claim.id)


@shared_task(queue="dataset")
def process_summary_vector_cleanup_task(cleanup_id: str) -> None:
    repository = _repository()
    claim = repository.claim_pending_cleanup(cleanup_id)
    if claim is None:
        return
    try:
        with _maintain_cleanup_claim(repository, claim):
            if claim.route is None:
                logger.error("Summary vector cleanup has no trustworthy persisted route, cleanup_id=%s", claim.id)
                repository.release_cleanup(claim)
                return
            with repository.guard_cleanup_delete(claim) as referenced:
                if not referenced:
                    vector_for_route(claim.route).delete_by_ids([claim.node_id])
            if not claim.writer_finished:
                # Deletion raced a potentially in-flight writer. Never
                # acknowledge based on a completion marker observed after this
                # delete; a later claim must delete once more before it can ACK.
                repository.release_cleanup(claim)
    except Exception:
        repository.release_cleanup(claim)
        raise


@shared_task(queue="dataset")
def drain_summary_vector_cleanup_outbox_task() -> None:
    repository = _repository()
    for cleanup_id in repository.list_pending_cleanup_ids(limit=_DRAIN_BATCH_SIZE):
        try:
            enqueue_summary_vector_cleanup(cleanup_id)
        finally:
            repository.mark_attempted(cleanup_id)

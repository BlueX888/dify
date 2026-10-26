"""Reconcile durable child-vector writer and cleanup intents."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from threading import Event, Thread

from celery import shared_task

from core.db.session_factory import session_factory
from core.rag.datasource.vdb.vector_factory import Vector
from models.dataset import Dataset
from repositories.knowledge.child_vector_cleanup_repository import (
    ChildVectorCleanupClaimLostError,
    SQLAlchemyChildVectorCleanupRepository,
)
from services.knowledge.entities.segments import ChildVectorCleanupClaim, ChildVectorRoute

logger = logging.getLogger(__name__)

_DRAIN_BATCH_SIZE = 100
_CLAIM_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


def _vector_for_route(route: ChildVectorRoute) -> Vector:
    dataset = Dataset(
        id=route.dataset_id,
        tenant_id=route.tenant_id,
        indexing_technique=route.indexing_technique,
        index_struct=route.index_struct,
        collection_binding_id=None,
    )
    vector = Vector(dataset, session=None, vector_type=route.vector_type)
    if vector.get_routing_target() != (route.vector_type, route.collection_name):
        raise RuntimeError("Child vector cleanup route did not resolve to its persisted target")
    return vector


def _repository() -> SQLAlchemyChildVectorCleanupRepository:
    return SQLAlchemyChildVectorCleanupRepository(session_factory=session_factory.get_session_maker())


def enqueue_child_vector_cleanup(cleanup_id: str) -> None:
    try:
        process_child_vector_cleanup_task.delay(cleanup_id)
    except Exception:
        logger.exception("Failed to publish child vector cleanup; periodic drain will retry, cleanup_id=%s", cleanup_id)


@contextmanager
def _maintain_cleanup_claim(
    repository: SQLAlchemyChildVectorCleanupRepository,
    claim: ChildVectorCleanupClaim,
    *,
    interval_seconds: float = _CLAIM_HEARTBEAT_SECONDS,
) -> Generator[None]:
    if interval_seconds <= 0:
        raise ValueError("child vector cleanup heartbeat interval must be positive")
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                repository.renew_cleanup(claim)
            except ChildVectorCleanupClaimLostError:
                logger.info("Child vector cleanup lost ownership, cleanup_id=%s", claim.id)
                return
            except Exception:
                logger.exception("Child vector cleanup heartbeat failed, cleanup_id=%s", claim.id)

    thread = Thread(target=heartbeat, name=f"child-vector-cleanup-heartbeat-{claim.id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning("Child vector cleanup heartbeat did not stop promptly, cleanup_id=%s", claim.id)


@shared_task(queue="dataset")
def process_child_vector_cleanup_task(cleanup_id: str) -> None:
    repository = _repository()
    claim = repository.claim_pending_cleanup(cleanup_id)
    if claim is None:
        return
    try:
        with _maintain_cleanup_claim(repository, claim):
            if claim.route is None:
                logger.error("Child vector cleanup has no trustworthy persisted route, cleanup_id=%s", claim.id)
                repository.release_cleanup(claim)
                return
            with repository.guard_cleanup_delete(claim) as referenced:
                if not referenced:
                    _vector_for_route(claim.route).delete_by_ids([claim.node_id])
            if not claim.writer_finished:
                # A call that returned an unknown result may still finish after
                # this delete. Keep the row for a later terminal cleanup pass.
                repository.release_cleanup(claim)
    except Exception:
        repository.release_cleanup(claim)
        raise


@shared_task(queue="dataset")
def drain_child_vector_cleanup_outbox_task() -> None:
    repository = _repository()
    for cleanup_id in repository.list_pending_cleanup_ids(limit=_DRAIN_BATCH_SIZE):
        try:
            enqueue_child_vector_cleanup(cleanup_id)
        finally:
            # Rotate every attempted delivery, including broker rejection, so
            # one slow or unavailable queue cannot hide newer cleanup work.
            repository.mark_attempted(cleanup_id)

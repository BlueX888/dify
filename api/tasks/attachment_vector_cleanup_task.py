"""Reconcile durable attachment-vector writer and cleanup receipts."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from threading import Event, Thread

from celery import shared_task

from core.db.session_factory import session_factory
from core.rag.datasource.vdb.vector_factory import Vector
from models.dataset import Dataset
from repositories.knowledge.attachment_vector_repository import (
    AttachmentVectorCleanupClaim,
    AttachmentVectorCleanupClaimLostError,
    AttachmentVectorCleanupGuard,
    SQLAlchemyAttachmentVectorStore,
)
from services.knowledge.entities.vector_route import VectorRoute

logger = logging.getLogger(__name__)

_DRAIN_BATCH_SIZE = 100
_CLAIM_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


def _vector_for_route(route: VectorRoute) -> Vector:
    dataset = Dataset(
        id=route.dataset_id,
        tenant_id=route.tenant_id,
        indexing_technique=route.indexing_technique,
        index_struct=route.index_struct,
        collection_binding_id=None,
    )
    vector = Vector(dataset, session=None, vector_type=route.vector_type)
    if vector.get_routing_target() != (route.vector_type, route.collection_name):
        raise RuntimeError("Attachment vector cleanup route did not resolve to its persisted target")
    return vector


def _live_binding_uses_route(guard: AttachmentVectorCleanupGuard, route: VectorRoute) -> bool:
    dataset = guard.dataset
    if dataset is None or dataset.indexing_technique != "high_quality" or not dataset.is_multimodal:
        return False
    try:
        current_vector_type = Vector.resolve_vector_type(dataset, session=guard.session)
        current = Vector(dataset, session=guard.session, vector_type=current_vector_type)
        return current.get_routing_target() == (route.vector_type, route.collection_name)
    except (KeyError, TypeError, ValueError):
        return False


def _repository() -> SQLAlchemyAttachmentVectorStore:
    return SQLAlchemyAttachmentVectorStore(session_factory=session_factory.get_session_maker())


def enqueue_attachment_vector_cleanup(cleanup_id: str) -> None:
    try:
        process_attachment_vector_cleanup_task.delay(cleanup_id)
    except Exception:
        logger.exception(
            "Failed to publish attachment vector cleanup; periodic drain will retry, cleanup_id=%s", cleanup_id
        )


@contextmanager
def _maintain_cleanup_claim(
    repository: SQLAlchemyAttachmentVectorStore,
    claim: AttachmentVectorCleanupClaim,
    *,
    interval_seconds: float = _CLAIM_HEARTBEAT_SECONDS,
) -> Generator[None]:
    if interval_seconds <= 0:
        raise ValueError("attachment vector cleanup heartbeat interval must be positive")
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                repository.renew_cleanup(claim)
            except AttachmentVectorCleanupClaimLostError:
                logger.info("Attachment vector cleanup lost ownership, cleanup_id=%s", claim.id)
                return
            except Exception:
                logger.exception("Attachment vector cleanup heartbeat failed, cleanup_id=%s", claim.id)

    thread = Thread(target=heartbeat, name=f"attachment-vector-cleanup-heartbeat-{claim.id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning("Attachment vector cleanup heartbeat did not stop promptly, cleanup_id=%s", claim.id)


@shared_task(queue="dataset")
def process_attachment_vector_cleanup_task(cleanup_id: str) -> None:
    repository = _repository()
    claim = repository.claim_pending_cleanup(cleanup_id)
    if claim is None:
        return
    try:
        with _maintain_cleanup_claim(repository, claim):
            with repository.guard_cleanup_delete(claim) as guard:
                referenced = guard.has_live_binding and _live_binding_uses_route(guard, claim.route)
                if not referenced:
                    _vector_for_route(claim.route).delete_by_ids([claim.attachment_id])
            if not claim.writer_finished:
                # UNKNOWN calls may still write after this delete. Keep the
                # receipt so a terminal writer outcome forces another pass.
                repository.release_cleanup(claim)
    except Exception:
        repository.release_cleanup(claim)
        raise


@shared_task(queue="dataset")
def drain_attachment_vector_cleanup_outbox_task() -> None:
    repository = _repository()
    for cleanup_id in repository.list_pending_cleanup_ids(limit=_DRAIN_BATCH_SIZE):
        try:
            enqueue_attachment_vector_cleanup(cleanup_id)
        finally:
            repository.mark_attempted(cleanup_id)

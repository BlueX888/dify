"""Execute durable knowledge-resource cleanup intents."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from threading import Event, Thread
from typing import Protocol

from celery import shared_task

from core.db.session_factory import session_factory
from core.rag.index_processor.constant.index_type import IndexStructureType
from repositories.knowledge.resource_cleanup_repository import (
    ResourceCleanupClaimLostError,
    ResourceCleanupIntent,
    ResourceCleanupType,
    SQLAlchemyResourceCleanupRepository,
)
from services.knowledge.entities.segments import ChildVectorRoute
from tasks.clean_dataset_task import run_dataset_cleanup, run_dataset_relational_cleanup, run_dataset_vector_cleanup
from tasks.clean_document_task import run_document_cleanup

logger = logging.getLogger(__name__)

_DRAIN_BATCH_SIZE = 100
_CLAIM_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


class CleanupSourceStillExistsError(RuntimeError):
    pass


class CleanupClaimStore(Protocol):
    def renew_claim(self, intent_id: str, *, claim_id: str) -> None: ...


def _repository() -> SQLAlchemyResourceCleanupRepository:
    return SQLAlchemyResourceCleanupRepository(session_factory=session_factory.get_session_maker())


def enqueue_resource_cleanup_intent(intent_id: str) -> None:
    """Best-effort fast path; the periodic outbox drain is the recovery path."""
    try:
        process_resource_cleanup_intent_task.delay(intent_id)
    except Exception:
        logger.exception(
            "Failed to publish resource cleanup intent; periodic drain will retry, intent_id=%s", intent_id
        )


def _run_cleanup(intent: ResourceCleanupIntent) -> ChildVectorRoute | None:
    doc_form = intent.doc_form or IndexStructureType.PARAGRAPH_INDEX
    if intent.resource_type == ResourceCleanupType.DATASET_VECTOR:
        if intent.vector_route is None:
            raise ValueError("Dataset vector cleanup intent has no persisted route")
        return run_dataset_vector_cleanup(intent.vector_route)
    if intent.resource_type == ResourceCleanupType.DATASET_RELATIONAL:
        if intent.resource_id != intent.dataset_id:
            raise ValueError("Dataset cleanup intent has an inconsistent owner chain")
        run_dataset_relational_cleanup(
            tenant_id=intent.tenant_id,
            dataset_id=intent.dataset_id,
            pipeline_id=intent.pipeline_id,
        )
        return None
    if intent.resource_type == ResourceCleanupType.DATASET:
        if intent.resource_id != intent.dataset_id:
            raise ValueError("Dataset cleanup intent has an inconsistent owner chain")
        return run_dataset_cleanup(
            tenant_id=intent.tenant_id,
            dataset_id=intent.dataset_id,
            indexing_technique=intent.indexing_technique,
            index_struct=intent.index_struct,
            collection_binding_id=intent.collection_binding_id,
            doc_form=doc_form,
            pipeline_id=intent.pipeline_id,
        )

    run_document_cleanup(
        tenant_id=intent.tenant_id,
        dataset_id=intent.dataset_id,
        document_id=intent.resource_id,
        doc_form=doc_form,
        file_id=intent.source_file_id,
    )
    return None


@contextmanager
def _maintain_cleanup_claim(
    repository: CleanupClaimStore,
    intent: ResourceCleanupIntent,
    *,
    interval_seconds: float = _CLAIM_HEARTBEAT_SECONDS,
) -> Generator[None]:
    """Renew one exact cleanup claim while external deletion work is running."""
    if interval_seconds <= 0:
        raise ValueError("cleanup claim heartbeat interval must be positive")
    if intent.claim_id is None:
        raise ValueError("cleanup intent has no claim")
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                repository.renew_claim(intent.id, claim_id=intent.claim_id)
            except ResourceCleanupClaimLostError:
                logger.info("Resource cleanup heartbeat lost ownership, intent_id=%s", intent.id)
                return
            except Exception:
                logger.exception("Resource cleanup heartbeat failed, intent_id=%s", intent.id)

    thread = Thread(target=heartbeat, name=f"resource-cleanup-heartbeat-{intent.id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning("Resource cleanup heartbeat did not stop promptly, intent_id=%s", intent.id)


@shared_task(queue="dataset")
def process_resource_cleanup_intent_task(intent_id: str) -> None:
    """Clean one committed deletion and acknowledge only complete work."""
    repository = _repository()
    intent = repository.claim(intent_id)
    if intent is None:
        return
    assert intent.claim_id is not None
    try:
        with _maintain_cleanup_claim(repository, intent):
            if repository.source_exists(intent):
                raise CleanupSourceStillExistsError(
                    f"Cleanup source still exists: {intent.resource_type}:{intent.resource_id}"
                )
            # Snapshot before the external deletion. A writer can finish and
            # recreate a node while collection deletion is still returning;
            # checking only afterward would discard its last recovery record.
            reconcile_again = (
                repository.has_unfinished_summary_writers(intent)
                or repository.has_unfinished_child_writers(intent)
                or repository.has_unfinished_attachment_writers(intent)
            )
            deleted_child_route = _run_cleanup(intent)
            if reconcile_again:
                repository.release_claim(intent.id, claim_id=intent.claim_id)
                logger.info("Retaining dataset cleanup for unfinished vector writes, intent_id=%s", intent.id)
            else:
                repository.complete(
                    intent,
                    claim_id=intent.claim_id,
                    deleted_child_route=deleted_child_route,
                )
    except Exception:
        repository.release_claim(intent.id, claim_id=intent.claim_id)
        raise


@shared_task(queue="dataset")
def drain_resource_cleanup_outbox_task() -> None:
    """Publish a fair bounded batch; records remain durable across broker failure."""
    repository = _repository()
    for intent_id in repository.list_pending_intent_ids(limit=_DRAIN_BATCH_SIZE):
        enqueue_resource_cleanup_intent(intent_id)
        # Rotate records even when the broker rejected the best-effort publish;
        # otherwise the first failed batch can permanently hide newer work.
        repository.mark_attempted(intent_id)

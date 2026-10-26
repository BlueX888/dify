"""Publish durable document-indexing intents."""

import logging

from celery import shared_task

from core.db.session_factory import session_factory
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    DocumentIndexingIntent,
    SQLAlchemyDocumentIndexingOutboxRepository,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.document_indexing_proxy.document_indexing_task_proxy import DocumentIndexingTaskProxy
from services.document_indexing_proxy.duplicate_document_indexing_task_proxy import DuplicateDocumentIndexingTaskProxy
from tasks.clean_notion_document_task import clean_notion_document_task
from tasks.document_indexing_update_task import document_indexing_update_task
from tasks.recover_document_indexing_task import recover_document_indexing_task
from tasks.retry_document_indexing_task import retry_document_indexing_task
from tasks.sync_website_document_indexing_task import sync_website_document_indexing_task

logger = logging.getLogger(__name__)

_DRAIN_BATCH_SIZE = 100


def _repository() -> SQLAlchemyDocumentIndexingOutboxRepository:
    return SQLAlchemyDocumentIndexingOutboxRepository(session_factory=session_factory.get_session_maker())


def _document_repository() -> SQLAlchemyDocumentRepository:
    return SQLAlchemyDocumentRepository(session_factory=session_factory.get_session_maker())


def _request_has_unpaused_documents(intent: DocumentIndexingIntent) -> bool:
    return _document_repository().indexing_request_has_unpaused_documents(
        request_id=intent.id,
        tenant_id=intent.tenant_id,
        dataset_id=intent.dataset_id,
    )


def enqueue_document_indexing_intent(intent_id: str) -> None:
    """Best-effort fast path; the periodic drain recovers rejected publishes."""
    try:
        process_document_indexing_intent_task.delay(intent_id)
    except Exception:
        logger.exception(
            "Failed to publish document indexing intent; periodic drain will retry, intent_id=%s", intent_id
        )


def _publish(intent: DocumentIndexingIntent) -> None:
    document_ids = list(intent.document_ids)
    if intent.action == DocumentIndexingAction.REMOVE_NOTION:
        clean_notion_document_task.delay(
            document_ids,
            intent.dataset_id,
            indexing_request_id=intent.id,
        )
    elif intent.action == DocumentIndexingAction.CREATE:
        DocumentIndexingTaskProxy(
            intent.tenant_id,
            intent.dataset_id,
            document_ids,
            indexing_request_id=intent.id,
        ).delay()
    elif intent.action == DocumentIndexingAction.DUPLICATE:
        DuplicateDocumentIndexingTaskProxy(
            intent.tenant_id,
            intent.dataset_id,
            document_ids,
            indexing_request_id=intent.id,
        ).delay()
    elif intent.action == DocumentIndexingAction.UPDATE:
        (document_id,) = intent.document_ids
        document_indexing_update_task.delay(
            intent.dataset_id,
            document_id,
            indexing_request_id=intent.id,
        )
    elif intent.action == DocumentIndexingAction.RECOVER:
        (document_id,) = intent.document_ids
        recover_document_indexing_task.delay(
            intent.dataset_id,
            document_id,
            indexing_request_id=intent.id,
        )
    elif intent.action == DocumentIndexingAction.RETRY:
        if intent.actor_id is None:
            raise ValueError(f"Retry indexing intent has no actor: {intent.id}")
        retry_document_indexing_task.delay(
            intent.dataset_id,
            document_ids,
            intent.actor_id,
            indexing_request_id=intent.id,
        )
    elif intent.action == DocumentIndexingAction.WEBSITE_SYNC:
        (document_id,) = intent.document_ids
        sync_website_document_indexing_task.delay(
            intent.dataset_id,
            document_id,
            indexing_request_id=intent.id,
        )


@shared_task(queue="dataset")
def process_document_indexing_intent_task(intent_id: str) -> None:
    """Publish one committed intent and acknowledge only accepted work."""
    repository = _repository()
    intent = repository.get_intent(intent_id)
    if intent is None:
        return
    repository.mark_attempted(intent.id)
    if intent.action != DocumentIndexingAction.REMOVE_NOTION:
        if not repository.request_is_current(intent):
            repository.complete(intent)
            return
        if not _request_has_unpaused_documents(intent):
            return
        if repository.has_active_worker(intent):
            return
    _publish(intent)


@shared_task(queue="dataset")
def drain_document_indexing_outbox_task() -> None:
    """Publish a fair bounded batch of durable indexing intents."""
    repository = _repository()
    for intent_id in repository.list_pending_intent_ids(limit=_DRAIN_BATCH_SIZE):
        intent = repository.get_intent(intent_id)
        if (
            intent is not None
            and intent.action != DocumentIndexingAction.REMOVE_NOTION
            and repository.request_is_current(intent)
            and not _request_has_unpaused_documents(intent)
        ):
            repository.mark_attempted(intent_id)
            continue
        enqueue_document_indexing_intent(intent_id)
        repository.mark_attempted(intent_id)

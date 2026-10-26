from datetime import datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

import tasks.document_indexing_outbox_task as task_module
from models import Dataset, Tenant
from models.dataset import Document
from models.model import DocumentIndexingOutbox
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    SQLAlchemyDocumentIndexingOutboxRepository,
    create_document_indexing_intents,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.document_indexing_proxy.batch_indexing_base import BatchDocumentIndexingProxy
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef
from tests.unit_tests.model_factories import make_dataset, make_document, make_tenant


def _create_intent(
    session_factory: sessionmaker[Session],
    *,
    action: DocumentIndexingAction = DocumentIndexingAction.CREATE,
    position: int = 1,
) -> str:
    document_id = f"document-{position}"
    ref = DatasetRef("tenant-1", "dataset-1")
    if action == DocumentIndexingAction.CREATE:
        jobs = DocumentIndexingJobs(ref, created=(document_id,))
    elif action == DocumentIndexingAction.DUPLICATE:
        jobs = DocumentIndexingJobs(ref, duplicated=(document_id,))
    elif action == DocumentIndexingAction.REMOVE_NOTION:
        jobs = DocumentIndexingJobs(ref, removed_notion=(document_id,))
    elif action == DocumentIndexingAction.WEBSITE_SYNC:
        jobs = DocumentIndexingJobs(ref, website_sync=document_id)
    else:
        jobs = DocumentIndexingJobs(ref, updated=document_id)
    with session_factory.begin() as session:
        if action != DocumentIndexingAction.REMOVE_NOTION:
            if session.get(Tenant, "tenant-1") is None:
                session.add(make_tenant(tenant_id="tenant-1"))
            if session.get(Dataset, "dataset-1") is None:
                session.add(make_dataset(dataset_id="dataset-1", tenant_id="tenant-1"))
            session.add(
                make_document(
                    document_id=f"document-{position}",
                    dataset_id="dataset-1",
                    tenant_id="tenant-1",
                )
            )
        (intent_id,) = create_document_indexing_intents(
            session,
            jobs,
        )
    return intent_id


def _use_repository(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
) -> SQLAlchemyDocumentIndexingOutboxRepository:
    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=session_factory)
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    document_repository = SQLAlchemyDocumentRepository(session_factory=session_factory)
    monkeypatch.setattr(task_module, "_document_repository", lambda: document_repository)
    return repository


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (DocumentIndexingAction.CREATE, ("create", "tenant-1", "dataset-1", ("document-1",))),
        (DocumentIndexingAction.DUPLICATE, ("duplicate", "tenant-1", "dataset-1", ("document-1",))),
        (DocumentIndexingAction.REMOVE_NOTION, ("remove_notion", "dataset-1", ("document-1",))),
        (DocumentIndexingAction.UPDATE, ("update", "dataset-1", ("document-1",))),
        (DocumentIndexingAction.WEBSITE_SYNC, ("website_sync", "dataset-1", ("document-1",))),
    ],
)
def test_process_routes_each_action_and_keeps_intent_until_the_worker_finishes(
    action: DocumentIndexingAction,
    expected: tuple[object, ...],
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_id = _create_intent(sqlite_session_factory, action=action)
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    calls: list[tuple[object, ...]] = []

    def create(proxy: BatchDocumentIndexingProxy) -> None:
        calls.append(
            (
                "create",
                proxy._tenant_id,
                proxy._dataset_id,
                tuple(proxy._document_ids),
                proxy._indexing_request_id,
            )
        )

    def duplicate(proxy: BatchDocumentIndexingProxy) -> None:
        calls.append(
            (
                "duplicate",
                proxy._tenant_id,
                proxy._dataset_id,
                tuple(proxy._document_ids),
                proxy._indexing_request_id,
            )
        )

    monkeypatch.setattr(task_module.DocumentIndexingTaskProxy, "delay", create)
    monkeypatch.setattr(task_module.DuplicateDocumentIndexingTaskProxy, "delay", duplicate)
    monkeypatch.setattr(
        task_module.clean_notion_document_task,
        "delay",
        lambda document_ids, dataset_id, *, indexing_request_id: calls.append(
            ("remove_notion", dataset_id, tuple(document_ids), indexing_request_id)
        ),
    )
    monkeypatch.setattr(
        task_module.document_indexing_update_task,
        "delay",
        lambda dataset_id, document_id, *, indexing_request_id: calls.append(
            ("update", dataset_id, (document_id,), indexing_request_id)
        ),
    )
    monkeypatch.setattr(
        task_module.sync_website_document_indexing_task,
        "delay",
        lambda dataset_id, document_id, *, indexing_request_id: calls.append(
            ("website_sync", dataset_id, (document_id,), indexing_request_id)
        ),
    )

    task_module.process_document_indexing_intent_task.run(intent_id)

    expected_with_request = (*expected, intent_id)
    assert calls == [expected_with_request]
    assert repository.get_intent(intent_id) is not None


def test_process_retains_intent_when_downstream_broker_rejects_publish(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_id = _create_intent(sqlite_session_factory)
    repository = _use_repository(monkeypatch, sqlite_session_factory)

    def fail(_proxy: BatchDocumentIndexingProxy) -> None:
        raise OSError("broker unavailable")

    monkeypatch.setattr(task_module.DocumentIndexingTaskProxy, "delay", fail)

    with pytest.raises(OSError, match="broker unavailable"):
        task_module.process_document_indexing_intent_task.run(intent_id)

    intent = repository.get_intent(intent_id)
    assert intent is not None
    with sqlite_session_factory() as session:
        row = session.get(DocumentIndexingOutbox, intent_id)
        assert row is not None
        assert row.last_attempted_at is not None


def test_stale_request_is_acknowledged_without_replaying_into_a_new_document_generation(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_intent_id = _create_intent(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        (new_intent_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(DatasetRef("tenant-1", "dataset-1"), updated="document-1"),
        )
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    published: list[object] = []
    monkeypatch.setattr(task_module, "_publish", published.append)

    task_module.process_document_indexing_intent_task.run(old_intent_id)

    assert published == []
    assert repository.get_intent(old_intent_id) is None
    assert repository.get_intent(new_intent_id) is not None
    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.indexing_request_id == new_intent_id


def test_active_worker_lease_suppresses_periodic_republication(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_id = _create_intent(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, "document-1")
        document.indexing_claim_id = "claim-1"
        document.indexing_claim_expires_at = datetime(2099, 1, 1)
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    published: list[object] = []
    monkeypatch.setattr(task_module, "_publish", published.append)

    task_module.process_document_indexing_intent_task.run(intent_id)

    assert published == []
    assert repository.get_intent(intent_id) is not None


def test_enqueue_swallows_initial_broker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_publish(_intent_id: str) -> None:
        raise OSError("broker unavailable")

    monkeypatch.setattr(task_module.process_document_indexing_intent_task, "delay", fail_publish)
    task_module.enqueue_document_indexing_intent("intent-1")


def test_periodic_drain_republishes_committed_intents(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_ids = [_create_intent(sqlite_session_factory, position=position) for position in range(1, 4)]
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    published: list[str] = []
    monkeypatch.setattr(task_module, "enqueue_document_indexing_intent", published.append)

    task_module.drain_document_indexing_outbox_task.run()

    assert set(published) == set(intent_ids)
    assert all(repository.get_intent(intent_id) is not None for intent_id in intent_ids)


def test_periodic_drain_does_not_publish_request_containing_only_paused_documents(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_id = _create_intent(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, "document-1")
        document.is_paused = True
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    published: list[str] = []
    monkeypatch.setattr(task_module, "enqueue_document_indexing_intent", published.append)

    task_module.drain_document_indexing_outbox_task.run()

    assert published == []
    assert repository.get_intent(intent_id) is not None

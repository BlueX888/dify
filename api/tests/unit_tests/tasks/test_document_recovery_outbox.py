"""Regression coverage for durable recovery and retry publications."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session, sessionmaker

import tasks.document_indexing_outbox_task as outbox_tasks
from enums import DeploymentEdition
from models.dataset import Document
from models.enums import IndexingStatus
from models.model import DocumentIndexingOutbox
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    SQLAlchemyDocumentIndexingOutboxRepository,
    create_document_indexing_intents,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef
from tasks.document_indexing_task import _document_indexing
from tasks.recover_document_indexing_task import recover_document_indexing_task
from tests.unit_tests.config_override import config_overrides_context
from tests.unit_tests.model_factories import make_dataset, make_document, make_tenant


def _persist_document(
    session_factory: sessionmaker[Session],
    *,
    status: str = IndexingStatus.WAITING,
) -> DatasetRef:
    ref = DatasetRef("tenant-1", "dataset-1")
    with session_factory.begin() as session:
        session.add(make_tenant(tenant_id=ref.tenant_id))
        session.add(make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id))
        document = make_document(
            document_id="document-1",
            tenant_id=ref.tenant_id,
            dataset_id=ref.dataset_id,
        )
        document.indexing_status = status
        document.is_paused = False
        session.add(document)
    return ref


def _create_intent(
    session_factory: sessionmaker[Session],
    ref: DatasetRef,
    action: DocumentIndexingAction,
) -> str:
    jobs = (
        DocumentIndexingJobs(ref, recovered="document-1")
        if action == DocumentIndexingAction.RECOVER
        else DocumentIndexingJobs(ref, retried=("document-1",), actor_id="actor-1")
    )
    with session_factory.begin() as session:
        (intent_id,) = create_document_indexing_intents(session, jobs)
    return intent_id


@pytest.mark.parametrize("action", [DocumentIndexingAction.RECOVER, DocumentIndexingAction.RETRY])
def test_broker_failure_keeps_recovery_intent_for_periodic_redelivery(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    action: DocumentIndexingAction,
) -> None:
    ref = _persist_document(sqlite_session_factory)
    intent_id = _create_intent(sqlite_session_factory, ref, action)
    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    monkeypatch.setattr(outbox_tasks, "_repository", lambda: repository)

    def reject_publish(_intent: object) -> None:
        raise OSError("broker unavailable")

    monkeypatch.setattr(outbox_tasks, "_publish", reject_publish)
    with pytest.raises(OSError, match="broker unavailable"):
        outbox_tasks.process_document_indexing_intent_task.run(intent_id)

    retained = repository.get_intent(intent_id)
    assert retained is not None
    assert retained.action == action
    if action == DocumentIndexingAction.RETRY:
        assert retained.actor_id == "actor-1"

    redelivered: list[str] = []
    monkeypatch.setattr(outbox_tasks, "enqueue_document_indexing_intent", redelivered.append)
    outbox_tasks.drain_document_indexing_outbox_task.run()

    assert intent_id in redelivered
    assert repository.get_intent(intent_id) is not None


@pytest.mark.parametrize("action", [DocumentIndexingAction.RECOVER, DocumentIndexingAction.RETRY])
def test_superseded_recovery_generation_is_discarded_without_publication(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    action: DocumentIndexingAction,
) -> None:
    ref = _persist_document(sqlite_session_factory)
    stale_intent_id = _create_intent(sqlite_session_factory, ref, action)
    current_intent_id = _create_intent(sqlite_session_factory, ref, action)
    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    monkeypatch.setattr(outbox_tasks, "_repository", lambda: repository)
    published: list[object] = []
    monkeypatch.setattr(outbox_tasks, "_publish", published.append)

    outbox_tasks.process_document_indexing_intent_task.run(stale_intent_id)

    assert published == []
    assert repository.get_intent(stale_intent_id) is None
    assert repository.get_intent(current_intent_id) is not None
    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.indexing_request_id == current_intent_id


@pytest.mark.parametrize(
    ("status", "method_name"),
    [
        (IndexingStatus.PARSING, "restart_claimed"),
        (IndexingStatus.CLEANING, "restart_claimed"),
        (IndexingStatus.SPLITTING, "restart_claimed"),
        (IndexingStatus.INDEXING, "run_in_indexing_status"),
        (IndexingStatus.ERROR, "restart_claimed"),
    ],
)
def test_recovery_resumes_owned_phase_and_completes_exact_request(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    method_name: str,
) -> None:
    ref = _persist_document(sqlite_session_factory, status=status)
    intent_id = _create_intent(sqlite_session_factory, ref, DocumentIndexingAction.RECOVER)
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    calls: list[tuple[str, str]] = []

    class CompletingIndexingService:
        def _complete(self, method: str, document_ref: DocumentRef, *, claim_id: str) -> None:
            calls.append((method, claim_id))
            repository.complete_indexing(document_ref, claim_id=claim_id, tokens=3, latency=0.1)

        def run_claimed(self, document_ref: DocumentRef, *, claim_id: str) -> None:
            self._complete("run_claimed", document_ref, claim_id=claim_id)

        def restart_claimed(self, document_ref: DocumentRef, *, claim_id: str) -> None:
            self._complete("restart_claimed", document_ref, claim_id=claim_id)

        def run_in_indexing_status(self, document_ref: DocumentRef, *, claim_id: str) -> None:
            self._complete("run_in_indexing_status", document_ref, claim_id=claim_id)

    monkeypatch.setattr(
        "tasks.recover_document_indexing_task.build_document_indexing_service",
        lambda **_kwargs: CompletingIndexingService(),
    )

    recover_document_indexing_task.run(
        ref.dataset_id,
        "document-1",
        indexing_request_id=intent_id,
    )

    assert len(calls) == 1
    assert calls[0][0] == method_name
    assert calls[0][1]
    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.indexing_status == IndexingStatus.COMPLETED
        assert document.indexing_request_id is None
        assert document.indexing_claim_id is None
        assert session.get(DocumentIndexingOutbox, intent_id) is None


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_claimed_worker_pause_releases_phase_and_recovery_executor_completes(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _persist_document(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        (create_intent_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, created=("document-1",)),
        )
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)

    class PausingIndexingService:
        def run_claimed(self, document_ref: DocumentRef, *, claim_id: str) -> None:
            with sqlite_session_factory.begin() as session:
                session.get_one(Document, document_ref.document_id).is_paused = True
            # DocumentIndexingService consumes ordinary backend failures after
            # asking the repository to persist them. A concurrent pause must
            # make that call release ownership rather than publish ERROR.
            repository.fail_indexing(document_ref, "provider failed after pause", claim_id=claim_id)

    monkeypatch.setattr(
        "tasks.document_indexing_task.build_document_indexing_service",
        lambda **_kwargs: PausingIndexingService(),
    )
    _document_indexing(ref.dataset_id, ["document-1"], indexing_request_id=create_intent_id)

    with sqlite_session_factory() as session:
        paused = session.get_one(Document, "document-1")
        assert paused.indexing_status == IndexingStatus.PARSING
        assert paused.indexing_request_id == create_intent_id
        assert paused.indexing_claim_id is None
        assert paused.error is None

    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, "document-1")
        document.is_paused = False
        (recover_intent_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, recovered=document.id),
        )

    executor_calls: list[str] = []

    class CompletingIndexingService:
        def run_claimed(self, document_ref: DocumentRef, *, claim_id: str) -> None:
            executor_calls.append("run_claimed")
            repository.complete_indexing(document_ref, claim_id=claim_id, tokens=5, latency=0.1)

        def restart_claimed(self, document_ref: DocumentRef, *, claim_id: str) -> None:
            executor_calls.append("restart_claimed")
            repository.complete_indexing(document_ref, claim_id=claim_id, tokens=5, latency=0.1)

    monkeypatch.setattr(
        "tasks.recover_document_indexing_task.build_document_indexing_service",
        lambda **_kwargs: CompletingIndexingService(),
    )
    recover_document_indexing_task.run(
        ref.dataset_id,
        "document-1",
        indexing_request_id=recover_intent_id,
    )

    assert executor_calls == ["restart_claimed"]
    with sqlite_session_factory() as session:
        recovered = session.get_one(Document, "document-1")
        assert recovered.indexing_status == IndexingStatus.COMPLETED
        assert recovered.indexing_request_id is None
        assert recovered.indexing_claim_id is None
        assert session.get(DocumentIndexingOutbox, recover_intent_id) is None


def test_outbox_routes_recovery_and_retry_with_their_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TaskCapture:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def delay(self, *args: object, **kwargs: object) -> None:
            self.calls.append((args, kwargs))

    recover_task = TaskCapture()
    retry_task = TaskCapture()
    monkeypatch.setattr(outbox_tasks, "recover_document_indexing_task", recover_task)
    monkeypatch.setattr(outbox_tasks, "retry_document_indexing_task", retry_task)
    recover_intent = outbox_tasks.DocumentIndexingIntent(
        id="recover-request",
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        action=DocumentIndexingAction.RECOVER,
        document_ids=("document-1",),
    )
    retry_intent = outbox_tasks.DocumentIndexingIntent(
        id="retry-request",
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        action=DocumentIndexingAction.RETRY,
        document_ids=("document-1",),
        actor_id="actor-1",
    )

    outbox_tasks._publish(recover_intent)
    outbox_tasks._publish(retry_intent)

    assert recover_task.calls == [(("dataset-1", "document-1"), {"indexing_request_id": "recover-request"})]
    assert retry_task.calls == [(("dataset-1", ["document-1"], "actor-1"), {"indexing_request_id": "retry-request"})]

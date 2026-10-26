from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, create_autospec

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.index_processor.constant.index_type import IndexTechniqueType
from enums import CloudPlan, DeploymentEdition
from models.account import Tenant
from models.dataset import Dataset, Document, DocumentSegment
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus, SegmentStatus
from models.model import DocumentIndexingOutbox
from repositories.knowledge.document_indexing_outbox_repository import create_document_indexing_intents
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef
from tasks import document_indexing_task as task_module
from tests.unit_tests.config_override import config_overrides_context


def _persist_indexing_scope(session: Session, *, document_statuses: dict[str, IndexingStatus]) -> Dataset:
    tenant = Tenant(name="Workspace")
    tenant.id = "workspace-1"
    dataset = Dataset(
        id="dataset-1",
        tenant_id=tenant.id,
        name="Dataset",
        created_by="account-1",
        indexing_technique=IndexTechniqueType.ECONOMY,
        data_source_type=DataSourceType.UPLOAD_FILE,
    )
    documents = [
        Document(
            id=document_id,
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            position=position,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch="batch-1",
            name=f"{document_id}.txt",
            created_from=DocumentCreatedFrom.WEB,
            created_by="account-1",
            indexing_status=status,
        )
        for position, (document_id, status) in enumerate(document_statuses.items(), start=1)
    ]
    session.add_all([tenant, dataset, *documents])
    session.commit()
    return dataset


class _CompletingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[DocumentRef, ...]] = []
        self.claimed_calls: list[tuple[DocumentRef, str]] = []
        self.restarted_calls: list[tuple[DocumentRef, str]] = []
        self._session_factory: sessionmaker[Session] | None = None

    def bind(self, database_client: sessionmaker[Session]) -> _CompletingRunner:
        self._session_factory = database_client
        return self

    def run(self, refs: Sequence[DocumentRef]) -> None:
        self.calls.append(tuple(refs))
        assert self._session_factory is not None
        with self._session_factory.begin() as session:
            for ref in refs:
                document = session.get_one(Document, ref.document_id)
                session.add(
                    DocumentSegment(
                        tenant_id=ref.dataset.tenant_id,
                        dataset_id=ref.dataset.dataset_id,
                        document_id=ref.document_id,
                        position=1,
                        content="chunk",
                        word_count=1,
                        tokens=1,
                        created_by=document.created_by,
                        status=SegmentStatus.COMPLETED,
                    )
                )
                document.indexing_status = IndexingStatus.COMPLETED

    def run_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
        self.claimed_calls.append((ref, claim_id))

    def restart_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
        self.restarted_calls.append((ref, claim_id))


@dataclass(frozen=True)
class _Subscription:
    plan: CloudPlan


@dataclass(frozen=True)
class _Billing:
    subscription: _Subscription


@dataclass(frozen=True)
class _VectorSpace:
    limit: int
    size: int


@dataclass(frozen=True)
class _Features:
    billing: _Billing
    vector_space: _VectorSpace


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_sequential_redelivery_does_not_create_duplicate_segments(
    sqlite_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _persist_indexing_scope(sqlite_session, document_statuses={"document-1": IndexingStatus.WAITING})
    runner = _CompletingRunner()

    def build_runner(
        *, session_factory: sessionmaker[Session], enforce_vector_space_admission: bool
    ) -> _CompletingRunner:
        assert enforce_vector_space_admission
        return runner.bind(session_factory)

    monkeypatch.setattr(task_module, "build_document_indexing_service", build_runner)

    task_module._document_indexing(dataset.id, ["document-1"])
    task_module._document_indexing(dataset.id, ["document-1"])

    sqlite_session.expire_all()
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == DatasetRef(dataset.tenant_id, dataset.id).document("document-1")
    assert sqlite_session.scalar(select(func.count()).select_from(DocumentSegment)) == 1
    assert sqlite_session.get_one(Document, "document-1").indexing_status == IndexingStatus.COMPLETED


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_old_create_redelivery_cannot_claim_a_new_update_generation(sqlite_session: Session) -> None:
    dataset = _persist_indexing_scope(sqlite_session, document_statuses={"document-1": IndexingStatus.WAITING})
    with sqlite_session.begin():
        (create_request_id,) = create_document_indexing_intents(
            sqlite_session,
            DocumentIndexingJobs(DatasetRef(dataset.tenant_id, dataset.id), created=("document-1",)),
        )

    with sqlite_session.begin():
        document = sqlite_session.get_one(Document, "document-1")
        document.indexing_status = IndexingStatus.COMPLETED
        document.indexing_request_id = None

    with sqlite_session.begin():
        document = sqlite_session.get_one(Document, "document-1")
        document.indexing_status = IndexingStatus.WAITING
        (update_request_id,) = create_document_indexing_intents(
            sqlite_session,
            DocumentIndexingJobs(DatasetRef(dataset.tenant_id, dataset.id), updated=document.id),
        )

    task_module._document_indexing(
        dataset.id,
        ["document-1"],
        indexing_request_id=create_request_id,
    )

    sqlite_session.expire_all()
    document = sqlite_session.get_one(Document, "document-1")
    assert document.indexing_status == IndexingStatus.WAITING
    assert document.indexing_request_id == update_request_id
    assert document.indexing_claim_id is None
    assert sqlite_session.get(DocumentIndexingOutbox, create_request_id) is None
    assert sqlite_session.get(DocumentIndexingOutbox, update_request_id) is not None


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_expired_create_worker_is_reclaimed_through_clean_restart(
    sqlite_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _persist_indexing_scope(sqlite_session, document_statuses={"document-1": IndexingStatus.WAITING})
    with sqlite_session.begin():
        (request_id,) = create_document_indexing_intents(
            sqlite_session,
            DocumentIndexingJobs(DatasetRef(dataset.tenant_id, dataset.id), created=("document-1",)),
        )
    runner = _CompletingRunner()

    def build_runner(
        *, session_factory: sessionmaker[Session], enforce_vector_space_admission: bool
    ) -> _CompletingRunner:
        assert enforce_vector_space_admission
        return runner.bind(session_factory)

    monkeypatch.setattr(task_module, "build_document_indexing_service", build_runner)

    task_module._document_indexing(dataset.id, ["document-1"], indexing_request_id=request_id)
    assert len(runner.claimed_calls) == 1
    assert runner.restarted_calls == []

    first_claim_id = runner.claimed_calls[0][1]
    with sqlite_session.begin():
        document = sqlite_session.get_one(Document, "document-1")
        assert document.indexing_claim_id == first_claim_id
        document.indexing_claim_expires_at = datetime(2020, 1, 1)

    task_module._document_indexing(dataset.id, ["document-1"], indexing_request_id=request_id)

    assert len(runner.restarted_calls) == 1
    restarted_ref, restarted_claim_id = runner.restarted_calls[0]
    assert restarted_ref == DatasetRef(dataset.tenant_id, dataset.id).document("document-1")
    assert restarted_claim_id != first_claim_id


def test_worker_completion_clears_its_request_marker(
    monkeypatch: pytest.MonkeyPatch,
    tenant_queue_commands: MagicMock,
) -> None:
    request_id = "request-1"
    worker = create_autospec(task_module._document_indexing)
    celery_task = create_autospec(task_module.normal_document_indexing_task)
    monkeypatch.setattr(task_module, "_document_indexing", worker)
    tenant_queue_commands.return_value = None

    task_module._document_indexing_with_tenant_queue(
        "tenant-1",
        "dataset-1",
        ["document-1"],
        celery_task,
        indexing_request_id=request_id,
    )

    worker.assert_called_once_with("dataset-1", ["document-1"], indexing_request_id=request_id)
    tenant_queue_commands.assert_any_call(
        "DEL",
        f"tenant_document_indexing_request:tenant-1:{request_id}",
    )


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
def test_cloud_quota_failure_only_fails_the_claimed_generation(
    sqlite_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _persist_indexing_scope(
        sqlite_session,
        document_statuses={
            "completed": IndexingStatus.COMPLETED,
            "waiting": IndexingStatus.WAITING,
        },
    )
    completed = sqlite_session.get_one(Document, "completed")
    completed.processing_started_at = datetime(2025, 1, 1)
    completed.completed_at = datetime(2025, 1, 2)
    sqlite_session.commit()
    features = _Features(
        billing=_Billing(subscription=_Subscription(plan=CloudPlan.PROFESSIONAL)),
        vector_space=_VectorSpace(limit=1, size=1),
    )
    monkeypatch.setattr(task_module.FeatureService, "get_features", lambda _tenant_id: features)

    task_module._document_indexing(dataset.id, ["completed", "waiting"])

    sqlite_session.expire_all()
    persisted_completed = sqlite_session.get_one(Document, "completed")
    persisted_waiting = sqlite_session.get_one(Document, "waiting")
    assert persisted_completed.indexing_status == IndexingStatus.COMPLETED
    assert persisted_completed.processing_started_at == datetime(2025, 1, 1)
    assert persisted_completed.completed_at == datetime(2025, 1, 2)
    assert persisted_waiting.indexing_status == IndexingStatus.ERROR
    assert persisted_waiting.error is not None
    assert "over the limit" in persisted_waiting.error


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
def test_cloud_quota_counts_only_locked_waiting_candidates(
    sqlite_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _persist_indexing_scope(
        sqlite_session,
        document_statuses={
            "completed": IndexingStatus.COMPLETED,
            "waiting": IndexingStatus.WAITING,
        },
    )
    runner = _CompletingRunner()
    features = _Features(
        billing=_Billing(subscription=_Subscription(plan=CloudPlan.SANDBOX)),
        vector_space=_VectorSpace(limit=10, size=0),
    )

    def build_runner(
        *, session_factory: sessionmaker[Session], enforce_vector_space_admission: bool
    ) -> _CompletingRunner:
        assert enforce_vector_space_admission
        return runner.bind(session_factory)

    monkeypatch.setattr(task_module.FeatureService, "get_features", lambda _tenant_id: features)
    monkeypatch.setattr(task_module, "build_document_indexing_service", build_runner)

    task_module._document_indexing(dataset.id, ["completed", "waiting", "waiting", "missing"])

    sqlite_session.expire_all()
    assert runner.calls == [(DatasetRef(dataset.tenant_id, dataset.id).document("waiting"),)]
    assert sqlite_session.get_one(Document, "completed").indexing_status == IndexingStatus.COMPLETED
    assert sqlite_session.get_one(Document, "waiting").indexing_status == IndexingStatus.COMPLETED

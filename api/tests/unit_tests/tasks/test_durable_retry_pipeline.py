from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import pytest
from sqlalchemy.orm import Session, sessionmaker

import services.rag_pipeline.rag_pipeline as pipeline_module
import tasks.retry_document_indexing_task as task_module
from core.app.apps.pipeline.pipeline_generator import PipelineGenerator
from models import Account, Dataset, Tenant, TenantAccountJoin, TenantAccountRole
from models.dataset import Document
from models.enums import IndexingStatus
from repositories.knowledge.document_indexing_outbox_repository import create_document_indexing_intents
from repositories.knowledge.document_repository import ClaimedIndexingRequest
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError
from tests.unit_tests.model_factories import make_account, make_dataset, make_document, make_tenant


@dataclass(frozen=True, slots=True)
class _HeartbeatCall:
    ref: object
    claim_id: str


@dataclass(frozen=True, slots=True)
class _PipelineRetryCall:
    dataset_id: str
    document_id: str
    indexing_request_id: str
    claim_id: str


@dataclass(frozen=True, slots=True)
class _RestartCall:
    ref: object
    claim_id: str


class _PipelineGeneratorStub:
    pass


@dataclass(frozen=True, slots=True)
class _KnowledgeServicesStub:
    pipeline_generator: PipelineGenerator


@dataclass(frozen=True, slots=True)
class _ApplicationServicesStub:
    knowledge: _KnowledgeServicesStub


def _claim() -> ClaimedIndexingRequest:
    return ClaimedIndexingRequest(
        ref=DatasetRef("tenant-1", "dataset-1").document("document-1"),
        claim_id="claim-1",
        recovery_required=False,
    )


def test_legacy_cleanup_guard_rejects_a_durable_request(sqlite_session: Session) -> None:
    tenant: Tenant = make_tenant(tenant_id="tenant-1")
    dataset: Dataset = make_dataset(dataset_id="dataset-1", tenant_id=tenant.id)
    document: Document = make_document(
        document_id="document-1",
        dataset_id=dataset.id,
        tenant_id=tenant.id,
    )
    document.indexing_status = IndexingStatus.PARSING
    document.indexing_request_id = "request-1"
    sqlite_session.add_all([tenant, dataset, document])
    sqlite_session.commit()

    with pytest.raises(DocumentIndexingClaimLostError):
        task_module._guard_retry_cleanup(
            sqlite_session,
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            document_id=document.id,
        )


def test_durable_cleanup_guard_rejects_an_expired_claim(sqlite_session: Session) -> None:
    tenant: Tenant = make_tenant(tenant_id="tenant-1")
    dataset: Dataset = make_dataset(dataset_id="dataset-1", tenant_id=tenant.id)
    document: Document = make_document(
        document_id="document-1",
        dataset_id=dataset.id,
        tenant_id=tenant.id,
    )
    document.indexing_status = IndexingStatus.PARSING
    document.indexing_request_id = "request-1"
    document.indexing_claim_id = "claim-1"
    document.indexing_claim_expires_at = datetime(2000, 1, 1)
    sqlite_session.add_all([tenant, dataset, document])
    sqlite_session.commit()

    with pytest.raises(DocumentIndexingClaimLostError):
        task_module._guard_retry_cleanup(
            sqlite_session,
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            document_id=document.id,
            indexing_request_id="request-1",
            claim_id="claim-1",
        )


def test_durable_pipeline_retry_finalizes_only_the_exact_claim(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim()
    with sqlite_session_factory.begin() as session:
        tenant = make_tenant(tenant_id=claim.ref.dataset.tenant_id)
        account = make_account(account_id="user-1", tenant=tenant)
        dataset = make_dataset(dataset_id=claim.ref.dataset.dataset_id, tenant_id=tenant.id)
        document = make_document(
            document_id=claim.ref.document_id,
            dataset_id=dataset.id,
            tenant_id=tenant.id,
        )
        document.indexing_status = IndexingStatus.PARSING
        document.indexing_request_id = "request-1"
        document.indexing_claim_id = claim.claim_id
        document.indexing_claim_expires_at = datetime(2099, 1, 1)
        session.add_all([tenant, account, dataset, document])

    documents = task_module.SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    heartbeat_calls: list[_HeartbeatCall] = []
    retry_calls: list[_PipelineRetryCall] = []

    @contextmanager
    def maintain_claim(_documents: object, ref: object, *, claim_id: str) -> Generator[None]:
        heartbeat_calls.append(_HeartbeatCall(ref, claim_id))
        yield

    class RagPipelineServiceStub:
        def __init__(self, session: Session) -> None:
            self._session = session

        def retry_error_document(
            self,
            dataset: Dataset,
            document: Document,
            _user: Account,
            *,
            generator: PipelineGenerator,
            indexing_request_id: str,
            claim_id: str,
            recovery_required: bool,
        ) -> None:
            del generator, recovery_required
            retry_calls.append(_PipelineRetryCall(dataset.id, document.id, indexing_request_id, claim_id))
            persisted = self._session.get_one(Document, document.id)
            persisted.indexing_status = IndexingStatus.COMPLETED
            persisted.tokens = 17
            persisted.indexing_latency = 2.5
            persisted.error = None
            self._session.commit()

    monkeypatch.setattr(pipeline_module, "maintain_indexing_claim", maintain_claim)
    monkeypatch.setattr(pipeline_module, "RagPipelineService", RagPipelineServiceStub)
    monkeypatch.setattr(pipeline_module.session_factory, "create_session", sqlite_session_factory)
    generator = cast(PipelineGenerator, _PipelineGeneratorStub())

    pipeline_module.run_durable_pipeline_indexing(
        documents=documents,
        claim=claim,
        indexing_request_id="request-1",
        user=account,
        pipeline_generator=generator,
    )

    assert heartbeat_calls == [_HeartbeatCall(claim.ref, claim.claim_id)]
    assert retry_calls == [_PipelineRetryCall("dataset-1", "document-1", "request-1", "claim-1")]
    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, claim.ref.document_id)
        assert persisted.indexing_status == IndexingStatus.COMPLETED
        assert persisted.tokens == 17
        assert persisted.indexing_latency == 2.5
        assert persisted.indexing_request_id is None
        assert persisted.indexing_claim_id is None


def test_durable_retry_claims_documents_one_at_a_time(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        tenant = make_tenant(tenant_id="tenant-1")
        user = make_account(account_id="user-1")
        dataset = make_dataset(dataset_id="dataset-1", tenant_id=tenant.id)
        document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=tenant.id)
        session.add_all(
            [
                tenant,
                user,
                TenantAccountJoin(
                    tenant_id=tenant.id,
                    account_id=user.id,
                    current=True,
                    role=TenantAccountRole.OWNER,
                ),
                dataset,
                document,
            ]
        )
        session.flush()
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(
                dataset=DatasetRef(tenant.id, dataset.id),
                retried=(document.id,),
                actor_id=user.id,
            ),
        )

    restart_calls: list[_RestartCall] = []
    deleted_keys: list[str] = []

    class IndexingServiceStub:
        @staticmethod
        def restart_claimed(ref: object, *, claim_id: str) -> None:
            restart_calls.append(_RestartCall(ref, claim_id))

    generator = cast(PipelineGenerator, _PipelineGeneratorStub())
    services = _ApplicationServicesStub(_KnowledgeServicesStub(generator))

    monkeypatch.setattr(task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(task_module.session_factory, "get_session_maker", lambda: sqlite_session_factory)
    monkeypatch.setattr(task_module, "build_document_indexing_service", lambda **_kwargs: IndexingServiceStub())
    monkeypatch.setattr(task_module, "pipeline_indexing_admission_error", lambda _tenant_id: None)
    monkeypatch.setattr(task_module.redis_client, "delete", deleted_keys.append)
    monkeypatch.setattr("extensions.ext_application_services.application_services", lambda: services)

    task_module._durable_retry_document_indexing(
        "dataset-1",
        ["document-1"],
        "user-1",
        indexing_request_id=request_id,
    )

    assert len(restart_calls) == 1
    assert restart_calls[0].ref == DatasetRef("tenant-1", "dataset-1").document("document-1")
    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, "document-1")
        assert persisted.indexing_request_id == request_id
        assert persisted.indexing_claim_id == restart_calls[0].claim_id
        assert persisted.indexing_status == IndexingStatus.PARSING
    assert deleted_keys == ["document_document-1_is_retried"]

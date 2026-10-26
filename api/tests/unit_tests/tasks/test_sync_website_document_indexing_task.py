import uuid
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import tasks.sync_website_document_indexing_task as task_module
from core.rag.index_processor.constant.index_type import IndexStructureType
from enums import DeploymentEdition
from models.dataset import Dataset, Document
from models.enums import DataSourceType, IndexingStatus
from models.model import DocumentIndexingOutbox
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    SQLAlchemyDocumentIndexingOutboxRepository,
    create_document_indexing_intents,
)
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef
from services.knowledge.indexing.website_sync_hint import website_sync_hint_key
from tests.unit_tests.config_override import config_overrides_context
from tests.unit_tests.model_factories import make_dataset, make_document, make_tenant


def _dataset(tenant_id: str) -> Dataset:
    return make_dataset(
        dataset_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        name="Website dataset",
        data_source_type=DataSourceType.WEBSITE_CRAWL,
        created_by=str(uuid.uuid4()),
    )


def _document(dataset: Dataset) -> Document:
    document = make_document(
        document_id=str(uuid.uuid4()),
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        data_source_type=DataSourceType.WEBSITE_CRAWL,
        name="Website document",
        created_by=str(uuid.uuid4()),
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    document.indexing_status = IndexingStatus.WAITING
    return document


def _persist_website_sync(
    session_factory: sessionmaker[Session],
) -> tuple[Dataset, Document, str]:
    dataset, document = _persist_waiting_website(session_factory)
    with session_factory.begin() as session:
        (intent_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(DatasetRef(dataset.tenant_id, dataset.id), website_sync=document.id),
        )
    return dataset, document, intent_id


def _persist_waiting_website(
    session_factory: sessionmaker[Session],
) -> tuple[Dataset, Document]:
    tenant_id = str(uuid.uuid4())
    dataset = _dataset(tenant_id)
    document = _document(dataset)
    with session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id=tenant_id), dataset, document])
    return dataset, document


class _HintRedis:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def register_script(self, _source: str) -> Callable[..., int]:
        def compare_and_delete(*, keys: list[str], args: list[str]) -> int:
            if self.values.get(keys[0]) != args[0]:
                return 0
            del self.values[keys[0]]
            return 1

        return compare_and_delete


class _ForbiddenRedis:
    @staticmethod
    def register_script(_source: str) -> Callable[..., int]:
        raise AssertionError("stale website-sync work must not inspect or clear a Redis hint")


class _IndexingServiceCapture:
    def __init__(self) -> None:
        self.restart_calls: list[tuple[DocumentRef, str]] = []

    def restart_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
        self.restart_calls.append((ref, claim_id))


@dataclass(frozen=True, slots=True)
class _VectorSpaceStub:
    limit: int
    size: int


@dataclass(frozen=True, slots=True)
class _FeaturesStub:
    vector_space: _VectorSpaceStub


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_legacy_message_cannot_bridge_document_outside_payload_dataset(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id = str(uuid.uuid4())
    requested_dataset = _dataset(tenant_id)
    foreign_dataset = _dataset(tenant_id)
    foreign_document = _document(foreign_dataset)
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id=tenant_id), requested_dataset, foreign_dataset, foreign_document])
    indexing_service = _IndexingServiceCapture()
    redis = _ForbiddenRedis()
    monkeypatch.setattr(task_module, "build_document_indexing_service", lambda **_kwargs: indexing_service)
    monkeypatch.setattr(task_module, "redis_client", redis)

    task_module.sync_website_document_indexing_task.run(requested_dataset.id, foreign_document.id)

    assert indexing_service.restart_calls == []
    with sqlite_session_factory() as session:
        assert session.scalar(select(DocumentIndexingOutbox.id)) is None
        stored = session.get_one(Document, foreign_document.id)
        assert stored.indexing_request_id is None
        assert stored.indexing_status == IndexingStatus.WAITING


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_legacy_message_creates_durable_request_and_enters_claimed_execution(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document = _persist_waiting_website(sqlite_session_factory)
    indexing_service = _IndexingServiceCapture()
    redis = _HintRedis({})
    monkeypatch.setattr(task_module, "build_document_indexing_service", lambda **_kwargs: indexing_service)
    monkeypatch.setattr(task_module, "redis_client", redis)

    task_module.sync_website_document_indexing_task.run(dataset.id, document.id)
    task_module.sync_website_document_indexing_task.run(dataset.id, document.id)

    [(ref, claim_id)] = indexing_service.restart_calls
    assert ref == DatasetRef(dataset.tenant_id, dataset.id).document(document.id)
    with sqlite_session_factory() as session:
        rows = session.query(DocumentIndexingOutbox).all()
        assert len(rows) == 1
        assert rows[0].action == DocumentIndexingAction.WEBSITE_SYNC.value
        stored = session.get_one(Document, document.id)
        assert stored.indexing_request_id == rows[0].id
        assert stored.indexing_claim_id == claim_id
        assert stored.indexing_status == IndexingStatus.PARSING


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_website_sync_uses_claimed_restart_for_cleanup_and_execution(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document, intent_id = _persist_website_sync(sqlite_session_factory)
    indexing_service = _IndexingServiceCapture()
    actual_key = website_sync_hint_key(document.id)
    payload_key = website_sync_hint_key("untrusted-payload-document")
    redis = _HintRedis({actual_key: intent_id, payload_key: "new-owner"})
    monkeypatch.setattr(task_module, "build_document_indexing_service", lambda **_kwargs: indexing_service)
    monkeypatch.setattr(task_module, "redis_client", redis)

    task_module.sync_website_document_indexing_task.run(
        "untrusted-payload-dataset",
        "untrusted-payload-document",
        indexing_request_id=intent_id,
    )

    [(ref, claim_id)] = indexing_service.restart_calls
    assert ref == DatasetRef(dataset.tenant_id, dataset.id).document(document.id)
    assert claim_id
    with sqlite_session_factory() as session:
        stored = session.get_one(Document, document.id)
        assert stored.indexing_status == IndexingStatus.PARSING
        assert stored.indexing_request_id == intent_id
        assert stored.indexing_claim_id == claim_id
        assert stored.indexing_claim_expires_at is not None
    assert actual_key not in redis.values
    assert redis.values[payload_key] == "new-owner"


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_superseded_sync_cannot_clean_or_overwrite_retry_generation(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document, sync_intent_id = _persist_website_sync(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        (retry_intent_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(
                DatasetRef(dataset.tenant_id, dataset.id),
                retried=(document.id,),
                actor_id="actor-1",
            ),
        )
    indexing_service = _IndexingServiceCapture()
    hint_key = website_sync_hint_key(document.id)
    redis = _HintRedis({hint_key: retry_intent_id})
    monkeypatch.setattr(task_module, "build_document_indexing_service", lambda **_kwargs: indexing_service)
    monkeypatch.setattr(task_module, "redis_client", redis)

    task_module.sync_website_document_indexing_task.run(
        dataset.id,
        document.id,
        indexing_request_id=sync_intent_id,
    )

    assert indexing_service.restart_calls == []
    assert redis.values[hint_key] == retry_intent_id
    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert repository.get_intent(sync_intent_id) is None
    assert repository.get_intent(retry_intent_id) is not None
    with sqlite_session_factory() as session:
        stored = session.get_one(Document, document.id)
        assert stored.indexing_status == IndexingStatus.WAITING
        assert stored.indexing_request_id == retry_intent_id
        assert stored.indexing_claim_id is None


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_legacy_task_cannot_replace_current_retry_request(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document = _persist_waiting_website(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        (retry_intent_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(
                DatasetRef(dataset.tenant_id, dataset.id),
                retried=(document.id,),
                actor_id="actor-1",
            ),
        )
    indexing_service = _IndexingServiceCapture()
    redis = _ForbiddenRedis()
    monkeypatch.setattr(task_module, "build_document_indexing_service", lambda **_kwargs: indexing_service)
    monkeypatch.setattr(task_module, "redis_client", redis)

    task_module.sync_website_document_indexing_task.run(dataset.id, document.id)

    assert indexing_service.restart_calls == []
    with sqlite_session_factory() as session:
        stored = session.get_one(Document, document.id)
        assert stored.indexing_request_id == retry_intent_id
        assert stored.indexing_claim_id is None
        assert stored.indexing_status == IndexingStatus.WAITING


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
def test_cloud_admission_failure_is_written_only_to_claimed_generation(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document, intent_id = _persist_website_sync(sqlite_session_factory)
    features = _FeaturesStub(vector_space=_VectorSpaceStub(limit=1, size=1))
    monkeypatch.setattr(task_module.FeatureService, "get_features", lambda _tenant_id: features)
    indexing_service = _IndexingServiceCapture()
    monkeypatch.setattr(task_module, "build_document_indexing_service", lambda **_kwargs: indexing_service)
    monkeypatch.setattr(
        task_module,
        "redis_client",
        _HintRedis({website_sync_hint_key(document.id): intent_id}),
    )

    task_module.sync_website_document_indexing_task.run(
        dataset.id,
        document.id,
        indexing_request_id=intent_id,
    )

    assert indexing_service.restart_calls == []
    with sqlite_session_factory() as session:
        stored = session.get_one(Document, document.id)
        assert stored.indexing_status == IndexingStatus.ERROR
        assert stored.indexing_request_id is None
        assert stored.indexing_claim_id is None
        assert "over the limit" in (stored.error or "")

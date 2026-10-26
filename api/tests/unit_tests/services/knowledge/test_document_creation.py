"""Exercise document creation with real SQL and assertions at external boundaries."""

from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import cast
from unittest.mock import Mock

import pytest
from redis.exceptions import LockNotOwnedError
from sqlalchemy import Engine, delete, event, select
from sqlalchemy.orm import Session, sessionmaker

from core.model_manager import ModelInstance
from core.rag.extractor.entity.extract_setting import WebsiteJobRef
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from enums import CloudPlan
from machinery.context import RequestContext
from models.account import Account, Tenant, TenantAccountJoin, TenantAccountRole
from models.credential_permission import CredentialPermission, CredentialType
from models.dataset import Dataset, DatasetProcessRule, Document, DocumentSegment
from models.enums import IndexingStatus, PermissionEnum, SegmentStatus
from models.model import DocumentIndexingOutbox
from models.oauth import DatasourceProvider
from repositories.data_source.credential_repository import SQLAlchemyDatasourceCredentialRepository
from repositories.data_source.website_crawl_job_repository import SQLAlchemyWebsiteCrawlJobRepository
from repositories.knowledge.document_indexing_outbox_repository import SQLAlchemyDocumentIndexingOutboxRepository
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.data_source.credential_gateway import (
    ActorAwareDatasourceCredentialGateway,
    RefreshedDatasourceCredential,
    TrustedStoredDatasourceCredentialGateway,
)
from services.data_source.entities.credential import DatasourceCredentialRecord
from services.data_source.provider_service import DatasourceProviderService
from services.data_source.website_crawl_cache import website_crawl_provider_instance_id
from services.data_source.website_service import WebsiteService
from services.document_indexing_proxy.batch_indexing_base import BatchDocumentIndexingProxy
from services.document_indexing_proxy.document_indexing_task_proxy import DocumentIndexingTaskProxy
from services.document_indexing_proxy.duplicate_document_indexing_task_proxy import DuplicateDocumentIndexingTaskProxy
from services.entities.feature_entities import FeatureModel
from services.errors.file import FileNotExistsError
from services.knowledge.documents.adapters import SQLAlchemyDocumentOperations
from services.knowledge.documents.application import DocumentProviderError
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.sources import NotionSourceResolver
from services.knowledge.indexing.errors import SourceCredentialUnavailableError
from services.knowledge.indexing.estimate import StoredSource
from tasks import document_indexing_outbox_task
from tests.unit_tests.model_factories import make_upload_file

CONTEXT = RequestContext("request", None, "actor", "tenant")
REF = DatasetRef("tenant", "dataset")


@dataclass
class _WebsiteCredentials:
    credentials: dict[str, object]

    def get_datasource_credentials(self, *, tenant_id: str, provider: str, plugin_id: str) -> dict[str, object]:
        assert tenant_id == "tenant"
        assert provider == "firecrawl"
        assert plugin_id == "langgenius/firecrawl_datasource"
        return dict(self.credentials)


class _PlainCredentialCodec:
    def decrypt(self, record: DatasourceCredentialRecord) -> dict[str, object]:
        return dict(record.encrypted_credentials)

    def encrypt(self, record: DatasourceCredentialRecord, credentials: Mapping[str, object]) -> dict[str, object]:
        del record
        return dict(credentials)


class _UnexpectedCredentialRefresher:
    def refresh(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        record: DatasourceCredentialRecord,
        credentials: Mapping[str, object],
    ) -> RefreshedDatasourceCredential:
        del workspace_id, actor_id, record, credentials
        raise AssertionError("non-expiring test credentials must not refresh")


def _notion_credential(
    credential_id: str,
    *,
    tenant_id: str = "tenant",
    owner_id: str = "actor",
    visibility: PermissionEnum = PermissionEnum.ONLY_ME,
) -> DatasourceProvider:
    credential = DatasourceProvider(
        tenant_id=tenant_id,
        name=credential_id,
        provider="notion_datasource",
        plugin_id="langgenius/notion_datasource",
        auth_type="oauth2",
        encrypted_credentials={"integration_secret": f"secret-{credential_id}"},
        user_id=owner_id,
        visibility=visibility,
        expires_at=-1,
    )
    credential.id = credential_id
    return credential


def settings(**values: object) -> dict[str, object]:
    return {
        "indexing_technique": "economy",
        "process_rule": {"mode": "automatic"},
        "data_source": {"info_list": {"data_source_type": "upload_file", "file_info_list": {"file_ids": ["file"]}}},
        **values,
    }


def notion_settings(*credential_ids: str, original_document_id: str | None = None) -> dict[str, object]:
    return settings(
        original_document_id=original_document_id,
        data_source={
            "info_list": {
                "data_source_type": "notion_import",
                "notion_info_list": [
                    {
                        "credential_id": credential_id,
                        "workspace_id": f"workspace-{position}",
                        "pages": [
                            {
                                "page_id": f"page-{position}",
                                "page_name": f"Page {position}",
                                "type": "page",
                            }
                        ],
                    }
                    for position, credential_id in enumerate(credential_ids, start=1)
                ],
            }
        },
    )


@pytest.fixture
def actor_credentials(
    sqlite_session_factory: sessionmaker[Session],
) -> ActorAwareDatasourceCredentialGateway:
    return ActorAwareDatasourceCredentialGateway(
        credentials=SQLAlchemyDatasourceCredentialRepository(session_factory=sqlite_session_factory),
        codec=_PlainCredentialCodec(),
        refresher=_UnexpectedCredentialRefresher(),
    )


@pytest.fixture
def operations(
    sqlite_session_factory: sessionmaker[Session],
    config_overrides: Callable[..., None],
    actor_credentials: ActorAwareDatasourceCredentialGateway,
) -> SQLAlchemyDocumentOperations:
    config_overrides(DEPLOYMENT_EDITION="CLOUD", RBAC_ENABLED=False)
    with sqlite_session_factory.begin() as session:
        actor = Account(name="Actor", email="actor@example.com")
        actor.id = "actor"
        other_actor = Account(name="Other Actor", email="other-actor@example.com")
        other_actor.id = "other-actor"
        tenant = Tenant(name="Tenant")
        tenant.id = "tenant"
        private_credential = _notion_credential("private-other", owner_id="other-actor")
        shared_credential = _notion_credential(
            "shared-other", owner_id="other-actor", visibility=PermissionEnum.ALL_TEAM
        )
        partial_credential = _notion_credential(
            "partial-other", owner_id="other-actor", visibility=PermissionEnum.PARTIAL_TEAM
        )
        session.add_all(
            [
                actor,
                other_actor,
                tenant,
                TenantAccountJoin(tenant_id="tenant", account_id="actor", role=TenantAccountRole.OWNER),
                TenantAccountJoin(tenant_id="tenant", account_id="other-actor", role=TenantAccountRole.NORMAL),
                Dataset(
                    id="dataset", tenant_id="tenant", name="Dataset", created_by="actor", indexing_technique="economy"
                ),
                make_upload_file(file_id="file", tenant_id="tenant", name="example.txt", created_by="actor"),
                make_upload_file(file_id="foreign-file", tenant_id="foreign", name="foreign.txt"),
                _notion_credential("credential"),
                private_credential,
                shared_credential,
                partial_credential,
                _notion_credential("foreign-credential", tenant_id="foreign", owner_id="other-actor"),
                CredentialPermission(
                    credential_id=partial_credential.id,
                    credential_type=CredentialType.DATASOURCE_PROVIDER,
                    account_id="actor",
                    tenant_id="tenant",
                ),
            ]
        )
    return SQLAlchemyDocumentOperations(
        session_factory=sqlite_session_factory,
        actor_credentials=actor_credentials,
    )


@pytest.fixture
def outside_transaction(sqlite_engine: Engine) -> Iterator[Callable[[], None]]:
    active: set[int] = set()

    def begin(connection: object) -> None:
        active.add(id(connection))

    def end(connection: object) -> None:
        active.discard(id(connection))

    def check() -> None:
        assert not active, "External I/O must not hold an open database transaction"

    listeners = [("begin", begin), ("commit", end), ("rollback", end)]
    for name, callback in listeners:
        event.listen(sqlite_engine, name, callback)
    yield check
    for name, callback in listeners:
        event.remove(sqlite_engine, name, callback)


@pytest.fixture
def external_calls(
    monkeypatch: pytest.MonkeyPatch,
    outside_transaction: Callable[[], None],
    sqlite_session_factory: sessionmaker[Session],
) -> list[tuple[str, tuple[str, ...]]]:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def features(_tenant_id: str, *, exclude_vector_space: bool) -> FeatureModel:
        outside_transaction()
        assert exclude_vector_space is True
        result = FeatureModel()
        result.billing.subscription.plan = CloudPlan.PROFESSIONAL
        return result

    @contextmanager
    def lock(_name: str, *, timeout: int) -> Generator[None]:
        outside_transaction()
        assert timeout == 600
        yield
        outside_transaction()

    def dispatched(kind: str, dataset_id: str, document_ids: tuple[str, ...]) -> None:
        outside_transaction()
        with sqlite_session_factory() as session:
            for document_id in document_ids:
                document = session.get(Document, document_id)
                assert document is not None
                assert document.dataset_id == dataset_id
                if kind != "clean":
                    assert document.indexing_status == IndexingStatus.WAITING
        calls.append((kind, document_ids))

    def create(proxy: BatchDocumentIndexingProxy) -> None:
        dispatched("create", proxy._dataset_id, tuple(proxy._document_ids))

    def duplicate(proxy: BatchDocumentIndexingProxy) -> None:
        dispatched("duplicate", proxy._dataset_id, tuple(proxy._document_ids))

    def update(dataset_id: str, document_id: str, *, indexing_request_id: str) -> None:
        assert indexing_request_id
        dispatched("update", dataset_id, (document_id,))

    def clean(document_ids: list[str], dataset_id: str, *, indexing_request_id: str) -> None:
        assert indexing_request_id
        dispatched("clean", dataset_id, tuple(document_ids))

    monkeypatch.setattr("services.knowledge.dataset_service.FeatureService.get_features", features)
    monkeypatch.setattr("services.knowledge.documents.adapters.redis_client.lock", lock)
    monkeypatch.setattr(DocumentIndexingTaskProxy, "delay", create)
    monkeypatch.setattr(DuplicateDocumentIndexingTaskProxy, "delay", duplicate)
    monkeypatch.setattr(
        "tasks.document_indexing_outbox_task.document_indexing_update_task.delay",
        update,
    )
    monkeypatch.setattr(
        "tasks.document_indexing_outbox_task.clean_notion_document_task.delay",
        clean,
    )
    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    monkeypatch.setattr(document_indexing_outbox_task, "_repository", lambda: repository)
    monkeypatch.setattr(
        "services.knowledge.dataset_service.enqueue_document_indexing_intent",
        document_indexing_outbox_task.process_document_indexing_intent_task.run,
    )
    return calls


@pytest.mark.parametrize("initialize", [False, True])
def test_upload_preserves_response_and_commits_before_dispatch(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
    initialize: bool,
) -> None:
    result = (
        operations.initialize_dataset(CONTEXT, settings())
        if initialize
        else operations.create_documents(CONTEXT, REF, settings())
    )
    (document,) = result["documents"]
    assert document["name"] == "example.txt"
    assert document["data_source_info_dict"] == {"upload_file_id": "file"}
    assert document["indexing_status"] == "waiting"
    assert result["batch"]
    assert external_calls == [("create", (document["id"],))]
    with sqlite_session_factory() as session:
        dataset = session.get(Dataset, result["dataset"]["id"])
        assert dataset is not None
        assert dataset.tenant_id == "tenant"
        assert dataset.name == ("example.txt..." if initialize else "Dataset")
        assert dataset.permission == "only_me"


def test_website_creation_persists_resolved_provider_instance(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
    actor_credentials: ActorAwareDatasourceCredentialGateway,
) -> None:
    del operations  # The fixture seeds the tenant, actor, and dataset graph.
    instance_id = website_crawl_provider_instance_id(provider="firecrawl", base_url="https://crawl-a.example/api")
    website_jobs = Mock()
    website_jobs.resolve_job_ref.return_value = WebsiteJobRef(
        tenant_id="tenant",
        provider="firecrawl",
        provider_instance_id=instance_id,
        job_id="job-1",
    )
    document_operations = SQLAlchemyDocumentOperations(
        session_factory=sqlite_session_factory,
        actor_credentials=actor_credentials,
        website_jobs=website_jobs,
    )
    config = settings(
        data_source={
            "info_list": {
                "data_source_type": "website_crawl",
                "website_info_list": {
                    "provider": "firecrawl",
                    "job_id": "job-1",
                    "source_handle": "source-handle",
                    "urls": ["https://source.example"],
                    "only_main_content": True,
                },
            }
        }
    )

    result = document_operations.create_documents(CONTEXT, REF, config)

    website_jobs.resolve_job_ref.assert_called_once_with(
        tenant_id="tenant",
        provider="firecrawl",
        job_id="job-1",
        source_handle="source-handle",
    )
    assert result["documents"][0]["data_source_info_dict"] == {
        "url": "https://source.example",
        "provider": "firecrawl",
        "provider_instance_id": instance_id,
        "job_id": "job-1",
        "only_main_content": True,
        "mode": "crawl",
    }
    assert external_calls == [("create", (result["documents"][0]["id"],))]


def test_website_creation_resolves_same_job_by_server_handle_across_instances(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
    actor_credentials: ActorAwareDatasourceCredentialGateway,
) -> None:
    del operations
    base_url_a = "https://crawl-a.example/api"
    base_url_b = "https://crawl-b.example/api"
    instance_a = website_crawl_provider_instance_id(provider="firecrawl", base_url=base_url_a)
    instance_b = website_crawl_provider_instance_id(provider="firecrawl", base_url=base_url_b)
    jobs = SQLAlchemyWebsiteCrawlJobRepository(session_factory=sqlite_session_factory)
    handle_a = jobs.register_created(
        tenant_id="tenant", provider="firecrawl", provider_instance_id=instance_a, job_id="shared-job"
    )
    handle_b = jobs.register_created(
        tenant_id="tenant", provider="firecrawl", provider_instance_id=instance_b, job_id="shared-job"
    )
    assert handle_a is not None
    assert handle_b is not None
    credentials = _WebsiteCredentials({"firecrawl_api_key": "key-a", "base_url": base_url_a})
    website = WebsiteService(providers=cast(DatasourceProviderService, credentials), crawl_jobs=jobs)
    document_operations = SQLAlchemyDocumentOperations(
        session_factory=sqlite_session_factory,
        actor_credentials=actor_credentials,
        website_jobs=website,
    )

    def website_settings(*, source_handle: str, url: str) -> dict[str, object]:
        return settings(
            data_source={
                "info_list": {
                    "data_source_type": "website_crawl",
                    "website_info_list": {
                        "provider": "firecrawl",
                        "job_id": "shared-job",
                        "source_handle": source_handle,
                        "urls": [url],
                    },
                }
            }
        )

    result_a = document_operations.create_documents(
        CONTEXT,
        REF,
        website_settings(source_handle=handle_a, url="https://source.example/a"),
    )
    credentials.credentials = {"firecrawl_api_key": "key-b", "base_url": base_url_b}
    result_b = document_operations.create_documents(
        CONTEXT,
        REF,
        website_settings(source_handle=handle_b, url="https://source.example/b"),
    )

    assert result_a["documents"][0]["data_source_info_dict"]["provider_instance_id"] == instance_a
    assert result_b["documents"][0]["data_source_info_dict"]["provider_instance_id"] == instance_b
    with pytest.raises(ValueError, match="endpoint has changed"):
        document_operations.create_documents(
            CONTEXT,
            REF,
            website_settings(source_handle=handle_a, url="https://source.example/mismatch"),
        )
    assert len(external_calls) == 2


def test_committed_creation_survives_initial_broker_failure_and_periodic_recovery(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_publish(_intent_id: str) -> None:
        raise OSError("broker unavailable")

    monkeypatch.setattr(
        "services.knowledge.dataset_service.enqueue_document_indexing_intent",
        document_indexing_outbox_task.enqueue_document_indexing_intent,
    )
    monkeypatch.setattr(document_indexing_outbox_task.process_document_indexing_intent_task, "delay", reject_publish)

    result = operations.create_documents(CONTEXT, REF, settings())
    document_id = result["documents"][0]["id"]

    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    (intent_id,) = repository.list_pending_intent_ids(limit=10)
    with sqlite_session_factory() as session:
        document = session.get_one(Document, document_id)
        assert document.indexing_status == IndexingStatus.WAITING
    assert external_calls == []

    monkeypatch.setattr(
        document_indexing_outbox_task.process_document_indexing_intent_task,
        "delay",
        document_indexing_outbox_task.process_document_indexing_intent_task.run,
    )
    document_indexing_outbox_task.drain_document_indexing_outbox_task.run()

    assert external_calls == [("create", (document_id,))]
    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (claim,) = documents.claim_indexing_request(
        intent_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )
    documents.complete_indexing(claim.ref, claim_id=claim.claim_id, tokens=1, latency=0.1)
    assert repository.get_intent(intent_id) is None


def test_commit_failure_rolls_back_documents_and_indexing_intents_without_publish(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    def reject_commit(_session: Session) -> None:
        raise RuntimeError("database unavailable")

    event.listen(sqlite_session_factory, "before_commit", reject_commit)
    try:
        with pytest.raises(RuntimeError, match="database unavailable"):
            operations.create_documents(CONTEXT, REF, settings())
    finally:
        event.remove(sqlite_session_factory, "before_commit", reject_commit)

    assert external_calls == []
    with sqlite_session_factory() as session:
        assert session.scalar(select(Document.id)) is None
    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert repository.list_pending_intent_ids(limit=10) == ()


def test_duplicate_upload_reuses_document_and_dispatches_duplicate_job(
    operations: SQLAlchemyDocumentOperations, external_calls: list[tuple[str, tuple[str, ...]]]
) -> None:
    first = operations.create_documents(CONTEXT, REF, settings())
    second = operations.create_documents(CONTEXT, REF, settings())
    first_id = first["documents"][0]["id"]
    assert second["documents"][0]["id"] == first_id
    assert external_calls == [("create", (first_id,)), ("duplicate", (first_id,))]


def test_reprocess_checks_model_outside_transaction_and_preserves_segment_reset(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    outside_transaction: Callable[[], None],
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = operations.create_documents(CONTEXT, REF, settings())
    document_id = created["documents"][0]["id"]
    with sqlite_session_factory.begin() as session:
        document = session.get(Document, document_id)
        assert document is not None
        document.indexing_status = IndexingStatus.COMPLETED
        document.completed_at = datetime(2026, 1, 1)
        dataset = session.get(Dataset, REF.dataset_id)
        assert dataset is not None
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.embedding_model = "embedding"
        dataset.embedding_model_provider = "provider"
        session.add(
            DocumentSegment(
                tenant_id="tenant",
                dataset_id="dataset",
                document_id=document_id,
                position=1,
                content="content",
                word_count=1,
                tokens=1,
                created_by="actor",
                status=SegmentStatus.COMPLETED,
            )
        )
    manager = Mock()
    manager.get_model_instance.side_effect = lambda **_kwargs: outside_transaction()
    monkeypatch.setattr("services.knowledge.dataset_service.ModelManager.for_tenant", lambda **_kwargs: manager)
    result = operations.create_documents(CONTEXT, REF, settings(original_document_id=document_id, name="Renamed"))
    assert result["documents"][0]["id"] == document_id
    assert result["documents"][0]["name"] == "Renamed"
    assert result["batch"] == created["batch"]
    assert external_calls[-1] == ("update", (document_id,))
    manager.get_model_instance.assert_called_once()
    with sqlite_session_factory() as session:
        segment = session.scalar(select(DocumentSegment).where(DocumentSegment.document_id == document_id))
        assert segment is not None
        assert segment.status == SegmentStatus.RE_SEGMENT


def test_first_high_quality_upload_resolves_default_model_without_transaction(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    outside_transaction: Callable[[], None],
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, REF.dataset_id)
        assert dataset is not None
        dataset.indexing_technique = None

    def model(**_kwargs: object) -> ModelInstance:
        outside_transaction()
        instance = object.__new__(ModelInstance)
        instance.model_name = "default-embedding"
        instance.provider = "default-provider"
        return instance

    class DefaultModelManager:
        def get_default_model_instance(self, **kwargs: object) -> ModelInstance:
            return model(**kwargs)

    manager = DefaultModelManager()
    monkeypatch.setattr("services.knowledge.dataset_service.ModelManager.for_tenant", lambda **_kwargs: manager)
    operations.create_documents(CONTEXT, REF, settings(indexing_technique="high_quality"))
    with sqlite_session_factory() as session:
        dataset = session.get(Dataset, REF.dataset_id)
        assert dataset is not None
        assert dataset.embedding_model == "default-embedding"
        assert dataset.embedding_model_provider == "default-provider"
        assert dataset.collection_binding_id is not None
    assert len(external_calls) == 1


def test_foreign_upload_is_rejected_without_dispatch(
    operations: SQLAlchemyDocumentOperations, external_calls: list[tuple[str, tuple[str, ...]]]
) -> None:
    config = settings(
        data_source={"info_list": {"data_source_type": "upload_file", "file_info_list": {"file_ids": ["foreign-file"]}}}
    )
    with pytest.raises(FileNotExistsError):
        operations.create_documents(CONTEXT, REF, config)
    assert external_calls == []


@pytest.mark.parametrize("credential_id", ["credential", "shared-other", "partial-other"])
def test_notion_creation_accepts_actor_owned_and_shared_credentials(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    credential_id: str,
) -> None:
    result = operations.create_documents(CONTEXT, REF, notion_settings(credential_id))

    document = result["documents"][0]
    assert document["data_source_info_dict"]["credential_id"] == credential_id
    assert external_calls == [("create", (document["id"],))]


@pytest.mark.parametrize("initialize", [False, True])
def test_private_notion_credential_is_rejected_before_create_or_initialize_writes(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
    initialize: bool,
) -> None:
    if initialize:
        create = partial(operations.initialize_dataset, CONTEXT, notion_settings("private-other"))
    else:
        create = partial(operations.create_documents, CONTEXT, REF, notion_settings("private-other"))

    with pytest.raises(DocumentProviderError, match="Credential not found"):
        create()

    assert external_calls == []
    with sqlite_session_factory() as session:
        assert session.scalars(select(Dataset.id).order_by(Dataset.id)).all() == ["dataset"]
        assert session.scalar(select(Document.id)) is None
        assert session.scalar(select(DocumentIndexingOutbox.id)) is None


@pytest.mark.parametrize("credential_id", ["foreign-credential", ""])
def test_notion_creation_rejects_cross_tenant_or_empty_credentials_without_writes(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
    credential_id: str,
) -> None:
    with pytest.raises(DocumentProviderError, match="Credential not found"):
        operations.create_documents(CONTEXT, REF, notion_settings(credential_id))

    assert external_calls == []
    with sqlite_session_factory() as session:
        assert session.scalar(select(Document.id)) is None
        assert session.scalar(select(DocumentIndexingOutbox.id)) is None


def test_multi_workspace_notion_selection_validates_every_credential_before_writes(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with pytest.raises(DocumentProviderError, match="Credential not found"):
        operations.create_documents(CONTEXT, REF, notion_settings("credential", "private-other"))

    assert external_calls == []
    with sqlite_session_factory() as session:
        assert session.scalar(select(Document.id)) is None
        assert session.scalar(select(DocumentIndexingOutbox.id)) is None


def test_notion_source_swap_rejects_private_credential_without_changing_document_or_intent(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    created = operations.create_documents(CONTEXT, REF, settings())
    document_id = created["documents"][0]["id"]
    with sqlite_session_factory() as session:
        before = session.get_one(Document, document_id)
        source_before = (before.data_source_type, before.data_source_info, before.name)
        intent_ids_before = tuple(
            session.scalars(select(DocumentIndexingOutbox.id).order_by(DocumentIndexingOutbox.id))
        )

    with pytest.raises(DocumentProviderError, match="Credential not found"):
        operations.create_documents(
            CONTEXT,
            REF,
            notion_settings("private-other", original_document_id=document_id),
        )

    assert len(external_calls) == 1
    with sqlite_session_factory() as session:
        document = session.get_one(Document, document_id)
        assert (document.data_source_type, document.data_source_info, document.name) == source_before
        assert (
            tuple(session.scalars(select(DocumentIndexingOutbox.id).order_by(DocumentIndexingOutbox.id)))
            == intent_ids_before
        )


def test_deleted_explicit_notion_credential_cannot_fall_back_to_global_token(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
    actor_credentials: ActorAwareDatasourceCredentialGateway,
) -> None:
    result = operations.create_documents(CONTEXT, REF, notion_settings("credential"))
    document_id = result["documents"][0]["id"]
    assert external_calls == [("create", (document_id,))]

    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, document_id)
        source_info = document.data_source_info_dict
        session.execute(delete(DatasourceProvider).where(DatasourceProvider.id == "credential"))

    stored_credentials = TrustedStoredDatasourceCredentialGateway(
        credentials=SQLAlchemyDatasourceCredentialRepository(session_factory=sqlite_session_factory),
        codec=_PlainCredentialCodec(),
        refresher=_UnexpectedCredentialRefresher(),
    )
    resolver = NotionSourceResolver(
        actor_credentials=actor_credentials,
        stored_credentials=stored_credentials,
        integration_token="global-integration-token",
    )
    source = StoredSource(
        document_ref=REF.document(document_id),
        source_type="notion_import",
        source_info=source_info,
        document_model="text_model",
    )

    with pytest.raises(SourceCredentialUnavailableError, match="Notion credential is unavailable"):
        resolver.resolve(source)


def test_notion_removal_is_dispatched_after_new_documents_commit(
    operations: SQLAlchemyDocumentOperations, external_calls: list[tuple[str, tuple[str, ...]]]
) -> None:
    def notion(page_id: str) -> dict[str, object]:
        return settings(
            data_source={
                "info_list": {
                    "data_source_type": "notion_import",
                    "notion_info_list": [
                        {
                            "credential_id": "credential",
                            "workspace_id": "workspace",
                            "pages": [{"page_id": page_id, "page_name": page_id, "type": "page"}],
                        }
                    ],
                }
            }
        )

    first = operations.create_documents(CONTEXT, REF, notion("old-page"))
    old_id = first["documents"][0]["id"]
    operations.create_documents(CONTEXT, REF, notion("new-page"))
    assert external_calls[-2] == ("clean", (old_id,))
    assert external_calls[-1][0] == "create"
    assert old_id not in external_calls[-1][1]


def test_cloud_quota_rejection_does_not_create_documents_or_dispatch(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
    config_overrides: Callable[..., None],
) -> None:
    config_overrides(BATCH_UPLOAD_LIMIT=0)
    with pytest.raises(ValueError, match="batch upload limit"):
        operations.create_documents(CONTEXT, REF, settings())
    assert external_calls == []
    with sqlite_session_factory() as session:
        assert session.scalar(select(Document.id)) is None


def test_response_read_failure_rolls_back_without_dispatch(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock(side_effect=ValueError("response read failed"))
    monkeypatch.setattr("services.knowledge.documents.adapters._created_response", response)
    with pytest.raises(ValueError, match="response read failed"):
        operations.create_documents(CONTEXT, REF, settings())
    assert external_calls == []
    with sqlite_session_factory() as session:
        assert session.scalar(select(Document.id)) is None
        assert session.scalar(select(DatasetProcessRule.id)) is None
        dataset = session.get(Dataset, REF.dataset_id)
        assert dataset is not None
        assert dataset.data_source_type is None
        assert dataset.indexing_technique == "economy"


def test_lost_creation_lock_keeps_empty_batch_response(
    operations: SQLAlchemyDocumentOperations,
    external_calls: list[tuple[str, tuple[str, ...]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def lock(_name: str, **_kwargs: object) -> Generator[None]:
        raise LockNotOwnedError("lock lost")
        yield  # pragma: no cover

    monkeypatch.setattr("services.knowledge.documents.adapters.redis_client.lock", lock)
    result = operations.create_documents(CONTEXT, REF, settings())
    assert result["documents"] == []
    assert result["batch"]
    assert result["dataset"]["id"] == REF.dataset_id
    assert external_calls == []


def test_billing_http_boundary_does_not_hold_a_transaction(
    operations: SQLAlchemyDocumentOperations,
    outside_transaction: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopBeforeNetworkError(Exception):
        pass

    def request(_method: str, url: str, **_kwargs: object) -> None:
        assert url.endswith("/subscription/info")
        outside_transaction()
        raise StopBeforeNetworkError()

    monkeypatch.setattr("services.billing_service._http_client.request", request)
    with pytest.raises(StopBeforeNetworkError):
        operations.create_documents(CONTEXT, REF, settings())
    outside_transaction()

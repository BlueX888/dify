"""Transaction regressions for upload-backed document creation.

SQLite cannot reproduce MySQL's ``REPEATABLE READ`` snapshots. These tests
exercise the portable contract around that implementation detail: a committed
upload from another session is accepted, while dataset configuration, process
rules, and documents remain one caller-owned transaction.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.exceptions import NotFound

from enums import DeploymentEdition
from enums.account import TenantAccountRole
from graphon.model_runtime.entities.model_entities import ModelType
from machinery.context import RequestContext
from models.account import Account, Tenant
from models.dataset import Dataset, DatasetCollectionBinding, DatasetProcessRule, Document
from models.model import DatasetIndexingOutbox, DocumentIndexingOutbox
from services.errors.file import FileNotExistsError
from services.knowledge import dataset_service as dataset_service_module
from services.knowledge.dataset_service import DatasetService, DocumentService, ExternalKnowledgeTemplateLookup
from services.knowledge.documents.adapters import SQLAlchemyDocumentOperations
from services.knowledge.entities.knowledge_entities import KnowledgeConfig
from services.knowledge.entities.resource_scope import DatasetRef
from tests.unit_tests.config_override import config_overrides_context
from tests.unit_tests.model_factories import make_account, make_document, make_tenant, make_upload_file

TENANT_ID = "tenant"
DATASET_ID = "dataset"
ACTOR_ID = "actor"
FILE_ID = "file"


class _UnexpectedActorCredentials:
    def resolve(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        credential_id: str,
        provider: str,
        plugin_id: str,
    ) -> dict[str, object]:
        del workspace_id, actor_id, credential_id, provider, plugin_id
        raise AssertionError("upload snapshot test did not select a datasource credential")


class _UnexpectedExternalTemplateLookup(ExternalKnowledgeTemplateLookup):
    def require_template(self, tenant_id: str, template_id: str, *, session: Session) -> None:
        del tenant_id, template_id, session
        raise AssertionError("internal dataset update must not resolve an external template")


@dataclass(frozen=True, slots=True)
class _EmbeddingModelBoundary:
    provider: str
    model_name: str


class _ModelManagerBoundary:
    def get_model_instance(
        self,
        *,
        tenant_id: str,
        provider: str,
        model_type: ModelType,
        model: str,
    ) -> _EmbeddingModelBoundary:
        assert tenant_id == TENANT_ID
        assert model_type == ModelType.TEXT_EMBEDDING
        return _EmbeddingModelBoundary(provider=provider, model_name=model)


def _account() -> Account:
    return make_account(
        account_id=ACTOR_ID,
        name="Actor",
        email="actor@example.com",
        tenant=make_tenant(tenant_id=TENANT_ID, name="Tenant"),
    )


def _config() -> KnowledgeConfig:
    return KnowledgeConfig.model_validate(
        {
            "indexing_technique": "economy",
            "process_rule": {"mode": "automatic"},
            "data_source": {
                "info_list": {
                    "data_source_type": "upload_file",
                    "file_info_list": {"file_ids": [FILE_ID]},
                }
            },
        }
    )


@pytest.fixture
def dataset_id(sqlite_session_factory: sessionmaker[Session]) -> str:
    with sqlite_session_factory.begin() as session:
        session.add(make_tenant(tenant_id=TENANT_ID, name="Tenant"))
        session.add(Dataset(id=DATASET_ID, tenant_id=TENANT_ID, name="Dataset", created_by=ACTOR_ID))
    return DATASET_ID


def _assert_no_creation_writes(session: Session) -> None:
    dataset = session.get(Dataset, DATASET_ID)
    assert dataset is not None
    assert dataset.data_source_type is None
    assert dataset.indexing_technique is None
    assert (
        session.scalar(
            select(func.count()).select_from(DatasetProcessRule).where(DatasetProcessRule.dataset_id == DATASET_ID)
        )
        == 0
    )
    assert session.scalar(select(func.count()).select_from(Document).where(Document.dataset_id == DATASET_ID)) == 0


def test_committed_upload_from_another_session_joins_atomic_creation(
    dataset_id: str, sqlite_session_factory: sessionmaker[Session]
) -> None:
    caller = sqlite_session_factory()
    try:
        stored_dataset = caller.get(Dataset, dataset_id)
        assert stored_dataset is not None

        # Mirrors FileService committing the upload after the document caller
        # has already performed its first read. The public wrapper starts a
        # fresh transaction after acquiring the creation lock on MySQL.
        with sqlite_session_factory.begin() as uploader:
            uploader.add(
                make_upload_file(
                    file_id=FILE_ID,
                    tenant_id=TENANT_ID,
                    name="guide.txt",
                    created_by=ACTOR_ID,
                )
            )

        documents, batch, jobs = DocumentService.save_prepared_documents(
            stored_dataset,
            _config(),
            _account(),
            session=caller,
        )

        assert len(documents) == 1
        assert documents[0].name == "guide.txt"
        assert documents[0].data_source_info_dict == {"upload_file_id": FILE_ID}
        assert batch
        assert jobs.created == (documents[0].id,)

        with sqlite_session_factory() as observer:
            _assert_no_creation_writes(observer)

        caller.commit()
    finally:
        caller.close()

    with sqlite_session_factory() as observer:
        stored_dataset = observer.get(Dataset, DATASET_ID)
        assert stored_dataset is not None
        assert stored_dataset.data_source_type == "upload_file"
        assert stored_dataset.indexing_technique == "economy"
        rule = observer.scalar(select(DatasetProcessRule).where(DatasetProcessRule.dataset_id == DATASET_ID))
        document = observer.scalar(select(Document).where(Document.dataset_id == DATASET_ID))
        assert rule is not None
        assert document is not None
        assert document.dataset_process_rule_id == rule.id
        assert document.data_source_info_dict == {"upload_file_id": FILE_ID}


def test_stale_first_document_request_cannot_overwrite_concurrent_dataset_generation(
    dataset_id: str,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = _account()
    account.role = TenantAccountRole.OWNER
    with sqlite_session_factory.begin() as seed:
        binding_b = DatasetCollectionBinding(
            provider_name="provider-b",
            model_name="model-b",
            type="dataset",
            collection_name="collection-b",
        )
        binding_b.id = "binding-b"
        seed.add_all(
            [
                make_upload_file(
                    file_id=FILE_ID,
                    tenant_id=TENANT_ID,
                    name="guide.txt",
                    created_by=ACTOR_ID,
                ),
                binding_b,
            ]
        )

    monkeypatch.setattr(dataset_service_module, "current_user", account)

    def model_manager_for_tenant(*, tenant_id: str) -> _ModelManagerBoundary:
        assert tenant_id == TENANT_ID
        return _ModelManagerBoundary()

    monkeypatch.setattr(
        dataset_service_module.ModelManager,
        "for_tenant",
        staticmethod(model_manager_for_tenant),
    )
    dispatched: list[str] = []
    monkeypatch.setattr(DatasetService, "dispatch_dataset_indexing", dispatched.extend)

    with sqlite_session_factory() as caller:
        stale_dataset = caller.get_one(Dataset, dataset_id)
        assert stale_dataset.indexing_technique is None

        with sqlite_session_factory() as patch_session:
            with DatasetService.update_dataset(
                dataset_id,
                {
                    "indexing_technique": "high_quality",
                    "embedding_model_provider": "provider-b",
                    "embedding_model": "model-b",
                },
                account,
                session=patch_session,
                external_templates=_UnexpectedExternalTemplateLookup(),
            ):
                pass

        assert len(dispatched) == 1
        generation_intent_id = dispatched[0]
        # A prepared first-document request still carries configuration A and
        # has dirty stale identity-map fields when PATCH B/G has already won.
        stale_dataset.indexing_technique = "high_quality"
        stale_dataset.embedding_model_provider = "provider-a"
        stale_dataset.embedding_model = "model-a"
        stale_dataset.collection_binding_id = "binding-a"
        prepared_a = _config().model_copy(
            update={
                "indexing_technique": "high_quality",
                "embedding_model_provider": "provider-a",
                "embedding_model": "model-a",
            }
        )

        with pytest.raises(ValueError, match="Prepared embedding provider differs"):
            DocumentService.save_prepared_documents(
                stale_dataset,
                prepared_a,
                account,
                session=caller,
            )
        caller.rollback()

    with sqlite_session_factory() as observer:
        current = observer.get_one(Dataset, dataset_id)
        intent = observer.get_one(DatasetIndexingOutbox, generation_intent_id)
        assert current.indexing_technique == "high_quality"
        assert current.embedding_model_provider == "provider-b"
        assert current.embedding_model == "model-b"
        assert current.collection_binding_id == "binding-b"
        assert current.indexing_request_id == intent.generation_id
        assert observer.scalar(select(func.count()).select_from(Document).where(Document.dataset_id == dataset_id)) == 0
        assert (
            observer.scalar(
                select(func.count()).select_from(DatasetProcessRule).where(DatasetProcessRule.dataset_id == dataset_id)
            )
            == 0
        )


def test_configured_dataset_accepts_omitted_prepared_configuration_without_changing_generation(
    dataset_id: str,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = _account()
    account.role = TenantAccountRole.OWNER
    with sqlite_session_factory.begin() as seed:
        binding_b = DatasetCollectionBinding(
            provider_name="provider-b",
            model_name="model-b",
            type="dataset",
            collection_name="collection-b",
        )
        binding_b.id = "binding-b"
        seed.add_all(
            [
                make_upload_file(
                    file_id=FILE_ID,
                    tenant_id=TENANT_ID,
                    name="guide.txt",
                    created_by=ACTOR_ID,
                ),
                binding_b,
            ]
        )

    monkeypatch.setattr(dataset_service_module, "current_user", account)

    def model_manager_for_tenant(*, tenant_id: str) -> _ModelManagerBoundary:
        assert tenant_id == TENANT_ID
        return _ModelManagerBoundary()

    monkeypatch.setattr(
        dataset_service_module.ModelManager,
        "for_tenant",
        staticmethod(model_manager_for_tenant),
    )
    dispatched: list[str] = []
    monkeypatch.setattr(DatasetService, "dispatch_dataset_indexing", dispatched.extend)
    with sqlite_session_factory() as patch_session:
        with DatasetService.update_dataset(
            dataset_id,
            {
                "indexing_technique": "high_quality",
                "embedding_model_provider": "provider-b",
                "embedding_model": "model-b",
            },
            account,
            session=patch_session,
            external_templates=_UnexpectedExternalTemplateLookup(),
        ):
            pass
    (generation_intent_id,) = dispatched

    omitted = _config().model_copy(
        update={
            "indexing_technique": None,
            "embedding_model_provider": None,
            "embedding_model": None,
        }
    )
    with sqlite_session_factory() as creator:
        dataset = creator.get_one(Dataset, dataset_id)
        documents, _batch, jobs = DocumentService.save_prepared_documents(
            dataset,
            omitted,
            account,
            session=creator,
        )
        assert jobs.created == (documents[0].id,)
        creator.commit()

    with sqlite_session_factory() as observer:
        current = observer.get_one(Dataset, dataset_id)
        intent = observer.get_one(DatasetIndexingOutbox, generation_intent_id)
        assert current.embedding_model_provider == "provider-b"
        assert current.embedding_model == "model-b"
        assert current.collection_binding_id == "binding-b"
        assert current.indexing_request_id == intent.generation_id
        document = observer.scalar(select(Document).where(Document.dataset_id == dataset_id))
        assert document is not None
        assert document.name == "guide.txt"
        assert observer.scalar(select(func.count()).select_from(DatasetProcessRule)) == 1


@pytest.mark.parametrize("file_tenant", [None, "other-tenant"])
def test_missing_or_foreign_upload_rolls_back_all_creation_writes(
    dataset_id: str,
    sqlite_session_factory: sessionmaker[Session],
    file_tenant: str | None,
) -> None:
    if file_tenant is not None:
        with sqlite_session_factory.begin() as uploader:
            uploader.add(make_upload_file(file_id=FILE_ID, tenant_id=file_tenant, created_by=ACTOR_ID))

    with sqlite_session_factory() as caller:
        stored_dataset = caller.get(Dataset, dataset_id)
        assert stored_dataset is not None
        with pytest.raises(FileNotExistsError, match="One or more files not found"):
            DocumentService.save_prepared_documents(
                stored_dataset,
                _config(),
                _account(),
                session=caller,
            )
        caller.rollback()

    with sqlite_session_factory() as observer:
        _assert_no_creation_writes(observer)


def test_response_failure_rolls_back_modern_creation_without_dispatch(
    dataset_id: str,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(make_upload_file(file_id=FILE_ID, tenant_id=TENANT_ID, created_by=ACTOR_ID))

    operations = SQLAlchemyDocumentOperations(
        session_factory=sqlite_session_factory,
        actor_credentials=_UnexpectedActorCredentials(),
    )
    account = _account()
    dispatched: list[object] = []

    def fail_response(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("response construction failed")

    monkeypatch.setattr(operations, "_actor", lambda _session, _context: account)
    monkeypatch.setattr("services.knowledge.documents.adapters._created_response", fail_response)
    monkeypatch.setattr(DocumentService, "dispatch_document_indexing", dispatched.append)

    with pytest.raises(RuntimeError, match="response construction failed"):
        operations._persist_creation(
            RequestContext("request", None, ACTOR_ID, TENANT_ID),
            DatasetRef(TENANT_ID, dataset_id),
            _config(),
            initialize=False,
            website_job_ref=None,
        )

    assert dispatched == []
    with sqlite_session_factory() as observer:
        _assert_no_creation_writes(observer)


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_creation_lock_uses_current_reads_without_replacing_caller_transaction(
    dataset_id: str,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as uploader:
        uploader.add(make_upload_file(file_id=FILE_ID, tenant_id=TENANT_ID, name="guide.txt", created_by=ACTOR_ID))

    @contextmanager
    def acquire_creation_lock(*_args: object, **_kwargs: object) -> Generator[None]:
        # Another request commits while this caller waits for the dataset lock.
        with sqlite_session_factory.begin() as writer:
            dataset = writer.get(Dataset, dataset_id)
            assert dataset is not None
            dataset.description = "Updated by the preceding request"
            writer.add(
                make_document(
                    document_id="existing-document",
                    tenant_id=TENANT_ID,
                    dataset_id=dataset_id,
                    name="guide.txt",
                    created_by=ACTOR_ID,
                    data_source_info='{"upload_file_id": "file"}',
                    doc_form="text_model",
                )
            )
        yield

    dispatched: list[object] = []
    monkeypatch.setattr("services.knowledge.dataset_service.redis_client.lock", acquire_creation_lock)
    monkeypatch.setattr(DocumentService, "dispatch_document_indexing", dispatched.append)

    with sqlite_session_factory() as caller:
        dataset = caller.get(Dataset, dataset_id)
        assert dataset is not None
        old_transaction = caller.get_transaction()
        assert old_transaction is not None
        assert dataset.description is None

        with DocumentService.save_document_with_dataset_id(
            dataset,
            _config().model_copy(update={"duplicate": True}),
            _account(),
            session=caller,
        ) as (documents, _batch):
            assert caller.get_transaction() is old_transaction
            assert caller.get_nested_transaction() is not None
            assert dataset.description == "Updated by the preceding request"
            assert [document.id for document in documents] == ["existing-document"]

    assert len(dispatched) == 1
    with sqlite_session_factory() as observer:
        assert observer.scalar(select(func.count()).select_from(Document)) == 1


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_creation_failure_rolls_back_savepoint_without_discarding_caller_state(
    dataset_id: str,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as uploader:
        uploader.add(make_upload_file(file_id=FILE_ID, tenant_id=TENANT_ID, name="guide.txt", created_by=ACTOR_ID))

    dispatched: list[object] = []
    monkeypatch.setattr(DocumentService, "dispatch_document_indexing", dispatched.append)

    with sqlite_session_factory() as caller:
        dataset = caller.get_one(Dataset, dataset_id)
        tenant = caller.get_one(Tenant, TENANT_ID)
        outer_transaction = caller.get_transaction()
        assert outer_transaction is not None
        tenant.name = "Caller staged tenant name"

        with pytest.raises(RuntimeError, match="response failed"):
            with DocumentService.save_document_with_dataset_id(
                dataset,
                _config(),
                _account(),
                session=caller,
            ):
                raise RuntimeError("response failed")

        assert caller.get_transaction() is outer_transaction
        assert caller.is_active
        assert tenant.name == "Caller staged tenant name"
        assert not inspect(tenant).expired
        assert caller.scalar(select(func.count()).select_from(Document).where(Document.dataset_id == dataset_id)) == 0
        assert caller.scalar(select(func.count()).select_from(DatasetProcessRule)) == 0
        assert caller.scalar(select(func.count()).select_from(DocumentIndexingOutbox)) == 0
        caller.commit()

    assert dispatched == []
    with sqlite_session_factory() as observer:
        assert observer.get_one(Tenant, TENANT_ID).name == "Caller staged tenant name"
        _assert_no_creation_writes(observer)
        assert observer.scalar(select(func.count()).select_from(DocumentIndexingOutbox)) == 0


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
def test_creation_lock_rejects_dataset_deleted_while_waiting(
    dataset_id: str,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def acquire_creation_lock(*_args: object, **_kwargs: object) -> Generator[None]:
        with sqlite_session_factory.begin() as writer:
            dataset = writer.get(Dataset, dataset_id)
            assert dataset is not None
            writer.delete(dataset)
        yield

    dispatched: list[object] = []
    monkeypatch.setattr("services.knowledge.dataset_service.redis_client.lock", acquire_creation_lock)
    monkeypatch.setattr(DocumentService, "dispatch_document_indexing", dispatched.append)

    with sqlite_session_factory() as caller:
        dataset = caller.get(Dataset, dataset_id)
        assert dataset is not None
        with pytest.raises(NotFound, match="Dataset not found"):
            with DocumentService.save_document_with_dataset_id(dataset, _config(), _account(), session=caller):
                pytest.fail("A deleted dataset must be rejected before writing documents")

    assert dispatched == []
    with sqlite_session_factory() as observer:
        assert observer.scalar(select(func.count()).select_from(Document)) == 0
        assert observer.scalar(select(func.count()).select_from(DatasetProcessRule)) == 0

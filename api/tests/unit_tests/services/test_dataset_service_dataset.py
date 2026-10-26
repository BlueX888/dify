"""SQLite-backed tests for dataset-level operations in :mod:`services.knowledge.dataset_service`.

Mapped objects in this module are real SQLAlchemy models. Provider runtimes,
RBAC clients, Celery tasks, and model-manager results remain mocked at their
external boundaries.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session, sessionmaker

import tasks.dataset_indexing_outbox_task as dataset_indexing_task_module
from core.errors.error import LLMBadRequestError, ProviderTokenNotInitError
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from graphon.model_runtime.entities.model_entities import ModelFeature, ModelType
from models import Account, Workflow
from models.dataset import (
    Dataset,
    DatasetCollectionBinding,
    DatasetPermission,
    DatasetPermissionEnum,
    ExternalKnowledgeApis,
    ExternalKnowledgeBindings,
    Pipeline,
)
from models.enums import DatasetRuntimeMode
from models.model import DatasetIndexingOutbox
from models.workflow import WorkflowType
from repositories.knowledge.dataset_indexing_outbox_repository import SQLAlchemyDatasetIndexingOutboxRepository
from repositories.knowledge.external_repository import SQLAlchemyExternalKnowledgeRepository
from services.enterprise import rbac_service
from services.entities.knowledge_entities.rag_pipeline_entities import (
    IconInfo,
    RagPipelineDatasetCreateEntity,
)
from services.errors.base import NoPermissionError
from services.errors.dataset import DatasetNameDuplicateError
from services.knowledge import dataset_service as dataset_service_module
from services.knowledge.dataset_service import (
    DatasetCollectionBindingService,
    DatasetPermissionService,
    DatasetService,
    ExternalKnowledgeTemplateLookup,
)
from services.knowledge.entities.datasets import DatasetIndexingRepresentation
from services.rag_pipeline.rag_pipeline_dsl_service import RagPipelineDslService
from tests.unit_tests.model_factories import make_account, make_dataset, make_tenant, make_workflow

from .dataset_service_test_helpers import (
    MagicMock,
    TenantAccountRole,
    _make_knowledge_configuration,
    _make_retrieval_model,
)


@dataclass(frozen=True, slots=True)
class _EmbeddingModelStub:
    provider: str
    model_name: str


class _ModelManagerStub:
    def get_model_instance(
        self,
        *,
        tenant_id: str,
        provider: str,
        model_type: object,
        model: str,
    ) -> _EmbeddingModelStub:
        del model_type
        assert tenant_id == "tenant-1"
        return _EmbeddingModelStub(provider=provider, model_name=model)


def _account(
    *,
    account_id: str = "user-1",
    tenant_id: str = "tenant-1",
    role: TenantAccountRole = TenantAccountRole.OWNER,
) -> Account:
    return make_account(
        account_id=account_id,
        name=f"User {account_id}",
        email=f"{account_id}@example.com",
        role=role,
        tenant=make_tenant(tenant_id=tenant_id, name=f"Tenant {tenant_id}"),
    )


def _dataset(
    *,
    dataset_id: str = "dataset-1",
    tenant_id: str = "tenant-1",
    name: str = "Dataset",
    maintainer: str = "user-1",
    permission: DatasetPermissionEnum = DatasetPermissionEnum.ALL_TEAM,
    provider: str = "vendor",
    indexing_technique: str = IndexTechniqueType.ECONOMY,
    chunk_structure: str | None = "text_model",
) -> Dataset:
    return make_dataset(
        dataset_id=dataset_id,
        tenant_id=tenant_id,
        name=name,
        description="",
        provider=provider,
        created_by=maintainer,
        maintainer=maintainer,
        permission=permission,
        indexing_technique=indexing_technique,
        chunk_structure=chunk_structure,
        embedding_model_provider="provider",
        embedding_model="embedding-model",
    )


def _configuration_representation(
    *,
    provider: str,
    model: str,
    binding_id: str,
) -> DatasetIndexingRepresentation:
    return DatasetIndexingRepresentation(
        indexing_technique="high_quality",
        index_struct=None,
        collection_binding_id=binding_id,
        embedding_model=model,
        embedding_model_provider=provider,
        keyword_number=None,
        is_multimodal=False,
        doc_form="text_model",
        summary_index_setting=None,
    )


def _seed_configuration_a(session: Session) -> Dataset:
    binding_a = DatasetCollectionBinding(
        provider_name="provider-a",
        model_name="model-a",
        type="dataset",
        collection_name="collection-a",
    )
    binding_a.id = "binding-a"
    binding_b = DatasetCollectionBinding(
        provider_name="provider-b",
        model_name="model-b",
        type="dataset",
        collection_name="collection-b",
    )
    binding_b.id = "binding-b"
    dataset = _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY)
    dataset.embedding_model_provider = "provider-a"
    dataset.embedding_model = "model-a"
    dataset.collection_binding_id = binding_a.id
    dataset.indexing_applied_representation = _configuration_representation(
        provider="provider-a",
        model="model-a",
        binding_id=binding_a.id,
    ).model_dump_json()
    session.add_all([binding_a, binding_b, dataset])
    session.commit()
    return dataset


def _commit_configuration_b(session_factory: sessionmaker[Session], *, account: Account) -> str:
    with session_factory() as writer:
        dataset = writer.get_one(Dataset, "dataset-1")
        (intent_id,) = DatasetService.update_rag_pipeline_dataset_settings(
            dataset,
            _make_knowledge_configuration(
                chunk_structure="text_model",
                indexing_technique=IndexTechniqueType.HIGH_QUALITY,
                embedding_model_provider="provider-b",
                embedding_model="model-b",
            ),
            has_published=True,
            session=writer,
            account=account,
        )
        writer.commit()
        return intent_id


def _assert_only_current_generation_claims(
    session_factory: sessionmaker[Session],
    *,
    stale_intent_id: str,
    current_intent_id: str,
) -> None:
    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=session_factory)
    assert repository.claim(stale_intent_id) is None
    current = repository.claim(current_intent_id)
    assert current is not None
    assert repository.release(current)


def _external_api(*, api_id: str = "api-1", tenant_id: str = "tenant-1") -> ExternalKnowledgeApis:
    api = ExternalKnowledgeApis(
        name="External API",
        description="",
        tenant_id=tenant_id,
        settings="{}",
        created_by="user-1",
        updated_by="user-1",
    )
    api.id = api_id
    return api


def _binding(
    *,
    binding_id: str = "binding-1",
    tenant_id: str = "tenant-1",
    dataset_id: str = "dataset-1",
    api_id: str = "api-1",
    knowledge_id: str = "knowledge-1",
) -> ExternalKnowledgeBindings:
    binding = ExternalKnowledgeBindings(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        external_knowledge_api_id=api_id,
        external_knowledge_id=knowledge_id,
        created_by="user-1",
    )
    binding.id = binding_id
    return binding


def _knowledge_index_graph(*, embedding_model: str, embedding_model_provider: str) -> str:
    return json.dumps(
        {
            "nodes": [
                {
                    "id": "knowledge-index-node",
                    "data": {
                        "type": "knowledge-index",
                        "embedding_model": embedding_model,
                        "embedding_model_provider": embedding_model_provider,
                    },
                }
            ],
            "edges": [],
        }
    )


def _persist_pipeline_dataset(session: Session) -> tuple[Dataset, Pipeline, Workflow, Workflow, str]:
    pipeline = Pipeline(
        tenant_id="tenant-1",
        name="Pipeline",
        description="",
        created_by="user-1",
    )
    pipeline.id = "pipeline-1"
    original_graph = _knowledge_index_graph(
        embedding_model="embedding-model",
        embedding_model_provider="provider",
    )
    published = make_workflow(
        workflow_id="published-workflow",
        tenant_id=pipeline.tenant_id,
        app_id=pipeline.id,
        workflow_type=WorkflowType.RAG_PIPELINE,
        version="published-v1",
        graph=original_graph,
        created_by="user-1",
    )
    draft = make_workflow(
        workflow_id="draft-workflow",
        tenant_id=pipeline.tenant_id,
        app_id=pipeline.id,
        workflow_type=WorkflowType.RAG_PIPELINE,
        graph=original_graph,
        created_by="user-1",
    )
    pipeline.workflow_id = published.id
    dataset = _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY)
    dataset.runtime_mode = DatasetRuntimeMode.RAG_PIPELINE
    dataset.pipeline_id = pipeline.id
    session.add_all([pipeline, dataset, published, draft])
    session.commit()
    return dataset, pipeline, published, draft, original_graph


@pytest.fixture
def external_templates(sqlite_session_factory: sessionmaker[Session]) -> ExternalKnowledgeTemplateLookup:
    return SQLAlchemyExternalKnowledgeRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def configuration_model_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    def manager_for_tenant(*, tenant_id: str) -> _ModelManagerStub:
        assert tenant_id == "tenant-1"
        return _ModelManagerStub()

    monkeypatch.setattr(
        dataset_service_module.ModelManager,
        "for_tenant",
        staticmethod(manager_for_tenant),
    )
    monkeypatch.setattr(
        DatasetService,
        "check_is_multimodal_model",
        staticmethod(lambda _tenant_id, _provider, _model: False),
    )


class TestDatasetServiceValidation:
    @pytest.mark.parametrize(
        ("dataset_doc_form", "incoming_doc_form"),
        [(None, "text_model"), ("text_model", "text_model")],
    )
    def test_check_doc_form_allows_matching_or_missing_dataset_doc_form(
        self,
        sqlite_session: Session,
        dataset_doc_form: str | None,
        incoming_doc_form: str,
    ) -> None:
        dataset = _dataset(chunk_structure=dataset_doc_form)
        sqlite_session.add(dataset)
        sqlite_session.commit()

        DatasetService.check_doc_form(dataset, incoming_doc_form, session=sqlite_session)

    def test_check_doc_form_rejects_mismatched_doc_form(self, sqlite_session: Session) -> None:
        dataset = _dataset(chunk_structure="qa_model")
        sqlite_session.add(dataset)
        sqlite_session.commit()

        with pytest.raises(ValueError, match="doc_form is different"):
            DatasetService.check_doc_form(dataset, "text_model", session=sqlite_session)

    @pytest.mark.parametrize("operator_check", [False, True])
    def test_dataset_permission_checks_ignore_foreign_tenant_binding(
        self, sqlite_session: Session, operator_check: bool
    ) -> None:
        dataset = _dataset(
            dataset_id="dataset-1",
            tenant_id="tenant-1",
            permission=DatasetPermissionEnum.PARTIAL_TEAM,
            maintainer="owner-1",
        )
        user = _account(
            account_id="user-1",
            tenant_id="tenant-1",
            role=TenantAccountRole.NORMAL,
        )
        sqlite_session.add_all(
            [
                dataset,
                DatasetPermission(dataset_id=dataset.id, account_id=user.id, tenant_id="tenant-2"),
            ]
        )
        sqlite_session.commit()

        if operator_check:
            with pytest.raises(NoPermissionError):
                DatasetService.check_dataset_operator_permission(user, dataset, session=sqlite_session)
        else:
            with pytest.raises(NoPermissionError):
                DatasetService.check_dataset_permission(dataset, user, sqlite_session)

    def test_check_dataset_model_setting_skips_non_high_quality_datasets(self) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.ECONOMY)

        with patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls:
            DatasetService.check_dataset_model_setting(dataset)

        model_manager_cls.assert_not_called()

    def test_check_dataset_model_setting_validates_high_quality_embedding(self) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY)

        with patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls:
            DatasetService.check_dataset_model_setting(dataset)

        model_manager_cls.for_tenant.return_value.get_model_instance.assert_called_once_with(
            tenant_id=dataset.tenant_id,
            provider=dataset.embedding_model_provider,
            model_type=ModelType.TEXT_EMBEDDING,
            model=dataset.embedding_model,
        )

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (LLMBadRequestError(), "No Embedding Model available"),
            (ProviderTokenNotInitError("token missing"), "token missing"),
        ],
    )
    def test_check_dataset_model_setting_wraps_provider_errors(self, error: Exception, message: str) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY)

        with patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = error
            with pytest.raises(ValueError, match=message):
                DatasetService.check_dataset_model_setting(dataset)

    @pytest.mark.parametrize(("features", "expected"), [([ModelFeature.VISION], True), ([], False)])
    def test_check_is_multimodal_model_reads_runtime_schema(self, features: list[ModelFeature], expected: bool) -> None:
        model_type_instance = MagicMock()
        model_type_instance.get_model_schema.return_value = SimpleNamespace(features=features)
        model_instance = SimpleNamespace(
            model_type_instance=model_type_instance,
            model_name="embedding-model",
            credentials={"api_key": "secret"},
        )

        with patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = model_instance
            result = DatasetService.check_is_multimodal_model("tenant-1", "provider", "embedding-model")

        assert result is expected

    def test_check_is_multimodal_model_rejects_missing_schema(self) -> None:
        model_type_instance = MagicMock()
        model_type_instance.get_model_schema.return_value = None
        model_instance = SimpleNamespace(
            model_type_instance=model_type_instance,
            model_name="embedding-model",
            credentials={},
        )

        with patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = model_instance
            with pytest.raises(ValueError, match="Model schema not found"):
                DatasetService.check_is_multimodal_model("tenant-1", "provider", "embedding-model")

    @pytest.mark.parametrize(
        ("method", "error", "message"),
        [
            (
                DatasetService.check_embedding_model_setting,
                ProviderTokenNotInitError("provider setup"),
                "provider setup",
            ),
            (DatasetService.check_reranking_model_setting, LLMBadRequestError(), "No Rerank Model available"),
        ],
    )
    def test_direct_model_setting_checks_wrap_runtime_errors(
        self, method: Callable[[str, str, str], None], error: Exception, message: str
    ) -> None:
        with patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = error
            with pytest.raises(ValueError, match=message):
                method("tenant-1", "provider", "model")


class TestDatasetServiceRetrieval:
    def test_get_dataset_for_tenant_rejects_cross_tenant_row(self, sqlite_session: Session) -> None:
        owned = _dataset()
        foreign = _dataset(dataset_id="dataset-2", tenant_id="tenant-2", name="Foreign")
        sqlite_session.add_all([owned, foreign])
        sqlite_session.commit()

        assert DatasetService.get_dataset_for_tenant(owned.id, "tenant-1", session=sqlite_session) is owned
        assert DatasetService.get_dataset_for_tenant(foreign.id, "tenant-1", session=sqlite_session) is None

    def test_get_datasets_applies_rbac_resource_scope_and_maintainer_override(
        self, application_tags, config_overrides: Callable[..., None], sqlite_session: Session
    ) -> None:
        user = _account(role=TenantAccountRole.NORMAL)
        accessible = _dataset(dataset_id="accessible", name="Accessible", maintainer="other")
        owned = _dataset(dataset_id="owned", name="Owned", maintainer=user.id)
        hidden = _dataset(dataset_id="hidden", name="Hidden", maintainer="other")
        foreign = _dataset(dataset_id="foreign", tenant_id="tenant-2", name="Foreign")
        sqlite_session.add_all([accessible, owned, hidden, foreign])
        sqlite_session.commit()

        config_overrides(RBAC_ENABLED=True)

        with (
            patch(
                "services.knowledge.dataset_service.enterprise_rbac_service.RBACService.MyPermissions.get",
                return_value=SimpleNamespace(workspace=SimpleNamespace(permission_keys=[])),
            ),
        ):
            datasets, total = DatasetService.get_datasets(
                page=1,
                per_page=20,
                session=sqlite_session,
                tenant_id="tenant-1",
                user=user,
                accessible_dataset_ids=[accessible.id],
                include_own_datasets=True,
                tags=application_tags,
            )

        assert total == 2
        assert {dataset.id for dataset in datasets} == {accessible.id, owned.id}

    def test_get_datasets_without_user_keeps_only_team_visible_rows(
        self, application_tags, config_overrides: Callable[..., None], sqlite_session: Session
    ) -> None:
        shared = _dataset(dataset_id="shared", name="Shared", permission=DatasetPermissionEnum.ALL_TEAM)
        private = _dataset(dataset_id="private", name="Private", permission=DatasetPermissionEnum.ONLY_ME)
        sqlite_session.add_all([shared, private])
        sqlite_session.commit()

        config_overrides(RBAC_ENABLED=False)
        datasets, total = DatasetService.get_datasets(
            page=1, per_page=20, session=sqlite_session, tenant_id="tenant-1", tags=application_tags
        )

        assert total == 1
        assert [dataset.id for dataset in datasets] == [shared.id]

    def test_get_datasets_by_ids_intersects_requested_and_accessible_ids(
        self, config_overrides: Callable[..., None], sqlite_session: Session
    ) -> None:
        user = _account(role=TenantAccountRole.NORMAL)
        accessible = _dataset(dataset_id="accessible", name="Accessible", maintainer="other")
        owned = _dataset(dataset_id="owned", name="Owned", maintainer=user.id)
        hidden = _dataset(dataset_id="hidden", name="Hidden", maintainer="other")
        sqlite_session.add_all([accessible, owned, hidden])
        sqlite_session.commit()

        config_overrides(RBAC_ENABLED=True)
        datasets, total = DatasetService.get_datasets_by_ids(
            [accessible.id, owned.id, hidden.id],
            "tenant-1",
            user=user,
            accessible_dataset_ids=[accessible.id, "not-requested"],
            include_own_datasets=True,
            session=sqlite_session,
        )

        assert total == 2
        assert {dataset.id for dataset in datasets} == {accessible.id, owned.id}

    def test_get_datasets_rbac_without_user_returns_no_rows(
        self, application_tags, config_overrides: Callable[..., None], sqlite_session: Session
    ) -> None:
        sqlite_session.add(_dataset())
        sqlite_session.commit()

        config_overrides(RBAC_ENABLED=True)
        datasets, total = DatasetService.get_datasets(
            page=1, per_page=20, session=sqlite_session, tenant_id="tenant-1", tags=application_tags
        )

        assert datasets == []
        assert total == 0

    def test_get_datasets_rbac_include_all_requires_workspace_permission(
        self, application_tags, config_overrides: Callable[..., None], sqlite_session: Session
    ) -> None:
        user = _account(role=TenantAccountRole.NORMAL)
        sqlite_session.add_all(
            [
                _dataset(dataset_id="one", name="One", maintainer="other"),
                _dataset(dataset_id="two", name="Two", maintainer="other"),
            ]
        )
        sqlite_session.commit()

        config_overrides(RBAC_ENABLED=True)
        with (
            patch(
                "services.knowledge.dataset_service.enterprise_rbac_service.RBACService.MyPermissions.get",
                return_value=SimpleNamespace(
                    workspace=SimpleNamespace(permission_keys=["dataset.create_and_management"])
                ),
            ),
        ):
            datasets, total = DatasetService.get_datasets(
                page=1,
                per_page=20,
                session=sqlite_session,
                tenant_id="tenant-1",
                user=user,
                include_all=True,
                tags=application_tags,
            )

        assert total == 2
        assert {dataset.id for dataset in datasets} == {"one", "two"}


class TestDatasetServiceCreationAndUpdate:
    def test_external_template_lookup_reads_pending_row_in_current_transaction(
        self, sqlite_session: Session, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        template = _external_api()
        sqlite_session.add(template)
        transaction_events: list[str] = []
        event.listen(sqlite_session, "after_commit", lambda _session: transaction_events.append("commit"))
        event.listen(sqlite_session, "after_rollback", lambda _session: transaction_events.append("rollback"))

        external_templates.require_template(template.tenant_id, template.id, session=sqlite_session)

        assert sqlite_session.get(ExternalKnowledgeApis, template.id) is template
        assert sqlite_session.in_transaction()
        assert transaction_events == []

    def test_create_empty_dataset_rejects_duplicate_name(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        sqlite_session.add(_dataset(name="Existing"))
        sqlite_session.commit()

        with pytest.raises(DatasetNameDuplicateError, match="already exists"):
            DatasetService.create_empty_dataset(
                "tenant-1",
                "Existing",
                "",
                IndexTechniqueType.ECONOMY,
                _account(),
                session=sqlite_session,
                external_templates=external_templates,
            )

    def test_create_empty_dataset_persists_default_embedding_and_creator_scope(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        *,
        external_templates: ExternalKnowledgeTemplateLookup,
    ) -> None:
        embedding_model = SimpleNamespace(provider="provider", model_name="default-embedding")

        with (
            patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls,
            patch(
                "services.knowledge.dataset_service.enterprise_rbac_service.try_sync_creator_access_policy_member_bindings"
            ),
        ):
            model_manager_cls.for_tenant.return_value.get_default_model_instance.return_value = embedding_model
            dataset = DatasetService.create_empty_dataset(
                "tenant-1",
                "Created",
                "Description",
                IndexTechniqueType.HIGH_QUALITY,
                _account(),
                session=sqlite_session,
                external_templates=external_templates,
            )

        with sqlite_session_factory() as observer:
            persisted = observer.get(Dataset, dataset.id)
            assert persisted is not None
            assert persisted.tenant_id == "tenant-1"
            assert persisted.maintainer == "user-1"
            assert persisted.embedding_model == "default-embedding"

    def test_create_empty_external_dataset_persists_tenant_owned_binding(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        sqlite_session.add_all([_external_api(), _external_api(api_id="api-foreign", tenant_id="tenant-2")])
        sqlite_session.commit()

        with patch(
            "services.knowledge.dataset_service.enterprise_rbac_service.try_sync_creator_access_policy_member_bindings"
        ):
            dataset = DatasetService.create_empty_dataset(
                "tenant-1",
                "External",
                "",
                IndexTechniqueType.ECONOMY,
                _account(),
                provider="external",
                external_knowledge_api_id="api-1",
                external_knowledge_id="knowledge-1",
                retrieval_model=_make_retrieval_model(),
                session=sqlite_session,
                external_templates=external_templates,
            )

        binding = sqlite_session.scalar(
            select(ExternalKnowledgeBindings).where(ExternalKnowledgeBindings.dataset_id == dataset.id)
        )
        assert binding is not None
        assert binding.tenant_id == dataset.tenant_id
        assert binding.external_knowledge_api_id == "api-1"

    def test_create_empty_external_dataset_rejects_foreign_api(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        sqlite_session.add(_external_api(api_id="api-foreign", tenant_id="tenant-2"))
        sqlite_session.commit()

        with (
            pytest.raises(ValueError, match="api template not found"),
            patch(
                "services.knowledge.dataset_service.enterprise_rbac_service.try_sync_creator_access_policy_member_bindings"
            ),
        ):
            DatasetService.create_empty_dataset(
                "tenant-1",
                "External",
                "",
                IndexTechniqueType.ECONOMY,
                _account(),
                provider="external",
                external_knowledge_api_id="api-foreign",
                external_knowledge_id="knowledge-1",
                session=sqlite_session,
                external_templates=external_templates,
            )

    def test_create_empty_rag_pipeline_dataset_generates_incremental_name(self, sqlite_session: Session) -> None:
        sqlite_session.add(_dataset(name="Untitled 1"))
        sqlite_session.commit()
        entity = RagPipelineDatasetCreateEntity(
            name="",
            description="Pipeline dataset",
            icon_info=IconInfo(icon_type="emoji", icon="📚", icon_background="#FFFFFF", icon_url=None),
            permission=DatasetPermissionEnum.ALL_TEAM,
        )

        with patch("services.knowledge.dataset_service.current_user", _account()):
            dataset = DatasetService.create_empty_rag_pipeline_dataset("tenant-1", entity, sqlite_session)

        assert dataset.name == "Untitled 2"
        pipeline = sqlite_session.get(Pipeline, dataset.pipeline_id)
        assert pipeline is not None
        assert pipeline.tenant_id == dataset.tenant_id

    def test_create_empty_rag_pipeline_dataset_rejects_duplicate_name(self, sqlite_session: Session) -> None:
        sqlite_session.add(_dataset(name="Existing"))
        sqlite_session.commit()
        entity = RagPipelineDatasetCreateEntity(
            name="Existing",
            description="",
            icon_info=IconInfo(icon_type="emoji", icon="📚", icon_background="#FFFFFF", icon_url=None),
            permission=DatasetPermissionEnum.ALL_TEAM,
        )

        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            pytest.raises(DatasetNameDuplicateError, match="already exists"),
        ):
            DatasetService.create_empty_rag_pipeline_dataset("tenant-1", entity, sqlite_session)

    def test_create_empty_rag_pipeline_dataset_requires_authenticated_account(self, sqlite_session: Session) -> None:
        account = _account()
        account.id = ""
        entity = RagPipelineDatasetCreateEntity(
            name="Dataset",
            description="",
            icon_info=IconInfo(icon_type="emoji", icon="📚", icon_background="#FFFFFF", icon_url=None),
            permission=DatasetPermissionEnum.ALL_TEAM,
        )

        with (
            patch("services.knowledge.dataset_service.current_user", account),
            pytest.raises(ValueError, match="Current user or current user id not found"),
        ):
            DatasetService.create_empty_rag_pipeline_dataset("tenant-1", entity, sqlite_session)

    def test_update_dataset_rejects_missing_and_duplicate_rows(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        current = _dataset(name="Current")
        duplicate = _dataset(dataset_id="dataset-2", name="Duplicate")
        sqlite_session.add_all([current, duplicate])
        sqlite_session.commit()

        with pytest.raises(ValueError, match="Dataset not found"):
            with DatasetService.update_dataset(
                "missing", {}, _account(), session=sqlite_session, external_templates=external_templates
            ):
                pytest.fail("Missing dataset must be rejected")
        with pytest.raises(ValueError, match="Dataset name already exists"):
            with DatasetService.update_dataset(
                current.id,
                {"name": duplicate.name},
                _account(),
                session=sqlite_session,
                external_templates=external_templates,
            ):
                pytest.fail("Duplicate dataset name must be rejected")
        with pytest.raises(NoPermissionError):
            with DatasetService.update_dataset(
                current.id,
                {},
                _account(tenant_id="tenant-2"),
                session=sqlite_session,
                external_templates=external_templates,
            ):
                pytest.fail("Foreign dataset must be rejected without refreshing its configuration")

    @pytest.mark.parametrize(
        ("provider", "helper_name"),
        [("external", "_update_external_dataset"), ("vendor", "_update_internal_dataset")],
    )
    def test_update_dataset_routes_by_provider(
        self,
        sqlite_session: Session,
        provider: str,
        helper_name: str,
        *,
        external_templates: ExternalKnowledgeTemplateLookup,
    ) -> None:
        dataset = _dataset(provider=provider)
        sqlite_session.add(dataset)
        sqlite_session.commit()

        with (
            patch.object(DatasetService, "check_dataset_permission"),
            patch.object(
                DatasetService, helper_name, return_value=dataset if provider == "external" else (dataset, None)
            ) as update_helper,
        ):
            with DatasetService.update_dataset(
                dataset.id,
                {"name": dataset.name},
                _account(),
                session=sqlite_session,
                external_templates=external_templates,
            ) as result:
                assert result is dataset
        update_helper.assert_called_once()

    def test_pipeline_dataset_update_rolls_back_workflow_changes_when_response_building_fails(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        *,
        external_templates: ExternalKnowledgeTemplateLookup,
    ) -> None:
        dataset, pipeline, published, draft, original_graph = _persist_pipeline_dataset(sqlite_session)

        with (
            patch.object(DatasetService, "_handle_indexing_technique_change", return_value="update"),
            patch("services.knowledge.dataset_service.enqueue_dataset_indexing_intent") as enqueue_intent,
            pytest.raises(RuntimeError, match="response materialization failed"),
        ):
            with DatasetService.update_dataset(
                dataset.id,
                {
                    "name": "Updated Dataset",
                    "indexing_technique": IndexTechniqueType.HIGH_QUALITY,
                    "embedding_model": "new-embedding-model",
                    "embedding_model_provider": "new-provider",
                },
                _account(),
                session=sqlite_session,
                external_templates=external_templates,
            ):
                raise RuntimeError("response materialization failed")

        enqueue_intent.assert_not_called()
        with sqlite_session_factory() as observer:
            persisted_dataset = observer.get(Dataset, dataset.id)
            assert persisted_dataset is not None
            assert persisted_dataset.name == "Dataset"
            assert persisted_dataset.embedding_model == "embedding-model"
            assert persisted_dataset.embedding_model_provider == "provider"
            assert observer.scalar(select(DatasetIndexingOutbox.id).limit(1)) is None
            workflows = observer.scalars(select(Workflow).where(Workflow.app_id == pipeline.id)).all()
            assert {workflow.id for workflow in workflows} == {published.id, draft.id}
            assert all(workflow.graph == original_graph for workflow in workflows)

    def test_pipeline_dataset_update_dispatches_after_dataset_and_workflows_commit(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        *,
        external_templates: ExternalKnowledgeTemplateLookup,
    ) -> None:
        dataset, pipeline, published, draft, original_graph = _persist_pipeline_dataset(sqlite_session)
        observed_committed_state: list[str] = []

        def observe_committed_state(intent_id: str) -> None:
            with sqlite_session_factory() as observer:
                intent = observer.get(DatasetIndexingOutbox, intent_id)
                assert intent is not None
                persisted_dataset = observer.get(Dataset, dataset.id)
                assert persisted_dataset is not None
                assert persisted_dataset.name == "Updated Dataset"
                assert persisted_dataset.embedding_model == "new-embedding-model"
                assert persisted_dataset.embedding_model_provider == "new-provider"

                workflows = observer.scalars(select(Workflow).where(Workflow.app_id == pipeline.id)).all()
                assert len(workflows) == 3
                persisted_published = next(workflow for workflow in workflows if workflow.id == published.id)
                persisted_draft = next(workflow for workflow in workflows if workflow.id == draft.id)
                new_versions = [workflow for workflow in workflows if workflow.id not in {published.id, draft.id}]
                assert persisted_published.graph == original_graph
                assert len(new_versions) == 1
                for workflow in [persisted_draft, new_versions[0]]:
                    node_data = json.loads(workflow.graph)["nodes"][0]["data"]
                    assert node_data["embedding_model"] == "new-embedding-model"
                    assert node_data["embedding_model_provider"] == "new-provider"
                assert persisted_dataset.indexing_request_id == intent.generation_id
                observed_committed_state.append(intent.operation)

        with (
            patch.object(DatasetService, "_handle_indexing_technique_change", return_value="update"),
            patch(
                "services.knowledge.dataset_service.enqueue_dataset_indexing_intent",
                side_effect=observe_committed_state,
            ) as enqueue_intent,
        ):
            with DatasetService.update_dataset(
                dataset.id,
                {
                    "name": "Updated Dataset",
                    "indexing_technique": IndexTechniqueType.HIGH_QUALITY,
                    "embedding_model": "new-embedding-model",
                    "embedding_model_provider": "new-provider",
                },
                _account(),
                session=sqlite_session,
                external_templates=external_templates,
            ):
                pass

        assert observed_committed_state == ["rebuild"]
        assert enqueue_intent.call_count == 1

    def test_update_external_dataset_metadata_only_preserves_binding_without_template_validation(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        dataset = _dataset(provider="external")
        binding = _binding()
        sqlite_session.add_all([dataset, binding])
        sqlite_session.commit()

        with DatasetService.update_dataset(
            dataset.id,
            {"name": "Renamed", "description": "Changed"},
            _account(),
            session=sqlite_session,
            external_templates=external_templates,
        ) as updated:
            assert updated is dataset
        assert dataset.name == "Renamed"
        assert dataset.description == "Changed"
        assert binding.external_knowledge_id == "knowledge-1"
        assert binding.external_knowledge_api_id == "api-1"

    @pytest.mark.parametrize(
        ("payload", "expected_knowledge_id", "expected_api_id"),
        [
            ({"external_knowledge_id": "knowledge-2"}, "knowledge-2", "api-1"),
            ({"external_knowledge_api_id": "api-2"}, "knowledge-1", "api-2"),
        ],
    )
    def test_update_external_dataset_merges_single_binding_field_with_existing_binding(
        self,
        sqlite_session: Session,
        payload: dict[str, str],
        expected_knowledge_id: str,
        expected_api_id: str,
        *,
        external_templates: ExternalKnowledgeTemplateLookup,
    ) -> None:
        dataset = _dataset(provider="external")
        binding = _binding()
        sqlite_session.add_all([dataset, binding, _external_api(), _external_api(api_id="api-2")])
        sqlite_session.commit()

        with DatasetService.update_dataset(
            dataset.id,
            payload,
            _account(),
            session=sqlite_session,
            external_templates=external_templates,
        ):
            pass

        assert binding.external_knowledge_id == expected_knowledge_id
        assert binding.external_knowledge_api_id == expected_api_id

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ({"external_knowledge_id": None}, "External knowledge id is required"),
            ({"external_knowledge_id": ""}, "External knowledge id is required"),
            ({"external_knowledge_api_id": None}, "External knowledge api id is required"),
            ({"external_knowledge_api_id": ""}, "External knowledge api id is required"),
        ],
    )
    def test_update_external_dataset_rejects_explicit_binding_clear(
        self,
        sqlite_session: Session,
        payload: dict[str, str | None],
        message: str,
        *,
        external_templates: ExternalKnowledgeTemplateLookup,
    ) -> None:
        dataset = _dataset(provider="external")
        binding = _binding()
        sqlite_session.add_all([dataset, binding, _external_api()])
        sqlite_session.commit()

        with pytest.raises(ValueError, match=message):
            with DatasetService.update_dataset(
                dataset.id,
                payload,
                _account(),
                session=sqlite_session,
                external_templates=external_templates,
            ):
                pytest.fail("Cleared external binding must be rejected")

        assert binding.external_knowledge_id == "knowledge-1"
        assert binding.external_knowledge_api_id == "api-1"

    def test_update_external_dataset_unchanged_binding_does_not_revalidate_template(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        dataset = _dataset(provider="external")
        binding = _binding()
        sqlite_session.add_all([dataset, binding])
        sqlite_session.commit()

        with DatasetService.update_dataset(
            dataset.id,
            {
                "external_knowledge_id": binding.external_knowledge_id,
                "external_knowledge_api_id": binding.external_knowledge_api_id,
            },
            _account(),
            session=sqlite_session,
            external_templates=external_templates,
        ):
            pass

        assert binding.external_knowledge_id == "knowledge-1"
        assert binding.external_knowledge_api_id == "api-1"

    def test_update_external_dataset_flushes_dataset_and_binding_without_committing(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        dataset = _dataset(provider="external")
        api = _external_api()
        binding = _binding()
        sqlite_session.add_all([dataset, api, binding])
        sqlite_session.commit()
        transaction_events: list[str] = []
        event.listen(sqlite_session, "after_commit", lambda _session: transaction_events.append("commit"))

        updated = DatasetService._update_external_dataset(
            dataset,
            {
                "name": "Updated",
                "description": "Changed",
                "external_knowledge_id": "knowledge-2",
                "external_knowledge_api_id": api.id,
            },
            _account(),
            sqlite_session,
            external_templates=external_templates,
        )

        assert updated is dataset
        assert dataset.name == "Updated"
        assert binding.external_knowledge_id == "knowledge-2"
        assert transaction_events == []

    def test_update_external_dataset_rejects_cross_tenant_api(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        dataset = _dataset(provider="external")
        foreign_api = _external_api(api_id="api-foreign", tenant_id="tenant-2")
        sqlite_session.add_all([dataset, foreign_api, _binding()])
        sqlite_session.commit()

        with pytest.raises(ValueError, match="api template not found"):
            DatasetService._update_external_dataset(
                dataset,
                {
                    "external_knowledge_id": "knowledge-2",
                    "external_knowledge_api_id": foreign_api.id,
                },
                _account(),
                sqlite_session,
                external_templates=external_templates,
            )

    def test_update_external_knowledge_binding_rejects_missing_row(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        with pytest.raises(ValueError, match="binding not found"):
            DatasetService._patch_external_knowledge_binding(
                _dataset(dataset_id="missing-dataset", provider="external"),
                {"external_knowledge_id": "knowledge-2", "external_knowledge_api_id": "api-2"},
                session=sqlite_session,
                external_templates=external_templates,
            )

    def test_update_external_knowledge_binding_rejects_cross_tenant_binding(
        self, sqlite_session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ) -> None:
        dataset = _dataset(provider="external")
        sqlite_session.add_all([dataset, _binding(tenant_id="tenant-2"), _external_api()])
        sqlite_session.commit()

        with pytest.raises(ValueError, match="binding not found"):
            DatasetService._patch_external_knowledge_binding(
                dataset,
                {"external_knowledge_id": "knowledge-2"},
                session=sqlite_session,
                external_templates=external_templates,
            )

    def test_update_internal_dataset_executes_real_update_without_committing(self, sqlite_session: Session) -> None:
        dataset = _dataset(name="Before")
        sqlite_session.add(dataset)
        sqlite_session.commit()
        transaction_events: list[str] = []
        event.listen(sqlite_session, "after_commit", lambda _session: transaction_events.append("commit"))

        with (
            patch.object(DatasetService, "_handle_indexing_technique_change", return_value="update"),
            patch.object(DatasetService, "_update_pipeline_knowledge_base_node_data"),
            patch("services.knowledge.dataset_service.enqueue_dataset_indexing_intent") as enqueue_intent,
        ):
            updated, indexing_job = DatasetService._update_internal_dataset(
                dataset,
                {"name": "After", "description": "Changed"},
                _account(),
                sqlite_session,
            )

        assert updated.name == "After"
        assert updated.description == "Changed"
        assert transaction_events == []
        enqueue_intent.assert_not_called()
        assert indexing_job is not None
        assert indexing_job.dataset.dataset_id == dataset.id
        assert indexing_job.source.indexing_technique == indexing_job.target.indexing_technique

    def test_update_pipeline_node_data_returns_for_non_pipeline_or_missing_pipeline(
        self, sqlite_session: Session
    ) -> None:
        ordinary = _dataset()
        missing_pipeline = _dataset(dataset_id="pipeline-dataset", name="Pipeline Dataset")
        missing_pipeline.runtime_mode = "rag_pipeline"
        missing_pipeline.pipeline_id = "missing-pipeline"
        sqlite_session.add_all([ordinary, missing_pipeline])
        sqlite_session.commit()

        DatasetService._update_pipeline_knowledge_base_node_data(ordinary, "user-1", sqlite_session)
        DatasetService._update_pipeline_knowledge_base_node_data(missing_pipeline, "user-1", sqlite_session)

    def test_update_pipeline_node_data_leaves_rollback_to_its_caller(self, sqlite_session: Session) -> None:
        pipeline = Pipeline(tenant_id="tenant-1", name="Pipeline", description="", created_by="user-1")
        pipeline.id = "pipeline-1"
        dataset = _dataset()
        dataset.runtime_mode = "rag_pipeline"
        dataset.pipeline_id = pipeline.id
        sqlite_session.add_all([pipeline, dataset])
        sqlite_session.commit()
        sqlite_session.begin()
        transaction_events: list[str] = []
        event.listen(
            sqlite_session,
            "after_soft_rollback",
            lambda _session, _previous_transaction: transaction_events.append("rollback"),
        )

        with patch("services.knowledge.dataset_service.RagPipelineService") as service_cls:
            service_cls.return_value.get_published_workflow.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError, match="boom"):
                DatasetService._update_pipeline_knowledge_base_node_data(dataset, "user-1", sqlite_session)

        assert transaction_events == []
        assert sqlite_session.in_transaction()
        sqlite_session.rollback()
        assert transaction_events == ["rollback"]


class TestDatasetServiceEmbeddingSettings:
    def test_handle_indexing_technique_change_returns_none_without_requested_change(
        self, unbound_session: Session
    ) -> None:
        assert DatasetService._handle_indexing_technique_change(_dataset(), {}, {}, unbound_session) is None

    def test_handle_indexing_technique_change_switches_to_economy(self, unbound_session: Session) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY)
        filtered_data: dict[str, object] = {}

        result = DatasetService._handle_indexing_technique_change(
            dataset,
            {"indexing_technique": IndexTechniqueType.ECONOMY},
            filtered_data,
            unbound_session,
        )

        assert result == "remove"
        assert filtered_data == {
            "embedding_model": None,
            "embedding_model_provider": None,
            "collection_binding_id": None,
        }

    def test_handle_indexing_technique_change_delegates_high_quality_configuration(
        self, unbound_session: Session
    ) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.ECONOMY)
        filtered_data: dict[str, object] = {}

        with patch.object(DatasetService, "_configure_embedding_model_for_high_quality") as configure:
            result = DatasetService._handle_indexing_technique_change(
                dataset,
                {"indexing_technique": IndexTechniqueType.HIGH_QUALITY},
                filtered_data,
                unbound_session,
            )

        assert result == "add"
        configure.assert_called_once()

    def test_handle_unchanged_indexing_preserves_existing_model(self, unbound_session: Session) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY)

        with patch.object(DatasetService, "_preserve_existing_embedding_settings") as preserve:
            result = DatasetService._handle_embedding_model_update_when_technique_unchanged(
                dataset,
                {},
                {},
                unbound_session,
            )

        assert result is None
        preserve.assert_called_once()

    def test_update_embedding_model_settings_delegates_changed_model(self, unbound_session: Session) -> None:
        dataset = _dataset()

        with patch.object(DatasetService, "_apply_new_embedding_settings") as apply_settings:
            result = DatasetService._update_embedding_model_settings(
                dataset,
                {"embedding_model_provider": "provider-2", "embedding_model": "model-2"},
                {},
                unbound_session,
            )

        assert result == "update"
        apply_settings.assert_called_once()

    def test_configure_high_quality_wraps_provider_error(self, unbound_session: Session) -> None:
        account = _account()
        with (
            patch("services.knowledge.dataset_service.current_user", account),
            patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls,
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = LLMBadRequestError()
            with pytest.raises(ValueError, match="No Embedding Model available"):
                DatasetService._configure_embedding_model_for_high_quality(
                    {"embedding_model_provider": "provider", "embedding_model": "model"},
                    {},
                    unbound_session,
                )

    def test_preserve_existing_embedding_settings(self) -> None:
        dataset = _dataset()
        filtered_data: dict[str, object] = {}

        DatasetService._preserve_existing_embedding_settings(dataset, filtered_data)

        assert filtered_data["embedding_model_provider"] == dataset.embedding_model_provider
        assert filtered_data["embedding_model"] == dataset.embedding_model

    def test_update_embedding_model_settings_returns_none_when_unchanged(self, unbound_session: Session) -> None:
        dataset = _dataset()

        result = DatasetService._update_embedding_model_settings(
            dataset,
            {
                "embedding_model_provider": dataset.embedding_model_provider,
                "embedding_model": dataset.embedding_model,
            },
            {},
            unbound_session,
        )

        assert result is None

    def test_apply_new_embedding_settings_uses_real_collection_binding(self, unbound_session: Session) -> None:
        dataset = _dataset()
        filtered_data: dict[str, object] = {}
        account = _account()
        embedding_model = _EmbeddingModelStub(provider="new-provider", model_name="new-model")
        collection_binding = DatasetCollectionBinding(
            provider_name="new-provider",
            model_name="new-model",
            type="dataset",
            collection_name="collection",
        )
        collection_binding.id = "collection-binding-1"

        with (
            patch("services.knowledge.dataset_service.current_user", account),
            patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls,
            patch.object(
                DatasetCollectionBindingService,
                "get_dataset_collection_binding",
                return_value=collection_binding,
            ),
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = embedding_model
            DatasetService._apply_new_embedding_settings(
                dataset,
                {"embedding_model_provider": "new-provider", "embedding_model": "new-model"},
                filtered_data,
                unbound_session,
            )

        assert filtered_data["embedding_model_provider"] == "new-provider"
        assert filtered_data["embedding_model"] == "new-model"
        assert filtered_data["collection_binding_id"] == collection_binding.id

    def test_apply_new_embedding_settings_preserves_existing_values_when_token_missing(
        self, unbound_session: Session
    ) -> None:
        dataset = _dataset()
        filtered_data: dict[str, object] = {}

        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls,
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = ProviderTokenNotInitError(
                "missing"
            )
            DatasetService._apply_new_embedding_settings(
                dataset,
                {"embedding_model_provider": "provider-2", "embedding_model": "model-2"},
                filtered_data,
                unbound_session,
            )

        assert filtered_data["embedding_model_provider"] == dataset.embedding_model_provider
        assert filtered_data["embedding_model"] == dataset.embedding_model

    @pytest.mark.parametrize(
        ("summary_setting", "expected"),
        [
            (None, False),
            ({"enable": False}, False),
            ({"enable": True, "model_name": "old-model", "model_provider_name": "provider"}, False),
            ({"enable": True, "model_name": "new-model", "model_provider_name": "provider"}, True),
        ],
    )
    def test_check_summary_index_setting_model_changed(
        self, summary_setting: dict[str, object] | None, expected: bool
    ) -> None:
        dataset = _dataset()
        dataset.summary_index_setting = {
            "enable": True,
            "model_name": "old-model",
            "model_provider_name": "provider",
        }

        data = {} if summary_setting is None else {"summary_index_setting": summary_setting}
        assert DatasetService._check_summary_index_setting_model_changed(dataset, data) is expected


class TestDatasetServiceRagPipelineSettings:
    def test_requires_current_tenant(self, unbound_session: Session) -> None:
        account = _account()
        account._current_tenant = None

        with (
            patch("services.knowledge.dataset_service.current_user", account),
            pytest.raises(ValueError, match="Current user or current tenant not found"),
        ):
            DatasetService.update_rag_pipeline_dataset_settings(
                _dataset(),
                _make_knowledge_configuration(),
                session=unbound_session,
            )

    def test_unpublished_high_quality_settings_use_real_merged_dataset(self, sqlite_session: Session) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.ECONOMY)
        sqlite_session.add(dataset)
        sqlite_session.commit()
        account = _account()
        embedding_model = SimpleNamespace(provider="provider-2", model_name="embedding-2")
        collection_binding = DatasetCollectionBinding(
            provider_name="provider-2",
            model_name="embedding-2",
            type="dataset",
            collection_name="collection",
        )
        collection_binding.id = "collection-binding-2"

        with (
            patch("services.knowledge.dataset_service.current_user", account),
            patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls,
            patch.object(DatasetService, "check_is_multimodal_model", return_value=True),
            patch.object(
                DatasetCollectionBindingService,
                "get_dataset_collection_binding",
                return_value=collection_binding,
            ),
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = embedding_model
            DatasetService.update_rag_pipeline_dataset_settings(
                dataset,
                _make_knowledge_configuration(
                    embedding_model_provider="provider-2",
                    embedding_model="embedding-2",
                    summary_index_setting={"enable": True},
                ),
                session=sqlite_session,
            )

        persisted = sqlite_session.get(Dataset, dataset.id)
        assert persisted is not None
        assert persisted.indexing_technique == IndexTechniqueType.HIGH_QUALITY
        assert persisted.embedding_model == "embedding-2"
        assert persisted.collection_binding_id == collection_binding.id
        assert persisted.is_multimodal is True

    def test_unpublished_economy_settings_update_keyword_number(self, sqlite_session: Session) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY)
        sqlite_session.add(dataset)
        sqlite_session.commit()

        with patch("services.knowledge.dataset_service.current_user", _account()):
            DatasetService.update_rag_pipeline_dataset_settings(
                dataset,
                _make_knowledge_configuration(
                    indexing_technique=IndexTechniqueType.ECONOMY,
                    keyword_number=17,
                ),
                session=sqlite_session,
            )

        persisted = sqlite_session.get(Dataset, dataset.id)
        assert persisted is not None
        assert persisted.indexing_technique == IndexTechniqueType.ECONOMY
        assert persisted.keyword_number == 17

    def test_published_economy_settings_stage_keyword_target_rebuild(self, sqlite_session: Session) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.ECONOMY, chunk_structure="paragraph")
        dataset.keyword_number = 4
        sqlite_session.add(dataset)
        sqlite_session.commit()

        with patch("services.knowledge.dataset_service.current_user", _account()):
            intent_ids = DatasetService.update_rag_pipeline_dataset_settings(
                dataset,
                _make_knowledge_configuration(
                    chunk_structure="paragraph",
                    indexing_technique=IndexTechniqueType.ECONOMY,
                    keyword_number=12,
                ),
                has_published=True,
                session=sqlite_session,
            )

        persisted = sqlite_session.get(Dataset, dataset.id)
        assert persisted is not None
        assert persisted.keyword_number == 12
        assert len(intent_ids) == 1
        row = sqlite_session.get_one(DatasetIndexingOutbox, intent_ids[0])
        target = DatasetIndexingRepresentation.model_validate_json(row.target_representation)
        assert target.indexing_technique == "economy"
        assert target.keyword_number == 12

    def test_published_embedding_change_survives_broker_failure_and_same_republish(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY, chunk_structure="paragraph")
        dataset.embedding_model_provider = "old-provider"
        dataset.embedding_model = "old-model"
        sqlite_session.add(dataset)
        sqlite_session.commit()
        embedding_model = _EmbeddingModelStub(provider="new-provider", model_name="new-model")
        binding = DatasetCollectionBinding(
            provider_name="new-provider",
            model_name="new-model",
            type="dataset",
            collection_name="new-collection",
        )
        binding.id = "new-binding"
        configuration = _make_knowledge_configuration(
            chunk_structure="paragraph",
            indexing_technique=IndexTechniqueType.HIGH_QUALITY,
            embedding_model_provider="new-provider",
            embedding_model="new-model",
        )
        repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sqlite_session_factory)
        monkeypatch.setattr(dataset_indexing_task_module, "_repository", lambda: repository)

        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls,
            patch.object(DatasetService, "check_is_multimodal_model", return_value=False),
            patch.object(DatasetCollectionBindingService, "get_dataset_collection_binding", return_value=binding),
            patch.object(
                dataset_indexing_task_module.process_dataset_indexing_intent_task,
                "delay",
                side_effect=OSError("broker unavailable"),
            ),
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = embedding_model
            intent_ids = DatasetService.update_rag_pipeline_dataset_settings(
                dataset,
                configuration,
                has_published=True,
                session=sqlite_session,
            )
            sqlite_session.commit()
            DatasetService.dispatch_dataset_indexing(intent_ids)
            repeated_ids = DatasetService.update_rag_pipeline_dataset_settings(
                dataset,
                configuration,
                has_published=True,
                session=sqlite_session,
            )
            sqlite_session.commit()

        assert len(intent_ids) == 1
        assert repeated_ids == ()
        with sqlite_session_factory() as observer:
            assert observer.get(DatasetIndexingOutbox, intent_ids[0]) is not None

        executed: list[str] = []
        monkeypatch.setattr(
            dataset_indexing_task_module.deal_dataset_vector_index_task,
            "run",
            lambda *_args, **_kwargs: executed.append("vector") or True,
        )
        monkeypatch.setattr(
            dataset_indexing_task_module.regenerate_summary_index_task,
            "run",
            lambda *_args, **_kwargs: executed.append("summary") or True,
        )
        monkeypatch.setattr(
            dataset_indexing_task_module.process_dataset_indexing_intent_task,
            "delay",
            dataset_indexing_task_module.process_dataset_indexing_intent_task.run,
        )

        dataset_indexing_task_module.drain_dataset_indexing_outbox_task.run()

        assert executed == ["vector", "summary"]
        with sqlite_session_factory() as observer:
            assert observer.get(DatasetIndexingOutbox, intent_ids[0]) is None

    def test_published_embedding_change_rolls_back_configuration_and_intent(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        dataset = _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY, chunk_structure="paragraph")
        dataset.embedding_model_provider = "old-provider"
        dataset.embedding_model = "old-model"
        sqlite_session.add(dataset)
        sqlite_session.commit()
        embedding_model = _EmbeddingModelStub(provider="new-provider", model_name="new-model")
        binding = DatasetCollectionBinding(
            provider_name="new-provider",
            model_name="new-model",
            type="dataset",
            collection_name="new-collection",
        )
        binding.id = "new-binding"

        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls,
            patch.object(DatasetService, "check_is_multimodal_model", return_value=False),
            patch.object(DatasetCollectionBindingService, "get_dataset_collection_binding", return_value=binding),
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = embedding_model
            intent_ids = DatasetService.update_rag_pipeline_dataset_settings(
                dataset,
                _make_knowledge_configuration(
                    chunk_structure="paragraph",
                    indexing_technique=IndexTechniqueType.HIGH_QUALITY,
                    embedding_model_provider="new-provider",
                    embedding_model="new-model",
                ),
                has_published=True,
                session=sqlite_session,
            )
            assert len(intent_ids) == 1
            sqlite_session.rollback()

        with sqlite_session_factory() as observer:
            persisted = observer.get_one(Dataset, dataset.id)
            assert persisted.embedding_model_provider == "old-provider"
            assert persisted.embedding_model == "old-model"
            assert observer.scalar(select(DatasetIndexingOutbox.id)) is None

    @pytest.mark.parametrize(
        ("dataset", "configuration", "message"),
        [
            (
                _dataset(chunk_structure="paragraph"),
                _make_knowledge_configuration(chunk_structure="sentence"),
                "Chunk structure is not allowed",
            ),
            (
                _dataset(indexing_technique=IndexTechniqueType.HIGH_QUALITY, chunk_structure="paragraph"),
                _make_knowledge_configuration(
                    chunk_structure="paragraph",
                    indexing_technique=IndexTechniqueType.ECONOMY,
                ),
                "not allowed to be updated to economy",
            ),
        ],
    )
    def test_published_settings_reject_incompatible_changes(
        self,
        sqlite_session: Session,
        dataset: Dataset,
        configuration,
        message: str,
    ) -> None:
        sqlite_session.add(dataset)
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            pytest.raises(ValueError, match=message),
        ):
            DatasetService.update_rag_pipeline_dataset_settings(
                dataset,
                configuration,
                has_published=True,
                session=sqlite_session,
            )


class TestDatasetConfigurationCurrentReads:
    def test_patch_refreshes_stale_identity_before_model_comparison_and_target_capture(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
        configuration_model_boundary: None,
        *,
        external_templates: ExternalKnowledgeTemplateLookup,
    ) -> None:
        del configuration_model_boundary
        account = _account()
        stale_dataset = _seed_configuration_a(sqlite_session)
        assert stale_dataset.embedding_model == "model-a"
        intent_b_id = _commit_configuration_b(sqlite_session_factory, account=account)
        stale_dataset.embedding_model = "dirty-stale-model"
        stale_dataset.embedding_model_provider = "dirty-stale-provider"
        stale_dataset.collection_binding_id = "binding-a"
        dispatched: list[str] = []
        monkeypatch.setattr(dataset_service_module, "current_user", account)
        monkeypatch.setattr(dataset_service_module, "enqueue_dataset_indexing_intent", dispatched.append)

        with DatasetService.update_dataset(
            stale_dataset.id,
            {
                "indexing_technique": IndexTechniqueType.HIGH_QUALITY,
                "embedding_model_provider": "provider-a",
                "embedding_model": "model-a",
            },
            account,
            session=sqlite_session,
            external_templates=external_templates,
        ) as updated:
            assert updated.embedding_model == "model-a"
            assert updated.embedding_model_provider == "provider-a"
            assert updated.collection_binding_id == "binding-a"

        (intent_c_id,) = dispatched
        assert intent_c_id != intent_b_id
        with sqlite_session_factory() as observer:
            current = observer.get_one(Dataset, stale_dataset.id)
            assert current.embedding_model == "model-a"
            assert current.embedding_model_provider == "provider-a"
            assert current.collection_binding_id == "binding-a"
            assert current.indexing_request_id is not None
            row = observer.get_one(DatasetIndexingOutbox, intent_c_id)
            target = DatasetIndexingRepresentation.model_validate_json(row.target_representation)
            assert target.embedding_model == "model-a"
            assert target.embedding_model_provider == "provider-a"
            assert target.collection_binding_id == "binding-a"
            assert set(observer.scalars(select(DatasetIndexingOutbox.id))) == {intent_b_id, intent_c_id}
        _assert_only_current_generation_claims(
            sqlite_session_factory,
            stale_intent_id=intent_b_id,
            current_intent_id=intent_c_id,
        )

    def test_patch_noop_uses_locked_current_fields_without_restoring_stale_model(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
        configuration_model_boundary: None,
        *,
        external_templates: ExternalKnowledgeTemplateLookup,
    ) -> None:
        del configuration_model_boundary
        account = _account()
        stale_dataset = _seed_configuration_a(sqlite_session)
        assert stale_dataset.embedding_model == "model-a"
        intent_b_id = _commit_configuration_b(sqlite_session_factory, account=account)
        stale_dataset.embedding_model = "dirty-stale-model"
        stale_dataset.embedding_model_provider = "dirty-stale-provider"
        stale_dataset.collection_binding_id = "binding-a"
        dispatched: list[str] = []
        monkeypatch.setattr(dataset_service_module, "current_user", account)
        monkeypatch.setattr(dataset_service_module, "enqueue_dataset_indexing_intent", dispatched.append)

        with DatasetService.update_dataset(
            stale_dataset.id,
            {"name": "Renamed", "indexing_technique": IndexTechniqueType.HIGH_QUALITY},
            account,
            session=sqlite_session,
            external_templates=external_templates,
        ):
            pass

        assert dispatched == []
        with sqlite_session_factory() as observer:
            current = observer.get_one(Dataset, stale_dataset.id)
            assert current.name == "Renamed"
            assert current.embedding_model == "model-b"
            assert current.embedding_model_provider == "provider-b"
            assert current.collection_binding_id == "binding-b"
            assert current.indexing_request_id == observer.get_one(DatasetIndexingOutbox, intent_b_id).generation_id
            assert observer.scalars(select(DatasetIndexingOutbox.id)).all() == [intent_b_id]

    @pytest.mark.parametrize("entry", ["rag", "dsl"])
    @pytest.mark.parametrize("desired", ["a", "b"])
    def test_rag_and_dsl_settings_refresh_stale_identity_before_comparison(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
        configuration_model_boundary: None,
        entry: str,
        desired: str,
    ) -> None:
        del configuration_model_boundary
        account = _account()
        monkeypatch.setattr(dataset_service_module, "current_user", _account(tenant_id="tenant-2"))
        stale_dataset = _seed_configuration_a(sqlite_session)
        assert stale_dataset.embedding_model == "model-a"
        intent_b_id = _commit_configuration_b(sqlite_session_factory, account=account)
        configuration = _make_knowledge_configuration(
            chunk_structure="text_model",
            indexing_technique=IndexTechniqueType.HIGH_QUALITY,
            embedding_model_provider=f"provider-{desired}",
            embedding_model=f"model-{desired}",
        )
        if desired == "b":
            stale_dataset.embedding_model = "dirty-stale-model"
            stale_dataset.embedding_model_provider = "dirty-stale-provider"
            stale_dataset.collection_binding_id = "binding-a"

        if entry == "rag":
            intent_ids = DatasetService.update_rag_pipeline_dataset_settings(
                stale_dataset,
                configuration,
                has_published=True,
                session=sqlite_session,
                account=account,
            )
        else:
            dsl_service = RagPipelineDslService(sqlite_session)
            dsl_service._apply_dataset_configuration(
                dataset=stale_dataset,
                knowledge_configuration=configuration,
                account=account,
                has_published=True,
            )
            intent_ids = dsl_service.pending_dataset_indexing_intent_ids
        sqlite_session.commit()

        intent_c_id: str | None = None
        with sqlite_session_factory() as observer:
            current = observer.get_one(Dataset, stale_dataset.id)
            if desired == "b":
                assert intent_ids == ()
                assert current.embedding_model == "model-b"
                assert current.embedding_model_provider == "provider-b"
                assert current.collection_binding_id == "binding-b"
                assert observer.scalars(select(DatasetIndexingOutbox.id)).all() == [intent_b_id]
            else:
                (intent_c_id,) = intent_ids
                assert intent_c_id != intent_b_id
                assert current.embedding_model == "model-a"
                assert current.embedding_model_provider == "provider-a"
                assert current.collection_binding_id == "binding-a"
                row = observer.get_one(DatasetIndexingOutbox, intent_c_id)
                target = DatasetIndexingRepresentation.model_validate_json(row.target_representation)
                assert target.embedding_model == "model-a"
                assert target.embedding_model_provider == "provider-a"
                assert target.collection_binding_id == "binding-a"
                assert set(observer.scalars(select(DatasetIndexingOutbox.id))) == {intent_b_id, intent_c_id}
        if desired == "a":
            assert intent_c_id is not None
            _assert_only_current_generation_claims(
                sqlite_session_factory,
                stale_intent_id=intent_b_id,
                current_intent_id=intent_c_id,
            )

    def test_dsl_rechecks_false_unpublished_hint_after_concurrent_generation_install(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        configuration_model_boundary: None,
    ) -> None:
        del configuration_model_boundary
        account = _account()
        stale_dataset = _seed_configuration_a(sqlite_session)
        stale_dataset.indexing_applied_representation = None
        stale_dataset.indexing_request_id = None
        sqlite_session.commit()
        # This is the state from which the DSL caller computed
        # ``has_published=False`` before the concurrent transaction installed B.
        assert stale_dataset.indexing_applied_representation is None
        assert stale_dataset.indexing_request_id is None

        intent_b_id = _commit_configuration_b(sqlite_session_factory, account=account)
        stale_dataset.embedding_model = "dirty-stale-model"
        stale_dataset.embedding_model_provider = "dirty-stale-provider"

        dsl_service = RagPipelineDslService(sqlite_session)
        dsl_service._apply_dataset_configuration(
            dataset=stale_dataset,
            knowledge_configuration=_make_knowledge_configuration(
                chunk_structure="text_model",
                indexing_technique=IndexTechniqueType.HIGH_QUALITY,
                embedding_model_provider="provider-a",
                embedding_model="model-a",
            ),
            account=account,
            has_published=False,
        )
        (intent_c_id,) = dsl_service.pending_dataset_indexing_intent_ids
        sqlite_session.commit()

        with sqlite_session_factory() as observer:
            current = observer.get_one(Dataset, stale_dataset.id)
            assert current.embedding_model == "model-a"
            assert current.embedding_model_provider == "provider-a"
            assert current.collection_binding_id == "binding-a"
            assert current.indexing_request_id == observer.get_one(DatasetIndexingOutbox, intent_c_id).generation_id
            row = observer.get_one(DatasetIndexingOutbox, intent_c_id)
            source = DatasetIndexingRepresentation.model_validate_json(row.source_representation)
            target = DatasetIndexingRepresentation.model_validate_json(row.target_representation)
            assert (source.embedding_model_provider, source.embedding_model, source.collection_binding_id) == (
                "provider-a",
                "model-a",
                "binding-a",
            )
            assert (target.embedding_model_provider, target.embedding_model, target.collection_binding_id) == (
                "provider-a",
                "model-a",
                "binding-a",
            )
            assert set(observer.scalars(select(DatasetIndexingOutbox.id))) == {intent_b_id, intent_c_id}
        _assert_only_current_generation_claims(
            sqlite_session_factory,
            stale_intent_id=intent_b_id,
            current_intent_id=intent_c_id,
        )


class TestDatasetPermissions:
    def test_check_dataset_permission_enforces_tenant_and_partial_members(self, sqlite_session: Session) -> None:
        dataset = _dataset(permission=DatasetPermissionEnum.PARTIAL_TEAM, maintainer="owner")
        permitted_user = _account(account_id="permitted", role=TenantAccountRole.NORMAL)
        denied_user = _account(account_id="denied", role=TenantAccountRole.NORMAL)
        foreign_user = _account(account_id="foreign", tenant_id="tenant-2", role=TenantAccountRole.NORMAL)
        sqlite_session.add_all(
            [
                dataset,
                DatasetPermission(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    account_id=permitted_user.id,
                ),
            ]
        )
        sqlite_session.commit()

        DatasetService.check_dataset_permission(dataset, permitted_user, sqlite_session)
        with pytest.raises(NoPermissionError):
            DatasetService.check_dataset_permission(dataset, denied_user, sqlite_session)
        with pytest.raises(NoPermissionError):
            DatasetService.check_dataset_permission(dataset, foreign_user, sqlite_session)

    def test_check_dataset_permission_skips_legacy_acl_in_rbac_mode(
        self, config_overrides: Callable[..., None], sqlite_session: Session
    ) -> None:
        dataset = _dataset(permission=DatasetPermissionEnum.ONLY_ME, maintainer="owner")
        non_member = _account(account_id="non-member", role=TenantAccountRole.NORMAL)
        foreign_user = _account(account_id="foreign", tenant_id="tenant-2", role=TenantAccountRole.NORMAL)
        sqlite_session.add(dataset)
        sqlite_session.commit()

        config_overrides(RBAC_ENABLED=True)

        DatasetService.check_dataset_permission(dataset, non_member, sqlite_session)
        with pytest.raises(NoPermissionError):
            DatasetService.check_dataset_permission(dataset, foreign_user, sqlite_session)

    def test_dataset_operator_cannot_change_permission_or_member_list(self, sqlite_session: Session) -> None:
        dataset = _dataset(permission=DatasetPermissionEnum.PARTIAL_TEAM)
        operator = _account(role=TenantAccountRole.DATASET_OPERATOR)
        sqlite_session.add_all(
            [
                dataset,
                DatasetPermission(tenant_id="tenant-1", dataset_id=dataset.id, account_id="member-1"),
            ]
        )
        sqlite_session.commit()

        with pytest.raises(NoPermissionError, match="cannot change"):
            DatasetPermissionService.check_permission(
                operator,
                dataset,
                DatasetPermissionEnum.ALL_TEAM,
                None,
                session=sqlite_session,
            )

    def test_non_editor_cannot_change_dataset_permissions(self, sqlite_session: Session) -> None:
        user = _account(role=TenantAccountRole.NORMAL)

        with pytest.raises(NoPermissionError, match="does not have permission"):
            DatasetPermissionService.check_permission(
                user,
                _dataset(),
                DatasetPermissionEnum.ALL_TEAM,
                None,
                session=sqlite_session,
            )

    def test_dataset_operator_can_keep_unchanged_partial_member_list(self, sqlite_session: Session) -> None:
        dataset = _dataset(permission=DatasetPermissionEnum.PARTIAL_TEAM)
        operator = _account(role=TenantAccountRole.DATASET_OPERATOR)
        sqlite_session.add_all(
            [
                dataset,
                DatasetPermission(tenant_id="tenant-1", dataset_id=dataset.id, account_id="member-1"),
            ]
        )
        sqlite_session.commit()

        DatasetPermissionService.check_permission(
            operator,
            dataset,
            DatasetPermissionEnum.PARTIAL_TEAM,
            [{"user_id": "member-1"}],
            session=sqlite_session,
        )
        with pytest.raises(ValueError, match="cannot change"):
            DatasetPermissionService.check_permission(
                operator,
                dataset,
                DatasetPermissionEnum.PARTIAL_TEAM,
                [{"user_id": "member-2"}],
                session=sqlite_session,
            )

    def test_update_partial_member_list_flush_failure_does_not_rollback_caller_session(
        self, sqlite_session: Session
    ) -> None:
        transaction_events: list[str] = []

        def fail_flush(_session, _flush_context, _instances) -> None:
            raise RuntimeError("flush failed")

        event.listen(sqlite_session, "before_flush", fail_flush)
        event.listen(sqlite_session, "after_rollback", lambda _session: transaction_events.append("rollback"))
        try:
            with pytest.raises(RuntimeError, match="flush failed"):
                DatasetPermissionService.update_partial_member_list(
                    "tenant-1",
                    "dataset-1",
                    [{"user_id": "member-1"}],
                    sqlite_session,
                )
        finally:
            event.remove(sqlite_session, "before_flush", fail_flush)

        assert transaction_events == []

    def test_clear_partial_member_list_execute_failure_does_not_rollback_caller_session(
        self, sqlite_session: Session
    ) -> None:
        transaction_events: list[str] = []

        def fail_execute(_orm_execute_state) -> None:
            raise RuntimeError("execute failed")

        event.listen(sqlite_session, "do_orm_execute", fail_execute)
        event.listen(sqlite_session, "after_rollback", lambda _session: transaction_events.append("rollback"))
        try:
            with pytest.raises(RuntimeError, match="execute failed"):
                DatasetPermissionService.clear_partial_member_list("dataset-1", sqlite_session)
        finally:
            event.remove(sqlite_session, "do_orm_execute", fail_execute)

        assert transaction_events == []


@pytest.mark.parametrize("rbac_enabled", [False, True])
@pytest.mark.parametrize("commit_fails", [False, True])
@pytest.mark.parametrize("expire_on_commit", [False, True])
def test_empty_pipeline_syncs_creator_after_commit(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    config_overrides: Callable[..., None],
    rbac_enabled: bool,
    commit_fails: bool,
    expire_on_commit: bool,
) -> None:
    config_overrides(RBAC_ENABLED=rbac_enabled)
    sqlite_session.expire_on_commit = expire_on_commit
    account = _account()
    entity = RagPipelineDatasetCreateEntity(
        name="New pipeline",
        description="",
        permission=DatasetPermissionEnum.ONLY_ME,
        icon_info=IconInfo(icon_type="emoji", icon="📚", icon_background="#FFFFFF", icon_url=None),
    )
    commit = sqlite_session.commit
    observed: list[tuple[str, str, object]] = []
    transactions_at_sync: list[bool] = []

    def commit_session() -> None:
        if commit_fails:
            raise RuntimeError("commit failed")
        commit()

    def sync_binding(_tenant_id: str, _account_id: str, *, resource_type: object, resource_id: str) -> list[object]:
        transactions_at_sync.append(sqlite_session.in_transaction())
        with sqlite_session_factory() as observer:
            stored = observer.get(Dataset, resource_id)
            if stored is not None:
                observed.append((stored.tenant_id, stored.created_by, resource_type))
        return []

    with (
        patch("services.knowledge.dataset_service.current_user", account),
        patch.object(sqlite_session, "commit", side_effect=commit_session),
        patch.object(
            rbac_service.RBACService.AccessPolicies,
            "sync_creator_access_policy_member_bindings",
            side_effect=sync_binding,
        ) as sync,
    ):
        if commit_fails:
            with pytest.raises(RuntimeError, match="commit failed"):
                DatasetService.create_empty_rag_pipeline_dataset("tenant-1", entity, sqlite_session)
        else:
            dataset = DatasetService.create_empty_rag_pipeline_dataset("tenant-1", entity, sqlite_session)
            if rbac_enabled:
                assert transactions_at_sync == [False]
                assert observed == [("tenant-1", account.id, rbac_service.RBACResourceType.DATASET)]
                sync.assert_called_once_with(
                    "tenant-1",
                    account.id,
                    resource_type=rbac_service.RBACResourceType.DATASET,
                    resource_id=dataset.id,
                )
        if commit_fails or not rbac_enabled:
            sync.assert_not_called()

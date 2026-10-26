from unittest.mock import Mock, create_autospec

import pytest
from flask import Flask
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from constants.resource_access_token import ResourceAccessTokenResourceType
from models import ApiToken, Dataset, DatasetApiTokenBinding, Tenant
from models.agent import Agent, AgentHomeSnapshot, AgentScope, AgentSource, AgentStatus, AgentWorkingResourceStatus
from models.dataset import DatasetPermission
from models.enums import ApiTokenType
from models.model import ResourceCleanupOutbox
from repositories.app.console_repository import ConsoleAppRepository
from repositories.knowledge.dataset_repository import SQLAlchemyDatasetRepository
from repositories.workspace.resource_cleanup_repository import WorkspaceResourceCleanupRepository
from services.resource_access_token_service import ResourceAccessTokenCleanupService
from services.workspace import resource_cleanup_service as module
from tasks.clean_dataset_task import run_dataset_cleanup


def _service(session_factory: sessionmaker[Session]) -> module.WorkspaceResourceCleanupService:
    return module.WorkspaceResourceCleanupService(
        session_factory=session_factory,
        apps=ConsoleAppRepository(session_factory=session_factory),
        datasets=SQLAlchemyDatasetRepository(session_factory=session_factory),
        agent_resources=WorkspaceResourceCleanupRepository(session_factory=session_factory),
    )


def test_agent_sweep_covers_all_scopes_and_retains_other_tenants(
    sqlite_session: Session, sqlite_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = [
        Agent(
            id="roster",
            tenant_id="target",
            name="Roster",
            scope=AgentScope.ROSTER,
            source=AgentSource.WORKFLOW,
            backing_app_id="missing-app",
        ),
        Agent(
            id="inline", tenant_id="target", name="Inline", scope=AgentScope.WORKFLOW_ONLY, source=AgentSource.WORKFLOW
        ),
        Agent(
            id="archived",
            tenant_id="target",
            name="Archived",
            scope=AgentScope.ROSTER,
            source=AgentSource.WORKFLOW,
            status=AgentStatus.ARCHIVED,
        ),
        Agent(id="other", tenant_id="other-tenant", name="Other", scope=AgentScope.ROSTER, source=AgentSource.WORKFLOW),
    ]
    snapshot = AgentHomeSnapshot(id="home", tenant_id="target", agent_id="inline", snapshot_ref="backend-home")
    sqlite_session.add_all([*agents, snapshot])
    sqlite_session.commit()
    collection = Mock()
    app_cleanup = Mock()
    monkeypatch.setattr(module, "enqueue_agent_resource_collection", collection)
    monkeypatch.setattr(module, "remove_app_and_related_data_task", app_cleanup)

    def assert_committed(**_kwargs: object) -> None:
        sqlite_session.expire_all()
        inline_agent = sqlite_session.get(Agent, "inline")
        assert inline_agent is not None
        assert inline_agent.status == AgentStatus.ARCHIVED
        home_snapshot = sqlite_session.get(AgentHomeSnapshot, "home")
        assert home_snapshot is not None
        assert home_snapshot.status == AgentWorkingResourceStatus.RETIRED

    collection.side_effect = assert_committed
    _service(sqlite_session_factory)._retire_agents("target")
    payload = collection.call_args.kwargs
    assert set(payload["purge_agent_ids"]) == {"roster", "inline", "archived"}
    assert payload["home_snapshot_ids"] == ["home"]
    other_agent = sqlite_session.get(Agent, "other")
    assert other_agent is not None
    assert other_agent.status == AgentStatus.ACTIVE
    app_cleanup.delay.assert_called_once_with(tenant_id="target", app_id="missing-app")


def test_dataset_deletion_schedules_even_without_index_configuration(
    sqlite_session: Session, sqlite_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    target_tenant = Tenant(name="Target")
    target_tenant.id = "target"
    other_tenant = Tenant(name="Other")
    other_tenant.id = "other"
    target_permission = DatasetPermission(tenant_id="target", dataset_id="dataset", account_id="reader")
    other_permission = DatasetPermission(tenant_id="other", dataset_id="other-dataset", account_id="reader")
    sqlite_session.add_all(
        [
            target_tenant,
            other_tenant,
            Dataset(id="dataset", tenant_id="target", name="Empty", created_by="owner"),
            Dataset(id="other-dataset", tenant_id="other", name="Other", created_by="owner"),
            target_permission,
            other_permission,
        ]
    )
    sqlite_session.commit()
    target_permission_id = target_permission.id
    other_permission_id = other_permission.id
    enqueued: list[str] = []
    cleanup = create_autospec(ResourceAccessTokenCleanupService, instance=True, spec_set=True)
    monkeypatch.setattr(module, "enqueue_resource_cleanup_intent", enqueued.append)

    def build_cleanup(*, session: Session) -> ResourceAccessTokenCleanupService:
        assert session.bind is sqlite_session.bind
        return cleanup

    monkeypatch.setattr(module, "build_resource_access_token_cleanup_service", build_cleanup)
    _service(sqlite_session_factory)._delete_dataset("target", "other-dataset")
    assert enqueued == []
    _service(sqlite_session_factory)._delete_dataset("target", "dataset")
    assert len(enqueued) == 1
    intent = sqlite_session.get(ResourceCleanupOutbox, enqueued[0])
    assert intent is not None
    assert (intent.tenant_id, intent.dataset_id, intent.resource_id) == ("target", "dataset", "dataset")
    cleanup.delete_resource_relations.assert_called_once_with(
        tenant_id="target",
        resource_type=ResourceAccessTokenResourceType.KNOWLEDGE,
        resource_id="dataset",
    )
    sqlite_session.expire_all()
    assert sqlite_session.get(Dataset, "dataset") is None
    assert sqlite_session.get(Dataset, "other-dataset") is not None
    assert sqlite_session.get(DatasetPermission, target_permission_id) is None
    assert sqlite_session.get(DatasetPermission, other_permission_id) is not None

    # The durable cleanup remains idempotent after the command removed ACLs.
    monkeypatch.setattr("tasks.clean_dataset_task._prepare_dataset_cleanup", lambda **_kwargs: None)
    run_dataset_cleanup(
        tenant_id=intent.tenant_id,
        dataset_id=intent.dataset_id,
        indexing_technique=intent.indexing_technique,
        index_struct=intent.index_struct,
        collection_binding_id=intent.collection_binding_id,
        doc_form=intent.doc_form or "text_model",
        pipeline_id=intent.pipeline_id,
    )
    sqlite_session.expire_all()
    assert sqlite_session.get(DatasetPermission, target_permission_id) is None
    assert sqlite_session.get(DatasetPermission, other_permission_id) is not None


def test_dataset_cleanup_commits_key_deletion_before_cache_invalidation(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = Tenant(name="Target")
    tenant.id = "target"
    dataset = Dataset(id="dataset", tenant_id=tenant.id, name="Empty", created_by="owner")
    key = ApiToken(id="key", tenant_id=tenant.id, type=ApiTokenType.DATASET, token="dataset-secret")
    sqlite_session.add_all([tenant, dataset, key])
    sqlite_session.flush()
    sqlite_session.add(DatasetApiTokenBinding(api_token_id=key.id, dataset_id=dataset.id))
    sqlite_session.commit()
    enqueued: list[str] = []
    invalidated: list[tuple[str, str]] = []

    monkeypatch.setattr(module, "enqueue_resource_cleanup_intent", enqueued.append)

    def invalidate(token: str, scope: str | None = None) -> bool:
        with sqlite_session_factory() as committed_session:
            assert committed_session.get(Dataset, dataset.id) is None
            assert committed_session.get(ApiToken, key.id) is None
            assert committed_session.scalar(select(ResourceCleanupOutbox.id)) is not None
        invalidated.append((token, scope or ""))
        return True

    monkeypatch.setattr(module.ApiTokenCache, "delete", invalidate)

    _service(sqlite_session_factory)._delete_dataset(tenant.id, dataset.id)

    assert len(enqueued) == 1
    assert invalidated == [(key.token, key.type)]


def test_dataset_cleanup_failure_rolls_back_dataset_and_intent(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = Tenant(name="Target")
    tenant.id = "target"
    dataset = Dataset(id="dataset", tenant_id=tenant.id, name="Empty", created_by="owner")
    sqlite_session.add_all([tenant, dataset])
    sqlite_session.commit()
    cleanup = create_autospec(ResourceAccessTokenCleanupService, instance=True, spec_set=True)
    cleanup.delete_resource_relations.side_effect = RuntimeError("relation cleanup failed")
    enqueued: list[str] = []
    monkeypatch.setattr(module, "enqueue_resource_cleanup_intent", enqueued.append)

    def build_cleanup(*, session: Session) -> ResourceAccessTokenCleanupService:
        assert session.bind is sqlite_session.bind
        return cleanup

    monkeypatch.setattr(module, "build_resource_access_token_cleanup_service", build_cleanup)

    with pytest.raises(RuntimeError, match="relation cleanup failed"):
        _service(sqlite_session_factory)._delete_dataset(tenant.id, dataset.id)

    sqlite_session.expire_all()
    assert sqlite_session.get(Dataset, dataset.id) is not None
    assert sqlite_session.scalar(select(ResourceCleanupOutbox)) is None
    assert enqueued == []


def test_agent_collection_failure_propagates_and_retirement_can_be_retried(
    sqlite_session: Session, sqlite_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_session.add(
        Agent(id="agent", tenant_id="target", name="Agent", scope=AgentScope.ROSTER, source=AgentSource.WORKFLOW)
    )
    sqlite_session.commit()
    collection = Mock(side_effect=RuntimeError("broker unavailable"))
    monkeypatch.setattr(module, "enqueue_agent_resource_collection", collection)
    with pytest.raises(RuntimeError, match="broker unavailable"):
        _service(sqlite_session_factory)._retire_agents("target")
    collection.side_effect = None
    _service(sqlite_session_factory)._retire_agents("target")
    assert collection.call_args.kwargs["purge_agent_ids"] == ["agent"]


def test_working_resources_are_retired_even_when_the_app_is_missing(
    sqlite_session: Session, sqlite_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from models.agent import (
        AgentConfigVersionKind,
        AgentWorkspace,
        AgentWorkspaceBinding,
        AgentWorkspaceOwnerType,
    )

    workspace = AgentWorkspace(
        id="working",
        tenant_id="target",
        app_id="missing-app",
        owner_type=AgentWorkspaceOwnerType.CONVERSATION,
        owner_id="conversation",
        owner_scope_key="conversation",
        backend_workspace_ref="backend-workspace",
    )
    binding = AgentWorkspaceBinding(
        id="binding",
        tenant_id="target",
        app_id="missing-app",
        workspace_id="working",
        agent_id="missing-agent",
        agent_config_version_id="config",
        agent_config_version_kind=AgentConfigVersionKind.SNAPSHOT,
        backend_binding_ref="backend-binding",
    )
    sqlite_session.add_all([workspace, binding])
    sqlite_session.commit()
    collection = Mock()
    monkeypatch.setattr(module, "enqueue_agent_resource_collection", collection)
    _service(sqlite_session_factory)._retire_agents("target")
    sqlite_session.expire_all()
    working_workspace = sqlite_session.get(AgentWorkspace, "working")
    assert working_workspace is not None
    assert working_workspace.status == AgentWorkingResourceStatus.RETIRED
    working_binding = sqlite_session.get(AgentWorkspaceBinding, "binding")
    assert working_binding is not None
    assert working_binding.status == AgentWorkingResourceStatus.RETIRED
    assert collection.call_args.kwargs["workspace_ids"] == ["working"]
    assert collection.call_args.kwargs["binding_ids"] == ["binding"]


def test_resource_failure_does_not_skip_the_other_groups(
    sqlite_session: Session, sqlite_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from models import App
    from models.enums import AppStatus
    from models.model import AppMode

    sqlite_session.add_all(
        [
            App(
                id="app",
                tenant_id="target",
                name="App",
                mode=AppMode.WORKFLOW,
                status=AppStatus.NORMAL,
                enable_site=True,
                enable_api=True,
            ),
            Dataset(id="dataset", tenant_id="target", name="Dataset", created_by="owner"),
        ]
    )
    sqlite_session.commit()
    app_service = Mock()
    app_service.return_value.delete_app.side_effect = RuntimeError("app failed")
    monkeypatch.setattr(module, "AppService", app_service)
    datasets = Mock()
    agents = Mock()
    monkeypatch.setattr(module.WorkspaceResourceCleanupService, "_delete_dataset", datasets)
    monkeypatch.setattr(module.WorkspaceResourceCleanupService, "_retire_agents", agents)
    with pytest.raises(RuntimeError, match="app:app"):
        _service(sqlite_session_factory).cleanup("target")
    datasets.assert_called_once_with("target", "dataset")
    agents.assert_called_once_with("target")


def test_agent_skill_snapshots_and_orphan_homes_are_included(
    sqlite_session: Session, sqlite_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from models.skill import AgentSkillBindingSnapshot

    sqlite_session.add_all(
        [
            AgentHomeSnapshot(id="orphan-home", tenant_id="target", agent_id="missing", snapshot_ref="backend"),
            AgentSkillBindingSnapshot(
                id="target-binding",
                tenant_id="target",
                agent_id="missing",
                config_snapshot_id="config",
                skill_id="skill",
                priority=0,
            ),
            AgentSkillBindingSnapshot(
                id="other-binding",
                tenant_id="other",
                agent_id="missing",
                config_snapshot_id="config",
                skill_id="skill",
                priority=0,
            ),
        ]
    )
    sqlite_session.commit()
    collection = Mock()
    monkeypatch.setattr(module, "enqueue_agent_resource_collection", collection)
    _service(sqlite_session_factory)._retire_agents("target")
    sqlite_session.expire_all()
    assert sqlite_session.get(AgentSkillBindingSnapshot, "target-binding") is None
    assert sqlite_session.get(AgentSkillBindingSnapshot, "other-binding") is not None
    orphan_home = sqlite_session.get(AgentHomeSnapshot, "orphan-home")
    assert orphan_home is not None
    assert orphan_home.status == AgentWorkingResourceStatus.RETIRED
    assert collection.call_args.kwargs["home_snapshot_ids"] == ["orphan-home"]


def test_direct_cleanup_does_not_resolve_a_console_login(
    app: Flask,
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from models import App
    from models.enums import AppStatus
    from models.model import AppMode
    from services import app_service

    sqlite_session.add(
        App(
            id="app",
            tenant_id="target",
            name="App",
            mode=AppMode.WORKFLOW,
            status=AppStatus.NORMAL,
            enable_site=True,
            enable_api=True,
        )
    )
    sqlite_session.commit()
    user = MagicMock()
    user.__bool__.side_effect = AssertionError("inner API must not load a console user")
    monkeypatch.setattr(app_service, "current_user", user)
    monkeypatch.setattr(app_service.app_was_deleted, "send", Mock())
    monkeypatch.setattr(app_service, "build_resource_access_token_cleanup_service", Mock())
    monkeypatch.setattr(app_service, "remove_app_and_related_data_task", Mock())
    retirement = Mock()
    monkeypatch.setattr(app_service, "WorkflowAgentRetirementService", retirement)
    monkeypatch.setattr(app_service, "enqueue_agent_resource_collection", Mock())
    features = Mock()
    features.is_webapp_auth_enabled.return_value = False
    monkeypatch.setattr(app_service, "SystemFeatureService", features)
    monkeypatch.setattr(app_service, "BillingService", Mock())
    monkeypatch.setattr(module, "enqueue_agent_resource_collection", Mock())
    with app.test_request_context(method="DELETE"):
        _service(sqlite_session_factory).cleanup("target")
    sqlite_session.expire_all()
    assert sqlite_session.get(App, "app") is None
    assert retirement.retire_unowned.call_args.kwargs["account_id"] is None
    user.__bool__.assert_not_called()


def test_tenant_home_snapshot_retirement_is_tenant_scoped_and_idempotent(sqlite_session: Session) -> None:
    from datetime import datetime

    from services.agent.home_snapshot_service import AgentHomeSnapshotService

    retired_at = datetime(2026, 1, 1)
    sqlite_session.add_all(
        [
            AgentHomeSnapshot(id="active", tenant_id="target", agent_id="agent", snapshot_ref="a"),
            AgentHomeSnapshot(
                id="retired",
                tenant_id="target",
                agent_id="missing",
                snapshot_ref="b",
                status=AgentWorkingResourceStatus.RETIRED,
                retired_at=retired_at,
            ),
            AgentHomeSnapshot(id="other", tenant_id="other", agent_id="agent", snapshot_ref="c"),
        ]
    )
    sqlite_session.commit()
    ids = AgentHomeSnapshotService.retire_all_for_tenant(session=sqlite_session, tenant_id="target")
    sqlite_session.commit()
    assert set(ids) == {"active", "retired"}
    active = sqlite_session.get(AgentHomeSnapshot, "active")
    retired = sqlite_session.get(AgentHomeSnapshot, "retired")
    other = sqlite_session.get(AgentHomeSnapshot, "other")
    assert active is not None
    assert active.status == AgentWorkingResourceStatus.RETIRED
    assert retired is not None
    assert retired.retired_at == retired_at
    assert other is not None
    assert other.status == AgentWorkingResourceStatus.ACTIVE

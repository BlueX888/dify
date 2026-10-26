from collections.abc import Generator, Mapping
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from typing_extensions import TypedDict

import tasks.trigger_processing_tasks as trigger_processing_tasks_module
from core.plugin.entities.plugin_daemon import CredentialType
from extensions.logstore.aliyun_logstore import AliyunLogStore
from extensions.logstore.repositories.logstore_api_workflow_run_repository import LogstoreAPIWorkflowRunRepository
from graphon.enums import WorkflowExecutionStatus
from models.enums import EndUserType
from models.model import EndUser
from models.trigger import TriggerSubscription, WorkflowPluginTrigger, WorkflowTriggerLog
from models.workflow import Workflow, WorkflowAppLog, WorkflowRun
from services.errors.app import QuotaExceededError
from services.workflow.entities import PluginTriggerMetadata
from tasks.trigger_processing_tasks import _record_trigger_failure_log, dispatch_triggered_workflow
from tasks.workflow_run_history_sync_task import drain_pending_workflow_run_histories
from tests.unit_tests.model_factories import make_end_user, make_workflow


def _workflow(*, app_id: str = "app-123") -> Workflow:
    return make_workflow(
        workflow_id="workflow-123",
        tenant_id="tenant-123",
        app_id=app_id,
        version="published",
        graph={
            "nodes": [
                {
                    "id": "node-123",
                    "data": {"type": trigger_processing_tasks_module.TRIGGER_PLUGIN_NODE_TYPE},
                }
            ],
            "edges": [],
        },
        created_by="user-123",
    )


def _end_user() -> EndUser:
    return make_end_user(
        end_user_id="end-user-123",
        tenant_id="tenant-123",
        app_id="app-123",
        end_user_type=EndUserType.TRIGGER,
        session_id="trigger-session",
    )


class _EndUserServiceStub:
    def __init__(self) -> None:
        self.result: Mapping[str, EndUser] = {}
        self.calls: list[tuple[EndUserType, str, list[str], str]] = []

    def create_end_user_batch(
        self,
        type: EndUserType,
        tenant_id: str,
        app_ids: list[str],
        user_id: str,
    ) -> Mapping[str, EndUser]:
        self.calls.append((type, tenant_id, app_ids, user_id))
        return self.result


class _RecordingLogstore:
    supports_pg_protocol = False

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def put_log(self, _name: str, fields: list[tuple[str, str]]) -> None:
        self.records.append(dict(fields))

    def get_logs(self, **_kwargs: object) -> list[dict[str, str]]:
        return self.records


@pytest.mark.parametrize("sqlite_session", [(WorkflowRun, WorkflowAppLog, WorkflowTriggerLog)], indirect=True)
def test_plugin_trigger_failure_registers_and_publishes_history_projection(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow()
    subscription = TriggerSubscription(
        tenant_id=workflow.tenant_id,
        user_id="account-123",
        name="Failure Subscription",
        endpoint_id="endpoint-123",
        provider_id="langgenius/test_plugin/test_plugin",
        parameters={},
        properties={},
        credentials={},
        credential_type=CredentialType.API_KEY,
    )
    subscription.id = "subscription-failure"
    plugin_trigger = WorkflowPluginTrigger(
        app_id=workflow.app_id,
        node_id="node-123",
        tenant_id=workflow.tenant_id,
        provider_id=subscription.provider_id,
        event_name="test_event",
        subscription_id=subscription.id,
    )
    plugin_trigger.id = "plugin-trigger-failure"
    metadata = PluginTriggerMetadata(
        endpoint_id=subscription.endpoint_id,
        plugin_unique_identifier="langgenius/test_plugin:0.0.1",
        provider_id=subscription.provider_id,
        event_name="test_event",
        icon_filename="icon.svg",
        icon_dark_filename="icon-dark.svg",
    )

    _record_trigger_failure_log(
        session=sqlite_session,
        workflow=workflow,
        plugin_trigger=plugin_trigger,
        subscription=subscription,
        trigger_metadata=metadata,
        end_user=None,
        error_message="plugin invocation failed",
        event_name="test_event",
        request_id="request-failure",
    )

    with sqlite_session_factory() as session:
        workflow_run = session.scalar(select(WorkflowRun))
        assert workflow_run is not None
        assert workflow_run.status == WorkflowExecutionStatus.FAILED
        assert workflow_run.history_revision == 1
        assert workflow_run.history_published_revision == 0
        assert workflow_run.history_sync_pending is True

    logstore = _RecordingLogstore()
    assert (
        drain_pending_workflow_run_histories(
            database_client=sqlite_session_factory,
            logstore_client=cast(AliyunLogStore, logstore),
            enable_graph=False,
        )
        == 1
    )
    assert [record["status"] for record in logstore.records] == [WorkflowExecutionStatus.FAILED.value]
    monkeypatch.setattr(AliyunLogStore, "_instance", cast(AliyunLogStore, logstore))
    monkeypatch.setattr(AliyunLogStore, "_initialized", True)
    history_reader = LogstoreAPIWorkflowRunRepository(sqlite_session_factory)  # pyrefly: ignore[bad-instantiation]
    history_reader._enable_dual_read = False
    found = history_reader.get_workflow_run_by_id(workflow.tenant_id, workflow.app_id, workflow_run.id)
    assert found is not None
    assert found.status == WorkflowExecutionStatus.FAILED


class _DispatchMocks(TypedDict, closed=True):
    get_workflows: MagicMock
    reserve: MagicMock
    quota_charge: MagicMock
    mark_rate_limited: MagicMock
    invoke_trigger_event: MagicMock
    invoke_response: MagicMock
    end_users: _EndUserServiceStub
    trigger_workflow_async: MagicMock


class TestDispatchTriggeredWorkflow:
    """Unit tests covering branch behaviours of ``dispatch_triggered_workflow``.

    The covered branches are:
    - workflow missing for ``plugin_trigger.app_id`` → log + ``continue``
    - ``QuotaService.reserve`` raising ``QuotaExceededError`` →
      ``mark_tenant_triggers_rate_limited`` + early ``return``
    - ``trigger_workflow_async`` succeeds →
      ``quota_charge.commit()`` + ``dispatched_count`` increments
    """

    @pytest.fixture
    def subscription(self) -> TriggerSubscription:
        subscription = TriggerSubscription(
            tenant_id="tenant-123",
            user_id="user-123",
            name="Test Subscription",
            endpoint_id="endpoint-123",
            provider_id="langgenius/test_plugin/test_plugin",
            parameters={},
            properties={},
            credentials={},
            credential_type=CredentialType.API_KEY,
        )
        subscription.id = "subscription-123"
        return subscription

    @pytest.fixture
    def plugin_trigger(self) -> WorkflowPluginTrigger:
        trigger = WorkflowPluginTrigger(
            app_id="app-123",
            node_id="node-123",
            tenant_id="tenant-123",
            provider_id="langgenius/test_plugin/test_plugin",
            event_name="test_event",
            subscription_id="subscription-123",
        )
        trigger.id = "plugin-trigger-123"
        return trigger

    @pytest.fixture
    def provider_controller(self) -> MagicMock:
        controller = MagicMock()
        controller.plugin_unique_identifier = "langgenius/test_plugin:0.0.1"
        controller.entity.identity.name = "Test Plugin"
        controller.entity.identity.icon = "icon.svg"
        controller.entity.identity.icon_dark = "icon_dark.svg"
        return controller

    @pytest.fixture
    def dispatch_mocks(
        self, plugin_trigger: WorkflowPluginTrigger, provider_controller: MagicMock
    ) -> Generator[_DispatchMocks]:
        """Patch all external dependencies reached by ``dispatch_triggered_workflow``.

        Defaults are configured so the code flow can reach the final async
        trigger block (line ~385); each test overrides specific handles
        (``get_workflows``, ``reserve``, ``end_users``, ...) to
        drive the path it targets.
        """
        invoke_response = MagicMock()
        invoke_response.cancelled = False
        invoke_response.variables = dict[str, object]()

        quota_charge = MagicMock()
        end_users = _EndUserServiceStub()

        with (
            patch.object(
                trigger_processing_tasks_module.TriggerHttpRequestCachingService,
                "get_request",
                return_value=MagicMock(),
            ),
            patch.object(
                trigger_processing_tasks_module.TriggerHttpRequestCachingService,
                "get_payload",
                return_value=MagicMock(),
            ),
            patch.object(
                trigger_processing_tasks_module.TriggerSubscriptionOperatorService,
                "get_subscriber_triggers",
                return_value=[plugin_trigger],
            ),
            patch.object(
                trigger_processing_tasks_module.TriggerManager,
                "get_trigger_provider",
                return_value=provider_controller,
            ),
            patch.object(
                trigger_processing_tasks_module.TriggerManager,
                "invoke_trigger_event",
                return_value=invoke_response,
            ) as invoke_trigger_event,
            patch.object(
                trigger_processing_tasks_module.TriggerEventNodeData,
                "model_validate",
                return_value=MagicMock(),
            ),
            patch.object(
                trigger_processing_tasks_module,
                "_get_published_workflows_by_app_ids",
            ) as get_workflows,
            patch.object(
                trigger_processing_tasks_module.QuotaService,
                "reserve",
                return_value=quota_charge,
            ) as reserve,
            patch.object(
                trigger_processing_tasks_module.AppTriggerService,
                "mark_tenant_triggers_rate_limited",
            ) as mark_rate_limited,
            patch.object(
                trigger_processing_tasks_module.AsyncWorkflowService,
                "trigger_workflow_async",
            ) as trigger_workflow_async,
        ):
            yield {
                "get_workflows": get_workflows,
                "reserve": reserve,
                "quota_charge": quota_charge,
                "mark_rate_limited": mark_rate_limited,
                "invoke_trigger_event": invoke_trigger_event,
                "invoke_response": invoke_response,
                "end_users": end_users,
                "trigger_workflow_async": trigger_workflow_async,
            }

    def test_dispatch_skips_when_workflow_missing(
        self, subscription: TriggerSubscription, dispatch_mocks: _DispatchMocks
    ) -> None:
        """Covers missing workflow → log + ``continue``."""
        dispatch_mocks["get_workflows"].return_value = dict[str, object]()

        dispatched = dispatch_triggered_workflow(
            user_id="user-123",
            subscription=subscription,
            event_name="test_event",
            request_id="request-123",
            end_users=dispatch_mocks["end_users"],
        )

        assert dispatched == 0
        dispatch_mocks["reserve"].assert_not_called()
        dispatch_mocks["invoke_trigger_event"].assert_not_called()
        dispatch_mocks["mark_rate_limited"].assert_not_called()

    def test_dispatch_marks_rate_limited_when_quota_exceeded(
        self, subscription: TriggerSubscription, plugin_trigger: WorkflowPluginTrigger, dispatch_mocks: _DispatchMocks
    ) -> None:
        """Covers QuotaExceededError → mark rate-limited + early return."""
        workflow = _workflow()
        dispatch_mocks["get_workflows"].return_value = {plugin_trigger.app_id: workflow}
        dispatch_mocks["reserve"].side_effect = QuotaExceededError(
            feature="trigger", tenant_id=subscription.tenant_id, required=1
        )

        dispatched = dispatch_triggered_workflow(
            user_id="user-123",
            subscription=subscription,
            event_name="test_event",
            request_id="request-123",
            end_users=dispatch_mocks["end_users"],
        )

        assert dispatched == 0
        dispatch_mocks["reserve"].assert_called_once()
        dispatch_mocks["mark_rate_limited"].assert_called_once_with(subscription.tenant_id)
        dispatch_mocks["invoke_trigger_event"].assert_not_called()

    def test_dispatch_commits_quota_and_counts_when_workflow_triggered(
        self, subscription: TriggerSubscription, plugin_trigger: WorkflowPluginTrigger, dispatch_mocks: _DispatchMocks
    ) -> None:
        """Happy path: end user exists and async trigger succeeds."""
        workflow = _workflow()
        dispatch_mocks["get_workflows"].return_value = {plugin_trigger.app_id: workflow}

        end_user = _end_user()
        dispatch_mocks["end_users"].result = {plugin_trigger.app_id: end_user}

        dispatched = dispatch_triggered_workflow(
            user_id="user-123",
            subscription=subscription,
            event_name="test_event",
            request_id="request-123",
            end_users=dispatch_mocks["end_users"],
        )

        assert dispatched == 1
        dispatch_mocks["trigger_workflow_async"].assert_called_once()
        _, kwargs = dispatch_mocks["trigger_workflow_async"].call_args
        assert kwargs["user"] is end_user
        dispatch_mocks["quota_charge"].commit.assert_called_once()
        dispatch_mocks["quota_charge"].refund.assert_not_called()
        dispatch_mocks["mark_rate_limited"].assert_not_called()

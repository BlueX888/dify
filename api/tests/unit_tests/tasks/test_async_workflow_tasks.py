from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event
from time import monotonic
from typing import Literal, cast
from unittest.mock import create_autospec

import pytest
from celery import Task
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from core.app.apps.execution_coordinator import clear_app_task_cancellation_signals
from core.app.apps.workflow.command_channels import CelerySignalCommandChannel
from core.rag.entities.retrieval_settings import ExternalDatasetRetriever
from core.workflow.system_variables import build_system_variables
from graphon.engine.command.protocol import CommandChannel
from graphon.engine_events import (
    GraphRunFailedEvent,
    GraphRunPartialSucceededEvent,
    GraphRunPausedEvent,
    GraphRunSucceededEvent,
)
from graphon.runtime import RuntimeState, VariablePool
from graphon.runtime.runtime_state_protocol import ReadOnlyRuntimeState
from models.enums import AppTriggerType, CreatorUserRole, EndUserType, WorkflowRunTriggeredFrom, WorkflowTriggerStatus
from models.model import App, AppMode, EndUser, Tenant
from models.trigger import WorkflowTriggerLog
from models.workflow import Workflow, WorkflowType
from repositories.sqlalchemy_workflow_trigger_log_repository import SQLAlchemyWorkflowTriggerLogRepository
from repositories.workflow_trigger_log_repository import WorkflowQuotaSettlementCandidate
from services.errors.app import WorkflowNotFoundError
from services.workflow.entities import TriggerData, WorkflowTaskData
from tasks import async_workflow_tasks
from tasks.app_generate.workflow_execute_task import _get_trigger_resume_layers


@dataclass(frozen=True, slots=True)
class _KnowledgeServicesStub:
    external_retrieval: ExternalDatasetRetriever


@dataclass(frozen=True, slots=True)
class _ApplicationServicesStub:
    knowledge: _KnowledgeServicesStub


def _persist_execution_dependencies(session: Session) -> None:
    tenant = Tenant(name="Async workflow tenant")
    tenant.id = "tenant-123"
    session.add_all(
        [
            tenant,
            App(
                id="app-123",
                tenant_id="tenant-123",
                name="Async workflow app",
                description="",
                mode=AppMode.WORKFLOW,
                enable_site=True,
                enable_api=True,
                max_active_requests=0,
            ),
            Workflow(
                id="workflow-123",
                tenant_id="tenant-123",
                app_id="app-123",
                type=WorkflowType.WORKFLOW,
                version="1",
                graph="{}",
                _features="{}",
                created_by="account-123",
            ),
            EndUser(
                id="end-user-123",
                tenant_id="tenant-123",
                app_id="app-123",
                type=EndUserType.BROWSER,
                name="End User",
                session_id="session-end-user-123",
            ),
        ]
    )


def _trigger_log(
    *,
    status: WorkflowTriggerStatus,
    execution_owner_id: str | None = None,
    execution_lease_expires_at: datetime | None = None,
) -> WorkflowTriggerLog:
    trigger_data = TriggerData(
        app_id="app-123",
        tenant_id="tenant-123",
        workflow_id="workflow-123",
        root_node_id="root-node-123",
        inputs={"name": "dify"},
        files=[],
        trigger_type=AppTriggerType.UNKNOWN,
        trigger_from=WorkflowRunTriggeredFrom.APP_RUN,
        trigger_metadata=None,
    )
    trigger_log = WorkflowTriggerLog(
        tenant_id=trigger_data.tenant_id,
        app_id=trigger_data.app_id,
        workflow_id=trigger_data.workflow_id or "workflow-123",
        workflow_run_id=None,
        root_node_id=trigger_data.root_node_id,
        trigger_metadata="{}",
        trigger_type=trigger_data.trigger_type,
        trigger_data=trigger_data.model_dump_json(),
        inputs=json.dumps(dict(trigger_data.inputs)),
        outputs=None,
        status=status,
        error=None,
        queue_name="sandbox",
        celery_task_id="task-123",
        created_by_role=CreatorUserRole.END_USER,
        created_by="end-user-123",
        execution_owner_id=execution_owner_id,
        execution_lease_expires_at=execution_lease_expires_at,
        retry_count=0,
        elapsed_time=None,
        total_tokens=None,
        triggered_at=None,
        finished_at=None,
    )
    trigger_log.id = "trigger-log-123"
    return trigger_log


def test_active_execution_lease_rejects_duplicate_delivery(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    lease_duration = timedelta(minutes=5)
    with sqlite_session_factory() as setup_session:
        setup_session.add(_trigger_log(status=WorkflowTriggerStatus.QUEUED))
        setup_session.commit()

    with sqlite_session_factory() as first_delivery:
        claimed = SQLAlchemyWorkflowTriggerLogRepository(first_delivery).claim_execution(
            "trigger-log-123",
            owner_id="owner-1",
            lease_duration=lease_duration,
        )
        assert claimed is not None
        first_delivery.commit()

    with sqlite_session_factory() as duplicate_delivery:
        claimed_again = SQLAlchemyWorkflowTriggerLogRepository(duplicate_delivery).claim_execution(
            "trigger-log-123",
            owner_id="owner-2",
            lease_duration=lease_duration,
        )
        duplicate_delivery.rollback()

    assert claimed_again is None


def test_heartbeat_delays_takeover_until_renewed_lease_expires(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    lease_duration = timedelta(seconds=30)
    with sqlite_session_factory() as setup_session:
        setup_session.add(_trigger_log(status=WorkflowTriggerStatus.QUEUED))
        setup_session.commit()

    with sqlite_session_factory() as first_delivery:
        repository = SQLAlchemyWorkflowTriggerLogRepository(first_delivery)
        assert repository.claim_execution(
            "trigger-log-123",
            owner_id="owner-1",
            lease_duration=lease_duration,
        )
        first_delivery.commit()

    with sqlite_session_factory() as heartbeat_session:
        renewed = SQLAlchemyWorkflowTriggerLogRepository(heartbeat_session).renew_execution_lease(
            "trigger-log-123",
            owner_id="owner-1",
            lease_duration=lease_duration,
        )
        heartbeat_session.commit()
    assert renewed

    with sqlite_session_factory() as early_redelivery:
        claim = SQLAlchemyWorkflowTriggerLogRepository(early_redelivery).claim_execution(
            "trigger-log-123",
            owner_id="owner-2",
            lease_duration=lease_duration,
        )
        early_redelivery.rollback()
    assert claim is None

    with sqlite_session_factory.begin() as expire_session:
        trigger_log = expire_session.get(WorkflowTriggerLog, "trigger-log-123")
        assert trigger_log is not None
        trigger_log.execution_lease_expires_at = datetime(2000, 1, 1)
        trigger_log.terminal_owner_id = "previous-owner"

    with sqlite_session_factory() as late_redelivery:
        claim = SQLAlchemyWorkflowTriggerLogRepository(late_redelivery).claim_execution(
            "trigger-log-123",
            owner_id="owner-2",
            lease_duration=lease_duration,
        )
        late_redelivery.commit()
    assert claim is not None
    assert claim.execution_owner_id == "owner-2"
    assert claim.terminal_owner_id is None


def test_expired_lease_takeover_fences_previous_owner_terminal_write(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    lease_duration = timedelta(seconds=30)
    with sqlite_session_factory() as setup_session:
        setup_session.add(_trigger_log(status=WorkflowTriggerStatus.QUEUED))
        setup_session.commit()

    with sqlite_session_factory() as first_delivery:
        assert SQLAlchemyWorkflowTriggerLogRepository(first_delivery).claim_execution(
            "trigger-log-123",
            owner_id="owner-1",
            lease_duration=lease_duration,
        )
        first_delivery.commit()

    with sqlite_session_factory.begin() as expire_session:
        trigger_log = expire_session.get(WorkflowTriggerLog, "trigger-log-123")
        assert trigger_log is not None
        trigger_log.execution_lease_expires_at = datetime(2000, 1, 1)

    with sqlite_session_factory() as redelivery:
        assert SQLAlchemyWorkflowTriggerLogRepository(redelivery).claim_execution(
            "trigger-log-123",
            owner_id="owner-2",
            lease_duration=lease_duration,
        )
        redelivery.commit()

    with sqlite_session_factory() as stale_worker:
        repository = SQLAlchemyWorkflowTriggerLogRepository(stale_worker)
        assert not repository.renew_execution_lease(
            "trigger-log-123",
            owner_id="owner-1",
            lease_duration=lease_duration,
        )
        assert not repository.finish_execution(
            "trigger-log-123",
            owner_id="owner-1",
            status=WorkflowTriggerStatus.SUCCEEDED,
            workflow_run_id="stale-run",
            outputs='{"stale":true}',
            error=None,
            elapsed_time=31,
            total_tokens=1,
            finished_at=datetime(2026, 10, 22, 12, 0),
        )
        assert not repository.fail_execution(
            "trigger-log-123",
            owner_id="owner-1",
            error="late failure",
            elapsed_time=31,
            finished_at=datetime(2026, 10, 22, 12, 0),
        )
        stale_worker.commit()

    with sqlite_session_factory() as verification:
        trigger_log = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert trigger_log is not None
        assert trigger_log.status == WorkflowTriggerStatus.RUNNING
        assert trigger_log.execution_owner_id == "owner-2"
        assert trigger_log.error is None


def test_heartbeat_stops_execution_after_ownership_is_lost(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory() as setup_session:
        setup_session.add(
            _trigger_log(
                status=WorkflowTriggerStatus.RUNNING,
                execution_owner_id="replacement-owner",
                execution_lease_expires_at=datetime(2100, 1, 1),
            )
        )
        setup_session.commit()

    stopped = Event()
    heartbeat = async_workflow_tasks._WorkflowExecutionLeaseHeartbeat(
        database_client=sqlite_session_factory,
        trigger_log_id="trigger-log-123",
        owner_id="stale-owner",
        lease_duration=timedelta(seconds=3),
        last_confirmed_at=monotonic(),
        on_ownership_lost=stopped.set,
    )

    with heartbeat:
        assert stopped.wait(timeout=2)


@pytest.mark.parametrize(
    ("event", "expected_status"),
    [
        (GraphRunPausedEvent(), WorkflowTriggerStatus.PAUSED),
        (GraphRunSucceededEvent(), WorkflowTriggerStatus.SUCCEEDED),
        (GraphRunFailedEvent(error="expected failure"), WorkflowTriggerStatus.FAILED),
    ],
)
def test_heartbeat_stops_quietly_after_trigger_post_layer_releases_execution(
    sqlite_session_factory: sessionmaker[Session],
    event: GraphRunPausedEvent | GraphRunSucceededEvent | GraphRunFailedEvent,
    expected_status: WorkflowTriggerStatus,
) -> None:
    with sqlite_session_factory.begin() as setup_session:
        setup_session.add(
            _trigger_log(
                status=WorkflowTriggerStatus.RUNNING,
                execution_owner_id="owner-1",
                execution_lease_expires_at=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

    runtime_state = RuntimeState(
        workflow_id="workflow-123",
        variable_pool=VariablePool.from_bootstrap(
            system_variables=build_system_variables(workflow_execution_id="workflow-run-123")
        ),
        start_at=0,
        outputs={"answer": "done"},
    )
    layer = async_workflow_tasks.TriggerPostLayer(
        datetime.now(UTC),
        "trigger-log-123",
        "owner-1",
    )
    layer.initialize(
        cast(ReadOnlyRuntimeState, runtime_state),
        CelerySignalCommandChannel(shutdown_state_getter=lambda: False, abort_reason="worker shutdown"),
    )

    stopped = Event()
    heartbeat = async_workflow_tasks._WorkflowExecutionLeaseHeartbeat(
        database_client=sqlite_session_factory,
        trigger_log_id="trigger-log-123",
        owner_id="owner-1",
        lease_duration=timedelta(milliseconds=300),
        last_confirmed_at=monotonic(),
        on_ownership_lost=stopped.set,
    )

    with heartbeat:
        layer.on_event(event)
        # Cross both a renewal interval and the original confirmed deadline.
        # A terminal-state-unaware heartbeat would inject a user-manual stop.
        assert not stopped.wait(timeout=0.45)

    with sqlite_session_factory() as verification:
        trigger_log = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert trigger_log is not None
        assert trigger_log.status == expected_status
        assert trigger_log.execution_owner_id is None
        assert trigger_log.execution_lease_expires_at is None
        assert trigger_log.terminal_owner_id == "owner-1"


def test_resume_keeps_pause_receipt_until_the_resumed_execution_finishes(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    trigger_log = _trigger_log(status=WorkflowTriggerStatus.PAUSED)
    trigger_log.workflow_run_id = "workflow-run-123"
    trigger_log.terminal_owner_id = "owner-1"
    with sqlite_session_factory.begin() as setup_session:
        setup_session.add(trigger_log)

    # Production clears task-level cancellation signals before resolving the
    # trigger layer for resume. Neither operation may erase the pause receipt:
    # the old heartbeat can still be winding down at this point.
    clear_app_task_cancellation_signals("workflow-task-123")
    layers = _get_trigger_resume_layers(sqlite_session_factory, "workflow-run-123")
    assert len(layers) == 1

    stopped = Event()
    stale_heartbeat = async_workflow_tasks._WorkflowExecutionLeaseHeartbeat(
        database_client=sqlite_session_factory,
        trigger_log_id="trigger-log-123",
        owner_id="owner-1",
        lease_duration=timedelta(seconds=3),
        last_confirmed_at=monotonic(),
        on_ownership_lost=stopped.set,
    )
    state = stale_heartbeat._read_current_lease_state()
    stale_heartbeat._reconcile_current_lease_state(state, observed_at=monotonic())
    assert not stopped.is_set()

    runtime_state = RuntimeState(
        workflow_id="workflow-123",
        variable_pool=VariablePool.from_bootstrap(
            system_variables=build_system_variables(workflow_execution_id="workflow-run-123")
        ),
        start_at=0,
        outputs={"answer": "resumed"},
    )
    resume_layer = layers[0]
    assert isinstance(resume_layer, async_workflow_tasks.TriggerPostLayer)
    resume_layer.initialize(
        cast(ReadOnlyRuntimeState, runtime_state),
        CelerySignalCommandChannel(shutdown_state_getter=lambda: False, abort_reason="worker shutdown"),
    )
    resume_layer.on_event(GraphRunSucceededEvent())

    with sqlite_session_factory() as verification:
        resumed = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert resumed is not None
        assert resumed.status == WorkflowTriggerStatus.SUCCEEDED
        assert resumed.outputs == '{"answer":"resumed"}'
        assert resumed.terminal_owner_id == "owner-1"


def test_heartbeat_stops_when_replacement_owner_commits_terminal_state(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as setup_session:
        setup_session.add(
            _trigger_log(
                status=WorkflowTriggerStatus.RUNNING,
                execution_owner_id="replacement-owner",
                execution_lease_expires_at=datetime.now(UTC) + timedelta(seconds=3),
            )
        )

    runtime_state = RuntimeState(
        workflow_id="workflow-123",
        variable_pool=VariablePool.from_bootstrap(
            system_variables=build_system_variables(workflow_execution_id="replacement-run")
        ),
        start_at=0,
        outputs={"answer": "replacement"},
    )
    replacement_layer = async_workflow_tasks.TriggerPostLayer(
        datetime.now(UTC),
        "trigger-log-123",
        "replacement-owner",
    )
    replacement_layer.initialize(
        cast(ReadOnlyRuntimeState, runtime_state),
        CelerySignalCommandChannel(shutdown_state_getter=lambda: False, abort_reason="worker shutdown"),
    )

    stopped = Event()
    stale_heartbeat = async_workflow_tasks._WorkflowExecutionLeaseHeartbeat(
        database_client=sqlite_session_factory,
        trigger_log_id="trigger-log-123",
        owner_id="stale-owner",
        lease_duration=timedelta(seconds=3),
        last_confirmed_at=monotonic(),
        on_ownership_lost=stopped.set,
    )

    replacement_layer.on_event(GraphRunSucceededEvent())
    state = stale_heartbeat._read_current_lease_state()
    stale_heartbeat._reconcile_current_lease_state(state, observed_at=monotonic())

    assert stopped.is_set()
    with sqlite_session_factory() as verification:
        trigger_log = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert trigger_log is not None
        assert trigger_log.status == WorkflowTriggerStatus.SUCCEEDED
        assert trigger_log.workflow_run_id == "replacement-run"
        assert trigger_log.terminal_owner_id == "replacement-owner"


def test_heartbeat_stops_execution_when_renewal_errors_outlive_the_confirmed_lease(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    first_renewal_succeeded = Event()
    renewal_attempts = 0

    def renew_then_fail(
        _repository: SQLAlchemyWorkflowTriggerLogRepository,
        _trigger_log_id: str,
        *,
        owner_id: str,
        lease_duration: timedelta,
    ) -> bool:
        nonlocal renewal_attempts
        assert owner_id == "owner-1"
        assert lease_duration == timedelta(milliseconds=300)
        renewal_attempts += 1
        if renewal_attempts == 1:
            first_renewal_succeeded.set()
            return True
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(SQLAlchemyWorkflowTriggerLogRepository, "renew_execution_lease", renew_then_fail)
    stopped = Event()
    heartbeat = async_workflow_tasks._WorkflowExecutionLeaseHeartbeat(
        database_client=sqlite_session_factory,
        trigger_log_id="trigger-log-123",
        owner_id="owner-1",
        lease_duration=timedelta(milliseconds=300),
        last_confirmed_at=monotonic(),
        on_ownership_lost=stopped.set,
    )

    with heartbeat:
        assert first_renewal_succeeded.wait(timeout=1)
        assert stopped.wait(timeout=1)

    assert renewal_attempts >= 2


def test_heartbeat_deadline_stops_execution_while_renewal_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    renewal_started = Event()
    release_renewal = Event()

    def blocked_renewal(
        _repository: SQLAlchemyWorkflowTriggerLogRepository,
        _trigger_log_id: str,
        *,
        owner_id: str,
        lease_duration: timedelta,
    ) -> bool:
        assert owner_id == "owner-1"
        assert lease_duration == timedelta(milliseconds=300)
        renewal_started.set()
        assert release_renewal.wait(timeout=1)
        return True

    monkeypatch.setattr(SQLAlchemyWorkflowTriggerLogRepository, "renew_execution_lease", blocked_renewal)
    stopped = Event()
    heartbeat = async_workflow_tasks._WorkflowExecutionLeaseHeartbeat(
        database_client=sqlite_session_factory,
        trigger_log_id="trigger-log-123",
        owner_id="owner-1",
        lease_duration=timedelta(milliseconds=300),
        last_confirmed_at=monotonic(),
        on_ownership_lost=stopped.set,
    )

    with heartbeat:
        assert renewal_started.wait(timeout=1)
        assert stopped.wait(timeout=1)
        release_renewal.set()


def test_pending_old_publish_retries_until_producer_queues_then_executes_once(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as setup_session:
        _persist_execution_dependencies(setup_session)
        setup_session.add(_trigger_log(status=WorkflowTriggerStatus.PENDING))

    task_data = WorkflowTaskData(workflow_trigger_log_id="trigger-log-123")
    external_retrieval = cast(ExternalDatasetRetriever, object())
    services = _ApplicationServicesStub(knowledge=_KnowledgeServicesStub(external_retrieval=external_retrieval))
    monkeypatch.setattr("extensions.ext_application_services.application_services", lambda: services)

    original_get_by_id = SQLAlchemyWorkflowTriggerLogRepository.get_by_id
    interleavings: list[str] = []

    def queue_before_fallback_read(
        repository: SQLAlchemyWorkflowTriggerLogRepository,
        trigger_log_id: str,
        tenant_id: str | None = None,
    ) -> WorkflowTriggerLog | None:
        if not interleavings:
            # SQLite serializes writers even when the worker's UPDATE matched
            # no rows. Release that no-op statement transaction so a second,
            # real Session can model the producer commit that PostgreSQL/MySQL
            # permit between the failed claim and fallback current-state read.
            repository.session.rollback()
            with sqlite_session_factory.begin() as producer_session:
                pending_log = producer_session.get(WorkflowTriggerLog, trigger_log_id)
                assert pending_log is not None
                assert pending_log.status == WorkflowTriggerStatus.PENDING
                assert pending_log.execution_owner_id is None
                queued = SQLAlchemyWorkflowTriggerLogRepository(producer_session).queue_pending_execution(
                    trigger_log_id,
                    celery_task_id="task-123",
                    triggered_at=datetime.now(UTC),
                )
                assert queued
            interleavings.append("producer-queued")
        return original_get_by_id(repository, trigger_log_id, tenant_id)

    monkeypatch.setattr(SQLAlchemyWorkflowTriggerLogRepository, "get_by_id", queue_before_fallback_read)

    retry_requests: list[tuple[int, int | None]] = []

    class RetryRequestedError(Exception):
        pass

    def retry_task(*, countdown: int, max_retries: int | None) -> None:
        retry_requests.append((countdown, max_retries))
        raise RetryRequestedError

    monkeypatch.setattr(async_workflow_tasks.execute_workflow_sandbox, "retry", retry_task)

    with pytest.raises(RetryRequestedError):
        async_workflow_tasks.execute_workflow_sandbox.run(task_data.model_dump(mode="json"))
    assert retry_requests == [(1, None)]
    assert interleavings == ["producer-queued"]

    executions: list[str] = []

    class CompletingWorkflowGenerator:
        def __init__(self, *, execution_driver: object, external_retrieval: ExternalDatasetRetriever) -> None:
            _ = execution_driver
            assert external_retrieval is external_retrieval_instance

        def generate(self, *, graph_engine_layers: list[object], **_kwargs: object) -> None:
            executions.append("executed")
            assert len(graph_engine_layers) == 1
            trigger_layer = graph_engine_layers[0]
            assert isinstance(trigger_layer, async_workflow_tasks.TriggerPostLayer)
            runtime_state = RuntimeState(
                workflow_id="workflow-123",
                variable_pool=VariablePool.from_bootstrap(
                    system_variables=build_system_variables(workflow_execution_id="workflow-run-123")
                ),
                start_at=0,
                outputs={"answer": "done"},
            )
            trigger_layer.initialize(
                cast(ReadOnlyRuntimeState, runtime_state),
                CelerySignalCommandChannel(shutdown_state_getter=lambda: False, abort_reason="worker shutdown"),
            )
            trigger_layer.on_event(GraphRunSucceededEvent())

    external_retrieval_instance = external_retrieval
    monkeypatch.setattr(async_workflow_tasks, "WorkflowAppGenerator", CompletingWorkflowGenerator)

    async_workflow_tasks.execute_workflow_sandbox.run(task_data.model_dump(mode="json"))

    assert executions == ["executed"]
    with sqlite_session_factory() as verification:
        completed = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert completed is not None
        assert completed.status == WorkflowTriggerStatus.SUCCEEDED
        assert completed.workflow_run_id == "workflow-run-123"
        assert completed.execution_owner_id is None
        assert completed.terminal_owner_id is not None


def test_aged_pending_dispatch_fails_and_late_legacy_queue_write_remains_fenced(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    trigger_log = _trigger_log(status=WorkflowTriggerStatus.PENDING)
    trigger_log.created_at = datetime(2000, 1, 1)
    with sqlite_session_factory.begin() as setup_session:
        setup_session.add(trigger_log)

    external_retrieval = cast(ExternalDatasetRetriever, object())
    services = _ApplicationServicesStub(knowledge=_KnowledgeServicesStub(external_retrieval=external_retrieval))
    monkeypatch.setattr("extensions.ext_application_services.application_services", lambda: services)

    def unexpected_retry(*, countdown: int, max_retries: int | None) -> None:
        pytest.fail(f"abandoned PENDING dispatch retried with {countdown=}, {max_retries=}")

    monkeypatch.setattr(async_workflow_tasks.execute_workflow_sandbox, "retry", unexpected_retry)
    task_data = WorkflowTaskData(workflow_trigger_log_id="trigger-log-123").model_dump(mode="json")

    with caplog.at_level(logging.ERROR, logger="tasks.async_workflow_tasks"):
        async_workflow_tasks.execute_workflow_sandbox.run(task_data)

    with sqlite_session_factory() as verification:
        failed = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert failed is not None
        assert failed.status == WorkflowTriggerStatus.FAILED
        assert failed.error == async_workflow_tasks._ABANDONED_PENDING_EXECUTION_ERROR
        assert failed.finished_at is not None
        assert failed.pending_dispatch_abandoned_at is not None
    assert "Abandoned async workflow dispatch was marked failed" in caplog.text

    with sqlite_session_factory() as current_producer:
        queued = SQLAlchemyWorkflowTriggerLogRepository(current_producer).queue_pending_execution(
            "trigger-log-123",
            celery_task_id="current-task-id",
            triggered_at=datetime.now(UTC),
        )
        current_producer.rollback()
    assert not queued

    # A rolling-deployment legacy producer does not know about the abandonment
    # marker and tries its old unconditional status write. The database fence
    # rejects the whole transition, preserving the observable FAILED result.
    def write_late_legacy_queue_state() -> None:
        with sqlite_session_factory.begin() as legacy_producer:
            late = legacy_producer.get(WorkflowTriggerLog, "trigger-log-123")
            assert late is not None
            late.status = WorkflowTriggerStatus.QUEUED
            late.celery_task_id = "legacy-task-id"
            late.triggered_at = datetime.now(UTC)

    with pytest.raises(IntegrityError):
        write_late_legacy_queue_state()

    async_workflow_tasks.execute_workflow_sandbox.run(task_data)

    with sqlite_session_factory() as verification:
        reconverged = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert reconverged is not None
        assert reconverged.status == WorkflowTriggerStatus.FAILED
        assert reconverged.error == async_workflow_tasks._ABANDONED_PENDING_EXECUTION_ERROR
        assert reconverged.execution_owner_id is None
        assert reconverged.pending_dispatch_abandoned_at is not None


def test_periodic_reconciliation_fails_unpublished_queued_dispatch_and_refunds_quota(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as setup_session:
        setup_session.add(_trigger_log(status=WorkflowTriggerStatus.PENDING))
    with sqlite_session_factory.begin() as prepare_session:
        repository = SQLAlchemyWorkflowTriggerLogRepository(prepare_session)
        assert repository.queue_pending_execution(
            "trigger-log-123",
            celery_task_id="unpublished-task",
            triggered_at=datetime.now(UTC),
            quota_reservation_id="reservation-123",
        )
        trigger_log = prepare_session.get_one(WorkflowTriggerLog, "trigger-log-123")
        trigger_log.dispatch_prepared_at = datetime(2000, 1, 1)

    settlements: list[WorkflowQuotaSettlementCandidate] = []

    def settle(candidate: WorkflowQuotaSettlementCandidate) -> Literal["committed", "refunded"]:
        settlements.append(candidate)
        return candidate.outcome

    monkeypatch.setattr(async_workflow_tasks.WorkflowTriggerQuotaService, "settle", staticmethod(settle))

    async_workflow_tasks.reconcile_async_workflow_dispatches.run()

    assert len(settlements) == 1
    assert settlements[0].reservation_id == "reservation-123"
    assert settlements[0].outcome == "refunded"
    with sqlite_session_factory() as verification:
        failed = verification.get_one(WorkflowTriggerLog, "trigger-log-123")
        assert failed.status == WorkflowTriggerStatus.FAILED
        assert failed.error == async_workflow_tasks._ABANDONED_QUEUED_DISPATCH_ERROR
        assert failed.dispatch_published_at is None
        assert failed.dispatch_abandoned_at is not None
        assert failed.finished_at is not None
        assert failed.quota_reservation_id == "reservation-123"
        assert failed.quota_settlement_status == "refunded"
        assert failed.quota_sync_pending is False
        assert not SQLAlchemyWorkflowTriggerLogRepository(verification).confirm_dispatch_published(
            "trigger-log-123",
            celery_task_id="unpublished-task",
        )


def test_worker_claim_wins_over_unpublished_dispatch_reaper_and_reconciler_commits_quota(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as setup_session:
        setup_session.add(_trigger_log(status=WorkflowTriggerStatus.PENDING))
    with sqlite_session_factory.begin() as prepare_session:
        repository = SQLAlchemyWorkflowTriggerLogRepository(prepare_session)
        assert repository.queue_pending_execution(
            "trigger-log-123",
            celery_task_id="delivered-task",
            triggered_at=datetime.now(UTC),
            quota_reservation_id="reservation-123",
        )
        trigger_log = prepare_session.get_one(WorkflowTriggerLog, "trigger-log-123")
        trigger_log.dispatch_prepared_at = datetime(2000, 1, 1)

    settlements: list[WorkflowQuotaSettlementCandidate] = []

    def settle(candidate: WorkflowQuotaSettlementCandidate) -> Literal["committed", "refunded"]:
        settlements.append(candidate)
        return candidate.outcome

    monkeypatch.setattr(async_workflow_tasks.WorkflowTriggerQuotaService, "settle", staticmethod(settle))

    with sqlite_session_factory() as worker_session:
        repository = SQLAlchemyWorkflowTriggerLogRepository(worker_session)
        claimed = repository.claim_execution(
            "trigger-log-123",
            owner_id="worker-owner",
            lease_duration=timedelta(minutes=5),
        )
        assert claimed is not None
        worker_session.commit()

    with sqlite_session_factory() as pending_verification:
        pending = pending_verification.get_one(WorkflowTriggerLog, "trigger-log-123")
        assert pending.status == WorkflowTriggerStatus.RUNNING
        assert pending.quota_sync_pending is True
        assert pending.quota_settlement_status is None

    async_workflow_tasks.reconcile_async_workflow_dispatches.run()

    assert len(settlements) == 1
    assert settlements[0].outcome == "committed"
    with sqlite_session_factory() as verification:
        running = verification.get_one(WorkflowTriggerLog, "trigger-log-123")
        assert running.status == WorkflowTriggerStatus.RUNNING
        assert running.execution_owner_id == "worker-owner"
        assert running.dispatch_published_at is None
        assert running.dispatch_abandoned_at is None
        assert running.quota_settlement_status == "committed"
        assert running.quota_sync_pending is False


def test_claimed_workflow_executes_before_durable_quota_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as setup_session:
        _persist_execution_dependencies(setup_session)
        setup_session.add(_trigger_log(status=WorkflowTriggerStatus.PENDING))
    with sqlite_session_factory.begin() as dispatch_session:
        repository = SQLAlchemyWorkflowTriggerLogRepository(dispatch_session)
        assert repository.queue_pending_execution(
            "trigger-log-123",
            celery_task_id="delivered-task",
            triggered_at=datetime.now(UTC),
            quota_reservation_id="reservation-123",
        )
        assert repository.confirm_dispatch_published(
            "trigger-log-123",
            celery_task_id="delivered-task",
        )

    synchronous_settlement_attempts: list[str] = []

    def fail_synchronous_settlement(
        _repository: SQLAlchemyWorkflowTriggerLogRepository,
        trigger_log_id: str,
        *,
        retry_interval: timedelta,
    ) -> WorkflowQuotaSettlementCandidate | None:
        _ = retry_interval
        synchronous_settlement_attempts.append(trigger_log_id)
        raise OperationalError("quota settlement database unavailable", {}, RuntimeError("database unavailable"))

    monkeypatch.setattr(
        SQLAlchemyWorkflowTriggerLogRepository,
        "claim_quota_settlement",
        fail_synchronous_settlement,
    )

    preexisting_thread_ids = {id(thread) for thread in threading.enumerate()}
    heartbeat_observations: list[bool] = []
    executions: list[str] = []

    class CompletingWorkflowGenerator:
        def __init__(self, *, execution_driver: object, external_retrieval: ExternalDatasetRetriever) -> None:
            _ = execution_driver
            assert external_retrieval is external_retrieval_instance

        def generate(self, *, graph_engine_layers: list[object], **_kwargs: object) -> None:
            executions.append("executed")
            heartbeat_observations.append(
                any(
                    thread.name == "workflow-lease-renew-trigger-log-123" and id(thread) not in preexisting_thread_ids
                    for thread in threading.enumerate()
                )
            )
            assert len(graph_engine_layers) == 1
            trigger_layer = graph_engine_layers[0]
            assert isinstance(trigger_layer, async_workflow_tasks.TriggerPostLayer)
            runtime_state = RuntimeState(
                workflow_id="workflow-123",
                variable_pool=VariablePool.from_bootstrap(
                    system_variables=build_system_variables(workflow_execution_id="workflow-run-123")
                ),
                start_at=0,
                outputs={"answer": "done"},
            )
            trigger_layer.initialize(
                cast(ReadOnlyRuntimeState, runtime_state),
                CelerySignalCommandChannel(shutdown_state_getter=lambda: False, abort_reason="worker shutdown"),
            )
            trigger_layer.on_event(GraphRunSucceededEvent())

    external_retrieval_instance = cast(ExternalDatasetRetriever, object())
    monkeypatch.setattr(async_workflow_tasks, "WorkflowAppGenerator", CompletingWorkflowGenerator)

    async_workflow_tasks._execute_workflow_common(
        WorkflowTaskData(workflow_trigger_log_id="trigger-log-123"),
        external_retrieval=external_retrieval_instance,
    )

    assert synchronous_settlement_attempts == []
    assert executions == ["executed"]
    assert heartbeat_observations == [True]
    with sqlite_session_factory() as pending_verification:
        completed = pending_verification.get_one(WorkflowTriggerLog, "trigger-log-123")
        assert completed.status == WorkflowTriggerStatus.SUCCEEDED
        assert completed.quota_sync_pending is True
        assert completed.quota_settlement_status is None

    settlements: list[WorkflowQuotaSettlementCandidate] = []

    def settle(candidate: WorkflowQuotaSettlementCandidate) -> Literal["committed", "refunded"]:
        settlements.append(candidate)
        return candidate.outcome

    monkeypatch.setattr(async_workflow_tasks.WorkflowTriggerQuotaService, "settle", staticmethod(settle))
    async_workflow_tasks.reconcile_async_workflow_dispatches.run()

    assert len(settlements) == 1
    assert settlements[0].trigger_log_id == "trigger-log-123"
    assert settlements[0].outcome == "committed"
    with sqlite_session_factory() as settled_verification:
        settled = settled_verification.get_one(WorkflowTriggerLog, "trigger-log-123")
        assert settled.status == WorkflowTriggerStatus.SUCCEEDED
        assert settled.quota_sync_pending is False
        assert settled.quota_settlement_status == "committed"


def test_quota_settlement_attempt_rotation_does_not_starve_later_candidates(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    first = _trigger_log(status=WorkflowTriggerStatus.PENDING)
    first.id = "trigger-a"
    second = _trigger_log(status=WorkflowTriggerStatus.PENDING)
    second.id = "trigger-b"
    with sqlite_session_factory.begin() as setup_session:
        setup_session.add_all([first, second])
    with sqlite_session_factory.begin() as prepare_session:
        repository = SQLAlchemyWorkflowTriggerLogRepository(prepare_session)
        for trigger_log_id in ("trigger-a", "trigger-b"):
            assert repository.queue_pending_execution(
                trigger_log_id,
                celery_task_id=f"task-{trigger_log_id}",
                triggered_at=datetime.now(UTC),
                quota_reservation_id=f"reservation-{trigger_log_id}",
            )
            assert repository.confirm_dispatch_published(
                trigger_log_id,
                celery_task_id=f"task-{trigger_log_id}",
            )

    with sqlite_session_factory.begin() as first_attempt:
        first_batch = SQLAlchemyWorkflowTriggerLogRepository(first_attempt).claim_quota_settlement_candidates(
            retry_interval=timedelta(minutes=1),
            limit=1,
        )
    with sqlite_session_factory.begin() as second_attempt:
        second_batch = SQLAlchemyWorkflowTriggerLogRepository(second_attempt).claim_quota_settlement_candidates(
            retry_interval=timedelta(minutes=1),
            limit=1,
        )

    assert [candidate.trigger_log_id for candidate in first_batch] == ["trigger-a"]
    assert [candidate.trigger_log_id for candidate in second_batch] == ["trigger-b"]
    with sqlite_session_factory() as verification:
        assert verification.get_one(WorkflowTriggerLog, "trigger-a").quota_sync_pending is True
        assert verification.get_one(WorkflowTriggerLog, "trigger-b").quota_sync_pending is True


@pytest.mark.parametrize(
    "status",
    [
        WorkflowTriggerStatus.PAUSED,
        WorkflowTriggerStatus.SUCCEEDED,
        WorkflowTriggerStatus.FAILED,
        WorkflowTriggerStatus.RATE_LIMITED,
        WorkflowTriggerStatus.RETRYING,
    ],
)
def test_non_executable_delivery_is_acknowledged_without_retry(
    sqlite_session_factory: sessionmaker[Session],
    status: WorkflowTriggerStatus,
) -> None:
    with sqlite_session_factory.begin() as setup_session:
        setup_session.add(_trigger_log(status=status))

    async_workflow_tasks._execute_workflow_common(
        WorkflowTaskData(workflow_trigger_log_id="trigger-log-123"),
        external_retrieval=cast(ExternalDatasetRetriever, object()),
    )

    with sqlite_session_factory() as verification:
        unchanged = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert unchanged is not None
        assert unchanged.status == status
        assert unchanged.execution_owner_id is None


def test_duplicate_worker_delivery_does_not_execute_workflow(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
) -> None:
    sqlite_session.add(
        _trigger_log(
            status=WorkflowTriggerStatus.RUNNING,
            execution_owner_id="active-owner",
            execution_lease_expires_at=datetime.now() + timedelta(minutes=5),
        )
    )
    sqlite_session.commit()
    generator = create_autospec(async_workflow_tasks.WorkflowAppGenerator, spec_set=True)
    monkeypatch.setattr(async_workflow_tasks, "WorkflowAppGenerator", generator)

    with pytest.raises(async_workflow_tasks._WorkflowExecutionLeaseActiveError) as error:
        async_workflow_tasks._execute_workflow_common(
            WorkflowTaskData(workflow_trigger_log_id="trigger-log-123"),
            external_retrieval=create_autospec(ExternalDatasetRetriever, instance=True, spec_set=True),
        )

    generator.assert_not_called()
    assert error.value.retry_after > 0
    sqlite_session.expire_all()
    trigger_log = sqlite_session.get(WorkflowTriggerLog, "trigger-log-123")
    assert trigger_log is not None
    assert trigger_log.status == WorkflowTriggerStatus.RUNNING
    assert trigger_log.execution_owner_id == "active-owner"


@pytest.mark.parametrize(
    "task",
    [
        async_workflow_tasks.execute_workflow_professional,
        async_workflow_tasks.execute_workflow_team,
        async_workflow_tasks.execute_workflow_sandbox,
    ],
)
def test_async_workflow_tasks_requeue_messages_when_worker_is_lost(task: Task[..., object]) -> None:
    assert task.acks_late is True
    assert task.reject_on_worker_lost is True
    assert task.max_retries is None


@pytest.mark.parametrize(
    "task",
    [
        async_workflow_tasks.execute_workflow_professional,
        async_workflow_tasks.execute_workflow_team,
        async_workflow_tasks.execute_workflow_sandbox,
    ],
)
def test_active_execution_lease_retries_delivery_after_lease_expiry(
    monkeypatch: pytest.MonkeyPatch,
    task: Task[..., object],
) -> None:
    retry_calls: list[tuple[int, int | None]] = []

    class RetryRequestedError(Exception):
        pass

    def execute(*_args, **_kwargs) -> None:
        raise async_workflow_tasks._WorkflowExecutionLeaseActiveError(17)

    def retry(*, countdown: int, max_retries: int | None) -> None:
        retry_calls.append((countdown, max_retries))
        raise RetryRequestedError

    external_retrieval = create_autospec(ExternalDatasetRetriever, instance=True, spec_set=True)
    services = _ApplicationServicesStub(knowledge=_KnowledgeServicesStub(external_retrieval=external_retrieval))
    monkeypatch.setattr("extensions.ext_application_services.application_services", lambda: services)
    monkeypatch.setattr(async_workflow_tasks, "_execute_workflow_common", execute)
    monkeypatch.setattr(task, "retry", retry)

    with pytest.raises(RetryRequestedError):
        task.run({"workflow_trigger_log_id": "trigger-log-id"})

    assert retry_calls == [(17, None)]


def test_unknown_pause_publication_retries_without_failing_claimed_trigger(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as setup_session:
        _persist_execution_dependencies(setup_session)
        setup_session.add(_trigger_log(status=WorkflowTriggerStatus.QUEUED))

    external_retrieval = cast(ExternalDatasetRetriever, object())
    services = _ApplicationServicesStub(knowledge=_KnowledgeServicesStub(external_retrieval=external_retrieval))
    monkeypatch.setattr("extensions.ext_application_services.application_services", lambda: services)

    class UnknownPausePublicationGenerator:
        def __init__(self, *, execution_driver: object, external_retrieval: ExternalDatasetRetriever) -> None:
            _ = execution_driver
            assert external_retrieval is external_retrieval_instance

        def generate(self, *, graph_engine_layers: list[object], **_kwargs: object) -> None:
            assert len(graph_engine_layers) == 1
            trigger_layer = graph_engine_layers[0]
            assert isinstance(trigger_layer, async_workflow_tasks.TriggerPostLayer)
            assert trigger_layer.defer_owned_pause()
            trigger_layer.mark_pause_publication_outcome_unknown()

    external_retrieval_instance = external_retrieval
    monkeypatch.setattr(async_workflow_tasks, "WorkflowAppGenerator", UnknownPausePublicationGenerator)

    retry_requests: list[tuple[int, int | None]] = []

    class RetryRequestedError(Exception):
        pass

    def retry_task(*, countdown: int, max_retries: int | None) -> None:
        retry_requests.append((countdown, max_retries))
        raise RetryRequestedError

    monkeypatch.setattr(async_workflow_tasks.execute_workflow_sandbox, "retry", retry_task)

    with pytest.raises(RetryRequestedError):
        async_workflow_tasks.execute_workflow_sandbox.run(
            WorkflowTaskData(workflow_trigger_log_id="trigger-log-123").model_dump(mode="json")
        )

    assert retry_requests == [(async_workflow_tasks.dify_config.ASYNC_WORKFLOW_EXECUTION_LEASE_SECONDS, None)]
    with sqlite_session_factory() as verification:
        trigger_log = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert trigger_log is not None
        assert trigger_log.status == WorkflowTriggerStatus.RUNNING
        assert trigger_log.execution_owner_id is not None
        assert trigger_log.execution_lease_expires_at is not None
        assert trigger_log.error is None
        assert trigger_log.finished_at is None
        assert trigger_log.terminal_owner_id is None


def test_known_generation_completion_does_not_force_pause_retry(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as setup_session:
        _persist_execution_dependencies(setup_session)
        setup_session.add(_trigger_log(status=WorkflowTriggerStatus.QUEUED))

    external_retrieval = cast(ExternalDatasetRetriever, object())
    services = _ApplicationServicesStub(knowledge=_KnowledgeServicesStub(external_retrieval=external_retrieval))
    monkeypatch.setattr("extensions.ext_application_services.application_services", lambda: services)

    class CompletingWorkflowGenerator:
        def __init__(self, *, execution_driver: object, external_retrieval: ExternalDatasetRetriever) -> None:
            _ = execution_driver
            assert external_retrieval is external_retrieval_instance

        def generate(self, *, graph_engine_layers: list[object], **_kwargs: object) -> None:
            assert len(graph_engine_layers) == 1
            trigger_layer = graph_engine_layers[0]
            assert isinstance(trigger_layer, async_workflow_tasks.TriggerPostLayer)
            assert not trigger_layer.pause_publication_outcome_unknown
            runtime_state = RuntimeState(
                workflow_id="workflow-123",
                variable_pool=VariablePool.from_bootstrap(
                    system_variables=build_system_variables(workflow_execution_id="workflow-run-123")
                ),
                start_at=0,
                outputs={"answer": "done"},
            )
            trigger_layer.initialize(
                cast(ReadOnlyRuntimeState, runtime_state),
                CelerySignalCommandChannel(shutdown_state_getter=lambda: False, abort_reason="worker shutdown"),
            )
            trigger_layer.on_event(GraphRunSucceededEvent())

    external_retrieval_instance = external_retrieval
    monkeypatch.setattr(async_workflow_tasks, "WorkflowAppGenerator", CompletingWorkflowGenerator)

    def unexpected_retry(*, countdown: int, max_retries: int | None) -> None:
        pytest.fail(f"known workflow completion retried with {countdown=}, {max_retries=}")

    monkeypatch.setattr(async_workflow_tasks.execute_workflow_sandbox, "retry", unexpected_retry)
    async_workflow_tasks.execute_workflow_sandbox.run(
        WorkflowTaskData(workflow_trigger_log_id="trigger-log-123").model_dump(mode="json")
    )

    with sqlite_session_factory() as verification:
        trigger_log = verification.get(WorkflowTriggerLog, "trigger-log-123")
        assert trigger_log is not None
        assert trigger_log.status == WorkflowTriggerStatus.SUCCEEDED
        assert trigger_log.workflow_run_id == "workflow-run-123"
        assert trigger_log.execution_owner_id is None
        assert trigger_log.terminal_owner_id is not None


def test_invalid_persisted_trigger_data_fails_claimed_trigger_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
) -> None:
    trigger_log = _trigger_log(status=WorkflowTriggerStatus.QUEUED)
    trigger_log.trigger_data = "not-json"
    sqlite_session.add(trigger_log)
    sqlite_session.commit()
    generator = create_autospec(async_workflow_tasks.WorkflowAppGenerator, spec_set=True)
    monkeypatch.setattr(async_workflow_tasks, "WorkflowAppGenerator", generator)

    async_workflow_tasks._execute_workflow_common(
        WorkflowTaskData(workflow_trigger_log_id="trigger-log-123"),
        external_retrieval=create_autospec(ExternalDatasetRetriever, instance=True, spec_set=True),
    )

    generator.assert_not_called()
    sqlite_session.expire_all()
    failed_log = sqlite_session.get(WorkflowTriggerLog, "trigger-log-123")
    assert failed_log is not None
    assert failed_log.status == WorkflowTriggerStatus.FAILED
    assert failed_log.error is not None
    assert "Invalid JSON" in failed_log.error
    assert failed_log.finished_at is not None
    assert failed_log.terminal_owner_id is not None


def test_partial_success_is_terminal_and_duplicate_delivery_does_not_execute_workflow(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(
            _trigger_log(
                status=WorkflowTriggerStatus.RUNNING,
                execution_owner_id="owner-1",
                execution_lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
            )
        )

    runtime_state = RuntimeState(
        workflow_id="workflow-123",
        variable_pool=VariablePool.from_bootstrap(
            system_variables=build_system_variables(workflow_execution_id="workflow-run-123")
        ),
        start_at=0,
        outputs={"answer": "partial"},
    )
    layer = async_workflow_tasks.TriggerPostLayer(
        datetime.now(UTC) - timedelta(seconds=1),
        "trigger-log-123",
        "owner-1",
    )
    layer.initialize(
        cast(ReadOnlyRuntimeState, runtime_state),
        create_autospec(CommandChannel, instance=True, spec_set=True),
    )
    layer.on_event(GraphRunPartialSucceededEvent(exceptions_count=1))

    with sqlite_session_factory() as session:
        trigger_log = session.get(WorkflowTriggerLog, "trigger-log-123")
        assert trigger_log is not None
        assert trigger_log.status == WorkflowTriggerStatus.SUCCEEDED
        assert trigger_log.execution_owner_id is None
        assert trigger_log.execution_lease_expires_at is None

    generator = create_autospec(async_workflow_tasks.WorkflowAppGenerator, spec_set=True)
    monkeypatch.setattr(async_workflow_tasks, "WorkflowAppGenerator", generator)
    async_workflow_tasks._execute_workflow_common(
        WorkflowTaskData(workflow_trigger_log_id="trigger-log-123"),
        external_retrieval=create_autospec(ExternalDatasetRetriever, instance=True, spec_set=True),
    )
    generator.assert_not_called()


@pytest.mark.parametrize(
    "task",
    [
        async_workflow_tasks.execute_workflow_professional,
        async_workflow_tasks.execute_workflow_team,
        async_workflow_tasks.execute_workflow_sandbox,
    ],
)
def test_execute_workflow_task_forwards_external_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    task: Task[..., object],
) -> None:
    external_retrieval = create_autospec(ExternalDatasetRetriever, instance=True, spec_set=True)
    services = _ApplicationServicesStub(knowledge=_KnowledgeServicesStub(external_retrieval=external_retrieval))
    execute = create_autospec(async_workflow_tasks._execute_workflow_common, spec_set=True)
    monkeypatch.setattr("extensions.ext_application_services.application_services", lambda: services)
    monkeypatch.setattr(async_workflow_tasks, "_execute_workflow_common", execute)

    task.run({"workflow_trigger_log_id": "trigger-log-id"})

    assert execute.call_args.kwargs["external_retrieval"] is external_retrieval


def test_workflow_not_found_error_message_is_a_single_formatted_string() -> None:
    # Regression: previously raised with printf-style positional args
    # ("...%s...", id1, id2), which made Exception store them as a tuple
    # and str(exc) rendered the literal tuple instead of a readable message.
    workflow_run_id = "wr-123"
    workflow_id = "wf-456"

    exc = WorkflowNotFoundError(f"Workflow not found: workflow_run_id={workflow_run_id}, workflow_id={workflow_id}")

    # The exception must stringify to the fully-interpolated message and must
    # not contain unresolved %s placeholders nor tuple punctuation.
    rendered = str(exc)
    assert rendered == "Workflow not found: workflow_run_id=wr-123, workflow_id=wf-456"
    assert "%s" not in rendered
    assert not rendered.startswith("(")
    assert exc.args == ("Workflow not found: workflow_run_id=wr-123, workflow_id=wf-456",)

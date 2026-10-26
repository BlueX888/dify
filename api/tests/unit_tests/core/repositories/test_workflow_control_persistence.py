"""Configured history backends retain synchronous, monotonic task control."""

import logging
from collections.abc import Callable

import pytest
from pytest_mock import MockerFixture
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from core.repositories.celery_workflow_execution_repository import CeleryWorkflowExecutionRepository
from core.repositories.sqlalchemy_workflow_execution_repository import SQLAlchemyWorkflowExecutionRepository
from core.repositories.sqlalchemy_workflow_node_execution_repository import SQLAlchemyWorkflowNodeExecutionRepository
from extensions.logstore.aliyun_logstore import AliyunLogStore
from extensions.logstore.repositories.logstore_api_workflow_node_execution_repository import (
    LogstoreAPIWorkflowNodeExecutionRepository,
)
from extensions.logstore.repositories.logstore_api_workflow_run_repository import LogstoreAPIWorkflowRunRepository
from extensions.logstore.repositories.logstore_workflow_execution_repository import LogstoreWorkflowExecutionRepository
from extensions.logstore.repositories.logstore_workflow_node_execution_repository import (
    LogstoreWorkflowNodeExecutionRepository,
)
from graphon.entities import WorkflowNodeExecution
from graphon.enums import WorkflowExecutionStatus, WorkflowNodeExecutionStatus, WorkflowType
from libs.datetime_utils import naive_utc_now
from models import Account, WorkflowRun
from models.enums import CreatorUserRole, WorkflowRunTriggeredFrom
from models.workflow import WorkflowNodeExecutionTriggeredFrom, WorkflowPause
from repositories.entities.workflow_execution import WorkflowExecution
from repositories.sqlalchemy_api_workflow_run_repository import DifyAPISQLAlchemyWorkflowRunRepository
from repositories.workflow_run_control import apply_workflow_stop_to_node
from tasks.workflow_execution_tasks import _update_workflow_run_from_execution


@pytest.fixture
def logstore(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture) -> AliyunLogStore:
    monkeypatch.setattr(AliyunLogStore, "_instance", None)
    monkeypatch.setattr(AliyunLogStore, "_initialized", False)
    monkeypatch.setenv("ALIYUN_SLS_ENDPOINT", "https://example.com")
    mocker.patch.object(AliyunLogStore, "_check_endpoint_connectivity")
    store = AliyunLogStore()
    store._use_pg_protocol = True
    mocker.patch.object(store, "put_log")
    return store


@pytest.mark.parametrize("backend", ["sql", "celery", "logstore"])
def test_task_owner_is_durable_before_start_and_delayed_writes_cannot_revive_stop(
    sqlite_session_factory: sessionmaker[Session], logstore: AliyunLogStore, mocker: MockerFixture, backend: str
) -> None:
    queued_save = mocker.patch(
        "core.repositories.celery_workflow_execution_repository.save_workflow_execution_task.delay"
    )
    get_logs = mocker.patch.object(logstore, "get_logs")
    repository_class = {
        "sql": SQLAlchemyWorkflowExecutionRepository,
        "celery": CeleryWorkflowExecutionRepository,
        "logstore": LogstoreWorkflowExecutionRepository,
    }[backend]
    user = Account(name="Test", email="test@example.com")
    user.id = "account-1"
    repository = repository_class(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    )
    if isinstance(repository, LogstoreWorkflowExecutionRepository):
        repository._enable_dual_write = False
    execution = WorkflowExecution(
        id_="run-1",
        workflow_id="workflow-1",
        workflow_type=WorkflowType.WORKFLOW,
        workflow_version="1",
        graph={"nodes": [], "edges": []},
        inputs={},
        started_at=naive_utc_now(),
    )
    repository.save_synchronously(execution)
    queued_save.assert_not_called()
    control = DifyAPISQLAlchemyWorkflowRunRepository(sqlite_session_factory)
    control.bind_workflow_task(tenant_id="tenant-1", app_id="app-1", workflow_run_id="run-1", task_id="task-1")
    with sqlite_session_factory() as session:
        run = session.get(WorkflowRun, "run-1")
        assert run is not None
        assert run.task_id == "task-1"
        assert (run.created_by_role, run.created_by) == (CreatorUserRole.ACCOUNT, "account-1")
    execution.status = WorkflowExecutionStatus.PAUSED
    repository.save_synchronously(execution)
    stopped = control.stop_paused_workflow_task(tenant_id="tenant-1", app_id="app-1", task_id="task-1", owner=None)
    assert stopped is not None
    assert stopped.status == WorkflowExecutionStatus.STOPPED

    # A writer already queued or running before the stop cannot restore RUNNING.
    execution.status = WorkflowExecutionStatus.RUNNING
    repository.save_synchronously(execution)
    with sqlite_session_factory.begin() as session:
        run = session.get(WorkflowRun, "run-1")
        assert run is not None
        _update_workflow_run_from_execution(run, execution)
    with sqlite_session_factory() as session:
        run = session.get(WorkflowRun, "run-1")
        assert run is not None
        assert run.status == WorkflowExecutionStatus.STOPPED
        assert run.stop_requested_at is not None
    if backend == "logstore":
        # LogStore implements the configured run-control subset of the history protocol.
        api = LogstoreAPIWorkflowRunRepository(sqlite_session_factory)  # pyrefly: ignore[bad-instantiation]
        api._enable_dual_read = False
        found = api.get_workflow_run_by_id("tenant-1", "app-1", "run-1")
        assert found is not None
        assert found.status == WorkflowExecutionStatus.STOPPED
        unscoped = api.get_workflow_run_by_id_without_tenant("run-1")
        assert unscoped is not None
        assert unscoped.status == WorkflowExecutionStatus.STOPPED
        get_logs.assert_not_called()


@pytest.mark.usefixtures("logstore")
def test_logstore_pause_lifecycle_uses_sql_control_with_dual_write_disabled(
    sqlite_session_factory: sessionmaker[Session], mocker: MockerFixture
) -> None:
    mocker.patch("repositories.sqlalchemy_api_workflow_run_repository.storage.save")
    mocker.patch("repositories.sqlalchemy_api_workflow_run_repository.storage.delete")
    user = Account(name="Test", email="test@example.com")
    user.id = "account-1"
    sql = SQLAlchemyWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    )
    sql.save_synchronously(
        WorkflowExecution(
            id_="run-1",
            workflow_id="workflow-1",
            workflow_type=WorkflowType.WORKFLOW,
            workflow_version="1",
            graph={},
            inputs={},
            started_at=naive_utc_now(),
        )
    )
    api = LogstoreAPIWorkflowRunRepository(sqlite_session_factory)  # pyrefly: ignore[bad-instantiation]
    api.bind_workflow_task(tenant_id="tenant-1", app_id="app-1", workflow_run_id="run-1", task_id="task-1")
    pause = api.create_workflow_pause("run-1", "account-1", "{}", [])
    persisted_pause = api.get_workflow_pause("run-1")
    assert persisted_pause is not None
    assert persisted_pause.id == pause.id
    resume_callbacks: list[None] = []
    resumed = api.resume_workflow_pause("run-1", pause, before_resume=lambda: resume_callbacks.append(None))
    assert resume_callbacks == [None]
    with sqlite_session_factory() as session:
        run = session.get(WorkflowRun, "run-1")
        assert run is not None
        assert run.history_revision == 2
        assert run.history_sync_pending is True
    api.delete_workflow_pause(resumed)
    assert api.get_workflow_pause("run-1") is None


@pytest.mark.usefixtures("logstore")
def test_atomic_pause_publishes_logstore_history_without_rewriting_sql_control(
    sqlite_session_factory: sessionmaker[Session],
    logstore: AliyunLogStore,
    config_overrides: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    from core.app.apps import workflow_app_runner
    from core.app.entities.queue_entities import QueueWorkflowPausedEvent
    from core.app.layers.pause_state_persist_layer import PauseStateLayerConfig
    from services.workflow_run_agg import WorkflowRunAgg
    from tests.unit_tests.services.workflow.test_workflow_run_agg import make_workflow_runner

    config_overrides(
        API_WORKFLOW_RUN_REPOSITORY=(
            "extensions.logstore.repositories.logstore_api_workflow_run_repository.LogstoreAPIWorkflowRunRepository"
        )
    )
    mocker.patch("repositories.sqlalchemy_api_workflow_run_repository.storage.save")
    mocker.patch("repositories.sqlalchemy_api_workflow_run_repository.storage.delete")
    monkeypatch.setattr(workflow_app_runner.dispatch_human_input_email_task, "apply_async", lambda **_kwargs: None)
    runner, _sql_execution_repository, node_repository = make_workflow_runner(
        sqlite_session_factory.kw["bind"], human_input=True
    )
    user = Account(name="History Tester", email="history@example.com")
    user.id = runner.application_generate_entity.user_id
    execution_repository = LogstoreWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id=runner.application_generate_entity.app_config.tenant_id,
        user=user,
        app_id=runner.application_generate_entity.app_config.app_id,
        triggered_from=WorkflowRunTriggeredFrom.DEBUGGING,
    )
    execution_repository._enable_dual_write = False
    sql_saves: list[WorkflowExecutionStatus] = []
    original_sql_save = execution_repository.sql_repository.save_with_model_callback

    def save_sql_control(execution: WorkflowExecution, callback: Callable[[Session, WorkflowRun], object]) -> object:
        sql_saves.append(execution.status)
        return original_sql_save(execution, callback)

    monkeypatch.setattr(execution_repository.sql_repository, "save_with_model_callback", save_sql_control)
    records: list[dict[str, str]] = []

    def append_history(_logstore_name: str, fields: list[tuple[str, str]]) -> None:
        records.append(dict(fields))

    monkeypatch.setattr(logstore, "put_log", append_history)

    WorkflowRunAgg.run(
        runner,
        PauseStateLayerConfig(sqlite_session_factory, runner.application_generate_entity.user_id),
        execution_repository,
        node_repository,
    )

    events = [message.event for message in runner._queue_manager.listen()]
    assert isinstance(events[-1], QueueWorkflowPausedEvent), events
    assert sql_saves == [WorkflowExecutionStatus.RUNNING]
    assert [record["status"] for record in records] == ["running", "paused"]
    assert int(records[1]["log_version"]) > int(records[0]["log_version"])
    with sqlite_session_factory() as session:
        run = session.get(WorkflowRun, runner.application_generate_entity.workflow_execution_id)
        assert run is not None
        assert run.status == WorkflowExecutionStatus.PAUSED
        assert run.history_revision == 2
        assert run.history_published_revision == 2
        assert run.history_sync_pending is False

    latest = max(records, key=lambda record: int(record["log_version"]))
    queries: list[str] = []

    def query_history(*, sql: str, **_kwargs: object) -> list[dict[str, object]]:
        normalized = " ".join(sql.lower().split())
        queries.append(normalized)
        if "group by status" in normalized:
            return [{"status": latest["status"], "count": 1}]
        if "count(*) as count" in normalized:
            return [{"count": 1 if "status='paused'" in normalized else 0}]
        latest_row: dict[str, object] = dict(latest)
        return [latest_row]

    monkeypatch.setattr(logstore, "execute_sql", query_history)
    api = LogstoreAPIWorkflowRunRepository(sqlite_session_factory)  # pyrefly: ignore[bad-instantiation]
    api._enable_dual_read = False
    page = api.get_paginated_workflow_runs(
        tenant_id=runner.application_generate_entity.app_config.tenant_id,
        app_id=runner.application_generate_entity.app_config.app_id,
        triggered_from=WorkflowRunTriggeredFrom.DEBUGGING,
        status=None,
    )
    filtered = api.get_paginated_workflow_runs(
        tenant_id=runner.application_generate_entity.app_config.tenant_id,
        app_id=runner.application_generate_entity.app_config.app_id,
        triggered_from=WorkflowRunTriggeredFrom.DEBUGGING,
        status="paused",
    )
    filtered_count = api.get_workflow_runs_count(
        tenant_id=runner.application_generate_entity.app_config.tenant_id,
        app_id=runner.application_generate_entity.app_config.app_id,
        triggered_from=WorkflowRunTriggeredFrom.DEBUGGING.value,
        status="paused",
    )
    all_count = api.get_workflow_runs_count(
        tenant_id=runner.application_generate_entity.app_config.tenant_id,
        app_id=runner.application_generate_entity.app_config.app_id,
        triggered_from=WorkflowRunTriggeredFrom.DEBUGGING.value,
    )
    assert [run.status for run in page.data] == [WorkflowExecutionStatus.PAUSED]
    assert [run.status for run in filtered.data] == [WorkflowExecutionStatus.PAUSED]
    assert filtered_count["total"] == 1
    assert all_count["total"] == 1
    assert any("where rn = 1 and status='paused'" in query for query in queries)
    assert any("where rn = 1 ) current where status='paused'" in query for query in queries)


@pytest.mark.usefixtures("logstore")
def test_logstore_pause_history_failure_preserves_sql_pause_and_recovery_evidence(
    sqlite_session_factory: sessionmaker[Session],
    logstore: AliyunLogStore,
    config_overrides: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from core.app.apps import workflow_app_runner
    from core.app.entities.queue_entities import QueueWorkflowPausedEvent
    from core.app.layers.pause_state_persist_layer import PauseStateLayerConfig
    from services.workflow_run_agg import WorkflowRunAgg
    from tests.unit_tests.services.workflow.test_workflow_run_agg import make_workflow_runner

    config_overrides(
        API_WORKFLOW_RUN_REPOSITORY=(
            "extensions.logstore.repositories.logstore_api_workflow_run_repository.LogstoreAPIWorkflowRunRepository"
        )
    )
    mocker.patch("repositories.sqlalchemy_api_workflow_run_repository.storage.save")
    mocker.patch("repositories.sqlalchemy_api_workflow_run_repository.storage.delete")
    monkeypatch.setattr(workflow_app_runner.dispatch_human_input_email_task, "apply_async", lambda **_kwargs: None)
    runner, _sql_execution_repository, node_repository = make_workflow_runner(
        sqlite_session_factory.kw["bind"], human_input=True
    )
    user = Account(name="History Tester", email="history-failure@example.com")
    user.id = runner.application_generate_entity.user_id
    execution_repository = LogstoreWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id=runner.application_generate_entity.app_config.tenant_id,
        user=user,
        app_id=runner.application_generate_entity.app_config.app_id,
        triggered_from=WorkflowRunTriggeredFrom.DEBUGGING,
    )
    execution_repository._enable_dual_write = False
    sql_saves: list[WorkflowExecutionStatus] = []
    original_sql_save = execution_repository.sql_repository.save_with_model_callback

    def save_sql_control(execution: WorkflowExecution, callback: Callable[[Session, WorkflowRun], object]) -> object:
        sql_saves.append(execution.status)
        return original_sql_save(execution, callback)

    monkeypatch.setattr(execution_repository.sql_repository, "save_with_model_callback", save_sql_control)

    def fail_paused_history(_logstore_name: str, fields: list[tuple[str, str]]) -> None:
        if dict(fields)["status"] == "paused":
            raise ConnectionError("logstore unavailable")

    monkeypatch.setattr(logstore, "put_log", fail_paused_history)
    with caplog.at_level(logging.ERROR, logger="services.workflow_run_agg"):
        WorkflowRunAgg.run(
            runner,
            PauseStateLayerConfig(sqlite_session_factory, runner.application_generate_entity.user_id),
            execution_repository,
            node_repository,
        )

    events = [message.event for message in runner._queue_manager.listen()]
    assert isinstance(events[-1], QueueWorkflowPausedEvent), events
    assert sql_saves == [WorkflowExecutionStatus.RUNNING]
    assert "Failed to publish committed workflow pause history" in caplog.text
    api = LogstoreAPIWorkflowRunRepository(sqlite_session_factory)  # pyrefly: ignore[bad-instantiation]
    api._enable_dual_read = False
    recovered = api.get_workflow_run_by_id(
        runner.application_generate_entity.app_config.tenant_id,
        runner.application_generate_entity.app_config.app_id,
        runner.application_generate_entity.workflow_execution_id,
    )
    assert recovered is not None
    assert recovered.status == WorkflowExecutionStatus.PAUSED
    with sqlite_session_factory() as session:
        run = session.get(WorkflowRun, runner.application_generate_entity.workflow_execution_id)
        assert run is not None
        assert run.status == WorkflowExecutionStatus.PAUSED
        assert run.history_sync_pending is True
        assert run.history_claim_id is None
        assert session.scalar(select(WorkflowPause).where(WorkflowPause.workflow_run_id == run.id)) is not None

    from tasks.workflow_run_history_sync_task import drain_pending_workflow_run_histories

    recovered_records: list[dict[str, str]] = []
    monkeypatch.setattr(logstore, "put_log", lambda _name, fields: recovered_records.append(dict(fields)))
    assert (
        drain_pending_workflow_run_histories(
            database_client=sqlite_session_factory,
            logstore_client=logstore,
            enable_graph=False,
        )
        == 1
    )
    assert [record["status"] for record in recovered_records] == ["paused"]
    with sqlite_session_factory() as session:
        run = session.get(WorkflowRun, runner.application_generate_entity.workflow_execution_id)
        assert run is not None
        assert run.history_sync_pending is False
        assert run.history_published_revision == run.history_revision


@pytest.mark.usefixtures("logstore")
def test_workflow_history_retry_rotates_past_a_failing_first_row(
    sqlite_session_factory: sessionmaker[Session],
    logstore: AliyunLogStore,
    mocker: MockerFixture,
) -> None:
    from repositories.workflow_run_history import (
        advance_workflow_run_history,
        claim_pending_workflow_run_histories,
    )
    from tasks.workflow_run_history_sync_task import drain_pending_workflow_run_histories

    user = Account(name="History Tester", email="history-fairness@example.com")
    user.id = "account-1"
    sql = SQLAlchemyWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    )
    run_ids = [f"run-{index:03d}" for index in range(101)]
    for run_id in run_ids:
        sql.save_synchronously(
            WorkflowExecution(
                id_=run_id,
                workflow_id="workflow-1",
                workflow_type=WorkflowType.WORKFLOW,
                workflow_version="1",
                graph={},
                inputs={},
                started_at=naive_utc_now(),
            )
        )
    with sqlite_session_factory.begin() as session:
        for run_id in run_ids:
            run = session.get(WorkflowRun, run_id)
            assert run is not None
            advance_workflow_run_history(run)

    attempted: list[str] = []

    def fail_first_run(_name: str, fields: list[tuple[str, str]]) -> None:
        run_id = dict(fields)["id"]
        attempted.append(run_id)
        if run_id == "run-000":
            raise ConnectionError("poison history row")

    mocker.patch.object(logstore, "put_log", side_effect=fail_first_run)
    assert (
        drain_pending_workflow_run_histories(
            database_client=sqlite_session_factory,
            logstore_client=logstore,
            enable_graph=False,
            limit=100,
        )
        == 99
    )
    assert (
        drain_pending_workflow_run_histories(
            database_client=sqlite_session_factory,
            logstore_client=logstore,
            enable_graph=False,
            limit=100,
        )
        == 1
    )
    assert attempted[:2] == ["run-000", "run-001"]
    assert attempted[100] == "run-100"

    with sqlite_session_factory.begin() as session:
        poison = session.get(WorkflowRun, "run-000")
        assert poison is not None
        assert poison.history_last_attempted_at is not None
        advance_workflow_run_history(poison)
        assert poison.history_last_attempted_at is None
    with sqlite_session_factory.begin() as session:
        (next_snapshot,) = claim_pending_workflow_run_histories(session, limit=1)
        assert next_snapshot.id == "run-000"


@pytest.mark.usefixtures("logstore")
def test_deleted_workflow_control_suppresses_pending_history_projection(
    sqlite_session_factory: sessionmaker[Session],
    logstore: AliyunLogStore,
    mocker: MockerFixture,
) -> None:
    from repositories.workflow_run_history import advance_workflow_run_history
    from tasks.workflow_run_history_sync_task import drain_pending_workflow_run_histories

    user = Account(name="History Tester", email="history-delete@example.com")
    user.id = "account-1"
    SQLAlchemyWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    ).save_synchronously(
        WorkflowExecution(
            id_="run-deleted",
            workflow_id="workflow-1",
            workflow_type=WorkflowType.WORKFLOW,
            workflow_version="1",
            graph={},
            inputs={},
            started_at=naive_utc_now(),
        )
    )
    with sqlite_session_factory.begin() as session:
        run = session.get(WorkflowRun, "run-deleted")
        assert run is not None
        advance_workflow_run_history(run)
        session.delete(run)

    append_history = mocker.patch.object(logstore, "put_log")
    assert (
        drain_pending_workflow_run_histories(
            database_client=sqlite_session_factory,
            logstore_client=logstore,
            enable_graph=False,
        )
        == 0
    )
    append_history.assert_not_called()


@pytest.mark.usefixtures("logstore")
def test_committed_pause_history_skips_a_control_row_already_resumed(
    sqlite_session_factory: sessionmaker[Session],
    logstore: AliyunLogStore,
    mocker: MockerFixture,
) -> None:
    mocker.patch("repositories.sqlalchemy_api_workflow_run_repository.storage.save")
    user = Account(name="History Tester", email="history-resume@example.com")
    user.id = "account-1"
    execution = WorkflowExecution(
        id_="run-resumed",
        workflow_id="workflow-1",
        workflow_type=WorkflowType.WORKFLOW,
        workflow_version="1",
        graph={},
        inputs={},
        started_at=naive_utc_now(),
    )
    SQLAlchemyWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    ).save_synchronously(execution)
    control = DifyAPISQLAlchemyWorkflowRunRepository(sqlite_session_factory)
    paused_execution = execution.model_copy(update={"status": WorkflowExecutionStatus.PAUSED})
    pause = control.publish_workflow_pause(
        workflow_execution=paused_execution,
        state_owner_user_id="account-1",
        state="{}",
        pause_reasons=(),
        trigger_updates=(),
    )
    control.resume_workflow_pause("run-resumed", pause)
    writer = LogstoreWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    )
    writer._enable_dual_write = False
    history_appends: list[list[tuple[str, str]]] = []
    mocker.patch.object(logstore, "put_log", side_effect=lambda _name, fields: history_appends.append(fields))

    assert pause.history_revision is not None
    assert pause.history_claim_id is not None
    writer.publish_committed_pause(
        paused_execution,
        history_revision=pause.history_revision,
        history_claim_id=pause.history_claim_id,
    )

    assert history_appends == []
    with sqlite_session_factory() as session:
        run = session.get(WorkflowRun, "run-resumed")
        assert run is not None
        assert run.status == WorkflowExecutionStatus.RUNNING


@pytest.mark.usefixtures("logstore")
def test_concurrent_resume_history_wins_even_when_paused_append_finishes_last(
    sqlite_session_factory: sessionmaker[Session],
    logstore: AliyunLogStore,
    mocker: MockerFixture,
) -> None:
    mocker.patch("repositories.sqlalchemy_api_workflow_run_repository.storage.save")
    user = Account(name="History Tester", email="history-order@example.com")
    user.id = "account-1"
    execution = WorkflowExecution(
        id_="run-history-order",
        workflow_id="workflow-1",
        workflow_type=WorkflowType.WORKFLOW,
        workflow_version="1",
        graph={},
        inputs={},
        started_at=naive_utc_now(),
    )
    SQLAlchemyWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    ).save_synchronously(execution)
    control = DifyAPISQLAlchemyWorkflowRunRepository(sqlite_session_factory)
    paused_execution = execution.model_copy(update={"status": WorkflowExecutionStatus.PAUSED})
    pause = control.publish_workflow_pause(
        workflow_execution=paused_execution,
        state_owner_user_id="account-1",
        state="{}",
        pause_reasons=(),
        trigger_updates=(),
    )
    writer = LogstoreWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    )
    writer._enable_dual_write = False
    records: list[dict[str, str]] = []
    resumed = False

    def append_after_resume(_logstore_name: str, fields: list[tuple[str, str]]) -> None:
        nonlocal resumed
        if not resumed:
            resumed = True
            control.resume_workflow_pause("run-history-order", pause)
            writer.save(execution.model_copy(update={"status": WorkflowExecutionStatus.RUNNING}))
        records.append(dict(fields))

    mocker.patch.object(logstore, "put_log", side_effect=append_after_resume)
    assert pause.history_revision is not None
    assert pause.history_claim_id is not None
    writer.publish_committed_pause(
        paused_execution,
        history_revision=pause.history_revision,
        history_claim_id=pause.history_claim_id,
    )

    assert [record["status"] for record in records] == ["running", "paused"]
    assert max(records, key=lambda record: int(record["log_version"]))["status"] == "running"
    with sqlite_session_factory() as session:
        run = session.get(WorkflowRun, "run-history-order")
        assert run is not None
        assert run.status == WorkflowExecutionStatus.RUNNING
        assert run.history_revision == 3
        assert run.history_published_revision == 3
        assert run.history_sync_pending is False


@pytest.mark.parametrize(
    ("record_type", "node_status"),
    [
        ("workflow", WorkflowNodeExecutionStatus.PAUSED),
        ("node", WorkflowNodeExecutionStatus.PAUSED),
        ("node", WorkflowNodeExecutionStatus.RETRY),
    ],
)
def test_logstore_stop_orders_before_delayed_upload_and_overrides_stale_indexes(
    sqlite_session_factory: sessionmaker[Session],
    logstore: AliyunLogStore,
    mocker: MockerFixture,
    record_type: str,
    node_status: WorkflowNodeExecutionStatus,
) -> None:
    user = Account(name="Test", email="test@example.com")
    user.id = "account-1"
    run = WorkflowExecution(
        accumulated_elapsed_time=3.5,
        id_="run-1",
        workflow_id="workflow-1",
        workflow_type=WorkflowType.WORKFLOW,
        workflow_version="1",
        graph={},
        inputs={},
        started_at=naive_utc_now(),
    )
    run.status = WorkflowExecutionStatus.PAUSED
    SQLAlchemyWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    ).save(run)
    control = DifyAPISQLAlchemyWorkflowRunRepository(sqlite_session_factory)
    control.bind_workflow_task(tenant_id="tenant-1", app_id="app-1", workflow_run_id="run-1", task_id="task-1")
    node = WorkflowNodeExecution(
        id="node-1",
        node_execution_id="node-1",
        workflow_id="workflow-1",
        workflow_execution_id="run-1",
        index=1,
        node_id="review",
        node_type="human-input",
        title="Review",
        created_at=run.started_at,
        status=node_status,
    )
    SQLAlchemyWorkflowNodeExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowNodeExecutionTriggeredFrom.WORKFLOW_RUN,
    ).save_synchronously(node)
    records: list[dict[str, str]] = []
    interrupted = False

    def append_after_concurrent_stop(_store: str, fields: list[tuple[str, str]]) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            stopped = control.stop_paused_workflow_task(
                tenant_id="tenant-1", app_id="app-1", task_id="task-1", owner=None
            )
            assert stopped is not None
            if record_type == "workflow":
                run_writer.save(
                    run.model_copy(
                        update={
                            "status": WorkflowExecutionStatus.STOPPED,
                            "finished_at": stopped.finished_at,
                            "error_message": stopped.error,
                        }
                    )
                )
            else:
                node_writer.save_many(
                    [
                        node.model_copy(
                            update={
                                "status": WorkflowNodeExecutionStatus.FAILED,
                                "finished_at": stopped.finished_at,
                                "error": stopped.error,
                            }
                        )
                    ]
                )
        records.append(dict(fields))

    mocker.patch.object(logstore, "put_log", side_effect=append_after_concurrent_stop)
    run_writer = LogstoreWorkflowExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
    )
    node_writer = LogstoreWorkflowNodeExecutionRepository(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        user=user,
        app_id="app-1",
        triggered_from=WorkflowNodeExecutionTriggeredFrom.WORKFLOW_RUN,
    )
    run_writer._enable_dual_write = node_writer._enable_dual_write = False
    if record_type == "workflow":
        run_writer.save(run)
        expected_status = "stopped"
        assert max(records, key=lambda row: int(row["log_version"]))["status"] == expected_status
        run.accumulated_elapsed_time = 999.0
        run_writer.save(run)  # Another late RUNNING/PAUSED write also remains terminal.
        assert float(records[-1]["elapsed_time"]) == 3.5
    else:
        node_writer.save(node)
        expected_status = "failed"
        assert max(records, key=lambda row: int(row["log_version"]))["status"] == expected_status
        stale_node = next(row for row in records if row["status"] == node_status)
        mocker.patch.object(logstore, "execute_sql", return_value=[stale_node])
        api = LogstoreAPIWorkflowNodeExecutionRepository(sqlite_session_factory)  # pyrefly: ignore[bad-instantiation]
        assert api.get_executions_by_workflow_run("tenant-1", "app-1", "run-1")[0].status == expected_status
        by_id = api.get_execution_by_id("node-1", "tenant-1")
        assert by_id is not None
        assert by_id.status == expected_status
        assert (
            api.get_execution_snapshots_by_workflow_run("tenant-1", "app-1", "workflow-1", "workflow-run", "run-1")[
                0
            ].status
            == expected_status
        )
        assert node_writer.get_by_workflow_execution("run-1")[0].status == expected_status
        node_writer.save(node)
    assert records[-1]["status"] == expected_status
    with sqlite_session_factory() as session:
        persisted_run = session.get(WorkflowRun, "run-1")
        assert persisted_run is not None
        assert persisted_run.status == WorkflowExecutionStatus.STOPPED
        assert persisted_run.history_revision >= 1
        if record_type == "workflow":
            assert persisted_run.history_published_revision == persisted_run.history_revision
            assert persisted_run.history_sync_pending is False
        else:
            assert persisted_run.history_sync_pending is True


@pytest.mark.parametrize(
    "status",
    [
        WorkflowNodeExecutionStatus.SUCCEEDED,
        WorkflowNodeExecutionStatus.FAILED,
        WorkflowNodeExecutionStatus.EXCEPTION,
        WorkflowNodeExecutionStatus.STOPPED,
    ],
)
def test_stop_preserves_completed_node_history(status: WorkflowNodeExecutionStatus) -> None:
    finished_at = naive_utc_now()
    stopped = WorkflowRun(
        id="run-1",
        tenant_id="tenant-1",
        app_id="app-1",
        workflow_id="workflow-1",
        type=WorkflowType.WORKFLOW,
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
        version="1",
        status=WorkflowExecutionStatus.STOPPED,
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="account-1",
        error="User requested stop",
        finished_at=finished_at,
        stop_requested_at=finished_at,
    )
    node = WorkflowNodeExecution(
        id="node-1",
        workflow_id="workflow-1",
        workflow_execution_id="run-1",
        index=1,
        node_id="answer",
        node_type="llm",
        title="Answer",
        status=status,
        outputs={"answer": "preserved"},
        error=None if status == WorkflowNodeExecutionStatus.SUCCEEDED else "Original terminal error",
        elapsed_time=2.5,
        created_at=finished_at,
        finished_at=finished_at,
    )
    original = node.model_dump()

    apply_workflow_stop_to_node(node, stopped)

    assert node.model_dump() == original

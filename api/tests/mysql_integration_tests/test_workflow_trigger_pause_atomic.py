"""MySQL transaction regressions for atomic workflow pause publication."""

from datetime import UTC, datetime, timedelta
from threading import Event, Thread
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import delete, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

import repositories.sqlalchemy_api_workflow_run_repository as workflow_run_repository_module
from graphon.enums import WorkflowExecutionStatus, WorkflowType
from models.enums import (
    AppTriggerType,
    CreatorUserRole,
    WorkflowRunTriggeredFrom,
    WorkflowTriggerStatus,
)
from models.trigger import WorkflowTriggerLog
from models.workflow import WorkflowPause, WorkflowPauseReason, WorkflowRun
from repositories.entities.workflow_execution import WorkflowExecution
from repositories.entities.workflow_pause import WorkflowTriggerPausePublication
from repositories.sqlalchemy_api_workflow_run_repository import DifyAPISQLAlchemyWorkflowRunRepository
from repositories.sqlalchemy_workflow_trigger_log_repository import SQLAlchemyWorkflowTriggerLogRepository
from repositories.workflow_run_history import (
    WorkflowRunHistorySnapshot,
    advance_and_claim_workflow_run_history,
    confirm_workflow_run_history_publication,
    workflow_history_database_now,
)


class _WorkflowPauseStorageBoundary:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def save(self, filename: str, data: bytes) -> None:
        self.objects[filename] = data

    def load(self, filename: str) -> bytes:
        return self.objects[filename]

    def delete(self, filename: str) -> None:
        self.objects.pop(filename, None)


def _repeatable_read_sessions(mysql_session: Session) -> tuple[Engine, sessionmaker[Session]]:
    engine = cast(Engine, mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ"))
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _join_if_started(thread: Thread) -> None:
    if thread.ident is not None:
        thread.join(timeout=10)


def test_pause_snapshot_and_trigger_owner_release_become_visible_in_one_commit(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader cannot observe PAUSED without its pause row and trigger owner release."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id = str(uuid4())
    app_id = str(uuid4())
    workflow_id = str(uuid4())
    workflow_run_id = str(uuid4())
    actor_id = str(uuid4())
    trigger_log_id = str(uuid4())
    owner_id = str(uuid4())
    now = datetime.now(UTC).replace(tzinfo=None)
    with sessions.begin() as seed:
        trigger_log = WorkflowTriggerLog(
            tenant_id=tenant_id,
            app_id=app_id,
            workflow_id=workflow_id,
            workflow_run_id=workflow_run_id,
            root_node_id=None,
            trigger_metadata="{}",
            trigger_type=AppTriggerType.TRIGGER_SCHEDULE,
            trigger_data="{}",
            inputs="{}",
            outputs=None,
            status=WorkflowTriggerStatus.RUNNING,
            error=None,
            queue_name="workflow",
            celery_task_id=None,
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by=actor_id,
            execution_owner_id=owner_id,
            execution_lease_expires_at=now + timedelta(minutes=5),
        )
        trigger_log.id = trigger_log_id
        seed.add_all(
            [
                WorkflowRun(
                    id=workflow_run_id,
                    tenant_id=tenant_id,
                    app_id=app_id,
                    workflow_id=workflow_id,
                    type=WorkflowType.WORKFLOW,
                    triggered_from=WorkflowRunTriggeredFrom.SCHEDULE,
                    version="1",
                    graph="{}",
                    inputs="{}",
                    status=WorkflowExecutionStatus.RUNNING,
                    created_by_role=CreatorUserRole.ACCOUNT,
                    created_by=actor_id,
                ),
                trigger_log,
            ]
        )

    storage = _WorkflowPauseStorageBoundary()
    monkeypatch.setattr(workflow_run_repository_module, "storage", storage)
    repository = DifyAPISQLAlchemyWorkflowRunRepository(session_maker=sessions)
    execution = WorkflowExecution(
        id_=workflow_run_id,
        workflow_id=workflow_id,
        workflow_version="1",
        workflow_type=WorkflowType.WORKFLOW,
        graph={},
        inputs={},
        outputs={"answer": "paused"},
        status=WorkflowExecutionStatus.PAUSED,
        total_tokens=11,
        total_steps=3,
        exceptions_count=0,
        started_at=now - timedelta(seconds=4),
        finished_at=now,
        accumulated_elapsed_time=4.0,
    )
    trigger_publication = WorkflowTriggerPausePublication(
        trigger_log_id=trigger_log_id,
        owner_id=owner_id,
        workflow_run_id=workflow_run_id,
        outputs='{"answer":"paused"}',
        elapsed_time=4.0,
        total_tokens=11,
        finished_at=now,
    )
    trigger_update_executed = Event()
    allow_commit = Event()
    publisher_finished = Event()
    publisher_errors: list[BaseException] = []
    pause_ids: list[str] = []

    def pause_after_trigger_update(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        normalized = " ".join(statement.lower().replace("`", "").split())
        if normalized.startswith("update workflow_trigger_logs"):
            trigger_update_executed.set()
            assert allow_commit.wait(timeout=10)

    def publish_pause() -> None:
        try:
            pause = repository.publish_workflow_pause(
                workflow_execution=execution,
                state_owner_user_id=actor_id,
                state='{"state":"paused"}',
                pause_reasons=(),
                trigger_updates=(trigger_publication,),
            )
            pause_ids.append(pause.id)
        except BaseException as error:
            publisher_errors.append(error)
        finally:
            publisher_finished.set()

    publisher = Thread(target=publish_pause)
    event.listen(engine, "after_cursor_execute", pause_after_trigger_update)
    try:
        publisher.start()
        assert trigger_update_executed.wait(timeout=10)
        assert not publisher_finished.wait(timeout=0.25)

        # The trigger update has executed in the publisher transaction, but a
        # real independent RR session still observes the complete old state.
        with sessions() as before_commit:
            run = before_commit.get_one(WorkflowRun, workflow_run_id)
            trigger = before_commit.get_one(WorkflowTriggerLog, trigger_log_id)
            pause = before_commit.scalar(select(WorkflowPause).where(WorkflowPause.workflow_run_id == workflow_run_id))
            assert run.status == WorkflowExecutionStatus.RUNNING
            assert pause is None
            assert trigger.status == WorkflowTriggerStatus.RUNNING
            assert trigger.execution_owner_id == owner_id

        allow_commit.set()
        _join_if_started(publisher)
        assert not publisher.is_alive()
        assert publisher_errors == []
        assert len(pause_ids) == 1

        with sessions() as after_commit:
            run = after_commit.get_one(WorkflowRun, workflow_run_id)
            trigger = after_commit.get_one(WorkflowTriggerLog, trigger_log_id)
            pause = after_commit.get_one(WorkflowPause, pause_ids[0])
            assert run.status == WorkflowExecutionStatus.PAUSED
            assert pause.workflow_run_id == workflow_run_id
            assert trigger.status == WorkflowTriggerStatus.PAUSED
            assert trigger.execution_owner_id is None
            assert trigger.execution_lease_expires_at is None
            assert trigger.terminal_owner_id == owner_id
            assert trigger.workflow_run_id == workflow_run_id
        assert list(storage.objects.values()) == [b'{"state":"paused"}']
    finally:
        allow_commit.set()
        _join_if_started(publisher)
        event.remove(engine, "after_cursor_execute", pause_after_trigger_update)
        with sessions.begin() as cleanup:
            pause_ids_to_delete = select(WorkflowPause.id).where(WorkflowPause.workflow_run_id == workflow_run_id)
            cleanup.execute(delete(WorkflowPauseReason).where(WorkflowPauseReason.pause_id.in_(pause_ids_to_delete)))
            cleanup.execute(delete(WorkflowPause).where(WorkflowPause.workflow_run_id == workflow_run_id))
            cleanup.execute(delete(WorkflowTriggerLog).where(WorkflowTriggerLog.id == trigger_log_id))
            cleanup.execute(delete(WorkflowRun).where(WorkflowRun.id == workflow_run_id))


def test_workflow_history_revision_allocation_serializes_and_stale_claim_cannot_ack(
    mysql_session: Session,
) -> None:
    """A delayed publisher cannot acknowledge a newer per-run history revision."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    workflow_run_id = str(uuid4())
    with sessions.begin() as seed:
        seed.add(
            WorkflowRun(
                id=workflow_run_id,
                tenant_id=str(uuid4()),
                app_id=str(uuid4()),
                workflow_id=str(uuid4()),
                type=WorkflowType.WORKFLOW,
                triggered_from=WorkflowRunTriggeredFrom.SCHEDULE,
                version="1",
                graph="{}",
                inputs="{}",
                status=WorkflowExecutionStatus.RUNNING,
                created_by_role=CreatorUserRole.ACCOUNT,
                created_by=str(uuid4()),
            )
        )

    first_claimed = Event()
    allow_first_commit = Event()
    second_select_executed = Event()
    second_finished = Event()
    first_snapshots: list[WorkflowRunHistorySnapshot] = []
    second_snapshots: list[WorkflowRunHistorySnapshot] = []
    writer_errors: list[BaseException] = []

    def observe_second_lock_attempt(_connection, _cursor, _statement, _parameters, context, _executemany) -> None:
        if context.execution_options.get("workflow_history_probe") == "second":
            second_select_executed.set()

    def allocate_first_revision() -> None:
        try:
            with sessions.begin() as session:
                run = session.scalar(select(WorkflowRun).where(WorkflowRun.id == workflow_run_id).with_for_update())
                assert run is not None
                snapshot = advance_and_claim_workflow_run_history(
                    run,
                    now=workflow_history_database_now(session),
                )
                first_snapshots.append(snapshot)
                first_claimed.set()
                assert allow_first_commit.wait(timeout=10)
        except BaseException as error:
            writer_errors.append(error)

    def allocate_second_revision() -> None:
        try:
            with sessions.begin() as session:
                stmt = (
                    select(WorkflowRun)
                    .where(WorkflowRun.id == workflow_run_id)
                    .with_for_update()
                    .execution_options(workflow_history_probe="second")
                )
                run = session.scalar(stmt)
                assert run is not None
                second_snapshots.append(
                    advance_and_claim_workflow_run_history(
                        run,
                        now=workflow_history_database_now(session),
                    )
                )
        except BaseException as error:
            writer_errors.append(error)
        finally:
            second_finished.set()

    first_writer = Thread(target=allocate_first_revision)
    second_writer = Thread(target=allocate_second_revision)
    event.listen(engine, "before_cursor_execute", observe_second_lock_attempt)
    try:
        first_writer.start()
        assert first_claimed.wait(timeout=10)
        second_writer.start()
        assert second_select_executed.wait(timeout=10)
        assert not second_finished.wait(timeout=0.25)

        allow_first_commit.set()
        _join_if_started(first_writer)
        _join_if_started(second_writer)
        assert not first_writer.is_alive()
        assert not second_writer.is_alive()
        assert writer_errors == []
        assert [snapshot.revision for snapshot in first_snapshots] == [1]
        assert [snapshot.revision for snapshot in second_snapshots] == [2]

        with sessions.begin() as acknowledge:
            assert confirm_workflow_run_history_publication(acknowledge, first_snapshots[0]) is False
            assert confirm_workflow_run_history_publication(acknowledge, second_snapshots[0]) is True

        with sessions() as verify:
            run = verify.get_one(WorkflowRun, workflow_run_id)
            assert run.history_revision == 2
            assert run.history_published_revision == 2
            assert run.history_claim_id is None
            assert run.history_claim_expires_at is None
            assert run.history_sync_pending is False
    finally:
        allow_first_commit.set()
        _join_if_started(first_writer)
        _join_if_started(second_writer)
        event.remove(engine, "before_cursor_execute", observe_second_lock_attempt)
        with sessions.begin() as cleanup:
            cleanup.execute(delete(WorkflowRun).where(WorkflowRun.id == workflow_run_id))


def test_abandoned_pending_marker_rejects_late_legacy_queue_write(
    mysql_session: Session,
) -> None:
    """MySQL must reject a stale producer's QUEUED write after the bounded PENDING recovery wins."""

    _engine, sessions = _repeatable_read_sessions(mysql_session)
    abandoned_id = str(uuid4())
    fresh_id = str(uuid4())
    tenant_id = str(uuid4())
    app_id = str(uuid4())
    workflow_id = str(uuid4())
    actor_id = str(uuid4())

    def pending_trigger(trigger_id: str) -> WorkflowTriggerLog:
        trigger = WorkflowTriggerLog(
            tenant_id=tenant_id,
            app_id=app_id,
            workflow_id=workflow_id,
            workflow_run_id=None,
            root_node_id=None,
            trigger_metadata="{}",
            trigger_type=AppTriggerType.TRIGGER_SCHEDULE,
            trigger_data="{}",
            inputs="{}",
            outputs=None,
            status=WorkflowTriggerStatus.PENDING,
            error=None,
            queue_name="workflow",
            celery_task_id=None,
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by=actor_id,
        )
        trigger.id = trigger_id
        return trigger

    with sessions.begin() as seed:
        seed.add_all([pending_trigger(abandoned_id), pending_trigger(fresh_id)])

    stale_producer = sessions()
    try:
        stale = stale_producer.get_one(WorkflowTriggerLog, abandoned_id)
        assert stale.status == WorkflowTriggerStatus.PENDING
        assert stale.pending_dispatch_abandoned_at is None

        with sessions.begin() as recovery:
            recovered = SQLAlchemyWorkflowTriggerLogRepository(recovery).fail_abandoned_pending_execution(
                abandoned_id,
                pending_timeout=timedelta(0),
                error="legacy pending dispatch timed out",
            )
            assert recovered is True

        # This Session retains the legacy producer's earlier PENDING identity
        # snapshot. Its old-style direct QUEUED update must hit the database
        # CHECK against the newly committed abandonment marker and roll back.
        stale.status = WorkflowTriggerStatus.QUEUED
        stale.celery_task_id = "late-legacy-task"
        with pytest.raises(IntegrityError):
            stale_producer.commit()
        stale_producer.rollback()

        with sessions() as verify_abandoned:
            abandoned = verify_abandoned.get_one(WorkflowTriggerLog, abandoned_id)
            assert abandoned.status == WorkflowTriggerStatus.FAILED
            assert abandoned.pending_dispatch_abandoned_at is not None
            assert abandoned.finished_at is not None
            assert abandoned.celery_task_id is None
            assert abandoned.error == "legacy pending dispatch timed out"

        queued_at = datetime.now(UTC).replace(tzinfo=None)
        with sessions.begin() as queue_session:
            queued = SQLAlchemyWorkflowTriggerLogRepository(queue_session).queue_pending_execution(
                fresh_id,
                celery_task_id="fresh-task",
                triggered_at=queued_at,
            )
            assert queued is True
        owner_id = str(uuid4())
        with sessions.begin() as claim_session:
            claimed = SQLAlchemyWorkflowTriggerLogRepository(claim_session).claim_execution(
                fresh_id,
                owner_id=owner_id,
                lease_duration=timedelta(minutes=5),
            )
            assert claimed is not None
            assert claimed.status == WorkflowTriggerStatus.RUNNING
            assert claimed.execution_owner_id == owner_id
            assert claimed.pending_dispatch_abandoned_at is None
    finally:
        stale_producer.close()
        with sessions.begin() as cleanup:
            cleanup.execute(delete(WorkflowTriggerLog).where(WorkflowTriggerLog.id.in_((abandoned_id, fresh_id))))


@pytest.mark.parametrize("winner", ["worker", "reaper", "published"])
def test_stale_unpublished_dispatch_reaper_respects_current_worker_or_receipt(
    mysql_session: Session,
    winner: str,
) -> None:
    """The reaper's RR candidate snapshot cannot override a newer claim or publication receipt."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    trigger_log_id = str(uuid4())
    owner_id = str(uuid4())
    task_id = f"dispatch-{uuid4()}"
    with sessions.begin() as seed:
        trigger = WorkflowTriggerLog(
            tenant_id=str(uuid4()),
            app_id=str(uuid4()),
            workflow_id=str(uuid4()),
            workflow_run_id=None,
            root_node_id=None,
            trigger_metadata="{}",
            trigger_type=AppTriggerType.TRIGGER_SCHEDULE,
            trigger_data="{}",
            inputs="{}",
            outputs=None,
            status=WorkflowTriggerStatus.QUEUED,
            error=None,
            queue_name="workflow",
            celery_task_id=task_id,
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by=str(uuid4()),
            dispatch_prepared_at=datetime(2000, 1, 1),
            quota_reservation_id=f"reservation-{uuid4()}",
        )
        trigger.id = trigger_log_id
        seed.add(trigger)

    candidates_loaded = Event()
    allow_reaper_cas = Event()
    reaper_finished = Event()
    reaper_counts: list[int] = []
    reaper_errors: list[BaseException] = []

    def pause_after_candidate_snapshot(
        _connection, _cursor, statement: str, _parameters, _context, _executemany
    ) -> None:
        normalized = " ".join(statement.lower().replace("`", "").split())
        if (
            winner != "reaper"
            and "select workflow_trigger_logs.id, workflow_trigger_logs.celery_task_id" in normalized
            and "dispatch_prepared_at" in normalized
        ):
            candidates_loaded.set()
            assert allow_reaper_cas.wait(timeout=10)

    def reap_dispatch() -> None:
        try:
            with sessions.begin() as reaper_session:
                count = SQLAlchemyWorkflowTriggerLogRepository(reaper_session).abandon_stale_unpublished_dispatches(
                    dispatch_timeout=timedelta(0),
                    error="broker publication receipt expired",
                    limit=100,
                )
                reaper_counts.append(count)
        except BaseException as error:
            reaper_errors.append(error)
        finally:
            reaper_finished.set()

    reaper = Thread(target=reap_dispatch)
    event.listen(engine, "after_cursor_execute", pause_after_candidate_snapshot)
    stale_worker = sessions()
    try:
        if winner == "reaper":
            stale = stale_worker.get_one(WorkflowTriggerLog, trigger_log_id)
            assert stale.status == WorkflowTriggerStatus.QUEUED
            reaper.start()
            _join_if_started(reaper)
            assert reaper_counts == [1]
            claimed = SQLAlchemyWorkflowTriggerLogRepository(stale_worker).claim_execution(
                trigger_log_id,
                owner_id=owner_id,
                lease_duration=timedelta(minutes=5),
            )
            assert claimed is None
            stale_worker.rollback()
        else:
            reaper.start()
            assert candidates_loaded.wait(timeout=10)
            assert not reaper_finished.wait(timeout=0.25)
            with sessions.begin() as competing:
                repository = SQLAlchemyWorkflowTriggerLogRepository(competing)
                if winner == "worker":
                    claimed = repository.claim_execution(
                        trigger_log_id,
                        owner_id=owner_id,
                        lease_duration=timedelta(minutes=5),
                    )
                    assert claimed is not None
                else:
                    assert repository.confirm_dispatch_published(trigger_log_id, celery_task_id=task_id) is True
            allow_reaper_cas.set()
            _join_if_started(reaper)
            assert reaper_counts == [0]

        assert not reaper.is_alive()
        assert reaper_errors == []
        with sessions() as verify:
            trigger = verify.get_one(WorkflowTriggerLog, trigger_log_id)
            if winner == "worker":
                assert trigger.status == WorkflowTriggerStatus.RUNNING
                assert trigger.execution_owner_id == owner_id
                assert trigger.dispatch_published_at is None
                assert trigger.dispatch_abandoned_at is None
            elif winner == "reaper":
                assert trigger.status == WorkflowTriggerStatus.FAILED
                assert trigger.execution_owner_id is None
                assert trigger.dispatch_published_at is None
                assert trigger.dispatch_abandoned_at is not None
                assert (
                    SQLAlchemyWorkflowTriggerLogRepository(verify).confirm_dispatch_published(
                        trigger_log_id,
                        celery_task_id=task_id,
                    )
                    is False
                )
            else:
                assert trigger.status == WorkflowTriggerStatus.QUEUED
                assert trigger.execution_owner_id is None
                assert trigger.dispatch_published_at is not None
                assert trigger.dispatch_abandoned_at is None
    finally:
        allow_reaper_cas.set()
        _join_if_started(reaper)
        stale_worker.close()
        event.remove(engine, "after_cursor_execute", pause_after_candidate_snapshot)
        with sessions.begin() as cleanup:
            cleanup.execute(delete(WorkflowTriggerLog).where(WorkflowTriggerLog.id == trigger_log_id))

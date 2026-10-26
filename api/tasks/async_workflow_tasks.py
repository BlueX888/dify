"""
Celery tasks for async workflow execution.

These tasks handle workflow execution for different subscription tiers
with appropriate retry policies and error handling.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Self
from uuid import uuid4

from celery import Task, shared_task
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from typing_extensions import TypedDict

from configs import dify_config
from core.app.apps.workflow.app_generator import WorkflowAppGenerator
from core.app.entities.app_invoke_entities import InvokeFrom
from core.app.layers.pause_state_persist_layer import PauseStateLayerConfig
from core.app.layers.trigger_post_layer import TriggerPostLayer
from core.db.session_factory import session_factory
from core.rag.entities.retrieval_settings import ExternalDatasetRetriever
from extensions.ext_redis import redis_client
from models.account import Account
from models.enums import CreatorUserRole, WorkflowTriggerStatus
from models.model import App, EndUser, Tenant
from models.trigger import WorkflowTriggerLog
from models.workflow import Workflow
from repositories.sqlalchemy_workflow_trigger_log_repository import SQLAlchemyWorkflowTriggerLogRepository
from repositories.workflow_trigger_log_repository import WorkflowExecutionLeaseState
from services.app_task_service import AppTaskControlService
from services.errors.app import WorkflowNotFoundError
from services.workflow.entities import (
    TriggerData,
    WorkflowTaskData,
)
from services.workflow_run_agg import WorkflowRunAgg
from services.workflow_trigger_quota_service import WorkflowTriggerQuotaService
from tasks.workflow_cfs_scheduler.entities import AsyncWorkflowQueue

logger = logging.getLogger(__name__)

_PENDING_EXECUTION_RETRY_SECONDS = 1
_QUOTA_SETTLEMENT_RETRY_SECONDS = 60
_DISPATCH_RECONCILIATION_BATCH_SIZE = 100
_ABANDONED_PENDING_EXECUTION_ERROR = (
    "Async workflow dispatch did not become executable before the compatibility deadline"
)
_ABANDONED_QUEUED_DISPATCH_ERROR = "Async workflow broker publication was not confirmed before the dispatch deadline"


class WorkflowGeneratorArgsDict(TypedDict):
    inputs: dict[str, Any]
    files: list[Any]


class _WorkflowExecutionLeaseActiveError(Exception):
    def __init__(self, retry_after: int) -> None:
        super().__init__("Workflow execution lease is still active")
        self.retry_after = retry_after


@shared_task(queue=AsyncWorkflowQueue.SANDBOX_QUEUE)
def reconcile_async_workflow_dispatches() -> None:
    """Fence unpublished dispatches and settle their durable quota receipts."""
    dispatch_timeout = timedelta(seconds=dify_config.ASYNC_WORKFLOW_EXECUTION_LEASE_SECONDS)
    with session_factory.create_session() as session:
        repository = SQLAlchemyWorkflowTriggerLogRepository(session)
        abandoned = repository.abandon_stale_unpublished_dispatches(
            dispatch_timeout=dispatch_timeout,
            error=_ABANDONED_QUEUED_DISPATCH_ERROR,
            limit=_DISPATCH_RECONCILIATION_BATCH_SIZE,
        )
        session.commit()
    if abandoned:
        logger.error("Abandoned %d async workflow dispatches without broker receipts", abandoned)

    with session_factory.create_session() as session:
        repository = SQLAlchemyWorkflowTriggerLogRepository(session)
        candidates = repository.claim_quota_settlement_candidates(
            retry_interval=timedelta(seconds=_QUOTA_SETTLEMENT_RETRY_SECONDS),
            limit=_DISPATCH_RECONCILIATION_BATCH_SIZE,
        )
        session.commit()

    for candidate in candidates:
        settlement_status = WorkflowTriggerQuotaService.settle(candidate)
        if settlement_status is None:
            continue
        with session_factory.create_session() as session:
            confirmed = SQLAlchemyWorkflowTriggerLogRepository(session).confirm_quota_settlement(
                candidate.trigger_log_id,
                attempted_at=candidate.attempted_at,
                settlement_status=settlement_status,
            )
            session.commit()
        if not confirmed:
            logger.info("Quota settlement attempt was superseded for workflow trigger %s", candidate.trigger_log_id)


@shared_task(
    bind=True,
    queue=AsyncWorkflowQueue.PROFESSIONAL_QUEUE,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=None,
)
def execute_workflow_professional(task: Task, task_data_dict: dict[str, Any]):
    """Execute workflow for professional tier with highest priority"""
    from extensions.ext_application_services import application_services

    task_data = WorkflowTaskData.model_validate(task_data_dict)
    try:
        _execute_workflow_common(
            task_data,
            external_retrieval=application_services().knowledge.external_retrieval,
        )
    except _WorkflowExecutionLeaseActiveError as error:
        raise task.retry(countdown=error.retry_after, max_retries=None) from None


@shared_task(
    bind=True,
    queue=AsyncWorkflowQueue.TEAM_QUEUE,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=None,
)
def execute_workflow_team(task: Task, task_data_dict: dict[str, Any]):
    """Execute workflow for team tier"""
    from extensions.ext_application_services import application_services

    task_data = WorkflowTaskData.model_validate(task_data_dict)
    try:
        _execute_workflow_common(
            task_data,
            external_retrieval=application_services().knowledge.external_retrieval,
        )
    except _WorkflowExecutionLeaseActiveError as error:
        raise task.retry(countdown=error.retry_after, max_retries=None) from None


@shared_task(
    bind=True,
    queue=AsyncWorkflowQueue.SANDBOX_QUEUE,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=None,
)
def execute_workflow_sandbox(task: Task, task_data_dict: dict[str, Any]):
    """Execute workflow for free tier with lower retry limit"""
    from extensions.ext_application_services import application_services

    task_data = WorkflowTaskData.model_validate(task_data_dict)
    try:
        _execute_workflow_common(
            task_data,
            external_retrieval=application_services().knowledge.external_retrieval,
        )
    except _WorkflowExecutionLeaseActiveError as error:
        raise task.retry(countdown=error.retry_after, max_retries=None) from None


def _build_generator_args(trigger_data: TriggerData) -> WorkflowGeneratorArgsDict:
    """Build args passed into WorkflowAppGenerator.generate for Celery executions."""
    return {
        "inputs": dict(trigger_data.inputs),
        "files": list(trigger_data.files),
    }


class _WorkflowExecutionLeaseHeartbeat:
    """Renew one trigger-log execution lease while its workflow is running."""

    def __init__(
        self,
        *,
        database_client: sessionmaker[Session],
        trigger_log_id: str,
        owner_id: str,
        lease_duration: timedelta,
        last_confirmed_at: float,
        on_ownership_lost: Callable[[], None],
    ) -> None:
        self._database_client = database_client
        self._trigger_log_id = trigger_log_id
        self._owner_id = owner_id
        self._lease_duration = lease_duration
        self._on_ownership_lost = on_ownership_lost
        lease_seconds = lease_duration.total_seconds()
        self._interval = min(60.0, max(0.05, lease_seconds / 3))
        self._lease_deadline = last_confirmed_at + lease_seconds
        self._stop = Event()
        self._deadline_changed = Event()
        self._state_lock = Lock()
        self._ownership_lost = False
        self._renewal_thread = Thread(target=self._renew, name=f"workflow-lease-renew-{trigger_log_id}", daemon=True)
        self._deadline_thread = Thread(
            target=self._watch_deadline,
            name=f"workflow-lease-deadline-{trigger_log_id}",
            daemon=True,
        )

    def __enter__(self) -> Self:
        self._deadline_thread.start()
        self._renewal_thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._stop.set()
        self._deadline_changed.set()
        self._deadline_thread.join(timeout=1)
        self._renewal_thread.join(timeout=1)

    def _notify_ownership_lost(self, *, expired_deadline: float | None = None) -> bool:
        with self._state_lock:
            if self._stop.is_set() or self._ownership_lost:
                return False
            if expired_deadline is not None and (
                self._lease_deadline != expired_deadline or monotonic() < expired_deadline
            ):
                return False
            self._ownership_lost = True
        self._stop.set()
        self._deadline_changed.set()
        self._on_ownership_lost()
        return True

    def _stop_after_normal_completion(self) -> None:
        """Stop renewing after this claim's persisted terminal transition."""
        with self._state_lock:
            if self._ownership_lost:
                return
            self._stop.set()
            self._deadline_changed.set()

    def _read_current_lease_state(self) -> WorkflowExecutionLeaseState | None:
        # Deliberately use a separate Session from the failed renewal. This is
        # a fresh transaction and the repository uses a locking current read,
        # so a committed TriggerPostLayer pause/finish cannot be hidden by a
        # repeatable-read snapshot established during renewal.
        with self._database_client() as session:
            state = SQLAlchemyWorkflowTriggerLogRepository(session).get_current_execution_lease_state(
                self._trigger_log_id
            )
            session.commit()
            return state

    def _reconcile_current_lease_state(self, state: WorkflowExecutionLeaseState | None, *, observed_at: float) -> None:
        if state is not None and state.status in {
            WorkflowTriggerStatus.PAUSED,
            WorkflowTriggerStatus.SUCCEEDED,
            WorkflowTriggerStatus.FAILED,
        }:
            if state.terminal_owner_id == self._owner_id:
                self._stop_after_normal_completion()
                return

        if (
            state is not None
            and state.status == WorkflowTriggerStatus.RUNNING
            and state.owner_id == self._owner_id
            and state.lease_expires_at is not None
        ):
            remaining = (state.lease_expires_at - state.database_now).total_seconds()
            if remaining > 0:
                # The persisted claim is still valid. This can happen when a
                # concurrent heartbeat for the same execution owner renewed it.
                # Anchor at the conservative pre-I/O boundary.
                with self._state_lock:
                    if self._stop.is_set() or self._ownership_lost:
                        return
                    self._lease_deadline = max(self._lease_deadline, observed_at + remaining)
                self._deadline_changed.set()
                return

        # Missing rows, non-running states, a different owner, and expired or
        # absent leases all mean this worker no longer has execution authority.
        self._notify_ownership_lost()

    def _watch_deadline(self) -> None:
        while not self._stop.is_set():
            self._deadline_changed.clear()
            with self._state_lock:
                deadline = self._lease_deadline
            remaining = deadline - monotonic()
            if remaining <= 0:
                observed_at = monotonic()
                try:
                    state = self._read_current_lease_state()
                except Exception:
                    logger.exception("Failed to confirm workflow execution lease at deadline: %s", self._trigger_log_id)
                    # A database outage must not let a worker run beyond its
                    # last confirmed lease. A concurrent successful renewal
                    # changes the deadline and suppresses this stale timeout.
                    if self._notify_ownership_lost(expired_deadline=deadline):
                        return
                    continue
                self._reconcile_current_lease_state(state, observed_at=observed_at)
                if self._stop.is_set():
                    return
                continue
            self._deadline_changed.wait(remaining)

    def _renew(self) -> None:
        while not self._stop.wait(self._interval):
            renewal_started_at = monotonic()
            with self._state_lock:
                deadline = self._lease_deadline
            if renewal_started_at >= deadline:
                try:
                    state = self._read_current_lease_state()
                except Exception:
                    logger.exception(
                        "Failed to confirm workflow execution lease before late renewal: %s", self._trigger_log_id
                    )
                    self._notify_ownership_lost(expired_deadline=deadline)
                    return
                self._reconcile_current_lease_state(state, observed_at=renewal_started_at)
                if self._stop.is_set():
                    return
                continue
            try:
                with self._database_client() as session:
                    renewed = SQLAlchemyWorkflowTriggerLogRepository(session).renew_execution_lease(
                        self._trigger_log_id,
                        owner_id=self._owner_id,
                        lease_duration=self._lease_duration,
                    )
                    if self._stop.is_set():
                        session.rollback()
                        return
                    session.commit()
                if self._stop.is_set():
                    return
                if not renewed:
                    observed_at = monotonic()
                    try:
                        state = self._read_current_lease_state()
                    except Exception:
                        # A transient verification failure is not evidence of
                        # takeover. Keep retrying until the confirmed deadline;
                        # the deadline watcher still enforces bounded ownership.
                        logger.exception(
                            "Failed to verify workflow execution lease after renewal rejection: %s",
                            self._trigger_log_id,
                        )
                        continue
                    self._reconcile_current_lease_state(state, observed_at=observed_at)
                    if self._stop.is_set():
                        return
                    continue
                # The database lease starts during the renewal transaction.
                # Measure from the conservative pre-I/O boundary so connection
                # stalls cannot extend this worker's local ownership window.
                with self._state_lock:
                    if self._ownership_lost:
                        return
                    if monotonic() >= self._lease_deadline:
                        expired = True
                    else:
                        expired = False
                        self._lease_deadline = renewal_started_at + self._lease_duration.total_seconds()
                if expired:
                    self._notify_ownership_lost(expired_deadline=deadline)
                    return
                self._deadline_changed.set()
            except Exception:
                logger.exception("Failed to renew workflow execution lease: %s", self._trigger_log_id)


def _execute_workflow_common(
    task_data: WorkflowTaskData,
    *,
    external_retrieval: ExternalDatasetRetriever,
):
    """Execute workflow with common logic and trigger log updates."""

    with session_factory.create_session() as session:
        trigger_log_repo = SQLAlchemyWorkflowTriggerLogRepository(session)

        execution_owner_id = str(uuid4())
        workflow_task_id = str(uuid4())
        lease_duration = timedelta(seconds=dify_config.ASYNC_WORKFLOW_EXECUTION_LEASE_SECONDS)
        claim_started_at = monotonic()
        # Commit the claim before loading workflow data or running side effects.
        # A redelivery can replace a worker only after its persisted lease expires.
        trigger_log = trigger_log_repo.claim_execution(
            task_data.workflow_trigger_log_id,
            owner_id=execution_owner_id,
            lease_duration=lease_duration,
        )

        if not trigger_log:
            abandoned = trigger_log_repo.fail_abandoned_pending_execution(
                task_data.workflow_trigger_log_id,
                pending_timeout=lease_duration,
                error=_ABANDONED_PENDING_EXECUTION_ERROR,
            )
            if abandoned:
                session.commit()
                logger.error(
                    "Abandoned async workflow dispatch was marked failed: %s",
                    task_data.workflow_trigger_log_id,
                )
                return
            current = trigger_log_repo.get_by_id(task_data.workflow_trigger_log_id)
            retry_after = None
            if current is not None:
                if current.pending_dispatch_abandoned_at is not None:
                    logger.error(
                        "Abandoned async workflow dispatch remains terminal: %s",
                        task_data.workflow_trigger_log_id,
                    )
                elif current.status in {WorkflowTriggerStatus.PENDING, WorkflowTriggerStatus.QUEUED}:
                    # Compatibility with messages published by older producers
                    # before they committed the executable QUEUED state. Never
                    # execute a row that this claim attempt did not acquire.
                    # The producer may commit QUEUED between the failed claim
                    # and this current-state read, so both states retry shortly.
                    retry_after = _PENDING_EXECUTION_RETRY_SECONDS
                elif current.status == WorkflowTriggerStatus.RUNNING:
                    retry_after = dify_config.ASYNC_WORKFLOW_EXECUTION_LEASE_SECONDS
            session.rollback()
            if retry_after is not None:
                raise _WorkflowExecutionLeaseActiveError(retry_after)
            return
        session.commit()

        start_time = datetime.now(UTC)

        try:
            with _WorkflowExecutionLeaseHeartbeat(
                database_client=session_factory.get_session_maker(),
                trigger_log_id=trigger_log.id,
                owner_id=execution_owner_id,
                lease_duration=lease_duration,
                last_confirmed_at=claim_started_at,
                on_ownership_lost=lambda: AppTaskControlService(
                    redis_client=redis_client
                ).stop_workflow_task_no_user_check(task_id=workflow_task_id),
            ):
                # Reconstruct execution data from trigger log
                trigger_data = TriggerData.model_validate_json(trigger_log.trigger_data)

                # Get app and workflow models
                app_model = session.scalar(select(App).where(App.id == trigger_log.app_id))

                if not app_model:
                    raise WorkflowNotFoundError(f"App not found: {trigger_log.app_id}")

                workflow = session.scalar(select(Workflow).where(Workflow.id == trigger_log.workflow_id))
                if not workflow:
                    raise WorkflowNotFoundError(f"Workflow not found: {trigger_log.workflow_id}")

                user = _get_user(session, trigger_log)

                # Execute workflow using WorkflowAppGenerator
                generator = WorkflowAppGenerator(
                    execution_driver=WorkflowRunAgg.run, external_retrieval=external_retrieval
                )

                # Adapt trigger inputs and files for the generator.
                args = _build_generator_args(trigger_data)

                pause_config = PauseStateLayerConfig(
                    session_factory=session_factory.get_session_maker(),
                    state_owner_user_id=workflow.created_by,
                )

                # NOTE (hj24)
                # Release the transaction before the blocking generate() call,
                # otherwise the connection stays "idle in transaction" for hours.
                session.commit()
                # NOTE END

                trigger_post_layer = TriggerPostLayer(
                    start_time,
                    trigger_log.id,
                    execution_owner_id,
                )
                generator.generate(
                    app_model=app_model,
                    workflow=workflow,
                    user=user,
                    args=args,
                    invoke_from=InvokeFrom.SERVICE_API,
                    streaming=False,
                    call_depth=0,
                    task_id=workflow_task_id,
                    triggered_from=trigger_data.trigger_from,
                    root_node_id=trigger_data.root_node_id,
                    graph_engine_layers=[
                        # TODO: Re-enable TimeSliceLayer after the HITL release.
                        trigger_post_layer,
                    ],
                    pause_state_config=pause_config,
                )
                if trigger_post_layer.pause_publication_outcome_unknown:
                    logger.warning(
                        "Workflow pause publication outcome is unknown; retrying trigger delivery: %s",
                        trigger_log.id,
                    )
                    raise _WorkflowExecutionLeaseActiveError(dify_config.ASYNC_WORKFLOW_EXECUTION_LEASE_SECONDS)

        except _WorkflowExecutionLeaseActiveError:
            raise
        except Exception as e:
            # Calculate elapsed time for failed execution
            elapsed_time = (datetime.now(UTC) - start_time).total_seconds()

            session.rollback()
            failed = trigger_log_repo.fail_execution(
                trigger_log.id,
                owner_id=execution_owner_id,
                error=str(e),
                elapsed_time=elapsed_time,
                finished_at=datetime.now(UTC),
            )
            session.commit()
            if not failed:
                logger.warning("Workflow execution ownership lost before failure update: %s", trigger_log.id)


def _get_user(session: Session, workflow_run: WorkflowTriggerLog) -> Account | EndUser:
    """Compose user from trigger log"""
    tenant = session.scalar(select(Tenant).where(Tenant.id == workflow_run.tenant_id))
    if not tenant:
        raise _TenantNotFoundError(
            "Tenant not found for WorkflowRun: tenant_id=%s, workflow_run_id=%s",
            workflow_run.tenant_id,
            workflow_run.id,
        )

    # Get user from trigger log
    if workflow_run.created_by_role == CreatorUserRole.ACCOUNT:
        user = session.scalar(select(Account).where(Account.id == workflow_run.created_by))
        if user:
            user.set_current_tenant_with_session(tenant, session=session)
    else:  # CreatorUserRole.END_USER
        user = session.scalar(select(EndUser).where(EndUser.id == workflow_run.created_by))

    if not user:
        raise _UserNotFoundError(
            "User not found: user_id=%s, created_by_role=%s, workflow_run_id=%s",
            workflow_run.created_by,
            workflow_run.created_by_role,
            workflow_run.id,
        )

    return user


class _TenantNotFoundError(Exception):
    pass


class _UserNotFoundError(Exception):
    pass

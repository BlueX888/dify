"""
SQLAlchemy implementation of WorkflowTriggerLogRepository.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal, cast, override

from sqlalchemy import and_, delete, func, or_, select, true, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from models.enums import WorkflowTriggerStatus
from models.trigger import WorkflowTriggerLog
from repositories.workflow_trigger_log_repository import (
    WorkflowExecutionLeaseState,
    WorkflowQuotaSettlementCandidate,
    WorkflowTriggerLogRepository,
)


class SQLAlchemyWorkflowTriggerLogRepository(WorkflowTriggerLogRepository):
    """
    SQLAlchemy implementation of WorkflowTriggerLogRepository.

    Optimized for large table operations with proper indexing and batch processing.
    """

    def __init__(self, session: Session):
        self.session = session

    @override
    def create(self, trigger_log: WorkflowTriggerLog) -> WorkflowTriggerLog:
        """Create a new trigger log entry."""
        self.session.add(trigger_log)
        self.session.flush()
        return trigger_log

    @override
    def update(self, trigger_log: WorkflowTriggerLog) -> WorkflowTriggerLog:
        """Update an existing trigger log entry."""
        self.session.merge(trigger_log)
        self.session.flush()
        return trigger_log

    @override
    def claim_execution(
        self,
        trigger_log_id: str,
        *,
        owner_id: str,
        lease_duration: timedelta,
    ) -> WorkflowTriggerLog | None:
        """Claim a queued trigger or replace the owner of an expired execution."""
        database_now = self.session.execute(select(func.current_timestamp())).scalar_one()
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(
                WorkflowTriggerLog.id == trigger_log_id,
                or_(
                    and_(
                        WorkflowTriggerLog.status == WorkflowTriggerStatus.QUEUED,
                        WorkflowTriggerLog.pending_dispatch_abandoned_at.is_(None),
                    ),
                    and_(
                        WorkflowTriggerLog.status == WorkflowTriggerStatus.RUNNING,
                        or_(
                            WorkflowTriggerLog.execution_lease_expires_at.is_(None),
                            WorkflowTriggerLog.execution_lease_expires_at <= func.current_timestamp(),
                        ),
                    ),
                ),
            )
            .values(
                status=WorkflowTriggerStatus.RUNNING,
                execution_owner_id=owner_id,
                execution_lease_expires_at=database_now + lease_duration,
                terminal_owner_id=None,
                quota_sync_pending=and_(
                    WorkflowTriggerLog.dispatch_prepared_at.is_not(None),
                    WorkflowTriggerLog.quota_settlement_status.is_(None),
                ),
            )
            .execution_options(synchronize_session=False)
        )
        if cast(CursorResult, result).rowcount != 1:
            return None
        return self.session.scalar(
            select(WorkflowTriggerLog)
            .where(WorkflowTriggerLog.id == trigger_log_id)
            .execution_options(populate_existing=True)
        )

    @override
    def fail_abandoned_pending_execution(
        self,
        trigger_log_id: str,
        *,
        pending_timeout: timedelta,
        error: str,
    ) -> bool:
        """Terminate a legacy dispatch that never became safely claimable."""
        database_now = self.session.execute(select(func.current_timestamp())).scalar_one()
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(
                WorkflowTriggerLog.id == trigger_log_id,
                WorkflowTriggerLog.created_at <= database_now - pending_timeout,
                WorkflowTriggerLog.status == WorkflowTriggerStatus.PENDING,
                WorkflowTriggerLog.pending_dispatch_abandoned_at.is_(None),
            )
            .values(
                status=WorkflowTriggerStatus.FAILED,
                error=error,
                finished_at=database_now,
                pending_dispatch_abandoned_at=database_now,
                execution_owner_id=None,
                execution_lease_expires_at=None,
                quota_sync_pending=False,
            )
            .execution_options(synchronize_session=False)
        )
        return cast(CursorResult, result).rowcount == 1

    @override
    def queue_pending_execution(
        self,
        trigger_log_id: str,
        *,
        celery_task_id: str,
        triggered_at: datetime,
        quota_reservation_id: str | None = None,
    ) -> bool:
        """Publish executable state only while the pending dispatch is live."""
        database_now = self.session.execute(select(func.current_timestamp())).scalar_one()
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(
                WorkflowTriggerLog.id == trigger_log_id,
                WorkflowTriggerLog.status == WorkflowTriggerStatus.PENDING,
                WorkflowTriggerLog.pending_dispatch_abandoned_at.is_(None),
            )
            .values(
                status=WorkflowTriggerStatus.QUEUED,
                celery_task_id=celery_task_id,
                triggered_at=triggered_at,
                dispatch_prepared_at=database_now,
                quota_reservation_id=quota_reservation_id,
                quota_sync_pending=False,
            )
            .execution_options(synchronize_session=False)
        )
        return cast(CursorResult, result).rowcount == 1

    @override
    def confirm_dispatch_published(self, trigger_log_id: str, *, celery_task_id: str) -> bool:
        """Persist an idempotent broker publication receipt for the exact dispatch."""
        database_now = self.session.execute(select(func.current_timestamp())).scalar_one()
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(
                WorkflowTriggerLog.id == trigger_log_id,
                WorkflowTriggerLog.celery_task_id == celery_task_id,
                WorkflowTriggerLog.dispatch_prepared_at.is_not(None),
                WorkflowTriggerLog.dispatch_published_at.is_(None),
                WorkflowTriggerLog.dispatch_abandoned_at.is_(None),
            )
            .values(
                dispatch_published_at=database_now,
                quota_sync_pending=WorkflowTriggerLog.quota_settlement_status.is_(None),
            )
            .execution_options(synchronize_session=False)
        )
        if cast(CursorResult, result).rowcount == 1:
            return True
        return (
            self.session.scalar(
                select(WorkflowTriggerLog.id).where(
                    WorkflowTriggerLog.id == trigger_log_id,
                    WorkflowTriggerLog.celery_task_id == celery_task_id,
                    WorkflowTriggerLog.dispatch_prepared_at.is_not(None),
                    WorkflowTriggerLog.dispatch_published_at.is_not(None),
                    WorkflowTriggerLog.dispatch_abandoned_at.is_(None),
                )
            )
            is not None
        )

    @override
    def abandon_stale_unpublished_dispatches(
        self,
        *,
        dispatch_timeout: timedelta,
        error: str,
        limit: int,
    ) -> int:
        """Fence a bounded batch of current-protocol dispatches with no broker receipt."""
        database_now = self.session.execute(select(func.current_timestamp())).scalar_one()
        cutoff = database_now - dispatch_timeout
        candidates = self.session.execute(
            select(WorkflowTriggerLog.id, WorkflowTriggerLog.celery_task_id)
            .where(
                WorkflowTriggerLog.status == WorkflowTriggerStatus.QUEUED,
                WorkflowTriggerLog.dispatch_prepared_at.is_not(None),
                WorkflowTriggerLog.dispatch_prepared_at <= cutoff,
                WorkflowTriggerLog.dispatch_published_at.is_(None),
                WorkflowTriggerLog.dispatch_abandoned_at.is_(None),
                WorkflowTriggerLog.celery_task_id.is_not(None),
            )
            .order_by(WorkflowTriggerLog.dispatch_prepared_at.asc(), WorkflowTriggerLog.id.asc())
            .limit(limit)
        ).all()
        abandoned = 0
        for trigger_log_id, celery_task_id in candidates:
            result = self.session.execute(
                update(WorkflowTriggerLog)
                .where(
                    WorkflowTriggerLog.id == trigger_log_id,
                    WorkflowTriggerLog.celery_task_id == celery_task_id,
                    WorkflowTriggerLog.status == WorkflowTriggerStatus.QUEUED,
                    WorkflowTriggerLog.dispatch_prepared_at.is_not(None),
                    WorkflowTriggerLog.dispatch_prepared_at <= cutoff,
                    WorkflowTriggerLog.dispatch_published_at.is_(None),
                    WorkflowTriggerLog.dispatch_abandoned_at.is_(None),
                )
                .values(
                    status=WorkflowTriggerStatus.FAILED,
                    error=error,
                    finished_at=database_now,
                    dispatch_abandoned_at=database_now,
                    execution_owner_id=None,
                    execution_lease_expires_at=None,
                    quota_sync_pending=WorkflowTriggerLog.quota_settlement_status.is_(None),
                )
                .execution_options(synchronize_session=False)
            )
            abandoned += int(cast(CursorResult, result).rowcount == 1)
        return abandoned

    @override
    def claim_quota_settlement_candidates(
        self,
        *,
        retry_interval: timedelta,
        limit: int,
    ) -> Sequence[WorkflowQuotaSettlementCandidate]:
        """Rotate unresolved settlements and lease each attempt without holding an RPC lock."""
        database_now = self.session.execute(select(func.current_timestamp())).scalar_one()
        retry_before = database_now - retry_interval
        retryable = or_(
            WorkflowTriggerLog.quota_settlement_attempted_at.is_(None),
            WorkflowTriggerLog.quota_settlement_attempted_at <= retry_before,
        )
        candidate_columns = (
            WorkflowTriggerLog.id,
            WorkflowTriggerLog.tenant_id,
            WorkflowTriggerLog.quota_reservation_id,
            WorkflowTriggerLog.dispatch_abandoned_at,
        )
        base_conditions = (
            WorkflowTriggerLog.quota_sync_pending == true(),
            WorkflowTriggerLog.quota_settlement_status.is_(None),
        )
        rows = list(
            self.session.execute(
                select(*candidate_columns)
                .where(
                    *base_conditions,
                    WorkflowTriggerLog.quota_settlement_attempted_at.is_(None),
                )
                .order_by(
                    WorkflowTriggerLog.dispatch_prepared_at.asc(),
                    WorkflowTriggerLog.id.asc(),
                )
                .limit(limit)
            ).all()
        )
        remaining = limit - len(rows)
        if remaining > 0:
            rows.extend(
                self.session.execute(
                    select(*candidate_columns)
                    .where(
                        *base_conditions,
                        WorkflowTriggerLog.quota_settlement_attempted_at.is_not(None),
                        WorkflowTriggerLog.quota_settlement_attempted_at <= retry_before,
                    )
                    .order_by(
                        WorkflowTriggerLog.quota_settlement_attempted_at.asc(),
                        WorkflowTriggerLog.dispatch_prepared_at.asc(),
                        WorkflowTriggerLog.id.asc(),
                    )
                    .limit(remaining)
                ).all()
            )

        claimed: list[WorkflowQuotaSettlementCandidate] = []
        for trigger_log_id, tenant_id, reservation_id, abandoned_at in rows:
            result = self.session.execute(
                update(WorkflowTriggerLog)
                .where(
                    WorkflowTriggerLog.id == trigger_log_id,
                    WorkflowTriggerLog.quota_sync_pending == true(),
                    WorkflowTriggerLog.quota_settlement_status.is_(None),
                    retryable,
                )
                .values(quota_settlement_attempted_at=database_now)
                .execution_options(synchronize_session=False)
            )
            if cast(CursorResult, result).rowcount != 1:
                continue
            claimed.append(
                WorkflowQuotaSettlementCandidate(
                    trigger_log_id=trigger_log_id,
                    tenant_id=tenant_id,
                    reservation_id=reservation_id,
                    outcome="refunded" if abandoned_at is not None else "committed",
                    attempted_at=database_now,
                )
            )
        return claimed

    @override
    def claim_quota_settlement(
        self,
        trigger_log_id: str,
        *,
        retry_interval: timedelta,
    ) -> WorkflowQuotaSettlementCandidate | None:
        """Lease one exact quota settlement attempt for producer or worker fast paths."""
        database_now = self.session.execute(select(func.current_timestamp())).scalar_one()
        retry_before = database_now - retry_interval
        retryable = or_(
            WorkflowTriggerLog.quota_settlement_attempted_at.is_(None),
            WorkflowTriggerLog.quota_settlement_attempted_at <= retry_before,
        )
        row = self.session.execute(
            select(
                WorkflowTriggerLog.tenant_id,
                WorkflowTriggerLog.quota_reservation_id,
                WorkflowTriggerLog.dispatch_abandoned_at,
            ).where(
                WorkflowTriggerLog.id == trigger_log_id,
                WorkflowTriggerLog.quota_sync_pending == true(),
                WorkflowTriggerLog.quota_settlement_status.is_(None),
                retryable,
            )
        ).one_or_none()
        if row is None:
            return None
        tenant_id, reservation_id, abandoned_at = row
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(
                WorkflowTriggerLog.id == trigger_log_id,
                WorkflowTriggerLog.quota_sync_pending == true(),
                WorkflowTriggerLog.quota_settlement_status.is_(None),
                retryable,
            )
            .values(quota_settlement_attempted_at=database_now)
            .execution_options(synchronize_session=False)
        )
        if cast(CursorResult, result).rowcount != 1:
            return None
        return WorkflowQuotaSettlementCandidate(
            trigger_log_id=trigger_log_id,
            tenant_id=tenant_id,
            reservation_id=reservation_id,
            outcome="refunded" if abandoned_at is not None else "committed",
            attempted_at=database_now,
        )

    @override
    def confirm_quota_settlement(
        self,
        trigger_log_id: str,
        *,
        attempted_at: datetime,
        settlement_status: Literal["committed", "refunded", "permanent_error"],
    ) -> bool:
        """Persist the terminal result only for the attempt that performed the RPC."""
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(
                WorkflowTriggerLog.id == trigger_log_id,
                WorkflowTriggerLog.quota_settlement_status.is_(None),
                WorkflowTriggerLog.quota_settlement_attempted_at == attempted_at,
            )
            .values(quota_settlement_status=settlement_status, quota_sync_pending=False)
            .execution_options(synchronize_session=False)
        )
        return cast(CursorResult, result).rowcount == 1

    @override
    def renew_execution_lease(
        self,
        trigger_log_id: str,
        *,
        owner_id: str,
        lease_duration: timedelta,
    ) -> bool:
        """Extend the lease without allowing a previous owner to reclaim it."""
        database_now = self.session.execute(select(func.current_timestamp())).scalar_one()
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(
                WorkflowTriggerLog.id == trigger_log_id,
                WorkflowTriggerLog.status == WorkflowTriggerStatus.RUNNING,
                WorkflowTriggerLog.execution_owner_id == owner_id,
            )
            .values(execution_lease_expires_at=database_now + lease_duration)
            .execution_options(synchronize_session=False)
        )
        return cast(CursorResult, result).rowcount == 1

    @override
    def get_current_execution_lease_state(self, trigger_log_id: str) -> WorkflowExecutionLeaseState | None:
        """Read lease state with a locking current read.

        The heartbeat calls this from a new Session after a failed renewal. A
        locking read is intentional: under MySQL's default repeatable-read
        isolation it observes the latest committed pause or terminal update,
        rather than an older consistent-read snapshot.
        """
        row = self.session.execute(
            select(
                WorkflowTriggerLog.status,
                WorkflowTriggerLog.execution_owner_id,
                WorkflowTriggerLog.execution_lease_expires_at,
                WorkflowTriggerLog.terminal_owner_id,
                func.current_timestamp(),
            )
            .where(WorkflowTriggerLog.id == trigger_log_id)
            .with_for_update()
        ).one_or_none()
        if row is None:
            return None

        status, owner_id, lease_expires_at, terminal_owner_id, database_now = row
        return WorkflowExecutionLeaseState(
            status=WorkflowTriggerStatus(status),
            owner_id=owner_id,
            lease_expires_at=lease_expires_at,
            terminal_owner_id=terminal_owner_id,
            database_now=database_now,
        )

    @override
    def finish_execution(
        self,
        trigger_log_id: str,
        *,
        owner_id: str | None,
        status: WorkflowTriggerStatus,
        workflow_run_id: str,
        outputs: str,
        error: str | None,
        elapsed_time: float,
        total_tokens: int,
        finished_at: datetime,
    ) -> bool:
        """Persist a terminal event only while the caller still owns the execution."""
        ownership = (
            and_(
                WorkflowTriggerLog.status == WorkflowTriggerStatus.PAUSED,
                WorkflowTriggerLog.execution_owner_id.is_(None),
            )
            if owner_id is None
            else and_(
                WorkflowTriggerLog.status == WorkflowTriggerStatus.RUNNING,
                WorkflowTriggerLog.execution_owner_id == owner_id,
            )
        )
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(WorkflowTriggerLog.id == trigger_log_id, ownership)
            .values(
                status=status,
                workflow_run_id=workflow_run_id,
                outputs=outputs,
                error=error,
                elapsed_time=func.coalesce(WorkflowTriggerLog.elapsed_time, 0) + elapsed_time,
                total_tokens=total_tokens,
                finished_at=finished_at,
                execution_owner_id=None,
                execution_lease_expires_at=None,
                terminal_owner_id=(owner_id if owner_id is not None else WorkflowTriggerLog.terminal_owner_id),
                quota_sync_pending=and_(
                    WorkflowTriggerLog.dispatch_prepared_at.is_not(None),
                    WorkflowTriggerLog.quota_settlement_status.is_(None),
                ),
            )
            .execution_options(synchronize_session=False)
        )
        return cast(CursorResult, result).rowcount == 1

    @override
    def fail_execution(
        self,
        trigger_log_id: str,
        *,
        owner_id: str,
        error: str,
        elapsed_time: float,
        finished_at: datetime,
    ) -> bool:
        """Fail a running trigger only while the caller still owns the execution."""
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(
                WorkflowTriggerLog.id == trigger_log_id,
                WorkflowTriggerLog.status == WorkflowTriggerStatus.RUNNING,
                WorkflowTriggerLog.execution_owner_id == owner_id,
            )
            .values(
                status=WorkflowTriggerStatus.FAILED,
                error=error,
                elapsed_time=func.coalesce(WorkflowTriggerLog.elapsed_time, 0) + elapsed_time,
                finished_at=finished_at,
                execution_owner_id=None,
                execution_lease_expires_at=None,
                terminal_owner_id=owner_id,
                quota_sync_pending=and_(
                    WorkflowTriggerLog.dispatch_prepared_at.is_not(None),
                    WorkflowTriggerLog.quota_settlement_status.is_(None),
                ),
            )
            .execution_options(synchronize_session=False)
        )
        return cast(CursorResult, result).rowcount == 1

    @override
    def mark_dispatch_failed(
        self,
        trigger_log_id: str,
        *,
        celery_task_id: str,
        error: str,
    ) -> bool:
        """Fail a trigger only if no worker has claimed the queued task."""
        database_now = self.session.execute(select(func.current_timestamp())).scalar_one()
        result = self.session.execute(
            update(WorkflowTriggerLog)
            .where(
                WorkflowTriggerLog.id == trigger_log_id,
                WorkflowTriggerLog.status == WorkflowTriggerStatus.QUEUED,
                WorkflowTriggerLog.celery_task_id == celery_task_id,
            )
            .values(
                status=WorkflowTriggerStatus.FAILED,
                error=error,
                finished_at=database_now,
                dispatch_abandoned_at=database_now,
                terminal_owner_id=None,
                quota_sync_pending=WorkflowTriggerLog.quota_settlement_status.is_(None),
            )
            .execution_options(synchronize_session=False)
        )
        return cast(CursorResult, result).rowcount == 1

    @override
    def get_by_id(self, trigger_log_id: str, tenant_id: str | None = None) -> WorkflowTriggerLog | None:
        """Get a trigger log by its ID."""
        query = select(WorkflowTriggerLog).where(WorkflowTriggerLog.id == trigger_log_id)

        if tenant_id:
            query = query.where(WorkflowTriggerLog.tenant_id == tenant_id)

        return self.session.scalar(query)

    def list_by_run_id(self, run_id: str) -> Sequence[WorkflowTriggerLog]:
        """List trigger logs for a workflow run."""
        query = select(WorkflowTriggerLog).where(WorkflowTriggerLog.workflow_run_id == run_id)
        return list(self.session.scalars(query).all())

    @override
    def get_failed_for_retry(
        self, tenant_id: str, max_retry_count: int = 3, limit: int = 100
    ) -> Sequence[WorkflowTriggerLog]:
        """Get failed trigger logs eligible for retry."""
        query = (
            select(WorkflowTriggerLog)
            .where(
                and_(
                    WorkflowTriggerLog.tenant_id == tenant_id,
                    WorkflowTriggerLog.status.in_([WorkflowTriggerStatus.FAILED, WorkflowTriggerStatus.RATE_LIMITED]),
                    WorkflowTriggerLog.retry_count < max_retry_count,
                )
            )
            .order_by(WorkflowTriggerLog.created_at.asc())
            .limit(limit)
        )

        return list(self.session.scalars(query).all())

    @override
    def get_recent_logs(
        self, tenant_id: str, app_id: str, hours: int = 24, limit: int = 100, offset: int = 0
    ) -> Sequence[WorkflowTriggerLog]:
        """Get recent trigger logs within specified hours."""
        since = datetime.now(UTC) - timedelta(hours=hours)

        query = (
            select(WorkflowTriggerLog)
            .where(
                and_(
                    WorkflowTriggerLog.tenant_id == tenant_id,
                    WorkflowTriggerLog.app_id == app_id,
                    WorkflowTriggerLog.created_at >= since,
                )
            )
            .order_by(WorkflowTriggerLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        return list(self.session.scalars(query).all())

    @override
    def get_by_workflow_run_id(self, workflow_run_id: str) -> WorkflowTriggerLog | None:
        """Get the trigger log associated with a workflow run."""
        query = (
            select(WorkflowTriggerLog)
            .where(WorkflowTriggerLog.workflow_run_id == workflow_run_id)
            .order_by(WorkflowTriggerLog.created_at.desc())
            .limit(1)
        )
        return self.session.scalar(query)

    @override
    def delete_by_run_ids(self, run_ids: Sequence[str]) -> int:
        """
        Delete trigger logs associated with the given workflow run ids.

        Args:
            run_ids: Collection of workflow run identifiers.

        Returns:
            Number of rows deleted.
        """
        if not run_ids:
            return 0

        result = self.session.execute(delete(WorkflowTriggerLog).where(WorkflowTriggerLog.workflow_run_id.in_(run_ids)))
        return cast(CursorResult, result).rowcount or 0

    def count_by_run_ids(self, run_ids: Sequence[str]) -> int:
        """
        Count trigger logs associated with the given workflow run ids.

        Args:
            run_ids: Collection of workflow run identifiers.

        Returns:
            Number of rows matched.
        """
        if not run_ids:
            return 0

        count = self.session.scalar(
            select(func.count()).select_from(WorkflowTriggerLog).where(WorkflowTriggerLog.workflow_run_id.in_(run_ids))
        )
        return int(count or 0)

"""
Repository protocol for WorkflowTriggerLog operations.

This module provides a protocol interface for operations on WorkflowTriggerLog,
designed to efficiently handle a potentially large volume of trigger logs with
proper indexing and batch operations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Protocol

from models.enums import WorkflowTriggerStatus
from models.trigger import WorkflowTriggerLog


class TriggerLogOrderBy(StrEnum):
    """Fields available for ordering trigger logs"""

    CREATED_AT = "created_at"
    TRIGGERED_AT = "triggered_at"
    FINISHED_AT = "finished_at"
    STATUS = "status"


@dataclass(frozen=True, slots=True)
class WorkflowExecutionLeaseState:
    """Current persisted ownership state for one trigger execution.

    ``terminal_owner_id`` is the claim token that committed the most recent
    pause or terminal state. It identifies an execution attempt, not a user.
    """

    status: WorkflowTriggerStatus
    owner_id: str | None
    lease_expires_at: datetime | None
    terminal_owner_id: str | None
    database_now: datetime


@dataclass(frozen=True, slots=True)
class WorkflowQuotaSettlementCandidate:
    """One claimed quota settlement attempt, safe to process outside a DB transaction."""

    trigger_log_id: str
    tenant_id: str
    reservation_id: str | None
    outcome: Literal["committed", "refunded"]
    attempted_at: datetime


class WorkflowTriggerLogRepository(Protocol):
    """
    Protocol for operations on WorkflowTriggerLog.

    This repository provides efficient access patterns for the trigger log table,
    which is expected to grow large over time. It includes:
    - Batch operations for cleanup
    - Efficient queries with proper indexing
    - Pagination support
    - Status-based filtering

    Implementation notes:
    - Leverage database indexes on (tenant_id, app_id), status, and created_at
    - Use batch operations for deletions to avoid locking
    - Support pagination for large result sets
    """

    def create(self, trigger_log: WorkflowTriggerLog) -> WorkflowTriggerLog:
        """
        Create a new trigger log entry.

        Args:
            trigger_log: The WorkflowTriggerLog instance to create

        Returns:
            The created WorkflowTriggerLog with generated ID
        """
        ...

    def update(self, trigger_log: WorkflowTriggerLog) -> WorkflowTriggerLog:
        """
        Update an existing trigger log entry.

        Args:
            trigger_log: The WorkflowTriggerLog instance to update

        Returns:
            The updated WorkflowTriggerLog
        """
        ...

    def claim_execution(
        self,
        trigger_log_id: str,
        *,
        owner_id: str,
        lease_duration: timedelta,
    ) -> WorkflowTriggerLog | None:
        """Atomically claim a queued trigger or take over an expired execution.

        Returns the claimed log, or ``None`` when another worker already
        owns an active lease or the trigger is terminal.
        """
        ...

    def fail_abandoned_pending_execution(
        self,
        trigger_log_id: str,
        *,
        pending_timeout: timedelta,
        error: str,
    ) -> bool:
        """Fail a stale PENDING dispatch and install its durable late-write fence."""
        ...

    def queue_pending_execution(
        self,
        trigger_log_id: str,
        *,
        celery_task_id: str,
        triggered_at: datetime,
        quota_reservation_id: str | None = None,
    ) -> bool:
        """Make a live, non-abandoned PENDING dispatch executable."""
        ...

    def confirm_dispatch_published(self, trigger_log_id: str, *, celery_task_id: str) -> bool:
        """Persist a broker publication receipt unless dispatch abandonment already won."""
        ...

    def abandon_stale_unpublished_dispatches(
        self,
        *,
        dispatch_timeout: timedelta,
        error: str,
        limit: int,
    ) -> int:
        """Fail a bounded batch of current-protocol QUEUED dispatches that were never published."""
        ...

    def claim_quota_settlement_candidates(
        self,
        *,
        retry_interval: timedelta,
        limit: int,
    ) -> Sequence[WorkflowQuotaSettlementCandidate]:
        """Claim a rotating batch of unresolved quota outcomes for out-of-transaction RPC."""
        ...

    def claim_quota_settlement(
        self,
        trigger_log_id: str,
        *,
        retry_interval: timedelta,
    ) -> WorkflowQuotaSettlementCandidate | None:
        """Claim one exact trigger's unresolved quota outcome."""
        ...

    def confirm_quota_settlement(
        self,
        trigger_log_id: str,
        *,
        attempted_at: datetime,
        settlement_status: Literal["committed", "refunded", "permanent_error"],
    ) -> bool:
        """Confirm a settlement only for the claimed attempt that completed it."""
        ...

    def renew_execution_lease(
        self,
        trigger_log_id: str,
        *,
        owner_id: str,
        lease_duration: timedelta,
    ) -> bool:
        """Extend a running execution lease only for its current owner."""
        ...

    def get_current_execution_lease_state(self, trigger_log_id: str) -> WorkflowExecutionLeaseState | None:
        """Read the current status and ownership of one trigger execution."""
        ...

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
        """Persist a terminal event only for the current execution owner."""
        ...

    def fail_execution(
        self,
        trigger_log_id: str,
        *,
        owner_id: str,
        error: str,
        elapsed_time: float,
        finished_at: datetime,
    ) -> bool:
        """Fail a running trigger only for the current execution owner."""
        ...

    def mark_dispatch_failed(
        self,
        trigger_log_id: str,
        *,
        celery_task_id: str,
        error: str,
    ) -> bool:
        """Fail an unclaimed queued trigger while it still belongs to a task."""
        ...

    def get_by_id(self, trigger_log_id: str, tenant_id: str | None = None) -> WorkflowTriggerLog | None:
        """
        Get a trigger log by its ID.

        Args:
            trigger_log_id: The trigger log identifier
            tenant_id: Optional tenant identifier for additional security

        Returns:
            The WorkflowTriggerLog if found, None otherwise
        """
        ...

    def get_failed_for_retry(
        self, tenant_id: str, max_retry_count: int = 3, limit: int = 100
    ) -> Sequence[WorkflowTriggerLog]:
        """
        Get failed trigger logs that are eligible for retry.

        Args:
            tenant_id: The tenant identifier
            max_retry_count: Maximum retry count to consider
            limit: Maximum number of results

        Returns:
            A sequence of WorkflowTriggerLog instances eligible for retry
        """
        ...

    def get_recent_logs(
        self, tenant_id: str, app_id: str, hours: int = 24, limit: int = 100, offset: int = 0
    ) -> Sequence[WorkflowTriggerLog]:
        """
        Get recent trigger logs within specified hours.

        Args:
            tenant_id: The tenant identifier
            app_id: The application identifier
            hours: Number of hours to look back
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            A sequence of recent WorkflowTriggerLog instances
        """
        ...

    def get_by_workflow_run_id(self, workflow_run_id: str) -> WorkflowTriggerLog | None:
        """
        Retrieve a trigger log associated with a specific workflow run.

        Args:
            workflow_run_id: Identifier of the workflow run

        Returns:
            The matching WorkflowTriggerLog if present, None otherwise
        """
        ...

    def delete_by_run_ids(self, run_ids: Sequence[str]) -> int:
        """
        Delete trigger logs for workflow run IDs.

        Args:
            run_ids: Workflow run IDs to delete

        Returns:
            Number of rows deleted
        """
        ...

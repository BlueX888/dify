"""
Domain entities for workflow pause management.

This module contains the domain model for workflow pause, which is used
by the core workflow module. These models are independent of the storage mechanism
and don't contain implementation details like tenant_id, app_id, etc.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from core.workflow.nodes.human_input.pause_reason import PauseReason as DifyPauseReason


@dataclass(frozen=True)
class WorkflowTriggerPausePublication:
    """Owner-fenced trigger fields committed with one durable workflow pause."""

    trigger_log_id: str
    owner_id: str
    workflow_run_id: str
    outputs: str
    elapsed_time: float
    total_tokens: int
    finished_at: datetime


class WorkflowPauseEntity(Protocol):
    """
    Protocol for workflow pause entities.

    This domain model represents a paused workflow execution state,
    without implementation details like tenant_id, app_id, etc.
    It provides the interface for managing workflow pause/resume operations
    and state persistence through file storage.

    The `WorkflowPauseEntity` is never reused. If a workflow execution pauses multiple times,
    it will generate multiple `WorkflowPauseEntity` records.
    """

    @property
    def id(self) -> str:
        """The identifier of current WorkflowPauseEntity"""
        ...

    @property
    def workflow_execution_id(self) -> str:
        """The identifier of the workflow execution record the pause associated with.
        Correspond to `WorkflowExecution.id`.
        """
        ...

    @property
    def history_revision(self) -> int | None:
        """Committed history revision, when this pause published one."""
        ...

    @property
    def history_claim_id(self) -> str | None:
        """Exact history publication claim owned by the current caller."""
        ...

    def get_state(self) -> bytes:
        """
        Retrieve the serialized workflow state from storage.

        This method should load and return the workflow execution state
        that was saved when the workflow was paused. The state contains
        all necessary information to resume the workflow execution.

        Returns:
            bytes: The serialized workflow state containing
            execution context, variable values, node states, etc.

        """
        ...

    @property
    def resumed_at(self) -> datetime | None:
        """`resumed_at` return the resumption time of the current pause, or `None` if
        the pause is not resumed yet.
        """
        ...

    @property
    def paused_at(self) -> datetime:
        """`paused_at` returns the creation time of the pause."""
        ...

    def get_pause_reasons(self) -> Sequence[DifyPauseReason]:
        """
        Retrieve detailed reasons for this pause.

        Returns a sequence of `PauseReason` objects describing the specific nodes and
        reasons for which the workflow execution was paused.
        """
        ...

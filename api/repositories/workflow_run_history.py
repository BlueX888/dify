"""Durable receipts for append-only workflow-run history publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy import func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from models.workflow import WorkflowRun

# Historical LogStore records use Unix nanoseconds (currently below 2e18).
# Reserve a signed-BIGINT namespace above them so a migrated run's first
# durable revision supersedes every legacy wall-clock record for that run.
WORKFLOW_HISTORY_VERSION_EPOCH = 4_000_000_000_000_000_000
WORKFLOW_HISTORY_CLAIM_TTL = timedelta(minutes=5)


def workflow_history_database_now(session: Session) -> datetime:
    """Use the shared database clock for lease and fairness timestamps."""
    now = session.scalar(select(func.current_timestamp()))
    if not isinstance(now, datetime):
        raise RuntimeError("database did not return a workflow history timestamp")
    return now.replace(tzinfo=None)


@dataclass(frozen=True)
class WorkflowRunHistorySnapshot:
    """One immutable SQL snapshot paired with its exact publication claim."""

    id: str
    revision: int
    claim_id: str
    tenant_id: str
    app_id: str
    workflow_id: str
    triggered_from: str
    type: str
    version: str
    graph: str
    inputs: str
    outputs: str
    status: str
    error: str
    total_tokens: int
    total_steps: int
    exceptions_count: int
    elapsed_time: float
    created_by_role: str
    created_by: str
    started_at: str
    finished_at: str

    @property
    def log_version(self) -> int:
        return WORKFLOW_HISTORY_VERSION_EPOCH + self.revision


def snapshot_claimed_workflow_run_history(model: WorkflowRun) -> WorkflowRunHistorySnapshot:
    if model.history_claim_id is None:
        raise ValueError("workflow history snapshot requires an active claim")
    return WorkflowRunHistorySnapshot(
        id=model.id,
        revision=model.history_revision,
        claim_id=model.history_claim_id,
        tenant_id=model.tenant_id,
        app_id=model.app_id,
        workflow_id=model.workflow_id,
        triggered_from=model.triggered_from.value,
        type=model.type.value,
        version=model.version,
        graph=model.graph or "{}",
        inputs=model.inputs or "{}",
        outputs=model.outputs or "{}",
        status=model.status.value,
        error=model.error or "",
        total_tokens=model.total_tokens or 0,
        total_steps=model.total_steps or 0,
        exceptions_count=model.exceptions_count or 0,
        elapsed_time=model.elapsed_time or 0,
        created_by_role=model.created_by_role.value,
        created_by=model.created_by,
        started_at=model.created_at.isoformat() if model.created_at else "",
        finished_at=model.finished_at.isoformat() if model.finished_at else "",
    )


def advance_workflow_run_history(model: WorkflowRun) -> int:
    """Advance the SQL projection and expose it to the reconciler."""
    model.history_revision = (model.history_revision or 0) + 1
    model.history_published_revision = model.history_published_revision or 0
    model.history_claim_id = None
    model.history_claim_expires_at = None
    model.history_last_attempted_at = None
    model.history_sync_pending = True
    return model.history_revision


def advance_and_claim_workflow_run_history(
    model: WorkflowRun,
    *,
    now: datetime | None = None,
) -> WorkflowRunHistorySnapshot:
    """Advance and claim the exact projected payload in the caller's transaction."""
    advance_workflow_run_history(model)
    return claim_current_workflow_run_history(model, now=now)


def claim_current_workflow_run_history(
    model: WorkflowRun,
    *,
    now: datetime | None = None,
) -> WorkflowRunHistorySnapshot:
    """Claim the already-current revision without creating another revision."""
    if now is None:
        raise ValueError("workflow history claims require the database timestamp")
    claimed_at = now
    model.history_claim_id = str(uuid4())
    model.history_claim_expires_at = claimed_at + WORKFLOW_HISTORY_CLAIM_TTL
    model.history_last_attempted_at = claimed_at
    return snapshot_claimed_workflow_run_history(model)


def claim_pending_workflow_run_histories(
    session: Session,
    *,
    limit: int,
    now: datetime | None = None,
) -> list[WorkflowRunHistorySnapshot]:
    """Atomically claim pending projections with NULL-first fair rotation."""
    claimed_at = now or workflow_history_database_now(session)
    stmt = (
        select(WorkflowRun)
        .where(
            WorkflowRun.history_sync_pending.is_(True),
            WorkflowRun.history_published_revision < WorkflowRun.history_revision,
            or_(
                WorkflowRun.history_claim_id.is_(None),
                WorkflowRun.history_claim_expires_at <= claimed_at,
            ),
        )
        .order_by(
            sa.case((WorkflowRun.history_last_attempted_at.is_(None), 0), else_=1),
            WorkflowRun.history_last_attempted_at,
            WorkflowRun.id,
        )
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    models = list(session.scalars(stmt))
    return [claim_current_workflow_run_history(model, now=claimed_at) for model in models]


def confirm_workflow_run_history_publication(session: Session, snapshot: WorkflowRunHistorySnapshot) -> bool:
    """ACK only the exact claim/revision that produced the remote record."""
    result = session.execute(
        update(WorkflowRun)
        .where(
            WorkflowRun.id == snapshot.id,
            WorkflowRun.tenant_id == snapshot.tenant_id,
            WorkflowRun.app_id == snapshot.app_id,
            WorkflowRun.history_revision == snapshot.revision,
            WorkflowRun.history_claim_id == snapshot.claim_id,
        )
        .values(
            history_published_revision=snapshot.revision,
            history_claim_id=None,
            history_claim_expires_at=None,
            history_sync_pending=False,
        )
    )
    return cast(CursorResult, result).rowcount == 1


def release_workflow_run_history_claim(session: Session, snapshot: WorkflowRunHistorySnapshot) -> bool:
    """Make a failed exact claim eligible for a later fair retry."""
    result = session.execute(
        update(WorkflowRun)
        .where(
            WorkflowRun.id == snapshot.id,
            WorkflowRun.tenant_id == snapshot.tenant_id,
            WorkflowRun.app_id == snapshot.app_id,
            WorkflowRun.history_revision == snapshot.revision,
            WorkflowRun.history_claim_id == snapshot.claim_id,
        )
        .values(history_claim_id=None, history_claim_expires_at=None)
    )
    return cast(CursorResult, result).rowcount == 1

"""Cross-dialect query-plan regression for workflow quota reconciliation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import delete, event, insert
from sqlalchemy.engine import CursorResult, Engine
from sqlalchemy.orm import Session, sessionmaker

from models.enums import AppTriggerType, CreatorUserRole, WorkflowTriggerStatus
from models.trigger import WorkflowTriggerLog
from repositories.sqlalchemy_workflow_trigger_log_repository import SQLAlchemyWorkflowTriggerLogRepository

_INDEX_NAME = "workflow_trigger_log_quota_sync_pending_idx"
_LEGACY_COUNT = 20_000
_SETTLED_COUNT = 5_000


@dataclass(frozen=True)
class _CapturedSelect:
    statement: str
    parameters: Mapping[str, Any] | Sequence[Any]


@dataclass(frozen=True)
class _ExplainPlan:
    document: object
    actual_rows: tuple[float, ...] = ()


def _walk_json(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _trigger_row(
    *,
    tenant_id: str,
    trigger_id: str,
    quota_sync_pending: bool,
    settlement_status: str | None,
    attempted_at: datetime | None,
    prepared_at: datetime | None,
) -> dict[str, object]:
    return {
        "id": trigger_id,
        "tenant_id": tenant_id,
        "app_id": str(uuid4()),
        "workflow_id": str(uuid4()),
        "workflow_run_id": None,
        "root_node_id": None,
        "trigger_metadata": "{}",
        "trigger_type": AppTriggerType.TRIGGER_SCHEDULE,
        "trigger_data": "{}",
        "inputs": "{}",
        "outputs": None,
        "status": WorkflowTriggerStatus.FAILED,
        "error": None,
        "queue_name": "workflow",
        "celery_task_id": None,
        "created_by_role": CreatorUserRole.ACCOUNT,
        "created_by": str(uuid4()),
        "quota_reservation_id": f"reservation-{trigger_id}",
        "quota_settlement_status": settlement_status,
        "quota_settlement_attempted_at": attempted_at,
        "quota_sync_pending": quota_sync_pending,
        "dispatch_prepared_at": prepared_at,
    }


def _copy_parameters(parameters: object) -> Mapping[str, Any] | Sequence[Any]:
    if isinstance(parameters, Mapping):
        return dict(parameters)
    if isinstance(parameters, (list, tuple)):
        return tuple(parameters)
    raise TypeError(f"unexpected DBAPI parameters: {type(parameters)!r}")


def _explain_actual_select(
    session: Session,
    *,
    captured: _CapturedSelect,
) -> _ExplainPlan:
    dialect = session.get_bind().dialect.name
    connection = session.connection()
    if dialect == "postgresql":
        return _ExplainPlan(
            document=connection.exec_driver_sql(
                f"EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS FALSE) {captured.statement}",
                captured.parameters,
            ).scalar_one()
        )
    if dialect == "mysql":
        raw = connection.exec_driver_sql(
            f"EXPLAIN FORMAT=JSON {captured.statement}",
            captured.parameters,
        ).scalar_one()
        if not isinstance(raw, str):
            raise TypeError("MySQL JSON EXPLAIN did not return text")
        analyzed = connection.exec_driver_sql(
            f"EXPLAIN ANALYZE {captured.statement}",
            captured.parameters,
        ).scalar_one()
        if not isinstance(analyzed, str):
            raise TypeError("MySQL EXPLAIN ANALYZE did not return text")
        actual_rows = tuple(
            float(value) for value in re.findall(r"\(actual time=[^)]* rows=([0-9.]+) loops=", analyzed)
        )
        return _ExplainPlan(document=json.loads(raw), actual_rows=actual_rows)
    raise AssertionError(f"unsupported database dialect: {dialect}")


def test_quota_settlement_candidates_use_pending_index_on_mysql_and_postgresql(
    database_session: Session,
) -> None:
    """Both production candidate SELECTs must avoid scanning the legacy population."""

    engine = cast(Engine, database_session.get_bind())
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id = str(uuid4())
    now = datetime.now(UTC).replace(tzinfo=None)
    old_attempt = now - timedelta(days=2)
    future_attempt = now + timedelta(days=2)
    prepared_at = now - timedelta(days=3)
    never_attempted_ids = [str(uuid4()) for _ in range(5)]
    retry_ids = [str(uuid4()) for _ in range(3)]
    future_ids = [str(uuid4()) for _ in range(2)]

    rows = [
        _trigger_row(
            tenant_id=tenant_id,
            trigger_id=str(uuid4()),
            quota_sync_pending=False,
            settlement_status=None,
            attempted_at=None,
            prepared_at=None,
        )
        for _ in range(_LEGACY_COUNT)
    ]
    rows.extend(
        _trigger_row(
            tenant_id=tenant_id,
            trigger_id=str(uuid4()),
            quota_sync_pending=False,
            settlement_status="committed",
            attempted_at=old_attempt,
            prepared_at=prepared_at,
        )
        for _ in range(_SETTLED_COUNT)
    )
    rows.extend(
        _trigger_row(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
            quota_sync_pending=True,
            settlement_status=None,
            attempted_at=None,
            prepared_at=prepared_at,
        )
        for trigger_id in never_attempted_ids
    )
    rows.extend(
        _trigger_row(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
            quota_sync_pending=True,
            settlement_status=None,
            attempted_at=old_attempt,
            prepared_at=prepared_at,
        )
        for trigger_id in retry_ids
    )
    rows.extend(
        _trigger_row(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
            quota_sync_pending=True,
            settlement_status=None,
            attempted_at=future_attempt,
            prepared_at=prepared_at,
        )
        for trigger_id in future_ids
    )

    with sessions.begin() as seed:
        seed.execute(insert(WorkflowTriggerLog), rows)
    with engine.begin() as connection:
        if engine.dialect.name == "mysql":
            connection.exec_driver_sql("ANALYZE TABLE workflow_trigger_logs")
        else:
            connection.exec_driver_sql("ANALYZE workflow_trigger_logs")

    captured_selects: list[_CapturedSelect] = []

    def capture_candidate_select(
        _connection,
        _cursor,
        statement: str,
        parameters: object,
        _context,
        _executemany,
    ) -> None:
        normalized = " ".join(statement.lower().replace("`", "").split())
        if (
            normalized.startswith("select workflow_trigger_logs.id")
            and "workflow_trigger_logs.quota_sync_pending" in normalized
            and "order by" in normalized
        ):
            captured_selects.append(_CapturedSelect(statement=statement, parameters=_copy_parameters(parameters)))

    event.listen(engine, "before_cursor_execute", capture_candidate_select)
    try:
        candidate_session = sessions()
        try:
            candidates = SQLAlchemyWorkflowTriggerLogRepository(candidate_session).claim_quota_settlement_candidates(
                retry_interval=timedelta(days=1),
                limit=8,
            )
            assert {candidate.trigger_log_id for candidate in candidates[:5]} == set(never_attempted_ids)
            assert {candidate.trigger_log_id for candidate in candidates[5:]} == set(retry_ids)
            assert not ({candidate.trigger_log_id for candidate in candidates} & set(future_ids))
        finally:
            candidate_session.rollback()
            candidate_session.close()

        assert len(captured_selects) == 2
        with sessions() as explain_session:
            plans = [_explain_actual_select(explain_session, captured=captured) for captured in captured_selects]

        for plan in plans:
            nodes = list(_walk_json(plan.document))
            if engine.dialect.name == "postgresql":
                indexed_nodes = [node for node in nodes if node.get("Index Name") == _INDEX_NAME]
                assert indexed_nodes
                assert max(int(node.get("Actual Rows", 0)) for node in indexed_nodes) <= 10
                assert all(node.get("Node Type") != "Sort" for node in nodes)
            else:
                indexed_nodes = [node for node in nodes if node.get("key") == _INDEX_NAME]
                assert indexed_nodes
                assert not any(node.get("using_filesort") is True for node in nodes)
                assert plan.actual_rows
                assert max(plan.actual_rows) <= 10
    finally:
        event.remove(engine, "before_cursor_execute", capture_candidate_select)
        with sessions.begin() as cleanup:
            result = cleanup.execute(delete(WorkflowTriggerLog).where(WorkflowTriggerLog.tenant_id == tenant_id))
            assert cast(CursorResult[Any], result).rowcount == len(rows)

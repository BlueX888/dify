"""Low-level, caller-transaction handoff of terminal vector route evidence."""

from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, func, select
from sqlalchemy.orm import Session

from models.dataset import ChildChunk, DocumentSegmentSummary
from models.model import ChildVectorCleanupOutbox, SummaryVectorCleanupOutbox
from repositories.knowledge.child_vector_route_repository import stored_vector_route

StoredVectorRoute = tuple[str, str]
ChildVectorCleanupTarget = tuple[str, str, StoredVectorRoute | None]


def _database_now(session: Session) -> datetime:
    value = session.scalar(select(func.current_timestamp(type_=DateTime())))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value.replace(tzinfo=None, microsecond=0)


def add_terminal_child_vector_cleanups(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    segment_id: str,
    targets: Sequence[ChildVectorCleanupTarget],
) -> None:
    """Persist terminal child nodes and invalidate any earlier owner atomically."""
    normalized = tuple(dict.fromkeys(targets))
    if not normalized:
        return
    node_ids = [node_id for _, node_id, _ in normalized]
    existing = {
        (row.node_id, row.vector_route_hash): row
        for row in session.scalars(
            select(ChildVectorCleanupOutbox)
            .where(
                ChildVectorCleanupOutbox.tenant_id == tenant_id,
                ChildVectorCleanupOutbox.node_id.in_(node_ids),
            )
            .with_for_update()
        )
    }
    database_now = _database_now(session)
    created: list[ChildVectorCleanupOutbox] = []
    for child_id, node_id, route in normalized:
        route_key = route[1] if route is not None else None
        row = existing.get((node_id, route_key))
        if row is None:
            created.append(
                ChildVectorCleanupOutbox(
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    segment_id=segment_id,
                    child_id=child_id,
                    node_id=node_id,
                    writer_id=str(uuid4()),
                    vector_route=route[0] if route is not None else None,
                    vector_route_hash=route_key,
                    writer_finished_at=database_now,
                )
            )
            continue
        # A detach is a new terminal cleanup generation. Replacing both stable
        # and renewable ownership prevents an earlier writer discard or cleanup
        # ACK from consuming the only receipt after the reference disappears.
        row.dataset_id = dataset_id
        row.document_id = document_id
        row.segment_id = segment_id
        row.child_id = child_id
        row.vector_route = route[0] if route is not None else None
        row.vector_route_hash = route_key
        if row.writer_finished_at is None:
            # The remote mutation may still arrive after this resource is
            # detached. Preserve its stable writer identity and unknown outcome,
            # while fencing publication and handing cleanup a fresh generation.
            row.cleanup_started_at = row.cleanup_started_at or database_now
            row.claim_id = None
            row.claim_expires_at = None
            row.last_attempted_at = None
            continue
        row.writer_id = str(uuid4())
        row.writer_finished_at = database_now
        row.cleanup_started_at = None
        row.claim_id = None
        row.claim_expires_at = None
        row.last_attempted_at = None
    session.add_all(created)


def add_terminal_child_row_cleanups(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    children: Sequence[ChildChunk],
) -> None:
    """Handoff already-locked child rows before their owning rows are deleted."""
    grouped: dict[tuple[str, str], list[ChildVectorCleanupTarget]] = {}
    for child in children:
        if child.tenant_id != tenant_id or child.dataset_id != dataset_id:
            raise ValueError("Child-vector cleanup row does not belong to the requested dataset")
        if child.index_node_id is None:
            continue
        route = stored_vector_route(
            route_serialized=child.vector_route,
            route_hash=child.vector_route_hash,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
        )
        grouped.setdefault((child.document_id, child.segment_id), []).append((child.id, child.index_node_id, route))
    for (document_id, segment_id), targets in grouped.items():
        add_terminal_child_vector_cleanups(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            targets=targets,
        )


def add_terminal_summary_vector_cleanups(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    summaries: Sequence[DocumentSegmentSummary],
) -> None:
    """Handoff already-locked summary rows before their owning rows are deleted."""
    candidates = [summary for summary in summaries if summary.summary_index_node_id]
    if not candidates:
        return
    database_now = _database_now(session)
    for summary in candidates:
        assert summary.summary_index_node_id is not None
        if summary.dataset_id != dataset_id:
            raise ValueError("Summary cleanup row does not belong to the requested dataset")
        route = stored_vector_route(
            route_serialized=summary.vector_route,
            route_hash=summary.vector_route_hash,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
        )
        existing = session.scalar(
            select(SummaryVectorCleanupOutbox)
            .where(
                SummaryVectorCleanupOutbox.tenant_id == tenant_id,
                SummaryVectorCleanupOutbox.node_id == summary.summary_index_node_id,
                (
                    SummaryVectorCleanupOutbox.vector_route_hash == route[1]
                    if route is not None
                    else SummaryVectorCleanupOutbox.vector_route_hash.is_(None)
                ),
            )
            .with_for_update()
        )
        if existing is not None:
            existing.dataset_id = dataset_id
            existing.document_id = summary.document_id
            existing.segment_id = summary.chunk_id
            existing.summary_id = summary.id
            existing.summary_version = summary.version
            existing.vector_route = route[0] if route is not None else None
            existing.vector_route_hash = route[1] if route is not None else None
            if existing.writer_finished_at is None:
                # A timeout is still an active writer even after its reference
                # becomes terminal. Fence publication without fabricating a
                # completion marker; a late finish will unlock final ACK.
                existing.cleanup_started_at = existing.cleanup_started_at or database_now
                existing.claim_id = None
                existing.claim_expires_at = None
                existing.last_attempted_at = None
                continue
            existing.writer_id = str(uuid4())
            existing.writer_finished_at = database_now
            existing.cleanup_started_at = None
            existing.claim_id = None
            existing.claim_expires_at = None
            existing.last_attempted_at = None
            continue
        session.add(
            SummaryVectorCleanupOutbox(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=summary.document_id,
                segment_id=summary.chunk_id,
                summary_id=summary.id,
                summary_version=summary.version,
                node_id=summary.summary_index_node_id,
                writer_id=str(uuid4()),
                vector_route=route[0] if route is not None else None,
                vector_route_hash=route[1] if route is not None else None,
                writer_finished_at=database_now,
            )
        )

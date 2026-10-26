"""Durable ownership for child-vector publication and superseded-node cleanup."""

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import cast
from uuid import uuid4

from sqlalchemy import DateTime, case, delete, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from libs.sqlalchemy_utils import database_operation_timeout
from models.dataset import ChildChunk, Dataset, Document, DocumentSegment
from models.model import ChildVectorCleanupOutbox
from repositories.knowledge.dataset_indexing_outbox_repository import (
    SQLAlchemyDatasetIndexingOutboxRepository,
    lock_dataset_indexing_generation_shared,
)
from repositories.knowledge.vector_cleanup_handoff_repository import add_terminal_child_vector_cleanups
from services.knowledge.entities.resource_scope import SegmentRef
from services.knowledge.entities.segments import (
    ChildVectorCleanupClaim,
    ChildVectorGuard,
    ChildVectorRoute,
    ChildVectorWriteIntent,
)

CHILD_VECTOR_CLEANUP_LEASE_SECONDS = 300
_RENEWAL_DATABASE_TIMEOUT_SECONDS = 1.0


class ChildVectorCleanupClaimLostError(RuntimeError):
    """A child-vector writer or cleanup worker no longer owns its lease."""


def _database_now(session: Session) -> datetime:
    value = session.scalar(select(func.current_timestamp(type_=DateTime())))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value.replace(tzinfo=None, microsecond=0)


def _claim_available(database_now: datetime):
    return or_(
        ChildVectorCleanupOutbox.claim_id.is_(None),
        ChildVectorCleanupOutbox.claim_expires_at.is_(None),
        ChildVectorCleanupOutbox.claim_expires_at <= database_now,
    )


def _route_from_row(row: ChildVectorCleanupOutbox) -> ChildVectorRoute | None:
    if row.vector_route is None or row.vector_route_hash is None:
        return None
    try:
        route = ChildVectorRoute.model_validate_json(row.vector_route)
    except ValueError:
        return None
    if route.key != row.vector_route_hash:
        return None
    if route.tenant_id != row.tenant_id or route.dataset_id != row.dataset_id:
        return None
    return route


def _pending_cleanup_ids_statement(database_now: datetime, *, limit: int):
    return (
        select(ChildVectorCleanupOutbox.id)
        .where(_claim_available(database_now))
        .order_by(
            case((ChildVectorCleanupOutbox.last_attempted_at.is_(None), 0), else_=1),
            ChildVectorCleanupOutbox.last_attempted_at,
            ChildVectorCleanupOutbox.created_at,
            ChildVectorCleanupOutbox.id,
        )
        .limit(limit)
    )


def _cleanup_dataset_lock_statement(claim: ChildVectorCleanupClaim):
    return (
        select(Dataset.id)
        .where(
            Dataset.id == claim.dataset_id,
            Dataset.tenant_id == claim.tenant_id,
        )
        .with_for_update(read=True)
    )


def _lock_guarded_dataset(session: Session, ref: SegmentRef, guard: ChildVectorGuard) -> Dataset:
    dataset = lock_dataset_indexing_generation_shared(
        session,
        dataset_id=ref.document.dataset.dataset_id,
        generation_id=guard.generation_id,
    )
    if dataset is None or dataset.tenant_id != ref.document.dataset.tenant_id:
        raise ChildVectorCleanupClaimLostError("child-vector dataset generation changed")
    current = ChildVectorGuard(
        required=guard.required,
        generation_id=dataset.indexing_request_id,
        indexing_technique=dataset.indexing_technique,
        embedding_model_provider=dataset.embedding_model_provider,
        embedding_model=dataset.embedding_model,
        index_struct=dataset.index_struct,
        collection_binding_id=dataset.collection_binding_id,
    )
    if current != guard:
        raise ChildVectorCleanupClaimLostError("child-vector dataset configuration changed")
    return dataset


class SQLAlchemyChildVectorCleanupRepository:
    def __init__(self, *, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def begin_writes(
        self,
        ref: SegmentRef,
        guard: ChildVectorGuard,
        targets: Sequence[tuple[str, str]],
        route: ChildVectorRoute,
    ) -> tuple[ChildVectorWriteIntent, ...]:
        """Commit writer identities before any vector mutation can start."""
        normalized = tuple(dict.fromkeys(targets))
        if len({child_id for child_id, _ in normalized}) != len(normalized):
            raise ValueError("A child vector write contains duplicate child IDs")
        if len({node_id for _, node_id in normalized}) != len(normalized):
            raise ValueError("A child vector write contains duplicate node IDs")
        if not normalized:
            return ()
        with self._session_factory() as session, session.begin():
            dataset = _lock_guarded_dataset(session, ref, guard)
            document = (
                session.scalar(
                    select(Document)
                    .where(
                        Document.id == ref.document.document_id,
                        Document.dataset_id == ref.document.dataset.dataset_id,
                        Document.tenant_id == ref.document.dataset.tenant_id,
                    )
                    .with_for_update()
                )
                if dataset is not None
                else None
            )
            segment = (
                session.scalar(
                    select(DocumentSegment)
                    .where(
                        DocumentSegment.id == ref.segment_id,
                        DocumentSegment.document_id == ref.document.document_id,
                        DocumentSegment.dataset_id == ref.document.dataset.dataset_id,
                        DocumentSegment.tenant_id == ref.document.dataset.tenant_id,
                    )
                    .with_for_update()
                )
                if document is not None
                else None
            )
            if document is None or segment is None:
                raise LookupError("Child-vector segment no longer exists")
            if guard.required != (segment.enabled and dataset.indexing_technique == "high_quality"):
                raise ChildVectorCleanupClaimLostError("child-vector segment state changed")
            if (
                route.tenant_id != dataset.tenant_id
                or route.dataset_id != dataset.id
                or route.indexing_technique != dataset.indexing_technique
                or route.collection_binding_id != dataset.collection_binding_id
                or route.doc_form != document.doc_form
            ):
                raise ChildVectorCleanupClaimLostError("child-vector publication route changed")
            SQLAlchemyDatasetIndexingOutboxRepository.record_generation_writer_route_locked(
                session,
                dataset=dataset,
                generation_id=guard.generation_id,
                route=route,
            )
            database_now = _database_now(session)
            rows = [
                ChildVectorCleanupOutbox(
                    tenant_id=ref.document.dataset.tenant_id,
                    dataset_id=ref.document.dataset.dataset_id,
                    document_id=ref.document.document_id,
                    segment_id=ref.segment_id,
                    child_id=child_id,
                    node_id=node_id,
                    writer_id=str(uuid4()),
                    vector_route=route.serialized,
                    vector_route_hash=route.key,
                    claim_id=str(uuid4()),
                    claim_expires_at=database_now + timedelta(seconds=CHILD_VECTOR_CLEANUP_LEASE_SECONDS),
                    last_attempted_at=database_now,
                )
                for child_id, node_id in normalized
            ]
            session.add_all(rows)
            session.flush()
            return tuple(self._to_write_intent(row) for row in rows)

    def discard_unsubmitted_writes(self, intents: Sequence[ChildVectorWriteIntent]) -> None:
        """Forget writers proven not to have crossed the vector mutation boundary."""
        if not intents:
            return
        with self._session_factory() as session, session.begin():
            for intent in intents:
                row = session.scalar(
                    select(ChildVectorCleanupOutbox)
                    .where(
                        ChildVectorCleanupOutbox.id == intent.id,
                        ChildVectorCleanupOutbox.tenant_id == intent.tenant_id,
                        ChildVectorCleanupOutbox.dataset_id == intent.dataset_id,
                        ChildVectorCleanupOutbox.node_id == intent.node_id,
                        ChildVectorCleanupOutbox.writer_id == intent.writer_id,
                    )
                    .with_for_update()
                )
                if row is None:
                    continue
                if row.cleanup_started_at is None:
                    session.delete(row)
                    continue
                # The resource handoff may represent an older vector at the same
                # physical identity. End writer uncertainty without erasing the
                # only durable cleanup receipt.
                row.writer_finished_at = row.writer_finished_at or _database_now(session)
                row.claim_id = None
                row.claim_expires_at = None

    def renew_writes(self, intents: Sequence[ChildVectorWriteIntent]) -> None:
        if not intents:
            return
        with self._session_factory() as session, session.begin():
            with database_operation_timeout(session, timeout_seconds=_RENEWAL_DATABASE_TIMEOUT_SECONDS):
                database_now = _database_now(session)
                for intent in intents:
                    cursor = session.execute(
                        update(ChildVectorCleanupOutbox)
                        .where(
                            ChildVectorCleanupOutbox.id == intent.id,
                            ChildVectorCleanupOutbox.tenant_id == intent.tenant_id,
                            ChildVectorCleanupOutbox.node_id == intent.node_id,
                            ChildVectorCleanupOutbox.writer_id == intent.writer_id,
                            ChildVectorCleanupOutbox.writer_finished_at.is_(None),
                            ChildVectorCleanupOutbox.cleanup_started_at.is_(None),
                            ChildVectorCleanupOutbox.claim_id == intent.claim_id,
                            ChildVectorCleanupOutbox.claim_expires_at > database_now,
                        )
                        .values(claim_expires_at=database_now + timedelta(seconds=CHILD_VECTOR_CLEANUP_LEASE_SECONDS))
                    )
                    if cast(CursorResult[object], cursor).rowcount != 1:
                        raise ChildVectorCleanupClaimLostError()

    def renew_publication(self, claims: Sequence[ChildVectorCleanupClaim]) -> None:
        """Renew finished writer claims until their child-row transaction commits."""
        if not claims:
            return
        with self._session_factory() as session, session.begin():
            with database_operation_timeout(session, timeout_seconds=_RENEWAL_DATABASE_TIMEOUT_SECONDS):
                database_now = _database_now(session)
                for claim in claims:
                    cursor = session.execute(
                        update(ChildVectorCleanupOutbox)
                        .where(
                            ChildVectorCleanupOutbox.id == claim.id,
                            ChildVectorCleanupOutbox.tenant_id == claim.tenant_id,
                            ChildVectorCleanupOutbox.node_id == claim.node_id,
                            ChildVectorCleanupOutbox.writer_finished_at.is_not(None),
                            ChildVectorCleanupOutbox.cleanup_started_at.is_(None),
                            ChildVectorCleanupOutbox.claim_id == claim.claim_id,
                            ChildVectorCleanupOutbox.claim_expires_at > database_now,
                        )
                        .values(claim_expires_at=database_now + timedelta(seconds=CHILD_VECTOR_CLEANUP_LEASE_SECONDS))
                    )
                    if cast(CursorResult[object], cursor).rowcount != 1:
                        raise ChildVectorCleanupClaimLostError()

    def finish_writes(self, intents: Sequence[ChildVectorWriteIntent]) -> tuple[ChildVectorCleanupClaim, ...]:
        """Mark returned vector calls terminal and retain only untouched claims."""
        if not intents:
            return ()
        claims: list[ChildVectorCleanupClaim] = []
        lost = False
        with self._session_factory() as session, session.begin():
            database_now = _database_now(session)
            rows = session.scalars(
                select(ChildVectorCleanupOutbox)
                .where(ChildVectorCleanupOutbox.id.in_([intent.id for intent in intents]))
                .order_by(ChildVectorCleanupOutbox.id)
                .with_for_update()
            ).all()
            rows_by_id = {row.id: row for row in rows}
            for intent in intents:
                row = rows_by_id.get(intent.id)
                if (
                    row is None
                    or row.tenant_id != intent.tenant_id
                    or row.node_id != intent.node_id
                    or row.writer_id != intent.writer_id
                    or _route_from_row(row) != intent.route
                ):
                    lost = True
                    continue
                if row.writer_finished_at is None:
                    row.writer_finished_at = database_now
                if (
                    row.claim_id != intent.claim_id
                    or row.cleanup_started_at is not None
                    or row.claim_expires_at is None
                    or row.claim_expires_at <= database_now
                ):
                    lost = True
                    continue
                row.claim_expires_at = database_now + timedelta(seconds=CHILD_VECTOR_CLEANUP_LEASE_SECONDS)
                row.last_attempted_at = database_now
                claims.append(self._to_claim(row))
        if lost or len(claims) != len(intents):
            raise ChildVectorCleanupClaimLostError("child-vector cleanup is owned by another worker")
        return tuple(claims)

    def release_publication(self, claims: Sequence[ChildVectorCleanupClaim]) -> None:
        """Make terminal, unpublished nodes immediately available to cleanup."""
        if not claims:
            return
        with self._session_factory() as session, session.begin():
            for claim in claims:
                session.execute(
                    update(ChildVectorCleanupOutbox)
                    .where(
                        ChildVectorCleanupOutbox.id == claim.id,
                        ChildVectorCleanupOutbox.tenant_id == claim.tenant_id,
                        ChildVectorCleanupOutbox.node_id == claim.node_id,
                        ChildVectorCleanupOutbox.claim_id == claim.claim_id,
                        ChildVectorCleanupOutbox.writer_finished_at.is_not(None),
                        ChildVectorCleanupOutbox.cleanup_started_at.is_(None),
                    )
                    .values(claim_id=None, claim_expires_at=None)
                )

    @staticmethod
    def consume_publication(
        session: Session,
        *,
        ref: SegmentRef,
        targets: Mapping[str, str],
        claims: Sequence[ChildVectorCleanupClaim],
    ) -> Mapping[str, ChildVectorRoute]:
        """Consume exact finished writers in the child-row publication transaction."""
        if len(targets) != len(claims):
            raise ChildVectorCleanupClaimLostError("child-vector publication claim count changed")
        if not targets:
            return {}
        database_now = _database_now(session)
        rows = session.scalars(
            select(ChildVectorCleanupOutbox)
            .where(ChildVectorCleanupOutbox.id.in_([claim.id for claim in claims]))
            .order_by(ChildVectorCleanupOutbox.id)
            .with_for_update()
        ).all()
        rows_by_id = {row.id: row for row in rows}
        routes: dict[str, ChildVectorRoute] = {}
        for claim in claims:
            row = rows_by_id.get(claim.id)
            route = _route_from_row(row) if row is not None else None
            if (
                row is None
                or row.tenant_id != ref.document.dataset.tenant_id
                or row.dataset_id != ref.document.dataset.dataset_id
                or row.document_id != ref.document.document_id
                or row.segment_id != ref.segment_id
                or row.node_id != claim.node_id
                or targets.get(row.child_id) != row.node_id
                or row.claim_id != claim.claim_id
                or row.writer_finished_at is None
                or row.cleanup_started_at is not None
                or row.claim_expires_at is None
                or row.claim_expires_at <= database_now
                or route is None
                or claim.route != route
            ):
                raise ChildVectorCleanupClaimLostError("child-vector publication claim is no longer current")
            routes[row.child_id] = route
        session.execute(delete(ChildVectorCleanupOutbox).where(ChildVectorCleanupOutbox.id.in_(rows_by_id)))
        return routes

    @staticmethod
    def add_terminal_cleanups(
        session: Session,
        *,
        ref: SegmentRef,
        obsolete_nodes: Sequence[tuple[str, str, ChildVectorRoute | None]],
    ) -> None:
        add_terminal_child_vector_cleanups(
            session,
            tenant_id=ref.document.dataset.tenant_id,
            dataset_id=ref.document.dataset.dataset_id,
            document_id=ref.document.document_id,
            segment_id=ref.segment_id,
            targets=tuple(
                (
                    child_id,
                    node_id,
                    (route.serialized, route.key) if route is not None else None,
                )
                for child_id, node_id, route in obsolete_nodes
            ),
        )

    def list_pending_cleanup_ids(self, *, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        with self._session_factory() as session:
            database_now = _database_now(session)
            return tuple(session.scalars(_pending_cleanup_ids_statement(database_now, limit=limit)))

    def mark_attempted(self, cleanup_id: str) -> None:
        """Rotate one publication attempt without changing cleanup ownership."""
        with self._session_factory() as session, session.begin():
            session.execute(
                update(ChildVectorCleanupOutbox)
                .where(ChildVectorCleanupOutbox.id == cleanup_id)
                .values(last_attempted_at=func.current_timestamp())
            )

    def claim_pending_cleanup(self, cleanup_id: str) -> ChildVectorCleanupClaim | None:
        claim_id = str(uuid4())
        with self._session_factory() as session, session.begin():
            database_now = _database_now(session)
            cursor = session.execute(
                update(ChildVectorCleanupOutbox)
                .where(ChildVectorCleanupOutbox.id == cleanup_id, _claim_available(database_now))
                .values(
                    claim_id=claim_id,
                    claim_expires_at=database_now + timedelta(seconds=CHILD_VECTOR_CLEANUP_LEASE_SECONDS),
                    last_attempted_at=database_now,
                    cleanup_started_at=func.coalesce(ChildVectorCleanupOutbox.cleanup_started_at, database_now),
                )
            )
            if cast(CursorResult[object], cursor).rowcount != 1:
                return None
            row = session.scalar(
                select(ChildVectorCleanupOutbox).where(
                    ChildVectorCleanupOutbox.id == cleanup_id,
                    ChildVectorCleanupOutbox.claim_id == claim_id,
                )
            )
            return self._to_claim(row) if row is not None else None

    def renew_cleanup(self, claim: ChildVectorCleanupClaim) -> None:
        with self._session_factory() as session, session.begin():
            with database_operation_timeout(session, timeout_seconds=_RENEWAL_DATABASE_TIMEOUT_SECONDS):
                database_now = _database_now(session)
                cursor = session.execute(
                    update(ChildVectorCleanupOutbox)
                    .where(
                        ChildVectorCleanupOutbox.id == claim.id,
                        ChildVectorCleanupOutbox.tenant_id == claim.tenant_id,
                        ChildVectorCleanupOutbox.node_id == claim.node_id,
                        ChildVectorCleanupOutbox.claim_id == claim.claim_id,
                        ChildVectorCleanupOutbox.claim_expires_at > database_now,
                    )
                    .values(claim_expires_at=database_now + timedelta(seconds=CHILD_VECTOR_CLEANUP_LEASE_SECONDS))
                )
                if cast(CursorResult[object], cursor).rowcount != 1:
                    raise ChildVectorCleanupClaimLostError()

    @contextmanager
    def guard_cleanup_delete(self, claim: ChildVectorCleanupClaim) -> Generator[bool]:
        """Hold references stable through vector deletion and terminal acknowledgement.

        Dataset rebuild and state transitions take the matching exclusive lock,
        so they cannot republish the same physical node after this check but
        before the caller's vector deletion. A live child reference is locked,
        so a concurrent detach cannot consume this row and discover the old
        cleanup receipt until the terminal ACK commits. Finished writers are
        acknowledged on normal context exit in this same transaction; unfinished
        writers remain durable for a later pass because their remote write may
        still arrive. The outbox claim is not locked during the external delete,
        so its independent heartbeat remains able to renew.
        """
        if claim.route is None:
            raise ChildVectorCleanupClaimLostError("child-vector cleanup has no valid route")
        with self._session_factory() as session:
            with session.begin():
                dataset_exists = session.scalar(_cleanup_dataset_lock_statement(claim)) is not None
                database_now = _database_now(session)
                current_claim = session.scalar(
                    select(ChildVectorCleanupOutbox.id).where(
                        ChildVectorCleanupOutbox.id == claim.id,
                        ChildVectorCleanupOutbox.tenant_id == claim.tenant_id,
                        ChildVectorCleanupOutbox.dataset_id == claim.dataset_id,
                        ChildVectorCleanupOutbox.node_id == claim.node_id,
                        ChildVectorCleanupOutbox.vector_route_hash == claim.route.key,
                        ChildVectorCleanupOutbox.claim_id == claim.claim_id,
                        ChildVectorCleanupOutbox.claim_expires_at > database_now,
                    )
                )
                if current_claim is None:
                    raise ChildVectorCleanupClaimLostError("child-vector cleanup claim is no longer current")
                if dataset_exists:
                    referenced = (
                        session.scalar(
                            select(ChildChunk.id)
                            .where(
                                ChildChunk.tenant_id == claim.tenant_id,
                                ChildChunk.dataset_id == claim.dataset_id,
                                ChildChunk.index_node_id == claim.node_id,
                                ChildChunk.vector_route_hash == claim.route.key,
                            )
                            .with_for_update()
                            .limit(1)
                        )
                        is not None
                    )
                else:
                    referenced = False
                yield referenced
                if claim.writer_finished and not self._finish_cleanup_in_session(session, claim):
                    raise ChildVectorCleanupClaimLostError("child-vector cleanup claim changed before acknowledgement")

    @staticmethod
    def _finish_cleanup_in_session(session: Session, claim: ChildVectorCleanupClaim) -> bool:
        route = claim.route
        if route is None:
            return False
        cursor = session.execute(
            delete(ChildVectorCleanupOutbox).where(
                ChildVectorCleanupOutbox.id == claim.id,
                ChildVectorCleanupOutbox.tenant_id == claim.tenant_id,
                ChildVectorCleanupOutbox.dataset_id == claim.dataset_id,
                ChildVectorCleanupOutbox.node_id == claim.node_id,
                ChildVectorCleanupOutbox.vector_route_hash == route.key,
                ChildVectorCleanupOutbox.claim_id == claim.claim_id,
                ChildVectorCleanupOutbox.writer_finished_at.is_not(None),
            )
        )
        return cast(CursorResult[object], cursor).rowcount == 1

    def release_cleanup(self, claim: ChildVectorCleanupClaim) -> bool:
        with self._session_factory() as session, session.begin():
            cursor = session.execute(
                update(ChildVectorCleanupOutbox)
                .where(
                    ChildVectorCleanupOutbox.id == claim.id,
                    ChildVectorCleanupOutbox.tenant_id == claim.tenant_id,
                    ChildVectorCleanupOutbox.node_id == claim.node_id,
                    ChildVectorCleanupOutbox.claim_id == claim.claim_id,
                )
                .values(claim_id=None, claim_expires_at=None)
            )
            return cast(CursorResult[object], cursor).rowcount == 1

    @staticmethod
    def _to_claim(row: ChildVectorCleanupOutbox) -> ChildVectorCleanupClaim:
        if row.claim_id is None:
            raise ValueError("child vector cleanup row has no claim")
        return ChildVectorCleanupClaim(
            id=row.id,
            tenant_id=row.tenant_id,
            dataset_id=row.dataset_id,
            node_id=row.node_id,
            claim_id=row.claim_id,
            writer_finished=row.writer_finished_at is not None,
            route=_route_from_row(row),
        )

    @staticmethod
    def _to_write_intent(row: ChildVectorCleanupOutbox) -> ChildVectorWriteIntent:
        if row.claim_id is None:
            raise ValueError("child vector cleanup row has no writer claim")
        route = _route_from_row(row)
        if route is None:
            raise ValueError("child vector writer row has no valid route")
        return ChildVectorWriteIntent(
            id=row.id,
            tenant_id=row.tenant_id,
            dataset_id=row.dataset_id,
            node_id=row.node_id,
            writer_id=row.writer_id,
            claim_id=row.claim_id,
            route=route,
        )

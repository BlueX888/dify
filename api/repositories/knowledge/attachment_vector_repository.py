"""Durable publication and cleanup ownership for shared attachment vectors."""

import logging
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Event, Thread
from typing import cast
from uuid import uuid4

from sqlalchemy import DateTime, and_, case, delete, func, not_, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from core.db.session_factory import session_factory as default_session_factory
from libs.sqlalchemy_utils import database_operation_timeout
from models.dataset import Dataset, SegmentAttachmentBinding
from models.model import AttachmentVectorCleanupOutbox
from repositories.knowledge.segment_cleanup_repository import (
    lock_segment_attachment_cleanup_dataset,
    lock_segment_attachment_publication_dataset,
)
from services.knowledge.entities.vector_route import VectorRoute

type AttachmentRef = tuple[str, str]
type MutationGuard = Callable[[Session], AbstractContextManager[None]]
type WriteBoundary = Callable[[Sequence[str]], None]
type PublishCallback = Callable[[frozenset[AttachmentRef], WriteBoundary, WriteBoundary], None]

ATTACHMENT_VECTOR_CLEANUP_LEASE_SECONDS = 300
_RENEWAL_DATABASE_TIMEOUT_SECONDS = 1.0
_WRITER_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0

logger = logging.getLogger(__name__)


class AttachmentVectorCleanupClaimLostError(RuntimeError):
    """An attachment-vector writer or cleanup worker no longer owns its lease."""


@dataclass(frozen=True, slots=True)
class AttachmentVectorWriteIntent:
    id: str
    tenant_id: str
    dataset_id: str
    attachment_id: str
    writer_id: str
    claim_id: str
    route: VectorRoute


@dataclass(frozen=True, slots=True)
class AttachmentVectorCleanupClaim:
    id: str
    tenant_id: str
    dataset_id: str
    attachment_id: str
    claim_id: str
    writer_finished: bool
    route: VectorRoute


@dataclass(frozen=True, slots=True)
class AttachmentVectorCleanupGuard:
    session: Session
    dataset: Dataset | None
    has_live_binding: bool


def _database_now(session: Session) -> datetime:
    value = session.scalar(select(func.current_timestamp(type_=DateTime())))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value.replace(tzinfo=None, microsecond=0)


def _claim_available(database_now: datetime):
    return or_(
        AttachmentVectorCleanupOutbox.claim_id.is_(None),
        AttachmentVectorCleanupOutbox.claim_expires_at.is_(None),
        AttachmentVectorCleanupOutbox.claim_expires_at <= database_now,
    )


def _route_from_row(row: AttachmentVectorCleanupOutbox) -> VectorRoute | None:
    try:
        route = VectorRoute.model_validate_json(row.vector_route)
    except ValueError:
        return None
    if route.key != row.vector_route_hash:
        return None
    if route.tenant_id != row.tenant_id or route.dataset_id != row.dataset_id:
        return None
    return route


class SQLAlchemyAttachmentVectorStore:
    """SQLAlchemy implementation of durable attachment publication and cleanup."""

    def __init__(self, *, session_factory: Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory or default_session_factory.get_session_maker()

    def begin_writes(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        attachment_ids: Sequence[str],
        route: VectorRoute,
    ) -> tuple[AttachmentVectorWriteIntent, ...]:
        """Commit one independent receipt per remote publication attempt.

        The caller already holds the Dataset publication lock. This short
        transaction deliberately touches only receipt rows, so MySQL cannot
        self-deadlock behind a queued Dataset UPDATE waiter.
        """
        normalized_ids = tuple(dict.fromkeys(attachment_ids))
        if not normalized_ids:
            return ()
        if route.tenant_id != tenant_id or route.dataset_id != dataset_id:
            raise ValueError("attachment vector route does not belong to the requested dataset")
        with self._session_factory() as session, session.begin():
            database_now = _database_now(session)
            rows = [
                AttachmentVectorCleanupOutbox(
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    attachment_id=attachment_id,
                    writer_id=str(uuid4()),
                    vector_route=route.serialized,
                    vector_route_hash=route.key,
                    claim_id=str(uuid4()),
                    claim_expires_at=database_now + timedelta(seconds=ATTACHMENT_VECTOR_CLEANUP_LEASE_SECONDS),
                    last_attempted_at=database_now,
                )
                for attachment_id in normalized_ids
            ]
            session.add_all(rows)
            session.flush()
            return tuple(self._to_write_intent(row) for row in rows)

    def discard_unsubmitted_writes(self, intents: Sequence[AttachmentVectorWriteIntent]) -> None:
        """Discard calls proven not submitted without erasing a cleanup handoff."""
        if not intents:
            return
        with self._session_factory() as session, session.begin():
            for intent in intents:
                row = session.scalar(
                    select(AttachmentVectorCleanupOutbox)
                    .where(
                        AttachmentVectorCleanupOutbox.id == intent.id,
                        AttachmentVectorCleanupOutbox.writer_id == intent.writer_id,
                    )
                    .with_for_update()
                )
                if row is None:
                    continue
                if row.cleanup_started_at is None:
                    session.delete(row)
                    continue
                row.writer_finished_at = row.writer_finished_at or _database_now(session)
                row.claim_id = None
                row.claim_expires_at = None

    def renew_writes(self, intents: Sequence[AttachmentVectorWriteIntent]) -> None:
        if not intents:
            return
        with self._session_factory() as session, session.begin():
            with database_operation_timeout(session, timeout_seconds=_RENEWAL_DATABASE_TIMEOUT_SECONDS):
                database_now = _database_now(session)
                for intent in intents:
                    cursor = session.execute(
                        update(AttachmentVectorCleanupOutbox)
                        .where(
                            AttachmentVectorCleanupOutbox.id == intent.id,
                            AttachmentVectorCleanupOutbox.writer_id == intent.writer_id,
                            AttachmentVectorCleanupOutbox.claim_id == intent.claim_id,
                            AttachmentVectorCleanupOutbox.cleanup_started_at.is_(None),
                            AttachmentVectorCleanupOutbox.claim_expires_at > database_now,
                        )
                        .values(
                            claim_expires_at=database_now + timedelta(seconds=ATTACHMENT_VECTOR_CLEANUP_LEASE_SECONDS)
                        )
                    )
                    if cast(CursorResult[object], cursor).rowcount != 1:
                        finished = session.scalar(
                            select(AttachmentVectorCleanupOutbox.id).where(
                                AttachmentVectorCleanupOutbox.id == intent.id,
                                AttachmentVectorCleanupOutbox.writer_id == intent.writer_id,
                                AttachmentVectorCleanupOutbox.writer_finished_at.is_not(None),
                            )
                        )
                        if finished is None:
                            raise AttachmentVectorCleanupClaimLostError()

    def finish_writes(self, intents: Sequence[AttachmentVectorWriteIntent]) -> tuple[AttachmentVectorCleanupClaim, ...]:
        """Mark returned calls terminal and retain only untouched writer claims."""
        if not intents:
            return ()
        claims: list[AttachmentVectorCleanupClaim] = []
        lost = False
        with self._session_factory() as session, session.begin():
            database_now = _database_now(session)
            rows = session.scalars(
                select(AttachmentVectorCleanupOutbox)
                .where(AttachmentVectorCleanupOutbox.id.in_([intent.id for intent in intents]))
                .order_by(AttachmentVectorCleanupOutbox.id)
                .with_for_update()
            ).all()
            rows_by_id = {row.id: row for row in rows}
            for intent in intents:
                row = rows_by_id.get(intent.id)
                if (
                    row is None
                    or row.writer_id != intent.writer_id
                    or row.attachment_id != intent.attachment_id
                    or _route_from_row(row) != intent.route
                ):
                    lost = True
                    continue
                row.writer_finished_at = row.writer_finished_at or database_now
                if (
                    row.claim_id != intent.claim_id
                    or row.cleanup_started_at is not None
                    or row.claim_expires_at is None
                    or row.claim_expires_at <= database_now
                ):
                    lost = True
                    continue
                row.claim_expires_at = database_now + timedelta(seconds=ATTACHMENT_VECTOR_CLEANUP_LEASE_SECONDS)
                row.last_attempted_at = database_now
                claims.append(self._to_claim(row))
        if lost or len(claims) != len(intents):
            raise AttachmentVectorCleanupClaimLostError("attachment vector cleanup already started")
        return tuple(claims)

    @staticmethod
    def consume_publication(session: Session, *, claims: Sequence[AttachmentVectorCleanupClaim]) -> None:
        """Consume exact successful receipts while their Dataset lock is held."""
        if not claims:
            return
        database_now = _database_now(session)
        rows = session.scalars(
            select(AttachmentVectorCleanupOutbox)
            .where(AttachmentVectorCleanupOutbox.id.in_([claim.id for claim in claims]))
            .order_by(AttachmentVectorCleanupOutbox.id)
            .with_for_update()
        ).all()
        rows_by_id = {row.id: row for row in rows}
        for claim in claims:
            row = rows_by_id.get(claim.id)
            route = _route_from_row(row) if row is not None else None
            binding_exists = (
                session.scalar(
                    select(SegmentAttachmentBinding.id)
                    .where(
                        SegmentAttachmentBinding.tenant_id == claim.tenant_id,
                        SegmentAttachmentBinding.dataset_id == claim.dataset_id,
                        SegmentAttachmentBinding.attachment_id == claim.attachment_id,
                    )
                    .limit(1)
                )
                is not None
            )
            if (
                row is None
                or route != claim.route
                or row.claim_id != claim.claim_id
                or row.writer_finished_at is None
                or row.cleanup_started_at is not None
                or row.claim_expires_at is None
                or row.claim_expires_at <= database_now
                or not binding_exists
            ):
                raise AttachmentVectorCleanupClaimLostError("attachment publication is no longer current")
        session.execute(delete(AttachmentVectorCleanupOutbox).where(AttachmentVectorCleanupOutbox.id.in_(rows_by_id)))

    def publish_bound(
        self,
        session: Session,
        *,
        tenant_id: str,
        dataset_id: str,
        collection_binding_id: str | None,
        indexing_technique: str,
        attachment_refs: Sequence[AttachmentRef],
        vector_type: str,
        collection_name: str,
        routing_index_struct: str | None = None,
        publish: PublishCallback,
        mutation_guard: MutationGuard | None = None,
    ) -> frozenset[AttachmentRef]:
        route = VectorRoute.capture(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            vector_type=vector_type,
            collection_name=collection_name,
            routing_index_struct=routing_index_struct,
            collection_binding_id=collection_binding_id,
            indexing_technique=indexing_technique,
            doc_form="attachment",
        )
        return publish_bound_attachment_vectors(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            attachment_refs=attachment_refs,
            route=route,
            publish=publish,
            mutation_guard=mutation_guard,
            store=self,
        )

    def delete_unreferenced(
        self,
        session: Session,
        *,
        tenant_id: str,
        dataset_id: str,
        collection_binding_id: str | None,
        indexing_technique: str,
        vector_type: str,
        collection_name: str,
        routing_index_struct: str | None = None,
        document_ids: Sequence[str] | None = None,
        segment_ids: Sequence[str] | None = None,
        candidate_attachment_ids: Sequence[str] | None = None,
        mutation_guard: MutationGuard | None = None,
        validate: Callable[[Session], None] | None = None,
    ) -> tuple[str, ...]:
        route = VectorRoute.capture(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            vector_type=vector_type,
            collection_name=collection_name,
            routing_index_struct=routing_index_struct,
            collection_binding_id=collection_binding_id,
            indexing_technique=indexing_technique,
            doc_form="attachment",
        )
        return delete_unreferenced_attachment_vectors(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_ids=document_ids,
            segment_ids=segment_ids,
            candidate_attachment_ids=candidate_attachment_ids,
            route=route,
            mutation_guard=mutation_guard,
            validate=validate,
        )

    def list_pending_cleanup_ids(self, *, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        with self._session_factory() as session:
            database_now = _database_now(session)
            return tuple(
                session.scalars(
                    select(AttachmentVectorCleanupOutbox.id)
                    .where(_claim_available(database_now))
                    .order_by(
                        case((AttachmentVectorCleanupOutbox.last_attempted_at.is_(None), 0), else_=1),
                        AttachmentVectorCleanupOutbox.last_attempted_at,
                        AttachmentVectorCleanupOutbox.created_at,
                        AttachmentVectorCleanupOutbox.id,
                    )
                    .limit(limit)
                )
            )

    def mark_attempted(self, cleanup_id: str) -> None:
        with self._session_factory() as session, session.begin():
            session.execute(
                update(AttachmentVectorCleanupOutbox)
                .where(AttachmentVectorCleanupOutbox.id == cleanup_id)
                .values(last_attempted_at=func.current_timestamp())
            )

    def claim_pending_cleanup(self, cleanup_id: str) -> AttachmentVectorCleanupClaim | None:
        claim_id = str(uuid4())
        with self._session_factory() as session, session.begin():
            database_now = _database_now(session)
            cursor = session.execute(
                update(AttachmentVectorCleanupOutbox)
                .where(AttachmentVectorCleanupOutbox.id == cleanup_id, _claim_available(database_now))
                .values(
                    claim_id=claim_id,
                    claim_expires_at=database_now + timedelta(seconds=ATTACHMENT_VECTOR_CLEANUP_LEASE_SECONDS),
                    last_attempted_at=database_now,
                    cleanup_started_at=func.coalesce(AttachmentVectorCleanupOutbox.cleanup_started_at, database_now),
                )
            )
            if cast(CursorResult[object], cursor).rowcount != 1:
                return None
            row = session.scalar(
                select(AttachmentVectorCleanupOutbox).where(
                    AttachmentVectorCleanupOutbox.id == cleanup_id,
                    AttachmentVectorCleanupOutbox.claim_id == claim_id,
                )
            )
            return self._to_claim(row) if row is not None else None

    def renew_cleanup(self, claim: AttachmentVectorCleanupClaim) -> None:
        with self._session_factory() as session, session.begin():
            with database_operation_timeout(session, timeout_seconds=_RENEWAL_DATABASE_TIMEOUT_SECONDS):
                database_now = _database_now(session)
                cursor = session.execute(
                    update(AttachmentVectorCleanupOutbox)
                    .where(
                        AttachmentVectorCleanupOutbox.id == claim.id,
                        AttachmentVectorCleanupOutbox.claim_id == claim.claim_id,
                        AttachmentVectorCleanupOutbox.claim_expires_at > database_now,
                    )
                    .values(claim_expires_at=database_now + timedelta(seconds=ATTACHMENT_VECTOR_CLEANUP_LEASE_SECONDS))
                )
                if cast(CursorResult[object], cursor).rowcount != 1:
                    raise AttachmentVectorCleanupClaimLostError()

    @contextmanager
    def guard_cleanup_delete(self, claim: AttachmentVectorCleanupClaim) -> Generator[AttachmentVectorCleanupGuard]:
        """Hold Dataset UPDATE from route/reference validation through delete."""
        with self._session_factory() as session, session.begin():
            dataset = session.scalar(
                select(Dataset)
                .where(Dataset.id == claim.dataset_id, Dataset.tenant_id == claim.tenant_id)
                .with_for_update()
            )
            database_now = _database_now(session)
            current_claim = session.scalar(
                select(AttachmentVectorCleanupOutbox.id).where(
                    AttachmentVectorCleanupOutbox.id == claim.id,
                    AttachmentVectorCleanupOutbox.tenant_id == claim.tenant_id,
                    AttachmentVectorCleanupOutbox.dataset_id == claim.dataset_id,
                    AttachmentVectorCleanupOutbox.attachment_id == claim.attachment_id,
                    AttachmentVectorCleanupOutbox.vector_route_hash == claim.route.key,
                    AttachmentVectorCleanupOutbox.claim_id == claim.claim_id,
                    AttachmentVectorCleanupOutbox.claim_expires_at > database_now,
                )
            )
            if current_claim is None:
                raise AttachmentVectorCleanupClaimLostError("attachment cleanup claim is no longer current")
            has_live_binding = False
            if dataset is not None:
                has_live_binding = (
                    session.scalar(
                        select(SegmentAttachmentBinding.id)
                        .where(
                            SegmentAttachmentBinding.tenant_id == claim.tenant_id,
                            SegmentAttachmentBinding.dataset_id == claim.dataset_id,
                            SegmentAttachmentBinding.attachment_id == claim.attachment_id,
                        )
                        .with_for_update()
                        .limit(1)
                    )
                    is not None
                )
            yield AttachmentVectorCleanupGuard(session=session, dataset=dataset, has_live_binding=has_live_binding)
            if claim.writer_finished and not self._finish_cleanup_in_session(session, claim):
                raise AttachmentVectorCleanupClaimLostError("attachment cleanup claim changed before acknowledgement")

    @staticmethod
    def _finish_cleanup_in_session(session: Session, claim: AttachmentVectorCleanupClaim) -> bool:
        cursor = session.execute(
            delete(AttachmentVectorCleanupOutbox).where(
                AttachmentVectorCleanupOutbox.id == claim.id,
                AttachmentVectorCleanupOutbox.tenant_id == claim.tenant_id,
                AttachmentVectorCleanupOutbox.dataset_id == claim.dataset_id,
                AttachmentVectorCleanupOutbox.attachment_id == claim.attachment_id,
                AttachmentVectorCleanupOutbox.vector_route_hash == claim.route.key,
                AttachmentVectorCleanupOutbox.claim_id == claim.claim_id,
                AttachmentVectorCleanupOutbox.writer_finished_at.is_not(None),
            )
        )
        return cast(CursorResult[object], cursor).rowcount == 1

    def release_cleanup(self, claim: AttachmentVectorCleanupClaim) -> bool:
        with self._session_factory() as session, session.begin():
            cursor = session.execute(
                update(AttachmentVectorCleanupOutbox)
                .where(
                    AttachmentVectorCleanupOutbox.id == claim.id,
                    AttachmentVectorCleanupOutbox.claim_id == claim.claim_id,
                )
                .values(claim_id=None, claim_expires_at=None)
            )
            return cast(CursorResult[object], cursor).rowcount == 1

    @staticmethod
    def _to_write_intent(row: AttachmentVectorCleanupOutbox) -> AttachmentVectorWriteIntent:
        route = _route_from_row(row)
        if route is None or row.claim_id is None:
            raise ValueError("attachment writer row has no trustworthy route or claim")
        return AttachmentVectorWriteIntent(
            id=row.id,
            tenant_id=row.tenant_id,
            dataset_id=row.dataset_id,
            attachment_id=row.attachment_id,
            writer_id=row.writer_id,
            claim_id=row.claim_id,
            route=route,
        )

    @staticmethod
    def _to_claim(row: AttachmentVectorCleanupOutbox) -> AttachmentVectorCleanupClaim:
        route = _route_from_row(row)
        if route is None or row.claim_id is None:
            raise ValueError("attachment cleanup row has no trustworthy route or claim")
        return AttachmentVectorCleanupClaim(
            id=row.id,
            tenant_id=row.tenant_id,
            dataset_id=row.dataset_id,
            attachment_id=row.attachment_id,
            claim_id=row.claim_id,
            writer_finished=row.writer_finished_at is not None,
            route=route,
        )


@contextmanager
def _maintain_writer_claims(
    store: SQLAlchemyAttachmentVectorStore,
    intents: Sequence[AttachmentVectorWriteIntent],
    *,
    interval_seconds: float = _WRITER_HEARTBEAT_SECONDS,
) -> Generator[None]:
    if not intents:
        yield
        return
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                store.renew_writes(intents)
            except AttachmentVectorCleanupClaimLostError:
                return
            except Exception:
                logger.exception("Attachment vector writer heartbeat failed")

    thread = Thread(target=heartbeat, name="attachment-vector-writer-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning("Attachment vector writer heartbeat did not stop promptly")


def publish_bound_attachment_vectors(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    attachment_refs: Sequence[AttachmentRef],
    route: VectorRoute,
    publish: PublishCallback,
    store: SQLAlchemyAttachmentVectorStore,
    mutation_guard: MutationGuard | None = None,
) -> frozenset[AttachmentRef]:
    """Publish bound attachments while retaining only submitted UNKNOWN calls."""
    normalized_refs = tuple(
        dict.fromkeys((document_id, attachment_id) for document_id, attachment_id in attachment_refs)
    )
    if not normalized_refs:
        return frozenset()
    lock_segment_attachment_publication_dataset(session, tenant_id=tenant_id, dataset_id=dataset_id)
    document_ids = {document_id for document_id, _ in normalized_refs}
    attachment_ids = {attachment_id for _, attachment_id in normalized_refs}
    persisted_refs = frozenset(
        session.execute(
            select(SegmentAttachmentBinding.document_id, SegmentAttachmentBinding.attachment_id)
            .where(
                SegmentAttachmentBinding.tenant_id == tenant_id,
                SegmentAttachmentBinding.dataset_id == dataset_id,
                SegmentAttachmentBinding.document_id.in_(document_ids),
                SegmentAttachmentBinding.attachment_id.in_(attachment_ids),
            )
            .distinct()
        ).tuples()
    )
    bound_refs = frozenset(ref for ref in normalized_refs if ref in persisted_refs)
    intents = store.begin_writes(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        attachment_ids=[attachment_id for _, attachment_id in bound_refs],
        route=route,
    )
    intents_by_attachment = {intent.attachment_id: intent for intent in intents}
    started_ids: set[str] = set()
    completed_ids: set[str] = set()
    completed_claims: dict[str, AttachmentVectorCleanupClaim] = {}

    def before_write(batch_attachment_ids: Sequence[str]) -> None:
        batch_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        for attachment_id in batch_attachment_ids:
            if attachment_id in started_ids or attachment_id in batch_ids:
                duplicate_ids.add(attachment_id)
            batch_ids.add(attachment_id)
        if duplicate_ids:
            raise AttachmentVectorCleanupClaimLostError(
                f"attachment write attempt submitted IDs more than once: {sorted(duplicate_ids)}"
            )
        unknown_ids = batch_ids.difference(intents_by_attachment)
        if unknown_ids:
            raise AttachmentVectorCleanupClaimLostError(
                f"attachment write contains unregistered IDs: {sorted(unknown_ids)}"
            )
        started_ids.update(batch_ids)

    def after_write(batch_attachment_ids: Sequence[str]) -> None:
        batch_ids = set(batch_attachment_ids)
        unknown_ids = batch_ids.difference(started_ids)
        if unknown_ids:
            raise AttachmentVectorCleanupClaimLostError(
                f"attachment write completed before registration: {sorted(unknown_ids)}"
            )
        newly_completed = batch_ids.difference(completed_ids)
        claims = store.finish_writes([intents_by_attachment[attachment_id] for attachment_id in newly_completed])
        completed_claims.update((claim.attachment_id, claim) for claim in claims)
        completed_ids.update(newly_completed)

    def discard_unsubmitted() -> None:
        store.discard_unsubmitted_writes(
            [intent for attachment_id, intent in intents_by_attachment.items() if attachment_id not in started_ids]
        )

    def finish_completed() -> None:
        store.consume_publication(session, claims=tuple(completed_claims.values()))

    guard = mutation_guard(session) if mutation_guard is not None else nullcontext()
    try:
        with guard, _maintain_writer_claims(store, intents):
            publish(bound_refs, before_write, after_write)
    except Exception:
        discard_unsubmitted()
        try:
            finish_completed()
        except AttachmentVectorCleanupClaimLostError:
            # after_write already persisted terminal outcomes before reporting
            # lost publication ownership, so cleanup remains convergent.
            logger.info("Completed attachment batch lost publication ownership")
        raise
    discard_unsubmitted()
    finish_completed()
    incomplete_ids = started_ids.difference(completed_ids)
    if incomplete_ids:
        raise AttachmentVectorCleanupClaimLostError(
            f"attachment write returned without completion: {sorted(incomplete_ids)}"
        )
    return bound_refs


def delete_unreferenced_attachment_vectors(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    route: VectorRoute,
    document_ids: Sequence[str] | None = None,
    segment_ids: Sequence[str] | None = None,
    candidate_attachment_ids: Sequence[str] | None = None,
    mutation_guard: MutationGuard | None = None,
    validate: Callable[[Session], None] | None = None,
) -> tuple[str, ...]:
    """Hand off current-route vectors only when every document binding is gone.

    The binding owner has not committed its relational detach yet, so external
    deletion here would leave a live binding without a vector if that database
    transaction rolled back. Instead this transaction terminalizes every
    UNKNOWN writer and adds a finished cleanup receipt. The periodic worker
    performs the physical delete only after the detach commit is observable.
    """
    normalized_document_ids = tuple(dict.fromkeys(document_ids)) if document_ids is not None else None
    normalized_segment_ids = tuple(dict.fromkeys(segment_ids)) if segment_ids is not None else None
    if normalized_document_ids is not None and not normalized_document_ids:
        return ()
    if normalized_segment_ids is not None and not normalized_segment_ids:
        return ()
    normalized_candidate_ids = (
        tuple(dict.fromkeys(candidate_attachment_ids)) if candidate_attachment_ids is not None else None
    )
    if normalized_candidate_ids is not None and not normalized_candidate_ids:
        return ()
    if normalized_document_ids is None and normalized_segment_ids is None and normalized_candidate_ids is None:
        raise ValueError("a cleanup scope or candidate_attachment_ids is required")
    if route.tenant_id != tenant_id or route.dataset_id != dataset_id:
        raise ValueError("attachment cleanup route does not belong to the requested dataset")

    lock_segment_attachment_cleanup_dataset(session, tenant_id=tenant_id, dataset_id=dataset_id)
    scope = [
        SegmentAttachmentBinding.tenant_id == tenant_id,
        SegmentAttachmentBinding.dataset_id == dataset_id,
    ]
    selected_scope = []
    if normalized_document_ids is not None:
        selected_scope.append(SegmentAttachmentBinding.document_id.in_(normalized_document_ids))
    if normalized_segment_ids is not None:
        selected_scope.append(SegmentAttachmentBinding.segment_id.in_(normalized_segment_ids))
    scope.extend(selected_scope)
    candidate_ids = normalized_candidate_ids or tuple(
        dict.fromkeys(
            session.scalars(
                select(SegmentAttachmentBinding.attachment_id)
                .where(*scope)
                .order_by(SegmentAttachmentBinding.attachment_id)
            )
        )
    )
    if not candidate_ids:
        return ()

    guard = mutation_guard(session) if mutation_guard is not None else nullcontext()
    with guard:
        if validate is not None:
            validate(session)
        retained_scope = [
            SegmentAttachmentBinding.tenant_id == tenant_id,
            SegmentAttachmentBinding.dataset_id == dataset_id,
            SegmentAttachmentBinding.attachment_id.in_(candidate_ids),
        ]
        if selected_scope:
            retained_scope.append(not_(and_(*selected_scope)))
        retained_ids = set(
            session.scalars(select(SegmentAttachmentBinding.attachment_id).where(*retained_scope).distinct())
        )
        orphan_ids = tuple(attachment_id for attachment_id in candidate_ids if attachment_id not in retained_ids)
        if not orphan_ids:
            return ()
        database_now = _database_now(session)
        existing_receipt_attachment_ids = set(
            session.scalars(
                select(AttachmentVectorCleanupOutbox.attachment_id)
                .where(
                    AttachmentVectorCleanupOutbox.tenant_id == tenant_id,
                    AttachmentVectorCleanupOutbox.dataset_id == dataset_id,
                    AttachmentVectorCleanupOutbox.attachment_id.in_(orphan_ids),
                    AttachmentVectorCleanupOutbox.vector_route_hash == route.key,
                )
                .with_for_update()
            )
        )
        session.execute(
            update(AttachmentVectorCleanupOutbox)
            .where(
                AttachmentVectorCleanupOutbox.tenant_id == tenant_id,
                AttachmentVectorCleanupOutbox.dataset_id == dataset_id,
                AttachmentVectorCleanupOutbox.attachment_id.in_(orphan_ids),
                AttachmentVectorCleanupOutbox.vector_route_hash == route.key,
            )
            .values(
                cleanup_started_at=func.coalesce(AttachmentVectorCleanupOutbox.cleanup_started_at, database_now),
                claim_id=None,
                claim_expires_at=None,
                last_attempted_at=None,
            )
        )
        session.add_all(
            [
                AttachmentVectorCleanupOutbox(
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    attachment_id=attachment_id,
                    writer_id=str(uuid4()),
                    vector_route=route.serialized,
                    vector_route_hash=route.key,
                    writer_finished_at=database_now,
                    cleanup_started_at=database_now,
                )
                for attachment_id in orphan_ids
                if attachment_id not in existing_receipt_attachment_ids
            ]
        )
        return orphan_ids

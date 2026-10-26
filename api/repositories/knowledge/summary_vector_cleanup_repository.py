"""Persist summary publication outcomes and durable vector-cleanup ownership."""

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import cast
from uuid import uuid4

from sqlalchemy import DateTime, case, delete, exists, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from libs.sqlalchemy_utils import database_operation_timeout
from models.dataset import Dataset, DocumentSegmentSummary
from models.enums import SummaryStatus
from models.model import SummaryVectorCleanupOutbox
from repositories.knowledge.dataset_indexing_outbox_repository import SQLAlchemyDatasetIndexingOutboxRepository
from repositories.knowledge.vector_cleanup_handoff_repository import add_terminal_summary_vector_cleanups
from services.knowledge.entities.summaries import (
    SummaryPublicationResult,
    SummaryVectorCleanupClaim,
    SummaryVectorGenerationLostError,
    SummaryVectorInput,
    SummaryVectorResult,
    SummaryVectorWriteIntent,
)
from services.knowledge.entities.vector_route import VectorRoute

SUMMARY_VECTOR_CLEANUP_LEASE_SECONDS = 300
_RENEWAL_DATABASE_TIMEOUT_SECONDS = 1.0


class SummaryVectorCleanupClaimLostError(RuntimeError):
    """The vector writer or cleanup worker no longer owns the persisted lease."""


def _database_now(session: Session) -> datetime:
    value = session.scalar(select(func.current_timestamp(type_=DateTime())))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value.replace(tzinfo=None, microsecond=0)


def _claim_available(database_now: datetime):
    return or_(
        SummaryVectorCleanupOutbox.claim_id.is_(None),
        SummaryVectorCleanupOutbox.claim_expires_at.is_(None),
        SummaryVectorCleanupOutbox.claim_expires_at <= database_now,
    )


def _pending_cleanup_ids_statement(database_now: datetime, *, limit: int):
    return (
        select(SummaryVectorCleanupOutbox.id)
        .where(_claim_available(database_now))
        .order_by(
            case((SummaryVectorCleanupOutbox.last_attempted_at.is_(None), 0), else_=1),
            SummaryVectorCleanupOutbox.last_attempted_at,
            SummaryVectorCleanupOutbox.created_at,
            SummaryVectorCleanupOutbox.id,
        )
        .limit(limit)
    )


def _route_from_row(row: SummaryVectorCleanupOutbox) -> VectorRoute | None:
    if row.vector_route is None or row.vector_route_hash is None:
        return None
    try:
        route = VectorRoute.model_validate_json(row.vector_route)
    except ValueError:
        return None
    if route.key != row.vector_route_hash or route.tenant_id != row.tenant_id or route.dataset_id != row.dataset_id:
        return None
    return route


def _route_from_summary(summary: DocumentSegmentSummary, *, tenant_id: str) -> VectorRoute | None:
    if summary.vector_route is None or summary.vector_route_hash is None:
        return None
    try:
        route = VectorRoute.model_validate_json(summary.vector_route)
    except ValueError:
        return None
    if route.key != summary.vector_route_hash or route.tenant_id != tenant_id or route.dataset_id != summary.dataset_id:
        return None
    return route


class SQLAlchemySummaryVectorCleanupRepository:
    def __init__(self, *, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def add_terminal_cleanups(
        session: Session,
        *,
        tenant_id: str,
        summaries: Sequence[DocumentSegmentSummary],
    ) -> None:
        """Transfer deleted summary nodes with their proven route, or explicit unknown evidence."""
        summaries_by_dataset: dict[str, list[DocumentSegmentSummary]] = {}
        for summary in summaries:
            summaries_by_dataset.setdefault(summary.dataset_id, []).append(summary)
        for dataset_id, dataset_summaries in summaries_by_dataset.items():
            add_terminal_summary_vector_cleanups(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                summaries=dataset_summaries,
            )

    def begin_write(
        self,
        command: SummaryVectorInput,
        result: SummaryVectorResult,
        route: VectorRoute,
    ) -> SummaryVectorWriteIntent:
        """Persist a stable writer identity before external vector I/O begins."""
        ref = command.ref
        writer_id = str(uuid4())
        claim_id = str(uuid4())
        with self._session_factory() as session, session.begin():
            # Serialize writer registration with Dataset deletion. Once this
            # shared lock is released, the committed outbox row is visible to
            # dataset cleanup before it may delete and acknowledge the parent.
            owned_dataset = session.scalar(
                select(Dataset)
                .where(
                    Dataset.id == ref.document.dataset.dataset_id,
                    Dataset.tenant_id == ref.document.dataset.tenant_id,
                )
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if owned_dataset is None:
                raise LookupError("summary vector dataset no longer exists")
            if owned_dataset.indexing_request_id != command.dataset_generation_id:
                raise SummaryVectorGenerationLostError()
            if route.tenant_id != ref.document.dataset.tenant_id or route.dataset_id != ref.document.dataset.dataset_id:
                raise ValueError("Summary vector route does not belong to the command dataset")
            summary_id = session.scalar(
                select(DocumentSegmentSummary.id)
                .where(
                    self._segment_exists(command),
                    DocumentSegmentSummary.id == command.summary_id,
                    DocumentSegmentSummary.version == command.version,
                    DocumentSegmentSummary.dataset_id == ref.document.dataset.dataset_id,
                    DocumentSegmentSummary.document_id == ref.document.document_id,
                    DocumentSegmentSummary.chunk_id == ref.segment_id,
                )
                .with_for_update()
            )
            if summary_id is None:
                raise SummaryVectorGenerationLostError()
            SQLAlchemyDatasetIndexingOutboxRepository.record_generation_writer_route_locked(
                session,
                dataset=owned_dataset,
                generation_id=command.dataset_generation_id,
                route=route,
            )
            row = session.scalar(
                select(SummaryVectorCleanupOutbox)
                .where(
                    SummaryVectorCleanupOutbox.tenant_id == ref.document.dataset.tenant_id,
                    SummaryVectorCleanupOutbox.node_id == result.node_id,
                )
                .with_for_update()
            )
            if row is None:
                database_now = _database_now(session)
                row = SummaryVectorCleanupOutbox(
                    tenant_id=ref.document.dataset.tenant_id,
                    dataset_id=ref.document.dataset.dataset_id,
                    document_id=ref.document.document_id,
                    segment_id=ref.segment_id,
                    summary_id=command.summary_id,
                    summary_version=command.version,
                    node_id=result.node_id,
                    writer_id=writer_id,
                    vector_route=route.serialized,
                    vector_route_hash=route.key,
                    claim_id=claim_id,
                    claim_expires_at=database_now + timedelta(seconds=SUMMARY_VECTOR_CLEANUP_LEASE_SECONDS),
                    last_attempted_at=database_now,
                )
                session.add(row)
                session.flush()
            else:
                raise SummaryVectorCleanupClaimLostError()
            return self._to_write_intent(row)

    def discard_unsubmitted_write(self, intent: SummaryVectorWriteIntent) -> None:
        """Forget a writer proven not to have crossed the vector mutation boundary.

        The stable writer identity, rather than the renewable publication claim,
        authorizes this terminal discard. A cleanup worker may already own the
        row and may even have started a harmless delete; neither requires this
        nonexistent node to remain durable. A missing row is already terminal.
        """
        with self._session_factory() as session, session.begin():
            row = session.scalar(
                select(SummaryVectorCleanupOutbox)
                .where(
                    SummaryVectorCleanupOutbox.id == intent.id,
                    SummaryVectorCleanupOutbox.tenant_id == intent.tenant_id,
                    SummaryVectorCleanupOutbox.dataset_id == intent.dataset_id,
                    SummaryVectorCleanupOutbox.node_id == intent.node_id,
                    SummaryVectorCleanupOutbox.writer_id == intent.writer_id,
                )
                .with_for_update()
            )
            if row is None:
                return
            if row.cleanup_started_at is None:
                session.delete(row)
                return
            # A terminal handoff may still need to remove a pre-existing vector
            # with this physical identity. Proven non-submission ends only the
            # writer uncertainty; it must not delete the handoff receipt.
            row.writer_finished_at = row.writer_finished_at or _database_now(session)
            row.claim_id = None
            row.claim_expires_at = None

    def finish_write(self, intent: SummaryVectorWriteIntent) -> SummaryVectorCleanupClaim:
        """Record that the external call returned and keep untouched writer ownership.

        The terminal marker is committed even when a cleanup worker currently
        owns the lease. That worker's claim remembers that it observed an
        unfinished writer, so it can delete the node but cannot acknowledge the
        row. The writer is not allowed to publish after any cleanup started; a
        later cleanup pass will delete once more before acknowledging it.
        """
        claim: SummaryVectorCleanupClaim | None = None
        with self._session_factory() as session, session.begin():
            row = session.scalar(
                select(SummaryVectorCleanupOutbox)
                .where(
                    SummaryVectorCleanupOutbox.id == intent.id,
                    SummaryVectorCleanupOutbox.tenant_id == intent.tenant_id,
                    SummaryVectorCleanupOutbox.node_id == intent.node_id,
                    SummaryVectorCleanupOutbox.writer_id == intent.writer_id,
                )
                .with_for_update()
            )
            if row is None:
                raise SummaryVectorCleanupClaimLostError()
            database_now = _database_now(session)
            if row.writer_finished_at is None:
                row.writer_finished_at = database_now
            # Once any cleanup worker has started an external delete, its lease
            # may expire while that delete is still in flight. Never publish this
            # node afterward, even if the old claim now looks available.
            if row.claim_id == intent.claim_id and row.cleanup_started_at is None:
                row.claim_expires_at = database_now + timedelta(seconds=SUMMARY_VECTOR_CLEANUP_LEASE_SECONDS)
                row.last_attempted_at = database_now
                claim = self._to_claim(row)
        if claim is None:
            raise SummaryVectorCleanupClaimLostError("cleanup claim is owned by another worker")
        return claim

    def renew_write(self, intent: SummaryVectorWriteIntent) -> None:
        """Renew only the original writer claim before any cleanup has started."""
        with self._session_factory() as session, session.begin():
            with database_operation_timeout(session, timeout_seconds=_RENEWAL_DATABASE_TIMEOUT_SECONDS):
                database_now = _database_now(session)
                cursor = session.execute(
                    update(SummaryVectorCleanupOutbox)
                    .where(
                        SummaryVectorCleanupOutbox.id == intent.id,
                        SummaryVectorCleanupOutbox.tenant_id == intent.tenant_id,
                        SummaryVectorCleanupOutbox.node_id == intent.node_id,
                        SummaryVectorCleanupOutbox.writer_id == intent.writer_id,
                        SummaryVectorCleanupOutbox.writer_finished_at.is_(None),
                        SummaryVectorCleanupOutbox.cleanup_started_at.is_(None),
                        SummaryVectorCleanupOutbox.claim_id == intent.claim_id,
                        SummaryVectorCleanupOutbox.claim_expires_at > database_now,
                    )
                    .values(claim_expires_at=database_now + timedelta(seconds=SUMMARY_VECTOR_CLEANUP_LEASE_SECONDS))
                )
                if cast(CursorResult[object], cursor).rowcount != 1:
                    raise SummaryVectorCleanupClaimLostError()

    def renew_cleanup(self, claim: SummaryVectorCleanupClaim) -> None:
        """Renew one exact cleanup claim while its external deletion runs."""
        with self._session_factory() as session, session.begin():
            with database_operation_timeout(session, timeout_seconds=_RENEWAL_DATABASE_TIMEOUT_SECONDS):
                database_now = _database_now(session)
                cursor = session.execute(
                    update(SummaryVectorCleanupOutbox)
                    .where(
                        SummaryVectorCleanupOutbox.id == claim.id,
                        SummaryVectorCleanupOutbox.tenant_id == claim.tenant_id,
                        SummaryVectorCleanupOutbox.node_id == claim.node_id,
                        SummaryVectorCleanupOutbox.claim_id == claim.claim_id,
                        SummaryVectorCleanupOutbox.claim_expires_at > database_now,
                    )
                    .values(claim_expires_at=database_now + timedelta(seconds=SUMMARY_VECTOR_CLEANUP_LEASE_SECONDS))
                )
                if cast(CursorResult[object], cursor).rowcount != 1:
                    raise SummaryVectorCleanupClaimLostError()

    def completed(
        self,
        command: SummaryVectorInput,
        result: SummaryVectorResult,
        claim: SummaryVectorCleanupClaim,
    ) -> SummaryPublicationResult:
        """Publish one exact generation and durably transfer its previous vector."""
        ref = command.ref
        with self._session_factory() as session, session.begin():
            database_now = _database_now(session)
            cleanup = session.scalar(
                select(SummaryVectorCleanupOutbox)
                .where(
                    SummaryVectorCleanupOutbox.id == claim.id,
                    SummaryVectorCleanupOutbox.tenant_id == claim.tenant_id,
                    SummaryVectorCleanupOutbox.node_id == result.node_id,
                    SummaryVectorCleanupOutbox.claim_id == claim.claim_id,
                    SummaryVectorCleanupOutbox.claim_expires_at > database_now,
                    SummaryVectorCleanupOutbox.writer_finished_at.is_not(None),
                    SummaryVectorCleanupOutbox.cleanup_started_at.is_(None),
                )
                .with_for_update()
            )
            if cleanup is None:
                return SummaryPublicationResult(False)
            cleanup_route = _route_from_row(cleanup)
            if cleanup_route is None or claim.route != cleanup_route:
                return SummaryPublicationResult(False)
            summary = session.scalar(
                select(DocumentSegmentSummary)
                .where(
                    self._segment_exists(command),
                    DocumentSegmentSummary.id == command.summary_id,
                    DocumentSegmentSummary.version == command.version,
                    DocumentSegmentSummary.dataset_id == ref.document.dataset.dataset_id,
                    DocumentSegmentSummary.document_id == ref.document.document_id,
                    DocumentSegmentSummary.chunk_id == ref.segment_id,
                )
                .with_for_update()
            )
            if summary is None:
                return SummaryPublicationResult(False)
            previous_node_id = summary.summary_index_node_id
            previous_route = _route_from_summary(summary, tenant_id=ref.document.dataset.tenant_id)
            cursor = session.execute(
                update(DocumentSegmentSummary)
                .where(
                    self._segment_exists(command),
                    DocumentSegmentSummary.id == command.summary_id,
                    DocumentSegmentSummary.version == command.version,
                    DocumentSegmentSummary.dataset_id == ref.document.dataset.dataset_id,
                    DocumentSegmentSummary.document_id == ref.document.document_id,
                    DocumentSegmentSummary.chunk_id == ref.segment_id,
                )
                .values(
                    summary_content=command.content,
                    summary_index_node_id=result.node_id,
                    summary_index_node_hash=result.content_hash,
                    vector_route=cleanup_route.serialized,
                    vector_route_hash=cleanup_route.key,
                    tokens=result.tokens,
                    status=SummaryStatus.COMPLETED,
                    error=None,
                    updated_at=database_now,
                )
            )
            if cast(CursorResult[object], cursor).rowcount != 1:
                return SummaryPublicationResult(False)
            previous_cleanup = self._claim_previous_node_cleanup(
                session,
                command=command,
                node_id=previous_node_id,
                route=previous_route,
                replacement_node_id=result.node_id,
                replacement_route=cleanup_route,
                database_now=database_now,
            )
            session.delete(cleanup)
            return SummaryPublicationResult(True, previous_cleanup)

    def failed(self, command: SummaryVectorInput, error: str) -> bool:
        """Mark only the exact current summary generation as failed."""
        ref = command.ref
        previous_node_predicate = (
            DocumentSegmentSummary.summary_index_node_id.is_(None)
            if command.previous_node_id is None
            else DocumentSegmentSummary.summary_index_node_id == command.previous_node_id
        )
        with self._session_factory() as session, session.begin():
            cursor = session.execute(
                update(DocumentSegmentSummary)
                .where(
                    self._segment_exists(command),
                    DocumentSegmentSummary.id == command.summary_id,
                    DocumentSegmentSummary.version == command.version,
                    DocumentSegmentSummary.dataset_id == ref.document.dataset.dataset_id,
                    DocumentSegmentSummary.document_id == ref.document.document_id,
                    DocumentSegmentSummary.chunk_id == ref.segment_id,
                    previous_node_predicate,
                )
                .values(status=SummaryStatus.ERROR, error=error, updated_at=_database_now(session))
            )
            return cast(CursorResult[object], cursor).rowcount == 1

    def _claim_previous_node_cleanup(
        self,
        session: Session,
        *,
        command: SummaryVectorInput,
        node_id: str | None,
        route: VectorRoute | None,
        replacement_node_id: str,
        replacement_route: VectorRoute,
        database_now: datetime,
    ) -> SummaryVectorCleanupClaim | None:
        """Persist and claim a superseded, known-terminal node in the publication transaction."""
        if not node_id or (node_id == replacement_node_id and route is not None and route.key == replacement_route.key):
            return None
        ref = command.ref
        existing = session.scalar(
            select(SummaryVectorCleanupOutbox)
            .where(
                SummaryVectorCleanupOutbox.tenant_id == ref.document.dataset.tenant_id,
                SummaryVectorCleanupOutbox.node_id == node_id,
                (
                    SummaryVectorCleanupOutbox.vector_route_hash == route.key
                    if route is not None
                    else SummaryVectorCleanupOutbox.vector_route_hash.is_(None)
                ),
            )
            .with_for_update()
        )
        if existing is not None:
            # Another writer or cleaner already owns this node. Preserve its
            # state machine; the periodic drainer will reconcile it.
            return None
        claim_id = str(uuid4())
        row = SummaryVectorCleanupOutbox(
            tenant_id=ref.document.dataset.tenant_id,
            dataset_id=ref.document.dataset.dataset_id,
            document_id=ref.document.document_id,
            segment_id=ref.segment_id,
            summary_id=command.summary_id,
            summary_version=command.version,
            node_id=node_id,
            writer_id=str(uuid4()),
            vector_route=route.serialized if route is not None else None,
            vector_route_hash=route.key if route is not None else None,
            writer_finished_at=database_now,
            cleanup_started_at=database_now,
            claim_id=claim_id,
            claim_expires_at=database_now + timedelta(seconds=SUMMARY_VECTOR_CLEANUP_LEASE_SECONDS),
            last_attempted_at=database_now,
        )
        session.add(row)
        session.flush()
        return self._to_claim(row)

    def is_published(
        self,
        command: SummaryVectorInput,
        result: SummaryVectorResult,
        claim: SummaryVectorCleanupClaim,
    ) -> bool:
        """Resolve an uncertain commit using the exact generation, node, and physical route.

        The application invokes this while its Dataset generation SHARE guard
        remains held. That outer owner lock prevents cleanup's Dataset UPDATE
        guard from crossing this read without requesting a second Dataset lock
        from this session (which could self-deadlock behind a queued writer).
        """
        ref = command.ref
        if claim.route is None:
            return False
        with self._session_factory() as session:
            return bool(
                session.scalar(
                    select(
                        exists().where(
                            self._segment_exists(command),
                            DocumentSegmentSummary.id == command.summary_id,
                            DocumentSegmentSummary.version == command.version,
                            DocumentSegmentSummary.dataset_id == ref.document.dataset.dataset_id,
                            DocumentSegmentSummary.document_id == ref.document.document_id,
                            DocumentSegmentSummary.chunk_id == ref.segment_id,
                            DocumentSegmentSummary.summary_content == command.content,
                            DocumentSegmentSummary.summary_index_node_id == result.node_id,
                            DocumentSegmentSummary.summary_index_node_hash == result.content_hash,
                            DocumentSegmentSummary.vector_route_hash == claim.route.key,
                            DocumentSegmentSummary.tokens == result.tokens,
                            DocumentSegmentSummary.status == SummaryStatus.COMPLETED,
                        )
                    )
                )
            )

    def release_cleanup(self, claim: SummaryVectorCleanupClaim) -> bool:
        with self._session_factory() as session, session.begin():
            cursor = session.execute(
                update(SummaryVectorCleanupOutbox)
                .where(
                    SummaryVectorCleanupOutbox.id == claim.id,
                    SummaryVectorCleanupOutbox.tenant_id == claim.tenant_id,
                    SummaryVectorCleanupOutbox.node_id == claim.node_id,
                    SummaryVectorCleanupOutbox.claim_id == claim.claim_id,
                )
                .values(claim_id=None, claim_expires_at=None)
            )
            return cast(CursorResult[object], cursor).rowcount == 1

    def list_pending_cleanup_ids(self, *, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        with self._session_factory() as session:
            database_now = _database_now(session)
            return tuple(session.scalars(_pending_cleanup_ids_statement(database_now, limit=limit)))

    def mark_attempted(self, cleanup_id: str) -> None:
        """Rotate every delivery attempt, including rejection by the broker."""
        with self._session_factory() as session, session.begin():
            session.execute(
                update(SummaryVectorCleanupOutbox)
                .where(SummaryVectorCleanupOutbox.id == cleanup_id)
                .values(last_attempted_at=_database_now(session))
            )

    def claim_pending_cleanup(self, cleanup_id: str) -> SummaryVectorCleanupClaim | None:
        claim_id = str(uuid4())
        with self._session_factory() as session, session.begin():
            database_now = _database_now(session)
            cursor = session.execute(
                update(SummaryVectorCleanupOutbox)
                .where(
                    SummaryVectorCleanupOutbox.id == cleanup_id,
                    _claim_available(database_now),
                )
                .values(
                    claim_id=claim_id,
                    claim_expires_at=database_now + timedelta(seconds=SUMMARY_VECTOR_CLEANUP_LEASE_SECONDS),
                    last_attempted_at=database_now,
                    cleanup_started_at=func.coalesce(
                        SummaryVectorCleanupOutbox.cleanup_started_at,
                        database_now,
                    ),
                )
            )
            if cast(CursorResult[object], cursor).rowcount != 1:
                return None
            row = session.scalar(
                select(SummaryVectorCleanupOutbox).where(
                    SummaryVectorCleanupOutbox.id == cleanup_id,
                    SummaryVectorCleanupOutbox.claim_id == claim_id,
                )
            )
            return self._to_claim(row) if row is not None else None

    @contextmanager
    def guard_cleanup_delete(self, claim: SummaryVectorCleanupClaim) -> Generator[bool]:
        """Freeze references through deletion and exact terminal acknowledgement.

        A referenced summary row stays locked until a finished cleanup receipt is
        acknowledged. Concurrent detach therefore observes the ACK before its
        handoff check and persists a fresh receipt for the newly unreferenced
        node. Unfinished writers are never acknowledged because their remote
        mutation may still arrive after this pass.
        """
        if claim.route is None:
            raise SummaryVectorCleanupClaimLostError("summary vector cleanup has no valid route")
        with self._session_factory() as session:
            with session.begin():
                dataset_exists = (
                    session.scalar(
                        select(Dataset.id)
                        .where(Dataset.id == claim.dataset_id, Dataset.tenant_id == claim.tenant_id)
                        .with_for_update()
                    )
                    is not None
                )
                database_now = _database_now(session)
                current_claim = session.scalar(
                    select(SummaryVectorCleanupOutbox.id).where(
                        SummaryVectorCleanupOutbox.id == claim.id,
                        SummaryVectorCleanupOutbox.tenant_id == claim.tenant_id,
                        SummaryVectorCleanupOutbox.dataset_id == claim.dataset_id,
                        SummaryVectorCleanupOutbox.node_id == claim.node_id,
                        SummaryVectorCleanupOutbox.vector_route_hash == claim.route.key,
                        SummaryVectorCleanupOutbox.claim_id == claim.claim_id,
                        SummaryVectorCleanupOutbox.claim_expires_at > database_now,
                    )
                )
                if current_claim is None:
                    raise SummaryVectorCleanupClaimLostError("summary vector cleanup claim is no longer current")
                if dataset_exists:
                    referenced = (
                        session.scalar(
                            select(DocumentSegmentSummary.id)
                            .where(
                                DocumentSegmentSummary.dataset_id == claim.dataset_id,
                                DocumentSegmentSummary.summary_index_node_id == claim.node_id,
                                DocumentSegmentSummary.vector_route_hash == claim.route.key,
                                DocumentSegmentSummary.enabled.is_(True),
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
                    raise SummaryVectorCleanupClaimLostError("summary cleanup claim changed before acknowledgement")

    @staticmethod
    def _finish_cleanup_in_session(session: Session, claim: SummaryVectorCleanupClaim) -> bool:
        route = claim.route
        if route is None:
            return False
        cursor = session.execute(
            delete(SummaryVectorCleanupOutbox).where(
                SummaryVectorCleanupOutbox.id == claim.id,
                SummaryVectorCleanupOutbox.tenant_id == claim.tenant_id,
                SummaryVectorCleanupOutbox.dataset_id == claim.dataset_id,
                SummaryVectorCleanupOutbox.node_id == claim.node_id,
                SummaryVectorCleanupOutbox.vector_route_hash == route.key,
                SummaryVectorCleanupOutbox.claim_id == claim.claim_id,
                SummaryVectorCleanupOutbox.writer_finished_at.is_not(None),
            )
        )
        return cast(CursorResult[object], cursor).rowcount == 1

    @staticmethod
    def _segment_exists(command: SummaryVectorInput):
        from models.dataset import Document, DocumentSegment

        ref = command.ref
        return exists(
            select(DocumentSegment.id)
            .join(Document, Document.id == DocumentSegment.document_id)
            .join(Dataset, Dataset.id == Document.dataset_id)
            .where(
                Dataset.id == ref.document.dataset.dataset_id,
                Dataset.tenant_id == ref.document.dataset.tenant_id,
                Document.id == ref.document.document_id,
                Document.tenant_id == ref.document.dataset.tenant_id,
                DocumentSegment.id == ref.segment_id,
                DocumentSegment.dataset_id == ref.document.dataset.dataset_id,
                DocumentSegment.tenant_id == ref.document.dataset.tenant_id,
            )
        )

    @staticmethod
    def _to_claim(row: SummaryVectorCleanupOutbox) -> SummaryVectorCleanupClaim:
        if row.claim_id is None:
            raise ValueError("summary vector cleanup row has no claim")
        return SummaryVectorCleanupClaim(
            id=row.id,
            tenant_id=row.tenant_id,
            dataset_id=row.dataset_id,
            node_id=row.node_id,
            claim_id=row.claim_id,
            writer_finished=row.writer_finished_at is not None,
            route=_route_from_row(row),
        )

    @staticmethod
    def _to_write_intent(row: SummaryVectorCleanupOutbox) -> SummaryVectorWriteIntent:
        if row.writer_id is None or row.claim_id is None:
            raise ValueError("summary vector cleanup row has no writer identity")
        route = _route_from_row(row)
        if route is None:
            raise ValueError("summary vector cleanup row has no valid route")
        return SummaryVectorWriteIntent(
            id=row.id,
            tenant_id=row.tenant_id,
            dataset_id=row.dataset_id,
            node_id=row.node_id,
            writer_id=row.writer_id,
            claim_id=row.claim_id,
            route=route,
        )

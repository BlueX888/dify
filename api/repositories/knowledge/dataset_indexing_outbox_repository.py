"""Persistence and ownership for durable dataset configuration rebuilds."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import cast
from uuid import uuid4

from sqlalchemy import DateTime, and_, case, delete, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, sessionmaker

from libs.sqlalchemy_utils import database_operation_timeout
from models.dataset import Dataset, Document
from models.model import DatasetIndexingOutbox
from services.knowledge.entities.datasets import (
    DatasetIndexingGenerationLostError,
    DatasetIndexingJob,
    DatasetIndexingRepresentation,
)
from services.knowledge.entities.vector_route import VectorRoute

__all__ = ["DatasetIndexingGenerationLostError"]

DATASET_INDEXING_EXECUTION_LEASE_SECONDS = 300
_RENEWAL_DATABASE_TIMEOUT_SECONDS = 1


class DatasetIndexingOperation(StrEnum):
    """One ordered vector-then-summary execution for a configuration generation."""

    REBUILD = "rebuild"


class DatasetIndexingClaimLostError(RuntimeError):
    pass


class DatasetIndexingWritersActiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetIndexingIntent:
    id: str
    tenant_id: str
    dataset_id: str
    generation_id: str
    action: str
    source: DatasetIndexingRepresentation
    target: DatasetIndexingRepresentation
    claim_id: str
    vector_completed: bool


def _database_now(session: Session) -> datetime:
    value = session.scalar(select(func.current_timestamp(type_=DateTime())))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value.replace(tzinfo=None, microsecond=0)


def _claim_available(database_now: datetime):
    return or_(
        DatasetIndexingOutbox.claim_id.is_(None),
        DatasetIndexingOutbox.claim_expires_at.is_(None),
        DatasetIndexingOutbox.claim_expires_at <= database_now,
    )


def create_dataset_indexing_intents(session: Session, job: DatasetIndexingJob | None) -> tuple[str, ...]:
    """Create one intent from locking current reads of applied and partial state.

    Callers may already have established a MySQL REPEATABLE READ snapshot.
    Dataset then Outbox ``FOR UPDATE`` reads deliberately bypass that snapshot,
    and ``populate_existing`` prevents the identity map from restoring stale
    applied or started-target receipts.
    """
    if job is None:
        return ()

    dataset = session.scalar(
        select(Dataset)
        .where(Dataset.id == job.dataset.dataset_id, Dataset.tenant_id == job.dataset.tenant_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if dataset is None:
        raise ValueError("Dataset not found while creating indexing intent")

    if dataset.indexing_applied_representation is None:
        applied_source = job.source.model_copy(update={"cleanup_routes": ()})
        dataset.indexing_applied_representation = applied_source.model_dump_json()
    else:
        applied_source = DatasetIndexingRepresentation.model_validate_json(dataset.indexing_applied_representation)
    if applied_source.vector_route is not None and (
        applied_source.vector_route.tenant_id != job.dataset.tenant_id
        or applied_source.vector_route.dataset_id != job.dataset.dataset_id
    ):
        raise ValueError("Applied dataset vector route belongs to another owner")

    cleanup_routes = {
        route.key: route
        for route in (
            *applied_source.cleanup_routes,
            *((applied_source.vector_route,) if applied_source.vector_route is not None else ()),
        )
    }
    # Keep Dataset -> Outbox lock order aligned with claim/execution paths.
    # This locking read must observe a partial target committed after the
    # caller's earlier consistent read, otherwise superseding it loses cleanup
    # evidence permanently.
    superseded_rows = session.scalars(
        select(DatasetIndexingOutbox)
        .where(
            DatasetIndexingOutbox.tenant_id == job.dataset.tenant_id,
            DatasetIndexingOutbox.dataset_id == job.dataset.dataset_id,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    ).all()
    for superseded in superseded_rows:
        for serialized in (superseded.source_representation, superseded.target_representation):
            representation = DatasetIndexingRepresentation.model_validate_json(serialized)
            routes = (
                *representation.cleanup_routes,
                *((representation.vector_route,) if representation.vector_route is not None else ()),
            )
            cleanup_routes.update({route.key: route for route in routes})
    source = applied_source.model_copy(update={"cleanup_routes": tuple(cleanup_routes.values())})
    target = job.target.model_copy(update={"vector_route": None, "cleanup_routes": ()})

    generation_id = str(uuid4())
    dataset.indexing_request_id = generation_id
    if job.source.indexing_technique != job.target.indexing_technique:
        action = "remove" if job.target.indexing_technique == "economy" else "add"
    else:
        action = "update"
    row = DatasetIndexingOutbox(
        tenant_id=job.dataset.tenant_id,
        dataset_id=job.dataset.dataset_id,
        generation_id=generation_id,
        operation=DatasetIndexingOperation.REBUILD.value,
        action=action,
        source_representation=source.model_dump_json(),
        target_representation=target.model_dump_json(),
    )
    session.add(row)
    session.flush()
    return (row.id,)


class SQLAlchemyDatasetIndexingOutboxRepository:
    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def list_pending_intent_ids(self, *, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        with self._session_factory() as session:
            database_now = _database_now(session)
            return tuple(
                session.scalars(
                    select(DatasetIndexingOutbox.id)
                    .where(_claim_available(database_now))
                    .order_by(
                        case((DatasetIndexingOutbox.last_attempted_at.is_(None), 0), else_=1),
                        DatasetIndexingOutbox.last_attempted_at,
                        DatasetIndexingOutbox.created_at,
                        DatasetIndexingOutbox.id,
                    )
                    .limit(limit)
                )
            )

    def claim(self, intent_id: str) -> DatasetIndexingIntent | None:
        """Claim execution after taking the Dataset lock used by actual mutations."""
        claim_id = str(uuid4())
        with self._session_factory.begin() as session:
            owner = session.execute(
                select(
                    DatasetIndexingOutbox.tenant_id,
                    DatasetIndexingOutbox.dataset_id,
                    DatasetIndexingOutbox.generation_id,
                ).where(DatasetIndexingOutbox.id == intent_id)
            ).one_or_none()
            if owner is None:
                return None
            tenant_id, dataset_id, generation_id = owner
            dataset = session.scalar(
                select(Dataset)
                .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            row = session.scalar(
                select(DatasetIndexingOutbox)
                .where(DatasetIndexingOutbox.id == intent_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if row is None:
                return None
            # Dataset deletion owns the explicit receipt handoff. A missing
            # parent without that proof must leave the route evidence durable
            # instead of silently discarding a possibly-started target.
            if dataset is None:
                return None
            if dataset.indexing_request_id != generation_id:
                session.delete(row)
                return None
            if row.operation != DatasetIndexingOperation.REBUILD.value or row.action is None:
                session.delete(row)
                return None
            database_now = _database_now(session)
            if not (row.claim_id is None or row.claim_expires_at is None or row.claim_expires_at <= database_now):
                return None
            row.claim_id = claim_id
            row.claim_expires_at = database_now + timedelta(seconds=DATASET_INDEXING_EXECUTION_LEASE_SECONDS)
            row.last_attempted_at = database_now
            session.flush()
            return DatasetIndexingIntent(
                id=row.id,
                tenant_id=row.tenant_id,
                dataset_id=row.dataset_id,
                generation_id=row.generation_id,
                action=row.action,
                source=DatasetIndexingRepresentation.model_validate_json(row.source_representation),
                target=DatasetIndexingRepresentation.model_validate_json(row.target_representation),
                claim_id=claim_id,
                vector_completed=row.vector_completed_at is not None,
            )

    def renew_claim(self, intent_id: str, *, claim_id: str) -> None:
        """Renew an exact execution claim without taking the Dataset mutation lock."""
        with self._session_factory.begin() as session:
            with database_operation_timeout(session, timeout_seconds=_RENEWAL_DATABASE_TIMEOUT_SECONDS):
                database_now = _database_now(session)
                cursor = session.execute(
                    update(DatasetIndexingOutbox)
                    .where(DatasetIndexingOutbox.id == intent_id, DatasetIndexingOutbox.claim_id == claim_id)
                    .values(claim_expires_at=database_now + timedelta(seconds=DATASET_INDEXING_EXECUTION_LEASE_SECONDS))
                )
            if cast(CursorResult[object], cursor).rowcount != 1:
                raise DatasetIndexingClaimLostError(intent_id)

    @staticmethod
    def refresh_locked_claim(session: Session, intent_id: str, *, claim_id: str) -> None:
        """Refresh while the caller holds the Dataset execution lock."""
        database_now = _database_now(session)
        cursor = session.execute(
            update(DatasetIndexingOutbox)
            .where(DatasetIndexingOutbox.id == intent_id, DatasetIndexingOutbox.claim_id == claim_id)
            .values(claim_expires_at=database_now + timedelta(seconds=DATASET_INDEXING_EXECUTION_LEASE_SECONDS))
        )
        if cast(CursorResult[object], cursor).rowcount != 1:
            raise DatasetIndexingClaimLostError(intent_id)

    @staticmethod
    def record_target_route_locked(
        session: Session,
        *,
        intent_id: str,
        tenant_id: str,
        dataset_id: str,
        generation_id: str,
        claim_id: str,
        route: VectorRoute,
    ) -> DatasetIndexingRepresentation:
        """Persist actual target evidence before any external vector mutation."""
        if route.tenant_id != tenant_id or route.dataset_id != dataset_id:
            raise ValueError("Target vector route belongs to another dataset")
        row = session.scalar(
            select(DatasetIndexingOutbox).where(
                DatasetIndexingOutbox.id == intent_id,
                DatasetIndexingOutbox.tenant_id == tenant_id,
                DatasetIndexingOutbox.dataset_id == dataset_id,
                DatasetIndexingOutbox.generation_id == generation_id,
                DatasetIndexingOutbox.claim_id == claim_id,
            )
        )
        if row is None:
            raise DatasetIndexingClaimLostError(intent_id)
        target = DatasetIndexingRepresentation.model_validate_json(row.target_representation)
        cleanup_routes = {cleanup.key: cleanup for cleanup in target.cleanup_routes}
        # Any prior started target may contain a partial or commit-unknown
        # publication. Reconcile it before retrying, even when the factory
        # resolves to the same physical destination again.
        if target.vector_route is not None:
            cleanup_routes[target.vector_route.key] = target.vector_route
        target = target.model_copy(update={"vector_route": route, "cleanup_routes": tuple(cleanup_routes.values())})
        row.target_representation = target.model_dump_json()
        session.flush()
        return target

    @staticmethod
    def record_generation_writer_route_locked(
        session: Session,
        *,
        dataset: Dataset,
        generation_id: str | None,
        route: VectorRoute,
    ) -> bool:
        """Persist an ordinary writer's physical target before external I/O.

        The caller must already hold the Dataset generation lock and the exact
        domain-writer lock (document claim, segment intent/version, child
        writer, or summary version). This method deliberately does not accept a
        rebuild claim: ordinary writers append cleanup evidence without
        becoming the rebuild owner or changing its logical target route.

        ``False`` means no pending receipt needed an update: either this is the
        legacy ``None`` generation, or the completed generation's applied
        receipt already proves the same physical target. A non-legacy writer
        whose actual route differs from that applied receipt is rejected when
        there is no pending outbox row to retain new cleanup evidence.
        """
        if route.tenant_id != dataset.tenant_id or route.dataset_id != dataset.id:
            raise ValueError("Writer vector route belongs to another dataset")
        if dataset.indexing_request_id != generation_id:
            raise DatasetIndexingGenerationLostError(dataset.id)
        if generation_id is None:
            return False
        row = session.scalar(
            select(DatasetIndexingOutbox)
            .where(
                DatasetIndexingOutbox.tenant_id == dataset.tenant_id,
                DatasetIndexingOutbox.dataset_id == dataset.id,
                DatasetIndexingOutbox.generation_id == generation_id,
                DatasetIndexingOutbox.operation == DatasetIndexingOperation.REBUILD.value,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row is None:
            if dataset.indexing_applied_representation is not None:
                applied = DatasetIndexingRepresentation.model_validate_json(dataset.indexing_applied_representation)
                if applied.vector_route is not None and applied.vector_route.key == route.key:
                    return False
            raise DatasetIndexingGenerationLostError(dataset.id)
        target = DatasetIndexingRepresentation.model_validate_json(row.target_representation)
        cleanup_routes = {cleanup.key: cleanup for cleanup in target.cleanup_routes}
        cleanup_routes[route.key] = route
        row.target_representation = target.model_copy(
            update={"cleanup_routes": tuple(cleanup_routes.values())}
        ).model_dump_json()
        session.flush()
        return True

    @staticmethod
    def record_applied_representation_locked(
        session: Session,
        dataset: Dataset,
        *,
        intent_id: str,
        generation_id: str,
        claim_id: str,
        target: DatasetIndexingRepresentation,
    ) -> None:
        """Advance the applied receipt in the vector phase's final transaction."""
        row = session.scalar(
            select(DatasetIndexingOutbox).where(
                DatasetIndexingOutbox.id == intent_id,
                DatasetIndexingOutbox.tenant_id == dataset.tenant_id,
                DatasetIndexingOutbox.dataset_id == dataset.id,
                DatasetIndexingOutbox.generation_id == generation_id,
                DatasetIndexingOutbox.claim_id == claim_id,
            )
        )
        if row is None or dataset.indexing_request_id != generation_id:
            raise DatasetIndexingClaimLostError(intent_id)
        stored_target = DatasetIndexingRepresentation.model_validate_json(row.target_representation)
        if stored_target != target:
            raise DatasetIndexingClaimLostError(intent_id)
        if stored_target.vector_route is not None and (
            stored_target.vector_route.tenant_id != dataset.tenant_id
            or stored_target.vector_route.dataset_id != dataset.id
        ):
            raise ValueError("Applied vector route belongs to another dataset")
        dataset.indexing_applied_representation = stored_target.model_copy(
            update={"cleanup_routes": ()}
        ).model_dump_json()

    def mark_vector_completed(self, intent: DatasetIndexingIntent) -> bool:
        """Persist the phase boundary only for the exact current execution owner."""
        with self._session_factory.begin() as session:
            dataset = session.scalar(
                select(Dataset)
                .where(Dataset.id == intent.dataset_id, Dataset.tenant_id == intent.tenant_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if dataset is None or dataset.indexing_request_id != intent.generation_id:
                return False
            cursor = session.execute(
                update(DatasetIndexingOutbox)
                .where(
                    DatasetIndexingOutbox.id == intent.id,
                    DatasetIndexingOutbox.tenant_id == intent.tenant_id,
                    DatasetIndexingOutbox.dataset_id == intent.dataset_id,
                    DatasetIndexingOutbox.generation_id == intent.generation_id,
                    DatasetIndexingOutbox.claim_id == intent.claim_id,
                )
                .values(vector_completed_at=func.current_timestamp())
            )
            return cast(CursorResult[object], cursor).rowcount == 1

    def complete(self, intent: DatasetIndexingIntent) -> bool:
        with self._session_factory.begin() as session:
            cursor = session.execute(
                delete(DatasetIndexingOutbox).where(
                    DatasetIndexingOutbox.id == intent.id,
                    DatasetIndexingOutbox.tenant_id == intent.tenant_id,
                    DatasetIndexingOutbox.dataset_id == intent.dataset_id,
                    DatasetIndexingOutbox.generation_id == intent.generation_id,
                    DatasetIndexingOutbox.claim_id == intent.claim_id,
                )
            )
            return cast(CursorResult[object], cursor).rowcount == 1

    def release(self, intent: DatasetIndexingIntent) -> bool:
        with self._session_factory.begin() as session:
            cursor = session.execute(
                update(DatasetIndexingOutbox)
                .where(DatasetIndexingOutbox.id == intent.id, DatasetIndexingOutbox.claim_id == intent.claim_id)
                .values(claim_id=None, claim_expires_at=None)
            )
            return cast(CursorResult[object], cursor).rowcount == 1

    def mark_attempted(self, intent_id: str) -> None:
        with self._session_factory.begin() as session:
            session.execute(
                update(DatasetIndexingOutbox)
                .where(DatasetIndexingOutbox.id == intent_id)
                .values(last_attempted_at=func.current_timestamp())
            )

    def find_current_intent_id(self, *, dataset_id: str, generation_id: str | None) -> str | None:
        """Resolve a trusted durable intent for a legacy queued task message."""
        with self._session_factory() as session:
            conditions = [
                Dataset.id == dataset_id,
                DatasetIndexingOutbox.dataset_id == Dataset.id,
                DatasetIndexingOutbox.generation_id == Dataset.indexing_request_id,
                DatasetIndexingOutbox.operation == DatasetIndexingOperation.REBUILD.value,
            ]
            if generation_id is not None:
                conditions.append(Dataset.indexing_request_id == generation_id)
            return session.scalar(
                select(DatasetIndexingOutbox.id)
                .join(Dataset, Dataset.id == DatasetIndexingOutbox.dataset_id)
                .where(*conditions)
                .limit(1)
            )


def dataset_indexing_generation_lock_statement(*, dataset_id: str, generation_id: str | None, read: bool = False):
    """Build a locking current read for one exact configuration generation."""
    generation_predicate = (
        Dataset.indexing_request_id.is_(None) if generation_id is None else Dataset.indexing_request_id == generation_id
    )
    return (
        select(Dataset)
        .where(Dataset.id == dataset_id, generation_predicate)
        .with_for_update(read=read)
        .execution_options(populate_existing=True)
    )


def lock_dataset_indexing_generation(session: Session, *, dataset_id: str, generation_id: str | None) -> Dataset | None:
    """Lock the current generation, rejecting tokenless work after durable adoption."""
    return session.scalar(
        dataset_indexing_generation_lock_statement(dataset_id=dataset_id, generation_id=generation_id)
    )


def lock_dataset_indexing_generation_shared(
    session: Session, *, dataset_id: str, generation_id: str | None
) -> Dataset | None:
    return session.scalar(
        dataset_indexing_generation_lock_statement(dataset_id=dataset_id, generation_id=generation_id, read=True)
    )


def _lock_dataset_indexing_execution(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    generation_id: str,
    intent_id: str,
    claim_id: str,
    read: bool,
    require_quiescent: bool,
) -> Dataset | None:
    """Fence I/O by Dataset first, then validate the exact execution owner.

    The outbox row is not locked across remote I/O so its heartbeat can renew.
    Claim takeover takes the Dataset exclusive lock first and therefore cannot
    pass this guard while a mutation is running.
    """
    dataset = session.scalar(
        select(Dataset)
        .where(
            Dataset.id == dataset_id,
            Dataset.tenant_id == tenant_id,
            Dataset.indexing_request_id == generation_id,
        )
        .with_for_update(read=read)
        .execution_options(populate_existing=True)
    )
    if dataset is None:
        return None
    owned_intent_id = session.scalar(
        select(DatasetIndexingOutbox.id).where(
            DatasetIndexingOutbox.id == intent_id,
            DatasetIndexingOutbox.tenant_id == tenant_id,
            DatasetIndexingOutbox.dataset_id == dataset_id,
            DatasetIndexingOutbox.generation_id == generation_id,
            DatasetIndexingOutbox.claim_id == claim_id,
        )
    )
    if owned_intent_id is None:
        return None
    if require_quiescent:
        database_now = _database_now(session)
        # The Dataset X lock proves that no generation-aware writer is inside
        # its actual mutation scope. Convert stale owners to the same explicit
        # full-restart state used when a worker detects parent-generation loss.
        # Import locally because document_repository depends on this module's
        # generation lock helpers.
        from repositories.knowledge.document_repository import transition_dataset_generation_owner_to_restart

        stale_documents = session.scalars(
            select(Document)
            .where(
                Document.tenant_id == tenant_id,
                Document.dataset_id == dataset_id,
                or_(
                    Document.indexing_claim_id.is_not(None),
                    Document.indexing_dataset_generation_id.is_not(None),
                ),
                or_(
                    Document.indexing_claim_id.is_(None),
                    Document.indexing_claim_expires_at.is_(None),
                    Document.indexing_claim_expires_at <= database_now,
                    Document.indexing_dataset_generation_id.is_(None),
                    Document.indexing_dataset_generation_id != generation_id,
                ),
            )
            .with_for_update(of=Document)
            .execution_options(populate_existing=True)
        ).all()
        for stale_document in stale_documents:
            transition_dataset_generation_owner_to_restart(
                stale_document,
                restarted_at=database_now,
            )
        active_writer = session.scalar(
            select(Document.id)
            .where(
                Document.tenant_id == tenant_id,
                Document.dataset_id == dataset_id,
                or_(
                    and_(
                        Document.indexing_claim_id.is_not(None),
                        Document.indexing_claim_expires_at > database_now,
                        Document.indexing_dataset_generation_id == generation_id,
                    ),
                    and_(
                        Document.indexing_request_id.is_not(None),
                        Document.enabled.is_(True),
                        Document.archived.is_(False),
                        Document.is_paused.is_not(True),
                    ),
                ),
            )
            .limit(1)
        )
        if active_writer is not None:
            raise DatasetIndexingWritersActiveError(dataset_id)
    return dataset


def lock_dataset_indexing_execution(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    generation_id: str,
    intent_id: str,
    claim_id: str,
    require_quiescent: bool = False,
) -> Dataset | None:
    return _lock_dataset_indexing_execution(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        generation_id=generation_id,
        intent_id=intent_id,
        claim_id=claim_id,
        read=False,
        require_quiescent=require_quiescent,
    )


def lock_dataset_indexing_execution_shared(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    generation_id: str,
    intent_id: str,
    claim_id: str,
) -> Dataset | None:
    return _lock_dataset_indexing_execution(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        generation_id=generation_id,
        intent_id=intent_id,
        claim_id=claim_id,
        read=True,
        require_quiescent=False,
    )

"""Persistence boundary for cleanup work created with resource deletion."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import DateTime, case, delete, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, sessionmaker

from constants.resource_access_token import ResourceAccessTokenResourceType
from libs.sqlalchemy_utils import database_operation_timeout
from models.dataset import Dataset, DatasetPermission, Document
from models.model import (
    AttachmentVectorCleanupOutbox,
    ChildVectorCleanupOutbox,
    DatasetIndexingOutbox,
    ResourceCleanupOutbox,
    SummaryVectorCleanupOutbox,
)
from repositories.knowledge import dataset_api_key_bindings
from repositories.knowledge.dataset_api_key_bindings import DeletedDatasetApiKey
from repositories.knowledge.dataset_read_repository import get_dataset_doc_form
from services.knowledge.entities.datasets import DatasetIndexingRepresentation
from services.knowledge.entities.segments import ChildVectorRoute
from services.knowledge.entities.vector_route import VectorRoute, vector_route_target_from_index_struct

RESOURCE_CLEANUP_LEASE_SECONDS = 300
_CLAIM_RENEWAL_DATABASE_TIMEOUT_SECONDS = 1.0


class ResourceCleanupType(StrEnum):
    DATASET = "dataset"
    DATASET_RELATIONAL = "dataset_rel"
    DATASET_VECTOR = "dataset_vector"
    DOCUMENT = "document"


@dataclass(frozen=True)
class ResourceCleanupIntent:
    id: str
    tenant_id: str
    dataset_id: str
    resource_type: ResourceCleanupType
    resource_id: str
    doc_form: str | None
    source_file_id: str | None
    indexing_technique: str | None
    index_struct: str | None
    collection_binding_id: str | None
    pipeline_id: str | None
    claim_id: str | None = None
    vector_route: VectorRoute | None = None


class ResourceCleanupClaimLostError(RuntimeError):
    """The cleanup worker no longer owns the persisted lease."""


class DatasetResourceRelationCleanup(Protocol):
    def delete_resource_relations(
        self,
        *,
        tenant_id: str,
        resource_type: ResourceAccessTokenResourceType,
        resource_id: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class DatasetDeletionResult:
    cleanup_intent_ids: tuple[str, ...]
    deleted_api_keys: tuple[DeletedDatasetApiKey, ...]


def _database_now(session: Session) -> datetime:
    value = session.scalar(select(func.current_timestamp(type_=DateTime())))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value.replace(tzinfo=None, microsecond=0)


def _claim_available(database_now: datetime):
    return or_(
        ResourceCleanupOutbox.claim_id.is_(None),
        ResourceCleanupOutbox.claim_expires_at.is_(None),
        ResourceCleanupOutbox.claim_expires_at <= database_now,
    )


def _validate_dataset_vector_route(route: VectorRoute, *, tenant_id: str, dataset_id: str) -> None:
    if route.tenant_id != tenant_id or route.dataset_id != dataset_id:
        raise ValueError("Dataset vector cleanup route belongs to another owner")


def _representation_routes(representation: DatasetIndexingRepresentation) -> tuple[VectorRoute, ...]:
    return (
        *representation.cleanup_routes,
        *((representation.vector_route,) if representation.vector_route is not None else ()),
    )


def _collect_dataset_vector_routes(
    dataset: Dataset,
    pending_indexing: Sequence[DatasetIndexingOutbox],
) -> tuple[VectorRoute, ...]:
    """Collect every known physical destination while Dataset and receipts are locked."""
    routes: dict[str, VectorRoute] = {}
    serialized_representations: list[str] = []
    if dataset.indexing_applied_representation is not None:
        serialized_representations.append(dataset.indexing_applied_representation)
    for row in pending_indexing:
        serialized_representations.extend((row.source_representation, row.target_representation))
    for serialized in serialized_representations:
        representation = DatasetIndexingRepresentation.model_validate_json(serialized)
        for route in _representation_routes(representation):
            _validate_dataset_vector_route(route, tenant_id=dataset.tenant_id, dataset_id=dataset.id)
            routes.setdefault(route.key, route)
    return tuple(sorted(routes.values(), key=lambda route: (route.vector_type, route.collection_name, route.key)))


def _create_dataset_vector_cleanup_intents(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    routes: Sequence[VectorRoute],
) -> tuple[str, ...]:
    rows: list[ResourceCleanupOutbox] = []
    for route in routes:
        _validate_dataset_vector_route(route, tenant_id=tenant_id, dataset_id=dataset_id)
        rows.append(
            ResourceCleanupOutbox(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                resource_type=ResourceCleanupType.DATASET_VECTOR.value,
                resource_id=str(uuid5(NAMESPACE_URL, f"dify:dataset-vector:{tenant_id}:{dataset_id}:{route.key}")),
                doc_form=route.doc_form,
                indexing_technique=route.indexing_technique,
                index_struct=route.index_struct,
                collection_binding_id=route.collection_binding_id,
            )
        )
    session.add_all(rows)
    session.flush()
    return tuple(row.id for row in rows)


def _vector_route_from_cleanup_row(row: ResourceCleanupOutbox) -> VectorRoute | None:
    if row.resource_type != ResourceCleanupType.DATASET_VECTOR.value:
        return None
    if row.index_struct is None or row.indexing_technique is None or row.doc_form is None:
        raise ValueError("Dataset vector cleanup intent is missing persisted route evidence")
    try:
        vector_type, collection_name = vector_route_target_from_index_struct(row.index_struct)
    except ValueError as error:
        raise ValueError("Dataset vector cleanup intent has invalid persisted route evidence") from error
    return VectorRoute(
        tenant_id=row.tenant_id,
        dataset_id=row.dataset_id,
        vector_type=vector_type,
        collection_name=collection_name,
        index_struct=row.index_struct,
        collection_binding_id=row.collection_binding_id,
        indexing_technique=row.indexing_technique,
        doc_form=row.doc_form,
    )


def create_dataset_cleanup_intent(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    doc_form: str | None,
    indexing_technique: str | None,
    index_struct: str | None,
    collection_binding_id: str | None,
    pipeline_id: str | None,
    resource_type: ResourceCleanupType = ResourceCleanupType.DATASET,
) -> str:
    """Persist dataset cleanup ownership in the caller's deletion transaction."""
    if resource_type not in (ResourceCleanupType.DATASET, ResourceCleanupType.DATASET_RELATIONAL):
        raise ValueError("Dataset parent cleanup requires a parent resource type")
    row = ResourceCleanupOutbox(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        resource_type=resource_type.value,
        resource_id=dataset_id,
        doc_form=doc_form,
        indexing_technique=indexing_technique,
        index_struct=index_struct,
        collection_binding_id=collection_binding_id,
        pipeline_id=pipeline_id,
    )
    session.add(row)
    session.flush()
    return row.id


def delete_dataset_for_cleanup(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    resource_relations: DatasetResourceRelationCleanup,
) -> DatasetDeletionResult | None:
    """Delete one owned Dataset and persist all relational cleanup atomically.

    Authorization is intentionally outside this command. Both member-driven
    deletion and workspace retirement enter here after their own authorization
    checks, and the caller remains the sole transaction/commit owner.
    """
    if not dataset_api_key_bindings.lock_workspace(session, tenant_id):
        return None
    dataset = session.scalar(
        select(Dataset)
        .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if dataset is None:
        return None
    pending_indexing = session.scalars(
        select(DatasetIndexingOutbox)
        .where(
            DatasetIndexingOutbox.tenant_id == tenant_id,
            DatasetIndexingOutbox.dataset_id == dataset.id,
        )
        .order_by(DatasetIndexingOutbox.id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).all()
    routes = _collect_dataset_vector_routes(dataset, pending_indexing)
    doc_form = get_dataset_doc_form(dataset, session=session)
    # A legacy Dataset without an applied receipt has no exact physical proof.
    # Keep its old logical snapshot on the parent intent as a retryable fallback;
    # once an applied receipt exists, only exact per-route intents may delete
    # vector collections and the desired Dataset columns are not evidence.
    legacy_fallback = dataset.indexing_applied_representation is None and not routes
    parent_intent_id = create_dataset_cleanup_intent(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        doc_form=doc_form,
        indexing_technique=dataset.indexing_technique if legacy_fallback else None,
        index_struct=dataset.index_struct if legacy_fallback else None,
        collection_binding_id=dataset.collection_binding_id if legacy_fallback else None,
        pipeline_id=dataset.pipeline_id,
        resource_type=(ResourceCleanupType.DATASET if legacy_fallback else ResourceCleanupType.DATASET_RELATIONAL),
    )
    vector_intent_ids = _create_dataset_vector_cleanup_intents(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        routes=routes,
    )
    # Exact physical evidence is now durable in this same transaction. Retire
    # the mutable indexing receipts only after the handoff rows were flushed.
    for row in pending_indexing:
        session.delete(row)
    session.flush()
    deleted_keys = tuple(dataset_api_key_bindings.delete_keys_scoped_only_to(session, str(dataset.id)))
    resource_relations.delete_resource_relations(
        tenant_id=tenant_id,
        resource_type=ResourceAccessTokenResourceType.KNOWLEDGE,
        resource_id=dataset.id,
    )
    session.execute(
        delete(DatasetPermission).where(
            DatasetPermission.tenant_id == tenant_id,
            DatasetPermission.dataset_id == dataset.id,
        )
    )
    session.delete(dataset)
    return DatasetDeletionResult((parent_intent_id, *vector_intent_ids), deleted_keys)


def create_document_cleanup_intent(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    doc_form: str | None,
    source_file_id: str | None,
) -> str:
    """Persist document cleanup ownership in the caller's deletion transaction."""
    row = ResourceCleanupOutbox(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        resource_type=ResourceCleanupType.DOCUMENT.value,
        resource_id=document_id,
        doc_form=doc_form,
        source_file_id=source_file_id,
    )
    session.add(row)
    session.flush()
    return row.id


class SQLAlchemyResourceCleanupRepository:
    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def list_pending_intent_ids(self, *, limit: int) -> tuple[str, ...]:
        """List claimable work while excluding cleanup with a live worker lease."""
        if limit <= 0:
            return ()
        with self._session_factory() as session:
            database_now = _database_now(session)
            return tuple(
                session.scalars(
                    select(ResourceCleanupOutbox.id)
                    .where(_claim_available(database_now))
                    .order_by(
                        case((ResourceCleanupOutbox.last_attempted_at.is_(None), 0), else_=1),
                        ResourceCleanupOutbox.last_attempted_at,
                        ResourceCleanupOutbox.created_at,
                        ResourceCleanupOutbox.id,
                    )
                    .limit(limit)
                )
            )

    def claim(self, intent_id: str) -> ResourceCleanupIntent | None:
        """Atomically lease one available intent to exactly one worker."""
        claim_id = str(uuid4())
        with self._session_factory.begin() as session:
            database_now = _database_now(session)
            cursor = session.execute(
                update(ResourceCleanupOutbox)
                .where(ResourceCleanupOutbox.id == intent_id, _claim_available(database_now))
                .values(
                    claim_id=claim_id,
                    claim_expires_at=database_now + timedelta(seconds=RESOURCE_CLEANUP_LEASE_SECONDS),
                    last_attempted_at=database_now,
                )
            )
            if cast(CursorResult[object], cursor).rowcount != 1:
                return None
            row = session.scalar(
                select(ResourceCleanupOutbox).where(
                    ResourceCleanupOutbox.id == intent_id,
                    ResourceCleanupOutbox.claim_id == claim_id,
                )
            )
            return self._to_intent(row) if row is not None else None

    def mark_attempted(self, intent_id: str) -> None:
        """Rotate a broker publication attempt without changing worker ownership."""
        with self._session_factory.begin() as session:
            session.execute(
                update(ResourceCleanupOutbox)
                .where(ResourceCleanupOutbox.id == intent_id)
                .values(last_attempted_at=func.current_timestamp())
            )

    def renew_claim(self, intent_id: str, *, claim_id: str) -> None:
        """Extend an exact live lease without reviving an expired worker."""
        with self._session_factory.begin() as session:
            with database_operation_timeout(
                session,
                timeout_seconds=_CLAIM_RENEWAL_DATABASE_TIMEOUT_SECONDS,
            ):
                database_now = _database_now(session)
                cursor = session.execute(
                    update(ResourceCleanupOutbox)
                    .where(
                        ResourceCleanupOutbox.id == intent_id,
                        ResourceCleanupOutbox.claim_id == claim_id,
                        ResourceCleanupOutbox.claim_expires_at > database_now,
                    )
                    .values(claim_expires_at=database_now + timedelta(seconds=RESOURCE_CLEANUP_LEASE_SECONDS))
                )
                if cast(CursorResult[object], cursor).rowcount != 1:
                    raise ResourceCleanupClaimLostError()

    def release_claim(self, intent_id: str, *, claim_id: str) -> bool:
        """Release only the caller's exact claim so another worker can retry."""
        with self._session_factory.begin() as session:
            cursor = session.execute(
                update(ResourceCleanupOutbox)
                .where(
                    ResourceCleanupOutbox.id == intent_id,
                    ResourceCleanupOutbox.claim_id == claim_id,
                )
                .values(claim_id=None, claim_expires_at=None)
            )
            return cast(CursorResult[object], cursor).rowcount == 1

    def get_intent(self, intent_id: str) -> ResourceCleanupIntent | None:
        with self._session_factory() as session:
            row = session.get(ResourceCleanupOutbox, intent_id)
            return self._to_intent(row) if row is not None else None

    def source_exists(self, intent: ResourceCleanupIntent) -> bool:
        """Reject cleanup while the resource deletion has not committed."""
        with self._session_factory() as session:
            if intent.resource_type in (
                ResourceCleanupType.DATASET,
                ResourceCleanupType.DATASET_RELATIONAL,
                ResourceCleanupType.DATASET_VECTOR,
            ):
                return (
                    session.scalar(
                        select(Dataset.id).where(
                            Dataset.id == intent.dataset_id,
                            Dataset.tenant_id == intent.tenant_id,
                        )
                    )
                    is not None
                )
            return (
                session.scalar(
                    select(Document.id).where(
                        Document.id == intent.resource_id,
                        Document.dataset_id == intent.dataset_id,
                        Document.tenant_id == intent.tenant_id,
                    )
                )
                is not None
            )

    def has_unfinished_summary_writers(self, intent: ResourceCleanupIntent) -> bool:
        """Snapshot external writes that may recreate a deleted dataset's vectors.

        Summary registration locks the live Dataset row before recording a
        writer, so no new writer can register after dataset deletion commits.
        Exact-route cleanup waits only for writers on its physical route. The
        relational parent retains unknown-route writers until their own durable
        cleanup evidence can identify what a late write may have published.
        Legacy dataset intents conservatively wait for every writer because
        their logical snapshot is the only collection-deletion evidence.
        """
        if intent.resource_type not in (
            ResourceCleanupType.DATASET,
            ResourceCleanupType.DATASET_RELATIONAL,
            ResourceCleanupType.DATASET_VECTOR,
        ):
            return False
        with self._session_factory() as session:
            conditions = [
                SummaryVectorCleanupOutbox.tenant_id == intent.tenant_id,
                SummaryVectorCleanupOutbox.dataset_id == intent.dataset_id,
                SummaryVectorCleanupOutbox.writer_finished_at.is_(None),
            ]
            if intent.resource_type == ResourceCleanupType.DATASET_VECTOR:
                if intent.vector_route is None:
                    raise ValueError("Dataset vector cleanup intent has no route")
                conditions.append(SummaryVectorCleanupOutbox.vector_route_hash == intent.vector_route.key)
            elif intent.resource_type == ResourceCleanupType.DATASET_RELATIONAL:
                conditions.append(SummaryVectorCleanupOutbox.vector_route_hash.is_(None))
            return session.scalar(select(SummaryVectorCleanupOutbox.id).where(*conditions).limit(1)) is not None

    def has_unfinished_child_writers(self, intent: ResourceCleanupIntent) -> bool:
        """Snapshot child-vector calls that may finish after collection deletion."""
        if intent.resource_type not in (
            ResourceCleanupType.DATASET,
            ResourceCleanupType.DATASET_RELATIONAL,
            ResourceCleanupType.DATASET_VECTOR,
        ):
            return False
        with self._session_factory() as session:
            conditions = [
                ChildVectorCleanupOutbox.tenant_id == intent.tenant_id,
                ChildVectorCleanupOutbox.dataset_id == intent.dataset_id,
                ChildVectorCleanupOutbox.writer_finished_at.is_(None),
            ]
            if intent.resource_type == ResourceCleanupType.DATASET_VECTOR:
                if intent.vector_route is None:
                    raise ValueError("Dataset vector cleanup intent has no route")
                conditions.append(ChildVectorCleanupOutbox.vector_route_hash == intent.vector_route.key)
            elif intent.resource_type == ResourceCleanupType.DATASET_RELATIONAL:
                conditions.append(ChildVectorCleanupOutbox.vector_route_hash.is_(None))
            return session.scalar(select(ChildVectorCleanupOutbox.id).where(*conditions).limit(1)) is not None

    def has_unfinished_attachment_writers(self, intent: ResourceCleanupIntent) -> bool:
        """Snapshot attachment calls that may recreate a deleted collection."""
        if intent.resource_type not in (
            ResourceCleanupType.DATASET,
            ResourceCleanupType.DATASET_RELATIONAL,
            ResourceCleanupType.DATASET_VECTOR,
        ):
            return False
        with self._session_factory() as session:
            conditions = [
                AttachmentVectorCleanupOutbox.tenant_id == intent.tenant_id,
                AttachmentVectorCleanupOutbox.dataset_id == intent.dataset_id,
                AttachmentVectorCleanupOutbox.writer_finished_at.is_(None),
            ]
            if intent.resource_type == ResourceCleanupType.DATASET_VECTOR:
                if intent.vector_route is None:
                    raise ValueError("Dataset vector cleanup intent has no route")
                conditions.append(AttachmentVectorCleanupOutbox.vector_route_hash == intent.vector_route.key)
            elif intent.resource_type == ResourceCleanupType.DATASET_RELATIONAL:
                # New attachment writers always capture a physical route. The
                # paired vector intent owns them; relational deletion must not
                # guess a route from already-removed Dataset configuration.
                return False
            return session.scalar(select(AttachmentVectorCleanupOutbox.id).where(*conditions).limit(1)) is not None

    def complete(
        self,
        intent: ResourceCleanupIntent,
        *,
        claim_id: str,
        deleted_child_route: ChildVectorRoute | None = None,
    ) -> None:
        """Acknowledge the exact live cleanup and terminal summary writers.

        Terminal summary and child rows are acknowledged only when their
        immutable physical route matches the collection this worker actually
        deleted. Older routes remain independently cleanable after the Dataset
        row and parent intent disappear. Unfinished writers remain durable so
        a late vector write forces another parent cleanup pass.
        """
        if intent.resource_type == ResourceCleanupType.DATASET_VECTOR:
            if intent.vector_route is None or deleted_child_route is None:
                raise ValueError("Dataset vector cleanup requires exact deleted route evidence")
            if deleted_child_route != intent.vector_route:
                raise ValueError("Deleted vector route does not match the cleanup intent")
        with self._session_factory.begin() as session:
            database_now = _database_now(session)
            cursor = session.execute(
                delete(ResourceCleanupOutbox).where(
                    ResourceCleanupOutbox.id == intent.id,
                    ResourceCleanupOutbox.tenant_id == intent.tenant_id,
                    ResourceCleanupOutbox.dataset_id == intent.dataset_id,
                    ResourceCleanupOutbox.resource_type == intent.resource_type.value,
                    ResourceCleanupOutbox.resource_id == intent.resource_id,
                    ResourceCleanupOutbox.claim_id == claim_id,
                    ResourceCleanupOutbox.claim_expires_at > database_now,
                )
            )
            if cast(CursorResult[object], cursor).rowcount != 1:
                raise ResourceCleanupClaimLostError()
            if intent.resource_type in (
                ResourceCleanupType.DATASET,
                ResourceCleanupType.DATASET_RELATIONAL,
                ResourceCleanupType.DATASET_VECTOR,
            ):
                if deleted_child_route is not None:
                    if (
                        deleted_child_route.tenant_id != intent.tenant_id
                        or deleted_child_route.dataset_id != intent.dataset_id
                    ):
                        raise ValueError("Deleted child-vector route does not belong to the cleanup intent")
                    session.execute(
                        delete(SummaryVectorCleanupOutbox).where(
                            SummaryVectorCleanupOutbox.tenant_id == intent.tenant_id,
                            SummaryVectorCleanupOutbox.dataset_id == intent.dataset_id,
                            SummaryVectorCleanupOutbox.vector_route_hash == deleted_child_route.key,
                            SummaryVectorCleanupOutbox.writer_finished_at.is_not(None),
                        )
                    )
                    session.execute(
                        delete(ChildVectorCleanupOutbox).where(
                            ChildVectorCleanupOutbox.tenant_id == intent.tenant_id,
                            ChildVectorCleanupOutbox.dataset_id == intent.dataset_id,
                            ChildVectorCleanupOutbox.vector_route_hash == deleted_child_route.key,
                            ChildVectorCleanupOutbox.writer_finished_at.is_not(None),
                        )
                    )
                    session.execute(
                        delete(AttachmentVectorCleanupOutbox).where(
                            AttachmentVectorCleanupOutbox.tenant_id == intent.tenant_id,
                            AttachmentVectorCleanupOutbox.dataset_id == intent.dataset_id,
                            AttachmentVectorCleanupOutbox.vector_route_hash == deleted_child_route.key,
                            AttachmentVectorCleanupOutbox.writer_finished_at.is_not(None),
                        )
                    )

    @staticmethod
    def _to_intent(row: ResourceCleanupOutbox) -> ResourceCleanupIntent:
        return ResourceCleanupIntent(
            id=row.id,
            tenant_id=row.tenant_id,
            dataset_id=row.dataset_id,
            resource_type=ResourceCleanupType(row.resource_type),
            resource_id=row.resource_id,
            doc_form=row.doc_form,
            source_file_id=row.source_file_id,
            indexing_technique=row.indexing_technique,
            index_struct=row.index_struct,
            collection_binding_id=row.collection_binding_id,
            pipeline_id=row.pipeline_id,
            claim_id=row.claim_id,
            vector_route=_vector_route_from_cleanup_row(row),
        )


def create_document_cleanup_intents(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    doc_form: str | None,
    source_file_ids: Mapping[str, str | None],
) -> tuple[str, ...]:
    """Persist one independently retryable intent for every deleted document."""
    rows = tuple(
        ResourceCleanupOutbox(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            resource_type=ResourceCleanupType.DOCUMENT.value,
            resource_id=document_id,
            doc_form=doc_form,
            source_file_id=source_file_ids.get(document_id),
        )
        for document_id in document_ids
    )
    session.add_all(rows)
    session.flush()
    return tuple(row.id for row in rows)

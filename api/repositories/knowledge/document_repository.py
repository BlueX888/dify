"""SQLAlchemy repository for tenant-owned document state."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import DateTime, delete, exists, func, select, update
from sqlalchemy.orm import Session, sessionmaker

from core.rag.entities import MetadataFilteringCondition
from libs.datetime_utils import naive_utc_now
from libs.sqlalchemy_utils import database_operation_timeout
from models import Account, DocumentIndexingOutbox, Tenant
from models.dataset import Dataset, DatasetProcessRule, Document, Pipeline
from models.enums import IndexingStatus
from repositories.knowledge.dataset_indexing_outbox_repository import lock_dataset_indexing_generation_shared
from repositories.knowledge.dataset_read_repository import get_pipeline_dataset, pipeline_dataset_query
from repositories.knowledge.metadata_filter import query_metadata_filter_document_ids
from services.knowledge.document_sync import NOTION_SYNC_LEASE_SECONDS, ClaimedNotionDocument, SyncDocumentRecord
from services.knowledge.entities.datasets import DatasetIndexingGenerationLostError
from services.knowledge.entities.hit_testing import RetrievalDocumentRecord
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef
from services.knowledge.indexing.errors import (
    DocumentIndexingClaimLostError,
    DocumentIsDeletedPausedError,
    DocumentIsPausedError,
)
from services.knowledge.indexing.estimate import EstimateDocumentRecord, StoredSource
from services.knowledge.indexing.execution import IndexingDocument

_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_CLAIM_RENEWAL_DATABASE_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class ClaimedIndexingRequest:
    ref: DocumentRef
    claim_id: str
    recovery_required: bool
    resume_status: str | None = None
    dataset_generation_id: str | None = None


def _database_now(session: Session) -> datetime:
    """Return the database clock in the precision persisted by Document leases."""
    bind = session.get_bind()
    timestamp = (
        func.clock_timestamp(type_=DateTime()) if bind.dialect.name == "postgresql" else func.current_timestamp()
    )
    value = session.scalar(select(timestamp))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value.replace(tzinfo=None, microsecond=0)


def has_active_indexing_claim(session: Session, document: Document) -> bool:
    """Check a locked document claim against the database clock."""
    if document.indexing_claim_id is None or document.indexing_claim_expires_at is None:
        return False
    return document.indexing_claim_expires_at > _database_now(session)


def _mapping(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return _MAPPING_ADAPTER.validate_json(value)
        return _MAPPING_ADAPTER.validate_python(value)
    except (TypeError, ValueError, ValidationError):
        return {}


def _estimate_document_record(document: Document) -> EstimateDocumentRecord:
    return EstimateDocumentRecord(
        id=document.id,
        workspace_id=document.tenant_id,
        dataset_id=document.dataset_id,
        data_source_type=str(document.data_source_type),
        data_source_info=_mapping(document.data_source_info),
        indexing_status=str(document.indexing_status),
        doc_form=str(document.doc_form),
        doc_language=document.doc_language,
        dataset_process_rule_id=document.dataset_process_rule_id,
    )


def _get_document(session: Session, document_ref: DocumentRef) -> Document | None:
    """Shared SQL for implementation adapters that already own a transaction."""
    return session.scalar(
        select(Document)
        .where(
            Document.id == document_ref.document_id,
            Document.dataset_id == document_ref.dataset.dataset_id,
            Document.tenant_id == document_ref.dataset.tenant_id,
        )
        .limit(1)
    )


def require_indexing_document(
    session: Session,
    ref: DocumentRef,
    *,
    claim_id: str | None = None,
    lock: bool = False,
    check_paused: bool = True,
) -> Document:
    """Validate the persisted owner chain before each indexing phase."""
    statement = (
        select(Document)
        .join(Dataset, Dataset.id == Document.dataset_id)
        .where(
            Document.id == ref.document_id,
            Document.dataset_id == ref.dataset.dataset_id,
            Document.tenant_id == ref.dataset.tenant_id,
            Dataset.tenant_id == ref.dataset.tenant_id,
        )
    )
    if lock:
        statement = statement.with_for_update(of=Document).execution_options(populate_existing=True)
    document = session.scalar(statement)
    if document is None:
        raise DocumentIsDeletedPausedError()
    if document.indexing_claim_id != claim_id:
        raise DocumentIndexingClaimLostError()
    if claim_id is None and document.indexing_request_id is not None:
        # Legacy workers do not own durable generations. Once any request is
        # installed, an unclaimed late write must not mutate or acknowledge it.
        raise DocumentIndexingClaimLostError()
    if claim_id is not None and not has_active_indexing_claim(session, document):
        raise DocumentIndexingClaimLostError()
    if check_paused and document.is_paused:
        raise DocumentIsPausedError()
    return document


def renew_indexing_claim(session: Session, ref: DocumentRef, *, claim_id: str) -> None:
    """Renew an exact active owner within the caller's short transaction."""
    document = require_indexing_document(
        session,
        ref,
        claim_id=claim_id,
        lock=True,
        check_paused=False,
    )
    document.indexing_claim_expires_at = _database_now(session) + timedelta(seconds=NOTION_SYNC_LEASE_SECONDS)


def refresh_locked_indexing_claim(
    session: Session,
    ref: DocumentRef,
    *,
    indexing_request_id: str | None,
    claim_id: str,
    allowed_statuses: Sequence[IndexingStatus],
) -> None:
    """Refresh an exact claim before releasing a lock held across external I/O.

    The row lock continuously excludes takeover, so this deliberately validates
    identity without rejecting a lease whose wall-clock expiry passed while the
    lock was held. The caller must name every phase in which its mutation may
    finish, preventing this exception from reviving an owner after a different
    state transition. Callers must acquire the same row lock with an active-
    lease check before starting the external operation.
    """
    statuses = tuple(dict.fromkeys(allowed_statuses))
    if not statuses:
        raise ValueError("at least one indexing status is required")
    document = session.scalar(
        select(Document)
        .where(
            Document.id == ref.document_id,
            Document.dataset_id == ref.dataset.dataset_id,
            Document.tenant_id == ref.dataset.tenant_id,
            Document.indexing_request_id == indexing_request_id,
            Document.indexing_claim_id == claim_id,
            Document.indexing_status.in_(statuses),
        )
        .with_for_update(of=Document)
        .execution_options(populate_existing=True)
    )
    if document is None:
        raise DocumentIndexingClaimLostError()
    document.indexing_claim_expires_at = _database_now(session) + timedelta(seconds=NOTION_SYNC_LEASE_SECONDS)


def transition_dataset_generation_owner_to_restart(
    document: Document,
    *,
    restarted_at: datetime,
) -> None:
    """Convert a fenced parent-generation owner into an explicit full-restart state.

    The caller must hold the Document row lock and must have validated the
    exact request, parent generation, and claim (or hold the parent Dataset X
    lock while invalidating stale owners). A durable request, when present,
    remains current while every worker token and phase-completion marker is
    removed. A non-null ``processing_started_at`` distinguishes this handoff
    from a first-time WAITING document so either a durable request worker or a
    requestless Notion sync cannot reuse partial segments.
    """
    document.indexing_status = IndexingStatus.WAITING
    document.indexing_claim_id = None
    document.indexing_claim_expires_at = None
    document.indexing_dataset_generation_id = None
    document.processing_started_at = document.processing_started_at or restarted_at
    document.parsing_completed_at = None
    document.cleaning_completed_at = None
    document.splitting_completed_at = None
    document.completed_at = None
    document.stopped_at = None
    document.error = None


class SQLAlchemyDocumentRepository:
    """Own SQL access to documents through their complete owner chain."""

    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def renew_indexing_document(self, document: IndexingDocument) -> None:
        """Fence imminent external writes by renewing the exact persisted owner."""
        if document.claim_id is None:
            return
        self.renew_indexing_claim(document.ref, claim_id=document.claim_id)

    def renew_indexing_claim(self, ref: DocumentRef, *, claim_id: str) -> None:
        """Renew an active claim without requiring the indexing payload."""
        with self._session_factory.begin() as session:
            with database_operation_timeout(
                session,
                timeout_seconds=_CLAIM_RENEWAL_DATABASE_TIMEOUT_SECONDS,
            ):
                renew_indexing_claim(session, ref, claim_id=claim_id)

    def release_dataset_generation_owner(
        self,
        ref: DocumentRef,
        *,
        indexing_request_id: str | None,
        claim_id: str,
        dataset_generation_id: str | None,
    ) -> None:
        """Fence an exact superseded owner and leave its request restartable.

        Postcondition: the request id is unchanged, status is WAITING with a
        restart marker, phase/terminal results are cleared, and no claim or
        captured parent generation remains. A stale caller cannot release a
        successor because all three identity components are matched under the
        Dataset-before-Document lock order.
        """
        with self._session_factory.begin() as session:
            dataset = session.scalar(
                select(Dataset)
                .where(Dataset.id == ref.dataset.dataset_id, Dataset.tenant_id == ref.dataset.tenant_id)
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if dataset is None:
                raise DocumentIsDeletedPausedError()
            request_condition = (
                Document.indexing_request_id == indexing_request_id
                if indexing_request_id is not None
                else Document.indexing_request_id.is_(None)
            )
            generation_condition = (
                Document.indexing_dataset_generation_id == dataset_generation_id
                if dataset_generation_id is not None
                else Document.indexing_dataset_generation_id.is_(None)
            )
            document = session.scalar(
                select(Document)
                .where(
                    Document.id == ref.document_id,
                    Document.dataset_id == ref.dataset.dataset_id,
                    Document.tenant_id == ref.dataset.tenant_id,
                    request_condition,
                    Document.indexing_claim_id == claim_id,
                    generation_condition,
                )
                .with_for_update(of=Document)
                .execution_options(populate_existing=True)
            )
            if document is None or dataset.indexing_request_id == dataset_generation_id:
                raise DocumentIndexingClaimLostError()
            transition_dataset_generation_owner_to_restart(
                document,
                restarted_at=_database_now(session),
            )

    def indexing_request_has_unpaused_documents(
        self,
        *,
        request_id: str,
        tenant_id: str,
        dataset_id: str,
    ) -> bool:
        """Return whether a current request has work that a worker may claim now."""
        with self._session_factory() as session:
            return (
                session.scalar(
                    select(Document.id)
                    .where(
                        Document.indexing_request_id == request_id,
                        Document.tenant_id == tenant_id,
                        Document.dataset_id == dataset_id,
                        Document.is_paused.is_not(True),
                    )
                    .limit(1)
                )
                is not None
            )

    def get_metadata_filter_document_ids(
        self,
        *,
        tenant_id: str,
        dataset_ids: Sequence[str],
        conditions: MetadataFilteringCondition | None,
    ) -> dict[str, list[str]]:
        model = conditions or MetadataFilteringCondition()
        with self._session_factory() as session:
            return query_metadata_filter_document_ids(
                session,
                tenant_id=tenant_id,
                dataset_ids=dataset_ids,
                logical_operator=model.logical_operator,
                conditions=model.conditions,
            )

    def get_hit_testing_document_filter(self, ref: DatasetRef, conditions: dict[str, object]) -> list[str]:
        model = MetadataFilteringCondition.model_validate(conditions)
        return self.get_metadata_filter_document_ids(
            tenant_id=ref.tenant_id,
            dataset_ids=(ref.dataset_id,),
            conditions=model,
        ).get(ref.dataset_id, [])

    def get_retrieval_documents(
        self, ref: DatasetRef, document_ids: Sequence[str]
    ) -> dict[str, RetrievalDocumentRecord]:
        if not document_ids:
            return {}
        with self._session_factory() as session:
            documents = session.scalars(
                select(Document)
                .join(Dataset, Dataset.id == Document.dataset_id)
                .where(
                    Document.id.in_(document_ids),
                    Document.dataset_id == ref.dataset_id,
                    Document.tenant_id == ref.tenant_id,
                    Dataset.tenant_id == ref.tenant_id,
                    Document.enabled.is_(True),
                    Document.archived.is_(False),
                    Document.indexing_status == "completed",
                )
            )
            return {
                document.id: RetrievalDocumentRecord(
                    document.id,
                    str(document.doc_form),
                    {
                        "id": document.id,
                        "data_source_type": document.data_source_type,
                        "name": document.name,
                        "doc_type": document.doc_type,
                        "doc_metadata": document.doc_metadata,
                    },
                )
                for document in documents
            }

    def get_pipeline_dataset(self, pipeline: Pipeline, *, session: Session, lock: bool) -> Dataset | None:
        """Read within the pipeline caller's transaction, including uncommitted changes."""
        if lock:
            return session.scalar(
                pipeline_dataset_query(pipeline.tenant_id, pipeline.id)
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
        return get_pipeline_dataset(pipeline, session=session)

    def get_indexing_document(self, ref: DocumentRef, *, claim_id: str | None = None) -> IndexingDocument:
        with self._session_factory() as session:
            document = require_indexing_document(session, ref, claim_id=claim_id)
            rule = session.scalar(
                select(DatasetProcessRule).where(
                    DatasetProcessRule.id == document.dataset_process_rule_id,
                    DatasetProcessRule.dataset_id == ref.dataset.dataset_id,
                )
            )
            return IndexingDocument(
                source=StoredSource.from_document(_estimate_document_record(document)),
                processing_rule=rule.to_dict() if rule else {},
                doc_language=document.doc_language or "English",
                need_summary=bool(document.need_summary),
                claim_id=claim_id,
                indexing_request_id=document.indexing_request_id,
                dataset_generation_id=document.indexing_dataset_generation_id,
            )

    def get_indexing_models(self, ref: DocumentRef, *, claim_id: str | None = None) -> tuple[Dataset, Document]:
        """Detached legacy processor inputs; never pass them to another thread."""
        with self._session_factory() as session:
            document = require_indexing_document(session, ref, claim_id=claim_id)
            dataset = session.scalar(
                select(Dataset).where(
                    Dataset.id == ref.dataset.dataset_id,
                    Dataset.tenant_id == ref.dataset.tenant_id,
                )
            )
            assert dataset is not None
            return dataset, document

    def get_indexing_user(self, ref: DocumentRef, *, claim_id: str | None = None) -> Account:
        with self._session_factory() as session:
            document = require_indexing_document(session, ref, claim_id=claim_id)
            account = session.get(Account, document.created_by)
            if account is None:
                raise ValueError("no current user found")
            account.set_tenant_id_with_session(ref.dataset.tenant_id, session=session)
            return account

    def claim_waiting_for_indexing(
        self,
        dataset_ref: DatasetRef,
        document_ids: Sequence[str],
        *,
        processing_started_at: datetime,
        admission_error: Callable[[int], str | None],
    ) -> tuple[DocumentRef, ...]:
        """Lock eligible documents, apply admission, and transition them in one transaction."""
        ordered_ids = tuple(dict.fromkeys(document_ids))
        if not ordered_ids:
            return ()
        owner_chain_exists = exists(
            select(Dataset.id)
            .join(Tenant, Tenant.id == Dataset.tenant_id)
            .where(
                Tenant.id == dataset_ref.tenant_id,
                Dataset.id == dataset_ref.dataset_id,
                Dataset.tenant_id == dataset_ref.tenant_id,
            )
        )
        with self._session_factory.begin() as session:
            eligible_ids = tuple(
                session.scalars(
                    select(Document.id)
                    .where(
                        owner_chain_exists,
                        Document.id.in_(ordered_ids),
                        Document.tenant_id == dataset_ref.tenant_id,
                        Document.dataset_id == dataset_ref.dataset_id,
                        Document.indexing_status == IndexingStatus.WAITING,
                        Document.indexing_request_id.is_(None),
                        Document.indexing_claim_id.is_(None),
                        Document.is_paused.is_not(True),
                    )
                    .order_by(Document.id)
                    .with_for_update()
                ).all()
            )
            if not eligible_ids:
                return ()

            error = admission_error(len(eligible_ids))
            if error is not None:
                session.execute(
                    update(Document)
                    .where(
                        Document.id.in_(eligible_ids),
                        Document.tenant_id == dataset_ref.tenant_id,
                        Document.dataset_id == dataset_ref.dataset_id,
                        Document.indexing_status == IndexingStatus.WAITING,
                        Document.indexing_request_id.is_(None),
                        Document.indexing_claim_id.is_(None),
                    )
                    .values(
                        indexing_status=IndexingStatus.ERROR,
                        error=error,
                        stopped_at=processing_started_at,
                    )
                )
                return ()

            session.execute(
                update(Document)
                .where(
                    Document.id.in_(eligible_ids),
                    Document.tenant_id == dataset_ref.tenant_id,
                    Document.dataset_id == dataset_ref.dataset_id,
                    Document.indexing_status == IndexingStatus.WAITING,
                    Document.indexing_request_id.is_(None),
                    Document.indexing_claim_id.is_(None),
                )
                .values(
                    indexing_status=IndexingStatus.PARSING,
                    processing_started_at=processing_started_at,
                )
            )
        eligible_id_set = set(eligible_ids)
        return tuple(dataset_ref.document(document_id) for document_id in ordered_ids if document_id in eligible_id_set)

    def claim_indexing_request(
        self,
        request_id: str,
        *,
        expected_action: str,
        processing_started_at: datetime,
        admission_error: Callable[[int], str | None],
    ) -> tuple[ClaimedIndexingRequest, ...]:
        """Claim one exact durable request document.

        A first-time WAITING row returns ``recovery_required=False``. WAITING
        rows carrying a prior ``processing_started_at`` marker, expired owners,
        and recoverable non-WAITING phases return ``recovery_required=True`` so
        callers must clean and reparse instead of resuming partial output.
        """
        with self._session_factory.begin() as session:
            request = session.scalar(
                select(DocumentIndexingOutbox).where(
                    DocumentIndexingOutbox.id == request_id,
                    DocumentIndexingOutbox.action == expected_action,
                )
            )
            if request is None:
                return ()
            dataset = session.scalar(
                select(Dataset)
                .where(Dataset.id == request.dataset_id, Dataset.tenant_id == request.tenant_id)
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if dataset is None:
                return ()
            dataset_generation_id = dataset.indexing_request_id
            requested_ids = tuple(dict.fromkeys(json.loads(request.document_ids)))
            documents = list(
                session.scalars(
                    select(Document)
                    .where(
                        Document.id.in_(requested_ids),
                        Document.tenant_id == request.tenant_id,
                        Document.dataset_id == request.dataset_id,
                        Document.indexing_request_id == request.id,
                    )
                    .order_by(Document.id)
                    .with_for_update()
                ).all()
            )
            if not documents:
                session.execute(
                    delete(DocumentIndexingOutbox).where(
                        DocumentIndexingOutbox.id == request.id,
                        DocumentIndexingOutbox.tenant_id == request.tenant_id,
                        DocumentIndexingOutbox.dataset_id == request.dataset_id,
                        DocumentIndexingOutbox.action == expected_action,
                    )
                )
                return ()

            database_now = _database_now(session)
            if any(
                document.indexing_claim_id is not None
                and document.indexing_claim_expires_at is not None
                and document.indexing_claim_expires_at > database_now
                for document in documents
            ):
                return ()
            recover_request = expected_action == "recover"
            pipeline_retry_request = expected_action == "retry"
            if recover_request:
                terminal_statuses = {IndexingStatus.COMPLETED}
            elif pipeline_retry_request:
                terminal_statuses = {IndexingStatus.ERROR}
            else:
                terminal_statuses = {IndexingStatus.COMPLETED, IndexingStatus.ERROR}
            terminal_documents = [document for document in documents if document.indexing_status in terminal_statuses]
            for document in terminal_documents:
                document.indexing_request_id = None
                document.indexing_claim_id = None
                document.indexing_claim_expires_at = None
                document.indexing_dataset_generation_id = None
            if terminal_documents:
                session.flush()
                documents = [document for document in documents if document not in terminal_documents]
                if not documents:
                    self._complete_request_if_finished(
                        session,
                        request_id=request.id,
                        tenant_id=request.tenant_id,
                        dataset_id=request.dataset_id,
                    )
                    return ()
            recoverable_statuses = {
                IndexingStatus.WAITING,
                IndexingStatus.PARSING,
                IndexingStatus.CLEANING,
                IndexingStatus.SPLITTING,
                IndexingStatus.INDEXING,
            }
            if recover_request:
                recoverable_statuses.add(IndexingStatus.ERROR)
            if pipeline_retry_request:
                recoverable_statuses.add(IndexingStatus.COMPLETED)
            eligible: list[tuple[Document, bool]] = []
            for document in documents:
                if document.is_paused:
                    continue
                pending_statuses = recoverable_statuses if recover_request else {IndexingStatus.WAITING}
                pending = document.indexing_status in pending_statuses and document.indexing_claim_id is None
                expired = (
                    document.indexing_claim_id is not None
                    and (
                        document.indexing_claim_expires_at is None or document.indexing_claim_expires_at <= database_now
                    )
                    and document.indexing_status in recoverable_statuses
                )
                restart_required = (
                    expired
                    or (
                        document.indexing_status == IndexingStatus.WAITING
                        and document.processing_started_at is not None
                    )
                    or (recover_request and document.indexing_status == IndexingStatus.ERROR)
                )
                if pending or expired:
                    eligible.append((document, restart_required))
            if not eligible:
                return ()

            admission_candidates = [candidate for candidate in eligible if not candidate[1]]
            error = admission_error(len(admission_candidates)) if admission_candidates and not eligible[0][1] else None
            if error is not None:
                for document, recovery_required in eligible:
                    if recovery_required:
                        continue
                    document.indexing_status = IndexingStatus.ERROR
                    document.error = error
                    document.stopped_at = processing_started_at
                    document.indexing_request_id = None
                    document.indexing_claim_id = None
                    document.indexing_claim_expires_at = None
                    document.indexing_dataset_generation_id = None
                session.flush()
                eligible = [candidate for candidate in eligible if candidate[1]]
                if not eligible:
                    self._complete_request_if_finished(
                        session,
                        request_id=request.id,
                        tenant_id=request.tenant_id,
                        dataset_id=request.dataset_id,
                    )
                    return ()

            claims: list[ClaimedIndexingRequest] = []
            # A worker processes request documents serially. Lease only its next
            # document so later documents retain their full execution window.
            for document, recovery_required in eligible[:1]:
                claim_id = str(uuid4())
                resume_status = str(document.indexing_status) if recover_request else None
                if not recover_request:
                    document.indexing_status = IndexingStatus.PARSING
                document.processing_started_at = processing_started_at
                document.indexing_claim_id = claim_id
                document.indexing_claim_expires_at = database_now + timedelta(seconds=NOTION_SYNC_LEASE_SECONDS)
                document.indexing_dataset_generation_id = dataset_generation_id
                claims.append(
                    ClaimedIndexingRequest(
                        ref=DatasetRef(document.tenant_id, document.dataset_id).document(document.id),
                        claim_id=claim_id,
                        recovery_required=recovery_required,
                        resume_status=resume_status,
                        dataset_generation_id=dataset_generation_id,
                    )
                )
            return tuple(claims)

    def mark_splitting(self, ref: DocumentRef, *, claim_id: str | None = None) -> None:
        with self._session_factory.begin() as session:
            document = require_indexing_document(session, ref, claim_id=claim_id, lock=True)
            document.indexing_status = IndexingStatus.SPLITTING
            document.parsing_completed_at = naive_utc_now()

    def mark_cleaning(self, ref: DocumentRef, *, claim_id: str | None = None) -> None:
        """Persist replacement cleanup as a resumable phase for exact owner."""
        with self._session_factory.begin() as session:
            document = require_indexing_document(
                session,
                ref,
                claim_id=claim_id,
                lock=True,
                # Persist the restart phase before the execution adapter
                # observes a pause so recovery knows cleanup is required.
                check_paused=False,
            )
            document.indexing_status = IndexingStatus.CLEANING

    def release_indexing_claim_for_pause(self, ref: DocumentRef, *, claim_id: str) -> bool:
        """Release a paused exact owner while retaining request and phase for resume."""
        try:
            with self._session_factory.begin() as session:
                document = require_indexing_document(
                    session,
                    ref,
                    claim_id=claim_id,
                    lock=True,
                    check_paused=False,
                )
                if not document.is_paused:
                    raise DocumentIndexingClaimLostError()
                document.indexing_claim_id = None
                document.indexing_claim_expires_at = None
                document.indexing_dataset_generation_id = None
            return True
        except DocumentIndexingClaimLostError:
            raise
        except DocumentIsDeletedPausedError:
            return False

    def complete_indexing(
        self,
        ref: DocumentRef,
        *,
        claim_id: str | None = None,
        tokens: int,
        latency: float,
        source_version: str | None = None,
        dataset_generation_id: str | None = None,
    ) -> None:
        """Publish success and atomically clear the complete task identity.

        A successful owner consumes its durable request only after the document
        result is committed. Parent-generation or claim loss raises without
        acknowledging a newer request.
        """
        with self._session_factory.begin() as session:
            dataset = lock_dataset_indexing_generation_shared(
                session,
                dataset_id=ref.dataset.dataset_id,
                generation_id=dataset_generation_id,
            )
            if dataset is None:
                owned_dataset_id = session.scalar(
                    select(Dataset.id).where(
                        Dataset.id == ref.dataset.dataset_id,
                        Dataset.tenant_id == ref.dataset.tenant_id,
                    )
                )
                if owned_dataset_id is None:
                    raise DocumentIsDeletedPausedError()
                raise DatasetIndexingGenerationLostError(ref.dataset.dataset_id)
            if dataset.tenant_id != ref.dataset.tenant_id:
                raise DocumentIsDeletedPausedError()
            document = require_indexing_document(session, ref, claim_id=claim_id, lock=True)
            if document.indexing_dataset_generation_id != dataset_generation_id:
                raise DatasetIndexingGenerationLostError(ref.dataset.dataset_id)
            request_id = document.indexing_request_id
            if source_version is not None and document.data_source_type == "notion_import":
                source_info = dict(_mapping(document.data_source_info) or {})
                source_info["last_edited_time"] = source_version
                document.data_source_info = json.dumps(source_info)
            document.indexing_status = IndexingStatus.COMPLETED
            document.tokens = tokens
            document.indexing_latency = latency
            document.completed_at = naive_utc_now()
            document.error = None
            document.indexing_request_id = None
            document.indexing_claim_id = None
            document.indexing_claim_expires_at = None
            document.indexing_dataset_generation_id = None
            session.flush()
            if request_id is not None:
                self._complete_request_if_finished(
                    session,
                    request_id=request_id,
                    tenant_id=ref.dataset.tenant_id,
                    dataset_id=ref.dataset.dataset_id,
                )

    def fail_indexing(
        self,
        ref: DocumentRef,
        error: str,
        *,
        claim_id: str | None = None,
        dataset_generation_id: str | None = None,
    ) -> None:
        """Publish terminal failure only for the exact current task identity.

        Paused owners retain their request and phase. Parent-generation loss is
        reported to the caller, which transitions the exact owner to WAITING
        through :meth:`release_dataset_generation_owner`.
        """
        try:
            with self._session_factory.begin() as session:
                dataset = lock_dataset_indexing_generation_shared(
                    session,
                    dataset_id=ref.dataset.dataset_id,
                    generation_id=dataset_generation_id,
                )
                if dataset is None:
                    owned_dataset_id = session.scalar(
                        select(Dataset.id).where(
                            Dataset.id == ref.dataset.dataset_id,
                            Dataset.tenant_id == ref.dataset.tenant_id,
                        )
                    )
                    if owned_dataset_id is None:
                        raise DocumentIsDeletedPausedError()
                    raise DatasetIndexingGenerationLostError(ref.dataset.dataset_id)
                if dataset.tenant_id != ref.dataset.tenant_id:
                    raise DocumentIsDeletedPausedError()
                document = require_indexing_document(
                    session,
                    ref,
                    claim_id=claim_id,
                    lock=True,
                    # A durable worker owns an exact lease and must be able to
                    # terminate it if the document is paused mid-run. Preserve
                    # the legacy pause guard for callers without a claim.
                    check_paused=claim_id is None,
                )
                if document.indexing_dataset_generation_id != dataset_generation_id:
                    raise DatasetIndexingGenerationLostError(ref.dataset.dataset_id)
                request_id = document.indexing_request_id
                if claim_id is not None and document.is_paused:
                    # Pause wins over a concurrent provider/backend failure.
                    # Keep both the durable request and the last committed
                    # phase so a later RECOVER generation can resume safely.
                    document.indexing_claim_id = None
                    document.indexing_claim_expires_at = None
                    document.indexing_dataset_generation_id = None
                    return
                document.indexing_status = IndexingStatus.ERROR
                document.error = error
                document.stopped_at = naive_utc_now()
                document.indexing_request_id = None
                document.indexing_claim_id = None
                document.indexing_claim_expires_at = None
                document.indexing_dataset_generation_id = None
                session.flush()
                if request_id is not None:
                    self._complete_request_if_finished(
                        session,
                        request_id=request_id,
                        tenant_id=ref.dataset.tenant_id,
                        dataset_id=ref.dataset.dataset_id,
                    )
        except DocumentIsDeletedPausedError:
            return

    @staticmethod
    def _complete_request_if_finished(
        session: Session,
        *,
        request_id: str,
        tenant_id: str,
        dataset_id: str,
    ) -> None:
        remaining = session.scalar(
            select(Document.id)
            .where(
                Document.indexing_request_id == request_id,
                Document.tenant_id == tenant_id,
                Document.dataset_id == dataset_id,
            )
            .limit(1)
        )
        if remaining is None:
            session.execute(
                delete(DocumentIndexingOutbox).where(
                    DocumentIndexingOutbox.id == request_id,
                    DocumentIndexingOutbox.tenant_id == tenant_id,
                    DocumentIndexingOutbox.dataset_id == dataset_id,
                )
            )

    def claim_notion_sync(
        self,
        ref: DocumentRef,
        *,
        claim_id: str,
    ) -> ClaimedNotionDocument | None:
        """Reserve a stable Notion document without holding a transaction across remote I/O."""
        with self._session_factory.begin() as session:
            dataset = session.scalar(
                select(Dataset)
                .where(Dataset.id == ref.dataset.dataset_id, Dataset.tenant_id == ref.dataset.tenant_id)
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if dataset is None:
                return None
            document = session.scalar(
                select(Document)
                .join(Dataset, Dataset.id == Document.dataset_id)
                .join(Tenant, Tenant.id == Dataset.tenant_id)
                .where(
                    Tenant.id == ref.dataset.tenant_id,
                    Dataset.id == ref.dataset.dataset_id,
                    Dataset.tenant_id == ref.dataset.tenant_id,
                    Document.id == ref.document_id,
                    Document.tenant_id == ref.dataset.tenant_id,
                    Document.dataset_id == ref.dataset.dataset_id,
                )
                .with_for_update(of=Document)
                .execution_options(populate_existing=True)
            )
            database_now = _database_now(session)
            if (
                document is None
                or document.data_source_type != "notion_import"
                or document.is_paused
                or not document.enabled
                or document.archived
            ):
                return None
            restart_waiting = (
                document.indexing_claim_id is None
                and document.indexing_status == IndexingStatus.WAITING
                and document.processing_started_at is not None
            )
            unclaimed_ready = restart_waiting or (
                document.indexing_claim_id is None
                and document.indexing_status
                in {
                    IndexingStatus.COMPLETED,
                    IndexingStatus.ERROR,
                }
            )
            expired_claim = document.indexing_claim_id is not None and not has_active_indexing_claim(session, document)
            if not unclaimed_ready and not expired_claim:
                return None

            source_info = _mapping(document.data_source_info)
            if source_info is None:
                raise ValueError("no notion page found")

            def required_string(key: str) -> str:
                value = source_info.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError("no notion page found")
                return value

            credential_value = source_info.get("credential_id")
            credential_id = credential_value if isinstance(credential_value, str) else None
            page_edited_value = source_info.get("last_edited_time")
            page_edited_time = page_edited_value if isinstance(page_edited_value, str) else None
            claim = ClaimedNotionDocument(
                ref=ref,
                claim_id=claim_id,
                workspace_id=required_string("notion_workspace_id"),
                page_id=required_string("notion_page_id"),
                page_type=required_string("type"),
                page_edited_time=page_edited_time,
                credential_id=credential_id,
                source_info=MappingProxyType(dict(source_info)),
                index_type=str(document.doc_form),
                # ERROR may predate success-only source-version publication.
                # Always rebuild it even if the remote timestamp matches the
                # stored value, which may have been advanced by an older worker.
                recovery_required=expired_claim or restart_waiting or document.indexing_status == IndexingStatus.ERROR,
                dataset_generation_id=dataset.indexing_request_id,
            )
            document.indexing_claim_id = claim_id
            document.indexing_claim_expires_at = database_now + timedelta(seconds=NOTION_SYNC_LEASE_SECONDS)
            document.indexing_dataset_generation_id = dataset.indexing_request_id
            return claim

    def renew_notion_sync(self, claim: ClaimedNotionDocument) -> bool:
        """Extend an active lease only while the caller remains its exact owner."""
        try:
            with self._session_factory.begin() as session:
                renew_indexing_claim(session, claim.ref, claim_id=claim.claim_id)
            return True
        except DocumentIsDeletedPausedError:
            return False

    def prepare_notion_sync(
        self,
        claim: ClaimedNotionDocument,
        *,
        processing_started_at: datetime,
    ) -> None:
        """Mark an owned changed page for indexing without publishing its source version."""
        with self._session_factory.begin() as session:
            document = require_indexing_document(
                session,
                claim.ref,
                claim_id=claim.claim_id,
                lock=True,
            )
            document.indexing_status = IndexingStatus.PARSING
            document.processing_started_at = processing_started_at

    def release_notion_sync(self, claim: ClaimedNotionDocument) -> bool:
        """Release only the caller's claim."""
        try:
            with self._session_factory.begin() as session:
                document = require_indexing_document(
                    session,
                    claim.ref,
                    claim_id=claim.claim_id,
                    lock=True,
                    check_paused=False,
                )
                document.indexing_claim_id = None
                document.indexing_claim_expires_at = None
                document.indexing_dataset_generation_id = None
            return True
        except DocumentIsDeletedPausedError:
            return False

    def exists(self, *, workspace_id: str, dataset_id: str, document_id: str) -> bool:
        with self._session_factory() as session:
            return bool(
                session.scalar(
                    select(
                        exists().where(
                            Document.tenant_id == workspace_id,
                            Document.dataset_id == dataset_id,
                            Document.id == document_id,
                        )
                    )
                )
            )

    def mark_failed(
        self,
        *,
        workspace_id: str,
        dataset_id: str,
        document_id: str,
        error: str,
        indexing_request_id: str | None = None,
        claim_id: str | None = None,
    ) -> None:
        if indexing_request_id is not None or claim_id is not None:
            if indexing_request_id is None or claim_id is None:
                raise DocumentIndexingClaimLostError()
            ref = DatasetRef(workspace_id, dataset_id).document(document_id)
            with self._session_factory.begin() as session:
                dataset = session.scalar(
                    select(Dataset)
                    .where(Dataset.id == dataset_id, Dataset.tenant_id == workspace_id)
                    .with_for_update(read=True)
                    .execution_options(populate_existing=True)
                )
                if dataset is None:
                    raise DocumentIsDeletedPausedError()
                document = require_indexing_document(
                    session,
                    ref,
                    claim_id=claim_id,
                    lock=True,
                    check_paused=False,
                )
                if document.indexing_request_id != indexing_request_id:
                    raise DocumentIndexingClaimLostError()
                if document.indexing_dataset_generation_id != dataset.indexing_request_id:
                    # A parent configuration generation superseded this
                    # pipeline attempt. Preserve the durable document request
                    # for a fresh claim instead of publishing a stale ERROR.
                    transition_dataset_generation_owner_to_restart(
                        document,
                        restarted_at=_database_now(session),
                    )
                    return
                if document.is_paused:
                    document.indexing_claim_id = None
                    document.indexing_claim_expires_at = None
                    document.indexing_dataset_generation_id = None
                    return
                document.indexing_status = IndexingStatus.ERROR
                document.error = error
                document.stopped_at = naive_utc_now()
                document.indexing_request_id = None
                document.indexing_claim_id = None
                document.indexing_claim_expires_at = None
                document.indexing_dataset_generation_id = None
                session.flush()
                self._complete_request_if_finished(
                    session,
                    request_id=indexing_request_id,
                    tenant_id=workspace_id,
                    dataset_id=dataset_id,
                )
            return
        with self._session_factory.begin() as session:
            session.execute(
                update(Document)
                .where(
                    Document.tenant_id == workspace_id,
                    Document.dataset_id == dataset_id,
                    Document.id == document_id,
                    Document.indexing_request_id.is_(None),
                    Document.indexing_claim_id.is_(None),
                )
                .values(indexing_status=IndexingStatus.ERROR, error=error)
            )

    def get_estimate_document(self, document_ref: DocumentRef) -> EstimateDocumentRecord | None:
        with self._session_factory() as session:
            document = _get_document(session, document_ref)
            return _estimate_document_record(document) if document is not None else None

    def list_estimate_documents_by_batch(
        self, dataset_ref: DatasetRef, batch: str
    ) -> tuple[EstimateDocumentRecord, ...]:
        with self._session_factory() as session:
            documents = session.scalars(
                select(Document).where(
                    Document.tenant_id == dataset_ref.tenant_id,
                    Document.dataset_id == dataset_ref.dataset_id,
                    Document.batch == batch,
                )
            ).all()
            return tuple(_estimate_document_record(document) for document in documents)

    def get_sync_document(self, document_ref: DocumentRef) -> SyncDocumentRecord | None:
        with self._session_factory() as session:
            document = _get_document(session, document_ref)
            if document is None:
                return None
            return SyncDocumentRecord(id=document.id, data_source_type=str(document.data_source_type))

    def list_active_notion_refs(self, dataset_ref: DatasetRef) -> tuple[DocumentRef, ...]:
        with self._session_factory() as session:
            return tuple(
                dataset_ref.document(document_id)
                for document_id in session.scalars(
                    select(Document.id).where(
                        Document.tenant_id == dataset_ref.tenant_id,
                        Document.dataset_id == dataset_ref.dataset_id,
                        Document.data_source_type == "notion_import",
                        Document.enabled.is_(True),
                        Document.archived.is_(False),
                    )
                ).all()
            )

    def list_bound_notion_page_ids(self, dataset_ref: DatasetRef) -> frozenset[str]:
        page_ids: set[str] = set()
        with self._session_factory() as session:
            documents = session.scalars(
                select(Document).where(
                    Document.tenant_id == dataset_ref.tenant_id,
                    Document.dataset_id == dataset_ref.dataset_id,
                    Document.data_source_type == "notion_import",
                    Document.enabled.is_(True),
                )
            ).all()
        for document in documents:
            source_info = _mapping(document.data_source_info)
            if source_info is None:
                continue
            page_id = source_info.get("notion_page_id")
            if isinstance(page_id, str) and page_id:
                page_ids.add(page_id)
        return frozenset(page_ids)

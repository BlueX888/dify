"""Persistence boundary for indexing publications owned by document writes."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from sqlalchemy import case, delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, sessionmaker

from models import Tenant
from models.dataset import Dataset, Document, DocumentPipelineExecutionLog
from models.enums import DataSourceType, IndexingStatus
from models.model import DocumentIndexingOutbox
from models.workflow import Workflow
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef


class DocumentIndexingAction(StrEnum):
    CREATE = "create"
    DUPLICATE = "duplicate"
    RECOVER = "recover"
    REMOVE_NOTION = "remove_notion"
    RETRY = "retry"
    UPDATE = "update"
    WEBSITE_SYNC = "website_sync"


@dataclass(frozen=True)
class DocumentIndexingIntent:
    id: str
    tenant_id: str
    dataset_id: str
    action: DocumentIndexingAction
    document_ids: tuple[str, ...]
    actor_id: str | None = None
    pipeline_execution_log_id: str | None = None


def create_document_indexing_intents(session: Session, jobs: DocumentIndexingJobs) -> tuple[str, ...]:
    """Persist independently retryable publications in the caller's transaction."""
    actions: tuple[tuple[DocumentIndexingAction, tuple[str, ...]], ...] = (
        (DocumentIndexingAction.REMOVE_NOTION, jobs.removed_notion),
        (DocumentIndexingAction.CREATE, jobs.created),
        (DocumentIndexingAction.DUPLICATE, jobs.duplicated),
        (DocumentIndexingAction.UPDATE, (jobs.updated,) if jobs.updated is not None else ()),
        (DocumentIndexingAction.WEBSITE_SYNC, (jobs.website_sync,) if jobs.website_sync is not None else ()),
        (DocumentIndexingAction.RECOVER, (jobs.recovered,) if jobs.recovered is not None else ()),
        (DocumentIndexingAction.RETRY, jobs.retried),
    )
    rows = tuple(
        DocumentIndexingOutbox(
            tenant_id=jobs.dataset.tenant_id,
            dataset_id=jobs.dataset.dataset_id,
            action=action.value,
            document_ids=json.dumps(document_ids),
            actor_id=jobs.actor_id if action == DocumentIndexingAction.RETRY else None,
        )
        for action, document_ids in actions
        if document_ids
    )
    session.add_all(rows)
    session.flush()
    for row in rows:
        document_ids = json.loads(row.document_ids)
        session.execute(
            update(Document)
            .where(
                Document.id.in_(document_ids),
                Document.tenant_id == row.tenant_id,
                Document.dataset_id == row.dataset_id,
            )
            # The request token is queue bookkeeping, not a user-visible
            # document mutation. Keep the timestamp assigned by the owning
            # save operation instead of firing Document.updated_at's onupdate.
            # Replacing a generation must fence the previous worker in the
            # same statement; otherwise its late completion can acknowledge
            # the newly installed request.
            .values(
                indexing_request_id=row.id,
                indexing_claim_id=None,
                indexing_claim_expires_at=None,
                indexing_dataset_generation_id=None,
                dataset_indexing_request_id=None,
                updated_at=Document.updated_at,
            )
        )
    return tuple(row.id for row in rows)


def create_pipeline_document_indexing_intents(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    pipeline_id: str,
    pipeline_execution_log_ids: Mapping[str, str],
    actor_id: str,
    source_actor_id: str | None = None,
) -> dict[str, str]:
    """Install one durable pipeline-replay owner for each WAITING document."""
    ordered_ids = tuple(sorted(pipeline_execution_log_ids))
    if not ordered_ids:
        return {}
    with session.no_autoflush:
        dataset = session.scalar(
            select(Dataset)
            .where(
                Dataset.id == dataset_id,
                Dataset.tenant_id == tenant_id,
                Dataset.pipeline_id == pipeline_id,
            )
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        if dataset is None:
            raise ValueError("Pipeline dataset not found")
        documents = tuple(
            session.scalars(
                select(Document)
                .where(
                    Document.id.in_(ordered_ids),
                    Document.tenant_id == tenant_id,
                    Document.dataset_id == dataset_id,
                )
                .order_by(Document.id)
                .with_for_update()
                .execution_options(populate_existing=True)
            ).all()
        )
    if tuple(document.id for document in documents) != ordered_ids:
        raise ValueError("Pipeline document not found")
    execution_logs = tuple(
        session.scalars(
            select(DocumentPipelineExecutionLog)
            .where(DocumentPipelineExecutionLog.id.in_(pipeline_execution_log_ids.values()))
            .order_by(DocumentPipelineExecutionLog.document_id)
            .with_for_update()
        ).all()
    )
    execution_logs_by_document_id = {execution_log.document_id: execution_log for execution_log in execution_logs}
    if set(execution_logs_by_document_id) != set(ordered_ids):
        raise ValueError("Pipeline execution log not found")
    requests: dict[str, str] = {}
    for document in documents:
        execution_log = execution_logs_by_document_id[document.id]
        if (
            execution_log.id != pipeline_execution_log_ids[document.id]
            or execution_log.pipeline_id != pipeline_id
            or execution_log.workflow_id is None
            or (source_actor_id is not None and execution_log.created_by != source_actor_id)
        ):
            raise ValueError("Pipeline execution log does not own document")
        document.indexing_status = IndexingStatus.WAITING
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(
                dataset=DatasetRef(tenant_id, dataset_id),
                retried=(document.id,),
                actor_id=actor_id,
            ),
        )
        session.get_one(DocumentIndexingOutbox, request_id).pipeline_execution_log_id = execution_log.id
        requests[document.id] = request_id
    return requests


def invalidate_notion_removal_intents(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Invalidate queued removals and return pages that need replacement indexing."""
    if not document_ids:
        return ()
    removal_request_ids = select(DocumentIndexingOutbox.id).where(
        DocumentIndexingOutbox.tenant_id == tenant_id,
        DocumentIndexingOutbox.dataset_id == dataset_id,
        DocumentIndexingOutbox.action == DocumentIndexingAction.REMOVE_NOTION.value,
    )
    invalidated_ids = tuple(
        session.scalars(
            select(Document.id)
            .where(
                Document.id.in_(document_ids),
                Document.tenant_id == tenant_id,
                Document.dataset_id == dataset_id,
                Document.indexing_request_id.in_(removal_request_ids),
            )
            .order_by(Document.id)
            .with_for_update()
        ).all()
    )
    if not invalidated_ids:
        return ()
    session.execute(
        update(Document)
        .where(
            Document.id.in_(invalidated_ids),
            Document.tenant_id == tenant_id,
            Document.dataset_id == dataset_id,
        )
        .values(
            indexing_request_id=None,
            indexing_claim_id=None,
            indexing_claim_expires_at=None,
            indexing_dataset_generation_id=None,
            indexing_status=IndexingStatus.WAITING,
            processing_started_at=None,
            parsing_completed_at=None,
            cleaning_completed_at=None,
            splitting_completed_at=None,
            completed_at=None,
            error=None,
            stopped_at=None,
            updated_at=Document.updated_at,
        )
    )
    return invalidated_ids


class SQLAlchemyDocumentIndexingOutboxRepository:
    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_or_create_legacy_pipeline_intent(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        pipeline_id: str,
        document_id: str,
        actor_id: str,
        datasource_type: str,
        datasource_info: dict[str, object],
        datasource_node_id: str,
        batch: str,
        workflow_id: str,
        original_document_id: str | None,
    ) -> DocumentIndexingIntent | None:
        """Bridge one pre-outbox pipeline delivery to an exact durable owner."""
        with self._session_factory.begin() as session:
            dataset = session.scalar(
                select(Dataset)
                .where(
                    Dataset.id == dataset_id,
                    Dataset.tenant_id == tenant_id,
                    Dataset.pipeline_id == pipeline_id,
                )
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if dataset is None:
                return None
            workflow_exists = session.scalar(
                select(Workflow.id).where(
                    Workflow.id == workflow_id,
                    Workflow.tenant_id == tenant_id,
                    Workflow.app_id == pipeline_id,
                    Workflow.version != Workflow.VERSION_DRAFT,
                )
            )
            if workflow_exists is None:
                return None
            document = session.scalar(
                select(Document)
                .where(
                    Document.id == document_id,
                    Document.tenant_id == tenant_id,
                    Document.dataset_id == dataset_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if document is None or document.indexing_request_id is not None:
                # Never attach an old embedded payload to a newer request. Its
                # outbox consumer will replay the current persisted log.
                return None
            source_log = session.scalar(
                select(DocumentPipelineExecutionLog)
                .where(
                    DocumentPipelineExecutionLog.document_id == document.id,
                    DocumentPipelineExecutionLog.pipeline_id == pipeline_id,
                    DocumentPipelineExecutionLog.created_by == actor_id,
                )
                .order_by(
                    DocumentPipelineExecutionLog.created_at.desc(),
                    DocumentPipelineExecutionLog.id.desc(),
                )
                .limit(1)
                .with_for_update()
            )
            if (
                source_log is None
                or str(source_log.datasource_type) != datasource_type
                or json.loads(source_log.datasource_info) != datasource_info
                or source_log.datasource_node_id != datasource_node_id
                or document.batch != batch
                or document.indexing_status != IndexingStatus.WAITING
                or document.indexing_claim_id is not None
                or document.indexing_claim_expires_at is not None
                or original_document_id not in (None, document.id)
            ):
                return None
            if source_log.workflow_id is None:
                source_log.workflow_id = workflow_id
                source_log.original_document_id = original_document_id
            elif source_log.workflow_id != workflow_id or source_log.original_document_id != original_document_id:
                return None
            (request_id,) = create_document_indexing_intents(
                session,
                DocumentIndexingJobs(
                    dataset=DatasetRef(tenant_id, dataset_id),
                    retried=(document.id,),
                    actor_id=actor_id,
                ),
            )
            row = session.get_one(DocumentIndexingOutbox, request_id)
            row.pipeline_execution_log_id = source_log.id
            return self._to_intent(row)

    def get_or_create_pipeline_retry_intent(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        pipeline_id: str,
        document_id: str,
        actor_id: str,
    ) -> DocumentIndexingIntent | None:
        """Bridge a legacy retry command without replacing an active owner."""
        with self._session_factory.begin() as session:
            dataset = session.scalar(
                select(Dataset)
                .where(
                    Dataset.id == dataset_id,
                    Dataset.tenant_id == tenant_id,
                    Dataset.pipeline_id == pipeline_id,
                )
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if dataset is None:
                return None
            document = session.scalar(
                select(Document)
                .where(
                    Document.id == document_id,
                    Document.tenant_id == tenant_id,
                    Document.dataset_id == dataset_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if document is None:
                return None
            if document.indexing_request_id is not None:
                row = session.scalar(
                    select(DocumentIndexingOutbox).where(
                        DocumentIndexingOutbox.id == document.indexing_request_id,
                        DocumentIndexingOutbox.tenant_id == tenant_id,
                        DocumentIndexingOutbox.dataset_id == dataset_id,
                        DocumentIndexingOutbox.action == DocumentIndexingAction.RETRY.value,
                        DocumentIndexingOutbox.actor_id == actor_id,
                    )
                )
                if row is None:
                    return None
                intent = self._to_intent(row)
                if intent.document_ids != (document.id,) or intent.pipeline_execution_log_id is None:
                    return None
                return intent
            source_log = session.scalar(
                select(DocumentPipelineExecutionLog)
                .where(
                    DocumentPipelineExecutionLog.document_id == document.id,
                    DocumentPipelineExecutionLog.pipeline_id == pipeline_id,
                )
                .order_by(
                    DocumentPipelineExecutionLog.created_at.desc(),
                    DocumentPipelineExecutionLog.id.desc(),
                )
                .limit(1)
                .with_for_update()
            )
            if (
                source_log is None
                or source_log.workflow_id is None
                or document.indexing_status != IndexingStatus.WAITING
                or document.indexing_claim_id is not None
                or document.indexing_claim_expires_at is not None
            ):
                return None
            (request_id,) = create_document_indexing_intents(
                session,
                DocumentIndexingJobs(
                    dataset=DatasetRef(tenant_id, dataset_id),
                    retried=(document.id,),
                    actor_id=actor_id,
                ),
            )
            row = session.get_one(DocumentIndexingOutbox, request_id)
            row.pipeline_execution_log_id = source_log.id
            return self._to_intent(row)

    def list_pending_intent_ids(self, *, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        with self._session_factory() as session:
            return tuple(
                session.scalars(
                    select(DocumentIndexingOutbox.id)
                    .order_by(
                        case((DocumentIndexingOutbox.last_attempted_at.is_(None), 0), else_=1),
                        DocumentIndexingOutbox.last_attempted_at,
                        DocumentIndexingOutbox.created_at,
                        DocumentIndexingOutbox.id,
                    )
                    .limit(limit)
                )
            )

    def mark_attempted(self, intent_id: str) -> None:
        with self._session_factory.begin() as session:
            session.execute(
                update(DocumentIndexingOutbox)
                .where(DocumentIndexingOutbox.id == intent_id)
                .values(last_attempted_at=func.current_timestamp())
            )

    def get_intent(self, intent_id: str) -> DocumentIndexingIntent | None:
        with self._session_factory() as session:
            row = session.get(DocumentIndexingOutbox, intent_id)
            return self._to_intent(row) if row is not None else None

    def get_pipeline_intent(
        self,
        intent_id: str,
        *,
        tenant_id: str,
        dataset_id: str,
        pipeline_id: str,
        document_id: str,
        actor_id: str,
        workflow_id: str,
        datasource_type: str,
        datasource_info: dict[str, object],
        datasource_node_id: str,
        batch: str,
        original_document_id: str | None,
    ) -> DocumentIndexingIntent | None:
        """Resolve a pipeline request through its complete persisted owner chain."""
        with self._session_factory.begin() as session:
            dataset = session.scalar(
                select(Dataset)
                .where(
                    Dataset.id == dataset_id,
                    Dataset.tenant_id == tenant_id,
                    Dataset.pipeline_id == pipeline_id,
                )
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if dataset is None:
                return None
            document = session.scalar(
                select(Document)
                .where(
                    Document.id == document_id,
                    Document.tenant_id == tenant_id,
                    Document.dataset_id == dataset_id,
                    Document.indexing_request_id == intent_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if document is None or document.batch != batch or original_document_id not in (None, document.id):
                return None
            row = session.scalar(
                select(DocumentIndexingOutbox)
                .where(
                    DocumentIndexingOutbox.id == intent_id,
                    DocumentIndexingOutbox.tenant_id == tenant_id,
                    DocumentIndexingOutbox.dataset_id == dataset_id,
                    DocumentIndexingOutbox.action == DocumentIndexingAction.RETRY.value,
                    DocumentIndexingOutbox.actor_id == actor_id,
                )
                .with_for_update()
            )
            if row is None:
                return None
            intent = self._to_intent(row)
            if intent.document_ids != (document_id,):
                return None
            if row.pipeline_execution_log_id is None:
                source_log = session.scalar(
                    select(DocumentPipelineExecutionLog)
                    .where(
                        DocumentPipelineExecutionLog.document_id == document.id,
                        DocumentPipelineExecutionLog.pipeline_id == pipeline_id,
                        DocumentPipelineExecutionLog.created_by == actor_id,
                    )
                    .order_by(
                        DocumentPipelineExecutionLog.created_at.desc(),
                        DocumentPipelineExecutionLog.id.desc(),
                    )
                    .limit(1)
                    .with_for_update()
                )
            else:
                source_log = session.scalar(
                    select(DocumentPipelineExecutionLog)
                    .where(
                        DocumentPipelineExecutionLog.id == row.pipeline_execution_log_id,
                        DocumentPipelineExecutionLog.document_id == document.id,
                        DocumentPipelineExecutionLog.pipeline_id == pipeline_id,
                        DocumentPipelineExecutionLog.created_by == actor_id,
                    )
                    .with_for_update()
                )
            if (
                source_log is None
                or str(source_log.datasource_type) != datasource_type
                or json.loads(source_log.datasource_info) != datasource_info
                or source_log.datasource_node_id != datasource_node_id
            ):
                return None
            workflow_exists = session.scalar(
                select(Workflow.id).where(
                    Workflow.id == workflow_id,
                    Workflow.tenant_id == tenant_id,
                    Workflow.app_id == pipeline_id,
                    Workflow.version != Workflow.VERSION_DRAFT,
                )
            )
            if workflow_exists is None:
                return None
            if source_log.workflow_id is None:
                source_log.workflow_id = workflow_id
                source_log.original_document_id = original_document_id
            elif source_log.workflow_id != workflow_id or source_log.original_document_id != original_document_id:
                return None
            row.pipeline_execution_log_id = source_log.id
            return self._to_intent(row)

    def get_or_create_legacy_website_sync_intent(
        self,
        *,
        dataset_id: str,
        document_id: str,
    ) -> DocumentIndexingIntent | None:
        """Bridge a pre-outbox website-sync message to one durable generation.

        Queue payload IDs are lookup selectors only. The locked database owner
        chain supplies the tenant and dataset scope used by the new intent.
        """
        with self._session_factory.begin() as session:
            document = session.scalar(
                select(Document)
                .join(
                    Dataset,
                    (Dataset.id == Document.dataset_id) & (Dataset.tenant_id == Document.tenant_id),
                )
                .join(Tenant, Tenant.id == Dataset.tenant_id)
                .where(
                    Dataset.id == dataset_id,
                    Document.id == document_id,
                    Document.dataset_id == dataset_id,
                    Document.data_source_type == DataSourceType.WEBSITE_CRAWL,
                    Document.indexing_status == IndexingStatus.WAITING,
                )
                .with_for_update(of=Document)
                .limit(1)
            )
            if document is None:
                return None
            if document.indexing_claim_id is not None or document.indexing_claim_expires_at is not None:
                return None

            if document.indexing_request_id is not None:
                row = session.scalar(
                    select(DocumentIndexingOutbox).where(
                        DocumentIndexingOutbox.id == document.indexing_request_id,
                        DocumentIndexingOutbox.tenant_id == document.tenant_id,
                        DocumentIndexingOutbox.dataset_id == document.dataset_id,
                        DocumentIndexingOutbox.action == DocumentIndexingAction.WEBSITE_SYNC.value,
                    )
                )
                if row is None:
                    return None
                intent = self._to_intent(row)
                return intent if intent.document_ids == (document.id,) else None

            row = DocumentIndexingOutbox(
                tenant_id=document.tenant_id,
                dataset_id=document.dataset_id,
                action=DocumentIndexingAction.WEBSITE_SYNC.value,
                document_ids=json.dumps([document.id]),
            )
            session.add(row)
            session.flush()
            claimed = session.execute(
                update(Document)
                .where(
                    Document.id == document.id,
                    Document.tenant_id == document.tenant_id,
                    Document.dataset_id == document.dataset_id,
                    Document.indexing_status == IndexingStatus.WAITING,
                    Document.indexing_request_id.is_(None),
                    Document.indexing_claim_id.is_(None),
                    Document.indexing_claim_expires_at.is_(None),
                )
                .values(
                    indexing_request_id=row.id,
                    dataset_indexing_request_id=None,
                    updated_at=Document.updated_at,
                )
            )
            if cast(CursorResult[object], claimed).rowcount != 1:
                session.delete(row)
                return None
            return self._to_intent(row)

    def request_is_current(self, intent: DocumentIndexingIntent) -> bool:
        with self._session_factory() as session:
            return (
                session.scalar(
                    select(Document.id)
                    .where(
                        Document.tenant_id == intent.tenant_id,
                        Document.dataset_id == intent.dataset_id,
                        Document.indexing_request_id == intent.id,
                    )
                    .limit(1)
                )
                is not None
            )

    def has_active_worker(self, intent: DocumentIndexingIntent) -> bool:
        with self._session_factory() as session:
            return (
                session.scalar(
                    select(Document.id)
                    .where(
                        Document.tenant_id == intent.tenant_id,
                        Document.dataset_id == intent.dataset_id,
                        Document.indexing_request_id == intent.id,
                        Document.indexing_claim_id.is_not(None),
                        Document.indexing_claim_expires_at > func.current_timestamp(),
                    )
                    .limit(1)
                )
                is not None
            )

    def complete(self, intent: DocumentIndexingIntent) -> None:
        with self._session_factory.begin() as session:
            session.execute(
                delete(DocumentIndexingOutbox).where(
                    DocumentIndexingOutbox.id == intent.id,
                    DocumentIndexingOutbox.tenant_id == intent.tenant_id,
                    DocumentIndexingOutbox.dataset_id == intent.dataset_id,
                    DocumentIndexingOutbox.action == intent.action.value,
                )
            )

    @staticmethod
    def _to_intent(row: DocumentIndexingOutbox) -> DocumentIndexingIntent:
        document_ids = json.loads(row.document_ids)
        if not isinstance(document_ids, list) or not all(isinstance(item, str) for item in document_ids):
            raise ValueError(f"Invalid document indexing outbox payload: {row.id}")
        return DocumentIndexingIntent(
            id=row.id,
            tenant_id=row.tenant_id,
            dataset_id=row.dataset_id,
            action=DocumentIndexingAction(row.action),
            document_ids=tuple(document_ids),
            actor_id=row.actor_id,
            pipeline_execution_log_id=row.pipeline_execution_log_id,
        )

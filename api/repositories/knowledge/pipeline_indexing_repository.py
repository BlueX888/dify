"""Fencing policy adapter for durable RAG-pipeline indexing attempts."""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta
from functools import partial

from sqlalchemy import DateTime, func, select
from sqlalchemy.orm import Session

from core.entities.document_task import (
    DocumentIndexingClaimLostError,
    DocumentIsDeletedPausedError,
    DocumentIsPausedError,
)
from models.dataset import Dataset, Document
from models.enums import IndexingStatus
from repositories.knowledge.dataset_indexing_outbox_repository import SQLAlchemyDatasetIndexingOutboxRepository
from services.knowledge.entities.datasets import DatasetIndexingGenerationLostError
from services.knowledge.entities.vector_route import VectorRoute

PipelineIndexingMutationGuard = Callable[[Session], AbstractContextManager[None]]
PipelineIndexingRouteRecorder = Callable[[str, str, str], None]
INDEXING_CLAIM_LEASE_SECONDS = 300


def _database_now(session: Session) -> datetime:
    bind = session.get_bind()
    timestamp = (
        func.clock_timestamp(type_=DateTime()) if bind.dialect.name == "postgresql" else func.current_timestamp()
    )
    value = session.scalar(select(timestamp))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value.replace(tzinfo=None, microsecond=0)


@contextmanager
def protect_pipeline_indexing_claim(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    indexing_request_id: str,
    claim_id: str,
) -> Generator[None]:
    """Lock, renew, and validate the exact durable RAG indexing generation."""
    dataset = session.scalar(
        select(Dataset)
        .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
        .with_for_update(read=True)
        .execution_options(populate_existing=True)
    )
    if dataset is None:
        raise DocumentIsDeletedPausedError()
    document = session.scalar(
        select(Document)
        .where(
            Document.id == document_id,
            Document.dataset_id == dataset_id,
            Document.tenant_id == tenant_id,
        )
        .with_for_update(of=Document)
        .execution_options(populate_existing=True)
    )
    if document is None:
        raise DocumentIsDeletedPausedError()
    if document.indexing_dataset_generation_id != dataset.indexing_request_id:
        raise DatasetIndexingGenerationLostError(dataset_id)
    database_now = _database_now(session)
    if (
        document.indexing_claim_id != claim_id
        or document.indexing_claim_expires_at is None
        or document.indexing_claim_expires_at <= database_now
    ):
        raise DocumentIndexingClaimLostError()
    if document.is_paused:
        raise DocumentIsPausedError()
    if document.indexing_request_id != indexing_request_id or document.indexing_status not in {
        IndexingStatus.PARSING,
        IndexingStatus.COMPLETED,
    }:
        raise DocumentIndexingClaimLostError()
    document.indexing_claim_expires_at = database_now + timedelta(seconds=INDEXING_CLAIM_LEASE_SECONDS)
    try:
        yield
    finally:
        locked_document = session.scalar(
            select(Document)
            .where(
                Document.id == document_id,
                Document.dataset_id == dataset_id,
                Document.tenant_id == tenant_id,
                Document.indexing_request_id == indexing_request_id,
                Document.indexing_claim_id == claim_id,
                Document.indexing_status.in_((IndexingStatus.PARSING, IndexingStatus.COMPLETED)),
            )
            .with_for_update(of=Document)
            .execution_options(populate_existing=True)
        )
        if locked_document is None:
            raise DocumentIndexingClaimLostError()
        locked_document.indexing_claim_expires_at = _database_now(session) + timedelta(
            seconds=INDEXING_CLAIM_LEASE_SECONDS
        )


def build_pipeline_indexing_guard(
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    indexing_request_id: str | None,
    claim_id: str | None,
) -> PipelineIndexingMutationGuard | None:
    """Return a mutation guard only when a complete trusted claim is present."""
    if indexing_request_id is None and claim_id is None:
        return None
    if indexing_request_id is None or claim_id is None:
        raise DocumentIndexingClaimLostError()
    return partial(
        protect_pipeline_indexing_claim,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        indexing_request_id=indexing_request_id,
        claim_id=claim_id,
    )


def build_pipeline_indexing_guard_from_ids(
    session: Session,
    *,
    dataset_id: str,
    document_id: str,
    indexing_request_id: str | None,
    claim_id: str | None,
) -> PipelineIndexingMutationGuard | None:
    """Resolve the trusted owner chain before adapting a workflow-node call."""
    if indexing_request_id is None and claim_id is None:
        return None
    tenant_id = session.scalar(
        select(Document.tenant_id).where(
            Document.id == document_id,
            Document.dataset_id == dataset_id,
        )
    )
    if tenant_id is None:
        raise DocumentIndexingClaimLostError()
    return build_pipeline_indexing_guard(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        indexing_request_id=indexing_request_id,
        claim_id=claim_id,
    )


def build_pipeline_indexing_route_recorder_from_ids(
    session: Session,
    *,
    dataset_id: str,
    document_id: str,
    indexing_request_id: str | None,
    claim_id: str | None,
    new_session: Callable[[], Session],
) -> PipelineIndexingRouteRecorder | None:
    """Build a short pre-I/O receipt transaction for one exact pipeline claim."""
    if indexing_request_id is None and claim_id is None:
        return None
    if indexing_request_id is None or claim_id is None:
        raise DocumentIndexingClaimLostError()
    tenant_id = session.scalar(
        select(Document.tenant_id).where(
            Document.id == document_id,
            Document.dataset_id == dataset_id,
        )
    )
    if tenant_id is None:
        raise DocumentIndexingClaimLostError()

    def record(vector_type: str, collection_name: str, doc_form: str) -> None:
        with new_session() as receipt_session, receipt_session.begin():
            with protect_pipeline_indexing_claim(
                receipt_session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                indexing_request_id=indexing_request_id,
                claim_id=claim_id,
            ):
                dataset = receipt_session.scalar(
                    select(Dataset).where(
                        Dataset.id == dataset_id,
                        Dataset.tenant_id == tenant_id,
                    )
                )
                if dataset is None:
                    raise DatasetIndexingGenerationLostError(dataset_id)
                route = VectorRoute.capture(
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    vector_type=vector_type,
                    collection_name=collection_name,
                    routing_index_struct=dataset.index_struct,
                    collection_binding_id=dataset.collection_binding_id,
                    indexing_technique=str(dataset.indexing_technique),
                    doc_form=doc_form,
                )
                SQLAlchemyDatasetIndexingOutboxRepository.record_generation_writer_route_locked(
                    receipt_session,
                    dataset=dataset,
                    generation_id=dataset.indexing_request_id,
                    route=route,
                )

    return record

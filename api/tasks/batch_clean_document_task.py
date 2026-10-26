import logging
import time
from collections.abc import Sequence

import click
from celery import shared_task
from sqlalchemy import delete, select

from core.db.session_factory import session_factory
from extensions.ext_storage import storage
from models.dataset import Dataset, DatasetMetadataBinding, Document
from repositories.knowledge.segment_cleanup_repository import (
    delete_document_segments_and_claim_uploads,
    lock_segment_attachment_cleanup_dataset,
)
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.indexing.adapters.cleanup import (
    clean_document_indexes,
    handoff_document_attachment_vectors,
)
from services.knowledge.upload_file_cleanup import cleanup_claimed_upload_files
from tasks.refresh_billing_vector_space_task import schedule_billing_vector_space_refresh

logger = logging.getLogger(__name__)


def run_documents_cleanup(
    *,
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    doc_form: str,
    file_ids: Sequence[str],
) -> None:
    """Clean deleted documents and durably claim their files in one database transaction."""
    normalized_document_ids = tuple(dict.fromkeys(document_ids))
    if not normalized_document_ids:
        return

    uploads = SQLAlchemyKnowledgeUploadRepository(session_factory=session_factory.get_session_maker())
    with session_factory.create_session() as session:
        live_document_ids = tuple(
            session.scalars(
                select(Document.id).where(
                    Document.tenant_id == tenant_id,
                    Document.dataset_id == dataset_id,
                    Document.id.in_(normalized_document_ids),
                )
            )
        )
        if live_document_ids:
            raise RuntimeError(f"Documents still exist and cannot be cleaned: {sorted(live_document_ids)}")

    cleaned_index_tenant_id = clean_document_indexes(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_ids=normalized_document_ids,
        doc_form=doc_form,
        new_session=session_factory.create_session,
    )

    with session_factory.create_session() as session, session.begin():
        lock_segment_attachment_cleanup_dataset(session, tenant_id=tenant_id, dataset_id=dataset_id)
        locked_dataset = session.scalar(
            select(Dataset)
            .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
            .execution_options(populate_existing=True)
        )
        live_document_ids = tuple(
            session.scalars(
                select(Document.id)
                .where(
                    Document.tenant_id == tenant_id,
                    Document.dataset_id == dataset_id,
                    Document.id.in_(normalized_document_ids),
                )
                .with_for_update()
            )
        )
        if live_document_ids:
            raise RuntimeError(f"Documents still exist and cannot be cleaned: {sorted(live_document_ids)}")
        if locked_dataset is not None:
            handoff_document_attachment_vectors(
                session,
                dataset=locked_dataset,
                document_ids=normalized_document_ids,
            )
        artifacts = delete_document_segments_and_claim_uploads(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_ids=normalized_document_ids,
            candidate_upload_file_ids=file_ids,
        )
        session.execute(
            delete(DatasetMetadataBinding).where(
                DatasetMetadataBinding.tenant_id == tenant_id,
                DatasetMetadataBinding.dataset_id == dataset_id,
                DatasetMetadataBinding.document_id.in_(normalized_document_ids),
            )
        )

    cleanup_claimed_upload_files(
        artifacts=artifacts,
        files=uploads,
        delete_blob=storage.delete,
        logger=logger,
    )
    if cleaned_index_tenant_id is not None:
        schedule_billing_vector_space_refresh(cleaned_index_tenant_id)


@shared_task(queue="dataset")
def batch_clean_document_task(
    document_ids: list[str],
    dataset_id: str,
    doc_form: str | None,
    file_ids: list[str],
) -> None:
    """Clean a batch of documents after their rows have been deleted."""
    logger.info(click.style("Start batch clean documents when documents deleted", fg="green"))
    start_at = time.perf_counter()
    if not doc_form:
        raise ValueError("doc_form is required")

    with session_factory.create_session() as session:
        dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
    if dataset is None:
        raise ValueError("Dataset not found; use run_documents_cleanup with the captured tenant scope")

    run_documents_cleanup(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset_id,
        document_ids=document_ids,
        doc_form=doc_form,
        file_ids=file_ids,
    )

    logger.info(
        click.style(
            f"Cleaned documents when documents deleted latency: {time.perf_counter() - start_at:.2f}s, "
            f"dataset_id: {dataset_id}, document_ids: {document_ids}",
            fg="green",
        )
    )

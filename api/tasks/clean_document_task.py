import logging
import time

import click
from celery import shared_task
from sqlalchemy import select

from core.db.session_factory import session_factory
from models.dataset import Dataset
from tasks.batch_clean_document_task import run_documents_cleanup

logger = logging.getLogger(__name__)


def run_document_cleanup(
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    doc_form: str,
    file_id: str | None,
) -> None:
    """Synchronous single-document cleanup entry point for durable intent workers."""
    run_documents_cleanup(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_ids=(document_id,),
        doc_form=doc_form,
        file_ids=(file_id,) if file_id else (),
    )


@shared_task(queue="dataset")
def clean_document_task(
    document_id: str,
    dataset_id: str,
    doc_form: str,
    file_id: str | None,
) -> None:
    """Clean one document after its row has been deleted."""
    logger.info(click.style(f"Start clean document when document deleted: {document_id}", fg="green"))
    start_at = time.perf_counter()
    with session_factory.create_session() as session:
        dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
    if dataset is None:
        raise ValueError("Dataset not found; use run_document_cleanup with the captured tenant scope")

    run_document_cleanup(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        doc_form=doc_form,
        file_id=file_id,
    )
    logger.info(
        click.style(
            f"Cleaned document when document deleted: {document_id} latency: {time.perf_counter() - start_at}",
            fg="green",
        )
    )

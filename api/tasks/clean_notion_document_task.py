import logging
import time
from functools import partial

import click
from celery import shared_task
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.db.session_factory import session_factory
from extensions.ext_storage import storage
from models.dataset import Dataset, DatasetMetadataBinding, Document
from repositories.knowledge.dataset_read_repository import get_dataset_doc_form
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    SQLAlchemyDocumentIndexingOutboxRepository,
)
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.indexing.adapters.cleanup import clean_document_for_reindex

logger = logging.getLogger(__name__)


class _NotionRemovalSupersededError(Exception):
    """The selected pages no longer match this removal generation."""


def _guard_notion_removal(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_ids: tuple[str, ...],
    indexing_request_id: str,
) -> None:
    current_ids = set(
        session.scalars(
            select(Document.id)
            .where(
                Document.id.in_(document_ids),
                Document.tenant_id == tenant_id,
                Document.dataset_id == dataset_id,
                Document.indexing_request_id == indexing_request_id,
            )
            .with_for_update()
        ).all()
    )
    if current_ids != set(document_ids):
        raise _NotionRemovalSupersededError()


@shared_task(queue="dataset")
def clean_notion_document_task(
    document_ids: list[str],
    dataset_id: str,
    indexing_request_id: str | None = None,
) -> None:
    """Clean only documents still owned by this Notion removal generation."""
    logger.info(click.style(f"Start clean document when import form notion document deleted: {dataset_id}", fg="green"))
    start_at = time.perf_counter()
    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=session_factory.get_session_maker())
    intent = None

    if indexing_request_id is not None:
        intent = repository.get_intent(indexing_request_id)
        if intent is None:
            return
        if intent.action != DocumentIndexingAction.REMOVE_NOTION or intent.dataset_id != dataset_id:
            logger.warning(
                "Ignoring mismatched Notion removal request, dataset_id=%s, indexing_request_id=%s",
                dataset_id,
                indexing_request_id,
            )
            return
        document_ids = list(intent.document_ids)

    with session_factory.create_session() as session:
        dataset_statement = select(Dataset).where(Dataset.id == dataset_id)
        if indexing_request_id is not None:
            assert intent is not None
            dataset_statement = dataset_statement.where(Dataset.tenant_id == intent.tenant_id)
        dataset = session.scalar(dataset_statement.limit(1))

        if not dataset:
            if indexing_request_id is not None:
                assert intent is not None
                repository.complete(intent)
                return
            raise Exception("Document has no dataset")
        tenant_id = dataset.tenant_id
        index_type = get_dataset_doc_form(dataset, session=session)
        if indexing_request_id is not None:
            document_ids = list(
                session.scalars(
                    select(Document.id).where(
                        Document.id.in_(document_ids),
                        Document.dataset_id == dataset_id,
                        Document.tenant_id == tenant_id,
                        Document.indexing_request_id == indexing_request_id,
                    )
                ).all()
            )

    if indexing_request_id is not None and not document_ids:
        assert intent is not None
        repository.complete(intent)
        return

    mutation_guard = (
        partial(
            _guard_notion_removal,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_ids=tuple(document_ids),
            indexing_request_id=indexing_request_id,
        )
        if indexing_request_id is not None
        else None
    )

    try:
        clean_document_for_reindex(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_ids=document_ids,
            doc_form=index_type,
            new_session=session_factory.create_session,
            uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=session_factory.get_session_maker()),
            delete_blob=storage.delete,
            logger=logger,
            mutation_guard=mutation_guard,
        )
    except _NotionRemovalSupersededError:
        with session_factory.create_session() as session:
            still_current = session.scalar(
                select(Document.id)
                .where(
                    Document.id.in_(document_ids),
                    Document.dataset_id == dataset_id,
                    Document.tenant_id == tenant_id,
                    Document.indexing_request_id == indexing_request_id,
                )
                .limit(1)
            )
        if still_current is None:
            assert intent is not None
            repository.complete(intent)
            return
        raise

    with session_factory.create_session() as session, session.begin():
        current_ids_statement = select(Document.id).where(
            Document.id.in_(document_ids),
            Document.dataset_id == dataset_id,
            Document.tenant_id == tenant_id,
        )
        if indexing_request_id is not None:
            current_ids_statement = current_ids_statement.where(Document.indexing_request_id == indexing_request_id)
        current_ids = tuple(session.scalars(current_ids_statement.with_for_update()).all())
        if current_ids:
            session.execute(
                delete(DatasetMetadataBinding).where(
                    DatasetMetadataBinding.tenant_id == tenant_id,
                    DatasetMetadataBinding.dataset_id == dataset_id,
                    DatasetMetadataBinding.document_id.in_(current_ids),
                )
            )
        session.execute(
            delete(Document).where(
                Document.id.in_(current_ids),
                Document.dataset_id == dataset_id,
                Document.tenant_id == tenant_id,
            )
        )

    end_at = time.perf_counter()
    logger.info(
        click.style(
            f"Clean document when import form notion document deleted end :: {dataset_id} latency: {end_at - start_at}",
            fg="green",
        )
    )
    if indexing_request_id is not None:
        assert intent is not None
        repository.complete(intent)

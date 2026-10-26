import logging
import time

import click
from celery import shared_task
from sqlalchemy import select, update

from core.db.session_factory import session_factory
from core.rag.index_processor.index_processor import IndexProcessorFactory
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset, DocumentSegment
from models.dataset import Document as DatasetDocument
from models.enums import IndexingStatus, SegmentStatus
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.child_vector_route_repository import SQLAlchemyChildVectorRouteStore
from repositories.knowledge.dataset_indexing_outbox_repository import lock_dataset_indexing_generation_shared
from repositories.knowledge.dataset_read_repository import (
    get_dataset_doc_form,
)

logger = logging.getLogger(__name__)


@shared_task(queue="dataset")
def create_segment_to_index_task(segment_id: str, keywords: list[str] | None = None):
    """
    Async create segment to index
    :param segment_id:
    :param keywords:
    Usage: create_segment_to_index_task.delay(segment_id)
    """
    logger.info(click.style(f"Start create segment to index: {segment_id}", fg="green"))
    start_at = time.perf_counter()

    with session_factory.create_session() as session:
        segment = session.scalar(select(DocumentSegment).where(DocumentSegment.id == segment_id).limit(1))
        if not segment:
            logger.info(click.style(f"Segment not found: {segment_id}", fg="red"))
            return

        dataset = session.scalar(
            select(Dataset)
            .where(Dataset.id == segment.dataset_id, Dataset.tenant_id == segment.tenant_id)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        if dataset is None:
            return
        dataset_generation_id = dataset.indexing_request_id
        dataset_document = session.scalar(
            select(DatasetDocument)
            .where(
                DatasetDocument.id == segment.document_id,
                DatasetDocument.dataset_id == dataset.id,
                DatasetDocument.tenant_id == dataset.tenant_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        segment = session.scalar(
            select(DocumentSegment)
            .where(
                DocumentSegment.id == segment_id,
                DocumentSegment.dataset_id == dataset.id,
                DocumentSegment.tenant_id == dataset.tenant_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if segment is None or segment.status != SegmentStatus.WAITING:
            return

        indexing_cache_key = f"segment_{segment.id}_indexing"

        try:
            # update segment status to indexing
            session.execute(
                update(DocumentSegment)
                .where(DocumentSegment.id == segment.id)
                .values(status=SegmentStatus.INDEXING, indexing_at=naive_utc_now())
            )
            document = Document(
                page_content=segment.content,
                metadata={
                    "doc_id": segment.index_node_id,
                    "doc_hash": segment.index_node_hash,
                    "document_id": segment.document_id,
                    "dataset_id": segment.dataset_id,
                },
            )

            if not dataset_document:
                logger.info(click.style(f"Segment {segment.id} has no document, pass.", fg="cyan"))
                return

            if (
                not dataset_document.enabled
                or dataset_document.archived
                or dataset_document.indexing_status != IndexingStatus.COMPLETED
            ):
                logger.info(click.style(f"Segment {segment.id} document status is invalid, pass.", fg="cyan"))
                return

            index_type = get_dataset_doc_form(dataset, session=session)
            index_processor = IndexProcessorFactory(
                index_type,
                attachment_vectors=SQLAlchemyAttachmentVectorStore(),
                child_vector_routes=SQLAlchemyChildVectorRouteStore(),
            ).init_index_processor()
            index_processor.load(dataset, [document], session=session)

            # update segment to completed
            session.execute(
                update(DocumentSegment)
                .where(DocumentSegment.id == segment.id)
                .values(status=SegmentStatus.COMPLETED, completed_at=naive_utc_now())
            )
            session.commit()

            end_at = time.perf_counter()
            logger.info(click.style(f"Segment created to index: {segment.id} latency: {end_at - start_at}", fg="green"))
        except Exception as e:
            logger.exception("create segment to index failed")
            session.rollback()
            current_dataset = lock_dataset_indexing_generation_shared(
                session, dataset_id=dataset.id, generation_id=dataset_generation_id
            )
            if current_dataset is None:
                return
            segment = session.scalar(
                select(DocumentSegment)
                .where(
                    DocumentSegment.id == segment_id,
                    DocumentSegment.dataset_id == current_dataset.id,
                    DocumentSegment.tenant_id == current_dataset.tenant_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if segment is None or segment.status != SegmentStatus.WAITING:
                return
            segment.enabled = False
            segment.disabled_at = naive_utc_now()
            segment.status = SegmentStatus.ERROR
            segment.error = str(e)
            session.commit()
        finally:
            redis_client.delete(indexing_cache_key)

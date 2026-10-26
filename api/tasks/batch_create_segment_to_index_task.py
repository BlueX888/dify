import logging
import tempfile
import time
import uuid
from pathlib import Path

import click
import pandas as pd
from celery import shared_task
from sqlalchemy import select

from core.credit_usage import CreditUsageCreatedBy
from core.db.session_factory import session_factory
from core.model_context import with_credit_usage_created_by
from core.model_manager import ModelManager
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from graphon.model_runtime.entities.model_entities import ModelType
from libs import helper
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset, Document
from models.enums import SegmentStatus
from models.model import UploadFile
from repositories.knowledge.segment_repository import SQLAlchemySegmentRepository
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.segments import SegmentCreateData

logger = logging.getLogger(__name__)


@shared_task(queue="dataset")
@with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
def batch_create_segment_to_index_task(
    job_id: str,
    upload_file_id: str,
    dataset_id: str,
    document_id: str,
    tenant_id: str,
    user_id: str,
):
    """
    Async batch create segment to index
    :param job_id:
    :param upload_file_id:
    :param dataset_id:
    :param document_id:
    :param tenant_id:
    :param user_id:

    Usage: batch_create_segment_to_index_task.delay(job_id, upload_file_id, dataset_id, document_id, tenant_id, user_id)
    """
    logger.info(click.style(f"Start batch create segment jobId: {job_id}", fg="green"))
    start_at = time.perf_counter()

    indexing_cache_key = f"segment_batch_import_{job_id}"

    try:
        with session_factory.create_session() as session:
            dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id))
            if not dataset:
                raise ValueError("Dataset not exist.")

            dataset_document = session.scalar(
                select(Document).where(
                    Document.id == document_id, Document.dataset_id == dataset_id, Document.tenant_id == tenant_id
                )
            )
            if not dataset_document:
                raise ValueError("Document not exist.")

            if (
                not dataset_document.enabled
                or dataset_document.archived
                or dataset_document.indexing_status != "completed"
            ):
                raise ValueError("Document is not available.")

            upload_file = session.scalar(
                select(UploadFile).where(UploadFile.id == upload_file_id, UploadFile.tenant_id == tenant_id)
            )
            if not upload_file:
                raise ValueError("UploadFile not found.")

            dataset_config = {
                "id": dataset.id,
                "indexing_technique": dataset.indexing_technique,
                "tenant_id": dataset.tenant_id,
                "embedding_model_provider": dataset.embedding_model_provider,
                "embedding_model": dataset.embedding_model,
            }

            document_form = dataset_document.doc_form

            upload_file_key = upload_file.key
    except Exception:
        logger.exception(
            "Segment batch import validation failed: job_id=%s dataset_id=%s document_id=%s tenant_id=%s",
            job_id,
            dataset_id,
            document_id,
            tenant_id,
        )
        redis_client.setex(indexing_cache_key, 600, "error")
        return

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            suffix = Path(upload_file_key).suffix
            file_path = f"{temp_dir}/{next(tempfile._get_candidate_names())}{suffix}"  # type: ignore
            storage.download(upload_file_key, file_path)

            df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
            content = []
            for _, row in df.iterrows():
                if document_form == IndexStructureType.QA_INDEX:
                    data = {"content": row.iloc[0], "answer": row.iloc[1]}
                else:
                    data = {"content": row.iloc[0]}
                content.append(data)
            if len(content) == 0:
                raise ValueError("The CSV file is empty.")

        segment_ids: list[str] = []
        embedding_model = None
        if dataset_config["indexing_technique"] == IndexTechniqueType.HIGH_QUALITY:
            model_manager = ModelManager.for_tenant(tenant_id=dataset_config["tenant_id"])
            embedding_model = model_manager.get_model_instance(
                tenant_id=dataset_config["tenant_id"],
                provider=dataset_config["embedding_model_provider"],
                model_type=ModelType.TEXT_EMBEDDING,
                model=dataset_config["embedding_model"],
            )

        if embedding_model:
            tokens_list = embedding_model.get_text_embedding_num_tokens(
                texts=[segment["content"] for segment in content]
            )
        else:
            tokens_list = [0] * len(content)

        now = naive_utc_now()
        persisted_segments: list[SegmentCreateData] = []
        for segment, tokens in zip(content, tokens_list, strict=True):
            segment_content = segment["content"]
            segment_id = str(uuid.uuid4())
            answer = segment.get("answer") if document_form == IndexStructureType.QA_INDEX else None
            persisted_segments.append(
                SegmentCreateData(
                    id=segment_id,
                    index_node_id=str(uuid.uuid4()),
                    index_node_hash=helper.generate_text_hash(segment_content),
                    content=segment_content,
                    answer=answer,
                    word_count=len(segment_content) + len(answer or ""),
                    tokens=tokens,
                    created_by=user_id,
                    indexing_at=now,
                    status=SegmentStatus.COMPLETED,
                    completed_at=now,
                )
            )
            segment_ids.append(segment_id)
        document_ref = DatasetRef(tenant_id, dataset_id).document(document_id)
        SQLAlchemySegmentRepository(session_factory=session_factory.get_session_maker()).create_segments(
            document_ref,
            persisted_segments,
        )

        # The registry imports this task while assembling the batch dispatcher.
        from extensions.ext_application_services import application_services

        application_services().knowledge.segments.mutations.index_segments(document_ref, segment_ids=segment_ids)
    except Exception:
        logger.exception(
            "Segment batch import failed: job_id=%s dataset_id=%s document_id=%s tenant_id=%s",
            job_id,
            dataset_id,
            document_id,
            tenant_id,
        )
        redis_client.setex(indexing_cache_key, 600, "error")
        raise
    else:
        redis_client.setex(indexing_cache_key, 600, "completed")
        end_at = time.perf_counter()
        logger.info(
            click.style(
                f"Segment batch created job: {job_id} latency: {end_at - start_at}",
                fg="green",
            )
        )

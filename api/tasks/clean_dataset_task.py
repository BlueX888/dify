import logging
import time
from collections.abc import Sequence

import click
from celery import shared_task
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.db.session_factory import session_factory
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.index_processor import IndexProcessorFactory
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from models import WorkflowType
from models.dataset import (
    AppDatasetJoin,
    Dataset,
    DatasetMetadata,
    DatasetMetadataBinding,
    DatasetPermission,
    DatasetProcessRule,
    DatasetQuery,
    Document,
    ExternalKnowledgeBindings,
    Pipeline,
)
from models.model import TagBinding
from models.workflow import Workflow
from repositories.knowledge.keyword_table_repository import delete_deleted_dataset_keyword_table
from repositories.knowledge.segment_cleanup_repository import (
    delete_document_segments_and_claim_uploads,
    delete_orphan_segments_and_claim_uploads,
)
from repositories.knowledge.upload_file_repository import (
    SQLAlchemyKnowledgeUploadRepository,
    UploadFileCleanupArtifact,
    get_document_source_upload_file_id,
)
from services.dataset_storage_cleanup_service import cleanup_dataset_crawl_cache
from services.knowledge.entities.segments import ChildVectorRoute
from services.knowledge.indexing.vector_route import vector_for_route
from services.knowledge.upload_file_cleanup import cleanup_claimed_upload_files
from tasks.refresh_billing_vector_space_task import schedule_billing_vector_space_refresh

logger = logging.getLogger(__name__)

_DELETE_BATCH_SIZE = 1000
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 30


def _cleanup_blobs(
    artifacts: Sequence[UploadFileCleanupArtifact],
    uploads: SQLAlchemyKnowledgeUploadRepository,
) -> None:
    """Delete storage objects after the transaction that created their durable outbox rows."""
    cleanup_claimed_upload_files(
        artifacts=artifacts,
        files=uploads,
        delete_blob=storage.delete,
        logger=logger,
    )


def _clean_index(dataset: Dataset, doc_form: str) -> ChildVectorRoute | None:
    """Remove the dataset index before deleting the relational retry evidence."""
    deleted_route = None
    with session_factory.create_session() as session:
        if dataset.indexing_technique == "high_quality":
            vector = Vector(dataset, session=session)
            vector_type, collection_name = vector.get_routing_target()
            deleted_route = ChildVectorRoute.capture(
                tenant_id=dataset.tenant_id,
                dataset_id=dataset.id,
                vector_type=vector_type,
                collection_name=collection_name,
                routing_index_struct=dataset.index_struct,
                collection_binding_id=dataset.collection_binding_id,
                indexing_technique=str(dataset.indexing_technique),
                doc_form=str(doc_form),
            )
        processor = IndexProcessorFactory(doc_form).init_index_processor()
        processor.clean(dataset, None, with_keywords=True, delete_child_chunks=True, session=session)
        session.commit()
    logger.info(click.style(f"Successfully cleaned vector database for dataset: {dataset.id}", fg="green"))
    return deleted_route


def _delete_keyword_table(*, tenant_id: str, dataset_id: str) -> None:
    """Remove keyword state using the trusted owner scope captured by the deletion outbox."""
    lock_name = f"keyword_indexing_lock_{dataset_id}"
    delete_deleted_dataset_keyword_table(
        session_factory=session_factory.get_session_maker(),
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        storage=storage,
        lock_factory=lambda: redis_client.lock(lock_name, timeout=600),
    )


def _delete_document_batch(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    documents: Sequence[Document],
) -> tuple[UploadFileCleanupArtifact, ...]:
    """Remove document ownership and create file cleanup records in the same transaction."""
    document_ids = tuple(document.id for document in documents)
    source_file_ids = tuple(
        file_id for document in documents if (file_id := get_document_source_upload_file_id(document)) is not None
    )
    session.execute(
        delete(DatasetMetadataBinding).where(
            DatasetMetadataBinding.tenant_id == tenant_id,
            DatasetMetadataBinding.dataset_id == dataset_id,
            DatasetMetadataBinding.document_id.in_(document_ids),
        )
    )
    session.execute(
        delete(Document).where(
            Document.tenant_id == tenant_id,
            Document.dataset_id == dataset_id,
            Document.id.in_(document_ids),
        )
    )
    return delete_document_segments_and_claim_uploads(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_ids=document_ids,
        candidate_upload_file_ids=source_file_ids,
    )


def _delete_documents(
    *,
    tenant_id: str,
    dataset_id: str,
    uploads: SQLAlchemyKnowledgeUploadRepository,
) -> None:
    """Delete documents in bounded transactions without losing their file ownership."""
    while True:
        with session_factory.create_session() as session, session.begin():
            documents = session.scalars(
                select(Document)
                .where(Document.tenant_id == tenant_id, Document.dataset_id == dataset_id)
                .order_by(Document.id)
                .limit(_DELETE_BATCH_SIZE)
                .with_for_update()
            ).all()
            if not documents:
                return
            artifacts = _delete_document_batch(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                documents=documents,
            )
        _cleanup_blobs(artifacts, uploads)


def _delete_orphan_segments_and_attachments(
    *,
    tenant_id: str,
    dataset_id: str,
    uploads: SQLAlchemyKnowledgeUploadRepository,
) -> None:
    """Remove dataset-owned rows left by an earlier partial cleanup."""
    with session_factory.create_session() as session, session.begin():
        artifacts = delete_orphan_segments_and_claim_uploads(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
        )
    _cleanup_blobs(artifacts, uploads)


def _delete_dataset_scoped_rows(*, tenant_id: str, dataset_id: str, pipeline_id: str | None) -> None:
    """Delete the remaining tenant-owned dataset tables in one transaction."""
    with session_factory.create_session() as session, session.begin():
        session.execute(delete(DatasetProcessRule).where(DatasetProcessRule.dataset_id == dataset_id))
        session.execute(delete(DatasetQuery).where(DatasetQuery.dataset_id == dataset_id))
        session.execute(delete(AppDatasetJoin).where(AppDatasetJoin.dataset_id == dataset_id))
        session.execute(
            delete(DatasetPermission).where(
                DatasetPermission.tenant_id == tenant_id,
                DatasetPermission.dataset_id == dataset_id,
            )
        )
        session.execute(
            delete(DatasetMetadata).where(
                DatasetMetadata.tenant_id == tenant_id,
                DatasetMetadata.dataset_id == dataset_id,
            )
        )
        session.execute(
            delete(DatasetMetadataBinding).where(
                DatasetMetadataBinding.tenant_id == tenant_id,
                DatasetMetadataBinding.dataset_id == dataset_id,
            )
        )
        session.execute(
            delete(ExternalKnowledgeBindings).where(
                ExternalKnowledgeBindings.tenant_id == tenant_id,
                ExternalKnowledgeBindings.dataset_id == dataset_id,
            )
        )
        session.execute(
            delete(TagBinding).where(
                TagBinding.tenant_id == tenant_id,
                TagBinding.target_id == dataset_id,
            )
        )
        if pipeline_id:
            session.execute(delete(Pipeline).where(Pipeline.id == pipeline_id, Pipeline.tenant_id == tenant_id))
            session.execute(
                delete(Workflow).where(
                    Workflow.tenant_id == tenant_id,
                    Workflow.app_id == pipeline_id,
                    Workflow.type == WorkflowType.RAG_PIPELINE,
                )
            )


def _require_dataset_deleted(*, tenant_id: str, dataset_id: str) -> None:
    """Refuse external cleanup until the owning deletion transaction commits."""
    with session_factory.create_session() as session:
        dataset_exists = session.scalar(
            select(Dataset.id).where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
        )
    if dataset_exists is not None:
        raise RuntimeError(f"Dataset still exists and cannot be cleaned: {tenant_id}:{dataset_id}")


def run_dataset_vector_cleanup(route: ChildVectorRoute) -> ChildVectorRoute:
    """Delete exactly one persisted physical destination and return its proof."""
    _require_dataset_deleted(tenant_id=route.tenant_id, dataset_id=route.dataset_id)
    # Resolve from immutable physical evidence with collection bindings disabled.
    # Live Dataset/binding state must never redirect a deletion before a later
    # proof check can notice the mismatch.
    vector_for_route(route).delete()
    logger.info(
        click.style(
            f"Successfully cleaned persisted vector route for dataset: {route.dataset_id}",
            fg="green",
        )
    )
    return route


def _prepare_dataset_cleanup(
    *,
    tenant_id: str,
    dataset_id: str,
    indexing_technique: str | None,
    index_struct: str | None,
    collection_binding_id: str | None,
    doc_form: str,
) -> ChildVectorRoute | None:
    """Verify committed deletion and remove external dataset state."""
    _require_dataset_deleted(tenant_id=tenant_id, dataset_id=dataset_id)
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        indexing_technique=indexing_technique,
        index_struct=index_struct,
        collection_binding_id=collection_binding_id,
    )
    cleanup_dataset_crawl_cache(tenant_id, dataset_id)
    deleted_route = _clean_index(dataset, doc_form)
    _delete_keyword_table(tenant_id=tenant_id, dataset_id=dataset_id)
    schedule_billing_vector_space_refresh(tenant_id)
    return deleted_route


def _prepare_dataset_relational_cleanup(*, tenant_id: str, dataset_id: str) -> None:
    """Remove non-vector state after vector disposition was durably handed off."""
    _require_dataset_deleted(tenant_id=tenant_id, dataset_id=dataset_id)
    cleanup_dataset_crawl_cache(tenant_id, dataset_id)
    _delete_keyword_table(tenant_id=tenant_id, dataset_id=dataset_id)
    schedule_billing_vector_space_refresh(tenant_id)


def _delete_dataset_relational_state(*, tenant_id: str, dataset_id: str, pipeline_id: str | None) -> None:
    uploads = SQLAlchemyKnowledgeUploadRepository(session_factory=session_factory.get_session_maker())
    _delete_documents(tenant_id=tenant_id, dataset_id=dataset_id, uploads=uploads)
    _delete_orphan_segments_and_attachments(tenant_id=tenant_id, dataset_id=dataset_id, uploads=uploads)
    _delete_dataset_scoped_rows(tenant_id=tenant_id, dataset_id=dataset_id, pipeline_id=pipeline_id)


def run_dataset_cleanup(
    *,
    tenant_id: str,
    dataset_id: str,
    indexing_technique: str | None,
    index_struct: str | None,
    collection_binding_id: str | None,
    doc_form: str,
    pipeline_id: str | None,
) -> ChildVectorRoute | None:
    """Execute one durable cleanup intent for a dataset that has committed deletion."""
    deleted_route = _prepare_dataset_cleanup(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        indexing_technique=indexing_technique,
        index_struct=index_struct,
        collection_binding_id=collection_binding_id,
        doc_form=doc_form,
    )
    _delete_dataset_relational_state(tenant_id=tenant_id, dataset_id=dataset_id, pipeline_id=pipeline_id)
    return deleted_route


def run_dataset_relational_cleanup(*, tenant_id: str, dataset_id: str, pipeline_id: str | None) -> None:
    """Clean a deleted dataset whose exact physical vector routes have separate intents."""
    _prepare_dataset_relational_cleanup(tenant_id=tenant_id, dataset_id=dataset_id)
    _delete_dataset_relational_state(tenant_id=tenant_id, dataset_id=dataset_id, pipeline_id=pipeline_id)


@shared_task(queue="dataset", bind=True, max_retries=_MAX_RETRIES, default_retry_delay=_RETRY_DELAY_SECONDS)
def clean_dataset_task(
    self,
    dataset_id: str,
    tenant_id: str,
    indexing_technique: str | None,
    index_struct: str | None,
    collection_binding_id: str | None,
    doc_form: str,
    pipeline_id: str | None = None,
    *,
    index_cleanup_done: bool = False,
) -> None:
    """Compatibility task for cleanup requests that already captured the complete owner scope."""
    logger.info(click.style(f"Start clean dataset when dataset deleted: {dataset_id}", fg="green"))
    start_at = time.perf_counter()
    try:
        if not index_cleanup_done:
            _prepare_dataset_cleanup(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                indexing_technique=indexing_technique,
                index_struct=index_struct,
                collection_binding_id=collection_binding_id,
                doc_form=doc_form,
            )
            index_cleanup_done = True
        _delete_dataset_relational_state(tenant_id=tenant_id, dataset_id=dataset_id, pipeline_id=pipeline_id)
    except Exception as exc:
        if self.request.retries >= _MAX_RETRIES:
            raise
        raise self.retry(
            exc=exc,
            args=(),
            kwargs={
                "dataset_id": dataset_id,
                "tenant_id": tenant_id,
                "indexing_technique": indexing_technique,
                "index_struct": index_struct,
                "collection_binding_id": collection_binding_id,
                "doc_form": doc_form,
                "pipeline_id": pipeline_id,
                "index_cleanup_done": index_cleanup_done,
            },
            countdown=_RETRY_DELAY_SECONDS * (2**self.request.retries),
        ) from exc
    logger.info(
        click.style(
            f"Cleaned dataset when dataset deleted: {dataset_id} latency: {time.perf_counter() - start_at}",
            fg="green",
        )
    )

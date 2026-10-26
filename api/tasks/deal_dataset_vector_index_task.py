import logging
import time

import click
from celery import shared_task
from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from core.db.session_factory import session_factory
from core.rag.datasource.keyword.jieba.jieba import Jieba
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.constant.doc_type import DocType
from core.rag.index_processor.constant.index_type import IndexStructureType
from core.rag.index_processor.index_processor import IndexProcessorFactory
from core.rag.models.document import AttachmentDocument, ChildDocument, Document
from models.dataset import Dataset, DocumentSegment
from models.dataset import Document as DatasetDocument
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.child_vector_route_repository import SQLAlchemyChildVectorRouteStore
from repositories.knowledge.dataset_indexing_outbox_repository import (
    DatasetIndexingWritersActiveError,
    SQLAlchemyDatasetIndexingOutboxRepository,
    lock_dataset_indexing_execution,
    lock_dataset_indexing_generation,
)
from repositories.knowledge.dataset_read_repository import get_dataset_doc_form, get_segment_child_chunks
from repositories.knowledge.segment_read_adapter import get_segment_attachments
from services.knowledge.entities.datasets import DatasetIndexingRepresentation
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.vector_route import capture_vector_route, vector_for_route

logger = logging.getLogger(__name__)


def _repository() -> SQLAlchemyDatasetIndexingOutboxRepository:
    return SQLAlchemyDatasetIndexingOutboxRepository(session_factory=session_factory.get_session_maker())


def _lock_owner(
    session: Session,
    *,
    dataset_id: str,
    indexing_request_id: str | None,
    indexing_intent_id: str | None,
    indexing_claim_id: str | None,
    tenant_id: str | None,
    require_quiescent: bool = False,
):
    execution_values = (indexing_intent_id, indexing_claim_id, tenant_id)
    if any(value is not None for value in execution_values):
        if indexing_request_id is None or any(value is None for value in execution_values):
            raise ValueError("A durable rebuild requires generation, intent, claim, and tenant ownership")
        assert indexing_intent_id is not None
        assert indexing_claim_id is not None
        assert tenant_id is not None
        return lock_dataset_indexing_execution(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            generation_id=indexing_request_id,
            intent_id=indexing_intent_id,
            claim_id=indexing_claim_id,
            require_quiescent=require_quiescent,
        )
    return lock_dataset_indexing_generation(
        session,
        dataset_id=dataset_id,
        generation_id=indexing_request_id,
    )


def _bridge_legacy_message(dataset_id: str, indexing_request_id: str | None) -> bool:
    """Redirect an old queued task to its current durable execution intent."""
    intent_id = _repository().find_current_intent_id(
        dataset_id=dataset_id,
        generation_id=indexing_request_id,
    )
    if intent_id is None:
        return False
    from tasks.dataset_indexing_outbox_task import enqueue_dataset_indexing_intent

    enqueue_dataset_indexing_intent(intent_id)
    return True


def _dataset_for_representation(
    dataset: Dataset,
    representation: DatasetIndexingRepresentation,
    *,
    route: VectorRoute | None = None,
) -> Dataset:
    """Build a detached cleanup view for the superseded physical representation."""
    return Dataset(
        id=dataset.id,
        tenant_id=dataset.tenant_id,
        name=dataset.name,
        created_by=dataset.created_by,
        provider=dataset.provider,
        permission=dataset.permission,
        indexing_technique=route.indexing_technique if route is not None else representation.indexing_technique,
        index_struct=route.index_struct if route is not None else representation.index_struct,
        embedding_model=representation.embedding_model,
        embedding_model_provider=representation.embedding_model_provider,
        keyword_number=representation.keyword_number,
        # An explicit physical route is authoritative. In particular, Qdrant
        # resolves a binding before index_struct and would otherwise target a
        # different collection than the durable cleanup receipt.
        collection_binding_id=None if route is not None else representation.collection_binding_id,
        is_multimodal=representation.is_multimodal,
        summary_index_setting=representation.summary_index_setting,
    )


def _load_document_vectors(session: Session, dataset, dataset_document, index_processor) -> None:
    segments = session.scalars(
        select(DocumentSegment)
        .where(
            DocumentSegment.document_id == dataset_document.id,
            DocumentSegment.enabled.is_(True),
        )
        .order_by(DocumentSegment.position.asc())
    ).all()
    if not segments:
        return
    documents: list[Document] = []
    multimodal_documents: list[AttachmentDocument] = []
    for segment in segments:
        document = Document(
            page_content=segment.content,
            metadata={
                "doc_id": segment.index_node_id,
                "doc_hash": segment.index_node_hash,
                "document_id": segment.document_id,
                "dataset_id": segment.dataset_id,
            },
        )
        if dataset_document.doc_form == IndexStructureType.PARENT_CHILD_INDEX:
            child_chunks = get_segment_child_chunks(segment, session=session)
            if child_chunks:
                document.children = [
                    ChildDocument(
                        page_content=child_chunk.content,
                        metadata={
                            "doc_id": child_chunk.index_node_id,
                            "doc_hash": child_chunk.index_node_hash,
                            "document_id": segment.document_id,
                            "dataset_id": segment.dataset_id,
                        },
                    )
                    for child_chunk in child_chunks
                ]
        if dataset.is_multimodal:
            for attachment in get_segment_attachments(segment, session=session):
                multimodal_documents.append(
                    AttachmentDocument(
                        page_content=attachment["name"],
                        metadata={
                            "doc_id": attachment["id"],
                            "doc_hash": "",
                            "document_id": segment.document_id,
                            "dataset_id": segment.dataset_id,
                            "doc_type": DocType.IMAGE,
                        },
                    )
                )
        documents.append(document)
    if dataset.indexing_technique == "economy":
        Jieba(dataset).add_texts(documents, session)
    else:
        index_processor.load(
            dataset,
            documents,
            multimodal_documents=multimodal_documents,
            with_keywords=False,
            session=session,
        )


@shared_task(queue="dataset")
def deal_dataset_vector_index_task(
    dataset_id: str,
    action: str,
    indexing_request_id: str | None = None,
    *,
    indexing_intent_id: str | None = None,
    indexing_claim_id: str | None = None,
    tenant_id: str | None = None,
    source_representation: str | None = None,
    target_representation: str | None = None,
) -> bool:
    """Rebuild one dataset while fencing every external mutation by execution owner."""
    if action not in {"add", "remove", "update"}:
        raise ValueError(f"Unsupported dataset vector indexing action: {action}")
    if indexing_intent_id is None and indexing_claim_id is None and tenant_id is None:
        if _bridge_legacy_message(dataset_id, indexing_request_id):
            return False
        logger.info(
            "Skipping legacy dataset rebuild without a current execution intent, dataset_id=%s, request_id=%s",
            dataset_id,
            indexing_request_id,
        )
        return False

    if source_representation is None or target_representation is None:
        raise ValueError("Durable dataset rebuild requires source and target representations")
    source = DatasetIndexingRepresentation.model_validate_json(source_representation)
    target = DatasetIndexingRepresentation.model_validate_json(target_representation)
    assert indexing_request_id is not None
    assert indexing_intent_id is not None
    assert indexing_claim_id is not None

    logger.info(click.style(f"Start deal dataset vector index: {dataset_id}", fg="green"))
    start_at = time.perf_counter()
    repository = _repository()
    assert tenant_id is not None

    # Resolve and durably record the physical target before any cleanup or
    # publication. A superseding generation can then reclaim a partial write,
    # while an intent that never starts has no false target receipt.
    with session_factory.create_session() as session:
        try:
            dataset = _lock_owner(
                session,
                dataset_id=dataset_id,
                indexing_request_id=indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
                tenant_id=tenant_id,
                require_quiescent=True,
            )
        except DatasetIndexingWritersActiveError:
            # The quiescence check may have fenced expired or superseded
            # document owners. Commit those restart handoffs before releasing
            # Dataset X; the durable rebuild claim is released by its caller.
            session.commit()
            raise
        if dataset is None:
            logger.info("Skipping unowned dataset vector rebuild, dataset_id=%s", dataset_id)
            return False
        if (
            str(dataset.indexing_technique) != target.indexing_technique
            or dataset.collection_binding_id != target.collection_binding_id
            or dataset.embedding_model != target.embedding_model
            or dataset.embedding_model_provider != target.embedding_model_provider
            or dataset.keyword_number != target.keyword_number
            or bool(dataset.is_multimodal) != target.is_multimodal
        ):
            logger.info("Skipping stale target representation, dataset_id=%s", dataset_id)
            return False
        if target.indexing_technique == "high_quality":
            target_doc_form = (
                target.doc_form or get_dataset_doc_form(dataset, session=session) or IndexStructureType.PARAGRAPH_INDEX
            )
            target_route = capture_vector_route(
                dataset,
                Vector(dataset, session=session),
                doc_form=target_doc_form,
            )
            target = repository.record_target_route_locked(
                session,
                intent_id=indexing_intent_id,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                generation_id=indexing_request_id,
                claim_id=indexing_claim_id,
                route=target_route,
            )
        repository.refresh_locked_claim(session, indexing_intent_id, claim_id=indexing_claim_id)
        session.commit()

    with session_factory.create_session() as session:
        try:
            dataset = _lock_owner(
                session,
                dataset_id=dataset_id,
                indexing_request_id=indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
                tenant_id=tenant_id,
                require_quiescent=True,
            )
        except DatasetIndexingWritersActiveError:
            session.commit()
            raise
        if dataset is None:
            logger.info("Skipping superseded dataset vector rebuild, dataset_id=%s", dataset_id)
            return False
        if target.indexing_technique == "high_quality" and target.vector_route is None:
            raise RuntimeError("High-quality dataset rebuild has no resolved target route")

        takeover_condition = or_(
            DatasetDocument.indexing_status == "completed",
            and_(
                DatasetDocument.dataset_indexing_request_id.is_not(None),
                DatasetDocument.indexing_status.in_(("indexing", "error")),
                DatasetDocument.indexing_request_id.is_(None),
                DatasetDocument.indexing_claim_id.is_(None),
            ),
        )
        session.execute(
            update(DatasetDocument)
            .where(
                DatasetDocument.dataset_id == dataset_id,
                DatasetDocument.dataset_indexing_request_id.is_not(None),
                DatasetDocument.indexing_request_id.is_(None),
                DatasetDocument.indexing_claim_id.is_(None),
                or_(DatasetDocument.enabled.is_(False), DatasetDocument.archived.is_(True)),
            )
            .values(indexing_status="completed", dataset_indexing_request_id=None, error=None)
        )
        session.execute(
            update(DatasetDocument)
            .where(
                DatasetDocument.dataset_id == dataset_id,
                DatasetDocument.enabled.is_(True),
                DatasetDocument.archived.is_(False),
                DatasetDocument.indexing_request_id.is_(None),
                DatasetDocument.indexing_claim_id.is_(None),
                takeover_condition,
            )
            .values(indexing_status="indexing", dataset_indexing_request_id=indexing_request_id, error=None)
        )
        dataset_documents = tuple(
            session.scalars(
                select(DatasetDocument)
                .where(
                    DatasetDocument.dataset_id == dataset_id,
                    DatasetDocument.dataset_indexing_request_id == indexing_request_id,
                    DatasetDocument.indexing_status == "indexing",
                    DatasetDocument.enabled.is_(True),
                    DatasetDocument.archived.is_(False),
                    DatasetDocument.indexing_request_id.is_(None),
                    DatasetDocument.indexing_claim_id.is_(None),
                )
                .order_by(DatasetDocument.position.asc())
                .with_for_update()
            ).all()
        )

        if source.indexing_technique == "economy":
            Jieba(_dataset_for_representation(dataset, source)).delete(session=session)
        cleanup_routes = {
            route.key: route
            for route in (
                *source.cleanup_routes,
                *((source.vector_route,) if source.vector_route is not None else ()),
                *target.cleanup_routes,
            )
        }
        if cleanup_routes:
            for route in cleanup_routes.values():
                if route.tenant_id != dataset.tenant_id or route.dataset_id != dataset.id:
                    raise ValueError("Cleanup vector route belongs to another dataset")
                # Dataset-wide cleanup is a physical collection operation.
                # ``doc_form`` describes the producer (body, child, attachment,
                # or summary) and must not be reinterpreted as a processor type.
                vector_for_route(route).delete()

        target_doc_form = (
            target.vector_route.doc_form
            if target.vector_route is not None
            else target.doc_form or get_dataset_doc_form(dataset, session=session) or IndexStructureType.PARAGRAPH_INDEX
        )
        target_processor = IndexProcessorFactory(
            target_doc_form,
            attachment_vectors=SQLAlchemyAttachmentVectorStore(),
            child_vector_routes=SQLAlchemyChildVectorRouteStore(),
        ).init_index_processor()
        target_dataset = (
            _dataset_for_representation(dataset, target, route=target.vector_route)
            if target.vector_route is not None
            else dataset
        )
        for dataset_document in dataset_documents:
            _load_document_vectors(session, target_dataset, dataset_document, target_processor)
            dataset_document.indexing_status = "completed"
            dataset_document.dataset_indexing_request_id = None
            dataset_document.error = None

        repository.record_applied_representation_locked(
            session,
            dataset,
            intent_id=indexing_intent_id,
            generation_id=indexing_request_id,
            claim_id=indexing_claim_id,
            target=target,
        )
        repository.refresh_locked_claim(session, indexing_intent_id, claim_id=indexing_claim_id)
        session.commit()

    logger.info(
        click.style(
            f"Deal dataset vector index: {dataset_id} latency: {time.perf_counter() - start_at}",
            fg="green",
        )
    )
    return True

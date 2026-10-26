"""Delete document indexes and their relational ownership in bounded phases."""

import logging
from collections.abc import Callable, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from configs import dify_config
from core.rag.datasource.keyword.jieba.jieba import Jieba
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from extensions.ext_redis import redis_client
from models.dataset import ChildChunk, Dataset, DocumentSegment, DocumentSegmentSummary, SegmentAttachmentBinding
from repositories.knowledge.attachment_vector_repository import delete_unreferenced_attachment_vectors
from repositories.knowledge.dataset_read_repository import get_dataset_keyword_table
from repositories.knowledge.keyword_table_repository import persist_keyword_table
from repositories.knowledge.segment_cleanup_repository import (
    delete_document_segments_and_claim_uploads,
    lock_segment_attachment_cleanup_dataset,
)
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.upload_file_cleanup import (
    cleanup_claimed_upload_files,
)


def handoff_document_attachment_vectors(
    session: Session,
    *,
    dataset: Dataset,
    document_ids: Sequence[str],
) -> tuple[str, ...]:
    """Persist post-commit attachment cleanup in the relational detach transaction."""
    if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
        return ()
    has_attachments = (
        session.scalar(
            select(SegmentAttachmentBinding.id)
            .where(
                SegmentAttachmentBinding.tenant_id == dataset.tenant_id,
                SegmentAttachmentBinding.dataset_id == dataset.id,
                SegmentAttachmentBinding.document_id.in_(document_ids),
            )
            .limit(1)
        )
        is not None
    )
    if not has_attachments:
        return ()
    vector_type = Vector.resolve_vector_type(dataset, session=session)
    attachment_vector = Vector(dataset, session=session, vector_type=vector_type)
    actual_vector_type, collection_name = attachment_vector.get_routing_target()
    attachment_route = VectorRoute.capture(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        vector_type=actual_vector_type,
        collection_name=collection_name,
        routing_index_struct=dataset.index_struct,
        collection_binding_id=dataset.collection_binding_id,
        indexing_technique=str(dataset.indexing_technique),
        doc_form="attachment",
    )
    return delete_unreferenced_attachment_vectors(
        session,
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_ids=document_ids,
        route=attachment_route,
    )


def clean_document_indexes(
    *,
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    doc_form: str | None,
    new_session: Callable[[], Session],
    mutation_guard: Callable[[Session], None] | None = None,
) -> str | None:
    """Clean indexes even when the owning document rows have already been deleted.

    Snapshot body rows before external I/O. Attachment vectors are handed off
    later in the same transaction that removes their bindings, so this phase
    must not delete them. Empty document selections never mean deleting the
    entire dataset index.
    """
    if not document_ids:
        return None
    with new_session() as session, session.begin():
        if mutation_guard is not None:
            mutation_guard(session)
        dataset = session.scalar(
            select(Dataset).where(
                Dataset.id == dataset_id,
                Dataset.tenant_id == tenant_id,
            )
        )
        if dataset is None:
            return None
        segments = session.scalars(
            select(DocumentSegment).where(
                DocumentSegment.tenant_id == tenant_id,
                DocumentSegment.dataset_id == dataset_id,
                DocumentSegment.document_id.in_(document_ids),
            )
        ).all()
        summaries = session.scalars(
            select(DocumentSegmentSummary).where(
                DocumentSegmentSummary.dataset_id == dataset_id,
                DocumentSegmentSummary.document_id.in_(document_ids),
            )
        ).all()
        children = session.scalars(
            select(ChildChunk).where(
                ChildChunk.tenant_id == tenant_id,
                ChildChunk.dataset_id == dataset_id,
                ChildChunk.document_id.in_(document_ids),
            )
        ).all()
        summary_node_ids = [summary.summary_index_node_id for summary in summaries if summary.summary_index_node_id]
        body_node_ids = (
            [child.index_node_id for child in children if child.index_node_id]
            if doc_form == IndexStructureType.PARENT_CHILD_INDEX
            else [segment.index_node_id for segment in segments if segment.index_node_id]
        )
        high_quality = dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY
        body_cleanup_node_ids = list(dict.fromkeys([*summary_node_ids, *body_node_ids]))
        vector_type = (
            Vector.resolve_vector_type(dataset, session=session) if high_quality and body_cleanup_node_ids else None
        )
        session.expunge(dataset)

    if high_quality:
        vector = Vector(dataset, session=None, vector_type=vector_type) if vector_type is not None else None
        if body_cleanup_node_ids:
            if mutation_guard is not None:
                with new_session() as session, session.begin():
                    mutation_guard(session)
            assert vector is not None
            vector.delete_by_ids(body_cleanup_node_ids)
    elif body_node_ids:
        # The keyword adapter owns its lock and storage I/O; callbacks only do SQL.
        def read() -> tuple[str, str | None]:
            with new_session() as session:
                row = get_dataset_keyword_table(dataset, session=session)
                return (
                    (row.data_source_type, row.keyword_table) if row else (dify_config.KEYWORD_DATA_SOURCE_TYPE, None)
                )

        def write(storage_type: str, data: str, keywords: Mapping[str, Sequence[str]]) -> None:
            with new_session() as session, session.begin():
                if mutation_guard is not None:
                    mutation_guard(session)
                persist_keyword_table(
                    session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    storage_type=storage_type,
                    data=data,
                    keywords=keywords,
                )

        def before_write() -> None:
            if mutation_guard is None:
                return
            with new_session() as session, session.begin():
                mutation_guard(session)

        Jieba(dataset).update_texts(
            [],
            read=read,
            write=write,
            lock=redis_client.lock(f"keyword_indexing_lock_{dataset_id}", timeout=600),
            before_write=before_write if mutation_guard is not None else None,
            cleanup_session_factory=new_session,
            deleted_ids=body_node_ids,
        )

    return tenant_id


def clean_document_for_reindex(
    *,
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    doc_form: str | None,
    new_session: Callable[[], Session],
    uploads: SQLAlchemyKnowledgeUploadRepository,
    delete_blob: Callable[[str], None],
    logger: logging.Logger,
    mutation_guard: Callable[[Session], None] | None = None,
) -> str | None:
    """Clean external indexes, then atomically remove rows and claim blobs."""
    cleaned_tenant_id = clean_document_indexes(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_ids=document_ids,
        doc_form=doc_form,
        new_session=new_session,
        mutation_guard=mutation_guard,
    )
    if cleaned_tenant_id is None:
        return None

    def delete_once():
        with new_session() as session, session.begin():
            lock_segment_attachment_cleanup_dataset(session, tenant_id=tenant_id, dataset_id=dataset_id)
            if mutation_guard is not None:
                mutation_guard(session)
            dataset = session.scalar(
                select(Dataset)
                .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
                .execution_options(populate_existing=True)
            )
            if dataset is not None:
                handoff_document_attachment_vectors(session, dataset=dataset, document_ids=document_ids)
            return delete_document_segments_and_claim_uploads(
                session,
                tenant_id=cleaned_tenant_id,
                dataset_id=dataset_id,
                document_ids=document_ids,
            )

    artifacts = delete_once()
    cleanup_claimed_upload_files(
        artifacts=artifacts,
        files=uploads,
        delete_blob=delete_blob,
        logger=logger,
    )
    return cleaned_tenant_id

"""Adapt legacy processors to committed document-indexing phases."""

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from typing import cast

from flask import Flask, current_app
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.exceptions import HTTPException
from werkzeug.local import LocalProxy

from configs import dify_config
from core.errors.error import (
    AppInvokeQuotaExceededError,
    InvokeRateLimitError,
    LLMError,
    ModelCurrentlyNotSupportError,
    ProviderTokenNotInitError,
    QuotaExceededError,
)
from core.model_manager import ModelManager
from core.plugin.impl.exc import PluginDaemonError
from core.rag.datasource.keyword.jieba.jieba import Jieba
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.embedding.token_counter import calculate_segment_token_counts
from core.rag.extractor.entity.extract_setting import CreatedUploadArtifact, ExtractionResult
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.index_processor.index_processor import IndexProcessorFactory
from core.rag.models.document import Document
from extensions.application_services.data_sources import build_data_source_credentials, build_website_service
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from extensions.otel import propagate_context
from graphon.model_runtime.entities.model_entities import ModelType
from libs import helper
from models.dataset import (
    ChildChunk,
    Dataset,
    DocumentSegment,
    DocumentSegmentSummary,
    SegmentAttachmentBinding,
)
from models.dataset import Document as DatasetDocument
from models.enums import DataSourceType, IndexingStatus
from repositories.data_source.website_source_repository import SQLAlchemyWebsiteSourceRepository
from repositories.knowledge.attachment_vector_repository import (
    SQLAlchemyAttachmentVectorStore,
    delete_unreferenced_attachment_vectors,
    publish_bound_attachment_vectors,
)
from repositories.knowledge.dataset_indexing_outbox_repository import (
    SQLAlchemyDatasetIndexingOutboxRepository,
    lock_dataset_indexing_generation_shared,
)
from repositories.knowledge.dataset_read_repository import get_dataset_keyword_table
from repositories.knowledge.document_repository import (
    SQLAlchemyDocumentRepository,
    refresh_locked_indexing_claim,
    require_indexing_document,
)
from repositories.knowledge.keyword_table_repository import persist_keyword_table
from repositories.knowledge.segment_cleanup_repository import (
    delete_document_segments_and_claim_uploads,
    lock_segment_attachment_cleanup_dataset,
)
from repositories.knowledge.segment_repository import SQLAlchemySegmentRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.entities.resource_scope import DocumentRef
from services.knowledge.entities.segments import ChildVectorRoute
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.adapters.cleanup import clean_document_for_reindex
from services.knowledge.indexing.adapters.sources import (
    CompositeStoredSourceResolver,
    FileSourceAdapter,
    NotionSourceResolver,
    WebsiteSourceAdapter,
)
from services.knowledge.indexing.errors import (
    DatasetIndexingGenerationLostError,
    DocumentIsDeletedPausedError,
    DocumentIsPausedError,
)
from services.knowledge.indexing.execution import DocumentIndexingService, IndexingDocument, IndexingTransformResult
from services.knowledge.indexing.vector_route import capture_vector_route
from services.knowledge.upload_file_cleanup import (
    cleanup_claimed_upload_files,
    cleanup_created_upload_artifacts,
    cleanup_unreferenced_upload_files,
    maintain_created_upload_artifact_writers,
)
from services.vector_space_admission_service import VectorSpaceAdmissionService

logger = logging.getLogger(__name__)


class IndexingExecutionAdapter:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        documents: SQLAlchemyDocumentRepository,
        segments: SQLAlchemySegmentRepository,
        uploads: SQLAlchemyKnowledgeUploadRepository,
        attachment_vectors: SQLAlchemyAttachmentVectorStore,
        sources: CompositeStoredSourceResolver,
    ) -> None:
        self._session_factory = session_factory
        self._documents = documents
        self._segments = segments
        self._uploads = uploads
        self._attachment_vectors = attachment_vectors
        self._sources = sources

    def _require_generation(
        self,
        session: Session,
        document: IndexingDocument,
        *,
        lock_document: bool,
    ) -> tuple[Dataset, DatasetDocument]:
        dataset = lock_dataset_indexing_generation_shared(
            session,
            dataset_id=document.ref.dataset.dataset_id,
            generation_id=document.dataset_generation_id,
        )
        if dataset is None or dataset.tenant_id != document.ref.dataset.tenant_id:
            raise DatasetIndexingGenerationLostError(document.ref.dataset.dataset_id)
        persisted = require_indexing_document(
            session,
            document.ref,
            claim_id=document.claim_id,
            lock=lock_document,
        )
        if persisted.indexing_dataset_generation_id != document.dataset_generation_id:
            raise DatasetIndexingGenerationLostError(document.ref.dataset.dataset_id)
        return dataset, persisted

    def _record_vector_route_before_write(self, document: IndexingDocument) -> VectorRoute | None:
        """Commit physical-route evidence before entering the external mutation guard."""
        if document.claim_id is None or document.dataset_generation_id is None:
            return None
        with self._session_factory.begin() as session:
            dataset, persisted = self._require_generation(session, document, lock_document=True)
            if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
                return None
            vector_type = Vector.resolve_vector_type(dataset, session=session)
            vector = Vector(dataset, session=None, vector_type=vector_type)
            route = capture_vector_route(dataset, vector, doc_form=str(persisted.doc_form))
            SQLAlchemyDatasetIndexingOutboxRepository.record_generation_writer_route_locked(
                session,
                dataset=dataset,
                generation_id=document.dataset_generation_id,
                route=route,
            )
            return route

    def describe_error(self, error: Exception) -> str:
        """Preserve provider and extractor descriptions without their transport prefixes."""
        if isinstance(
            error,
            (
                LLMError,
                ProviderTokenNotInitError,
                QuotaExceededError,
                AppInvokeQuotaExceededError,
                ModelCurrentlyNotSupportError,
                InvokeRateLimitError,
                PluginDaemonError,
                HTTPException,
            ),
        ):
            return str(error.description)
        return str(error)

    def check_paused(self, ref: DocumentRef) -> None:
        if redis_client.get(f"document_{ref.document_id}_is_paused"):
            raise DocumentIsPausedError(f"Document paused, document id: {ref.document_id}")

    def clean_for_reindex(self, document: IndexingDocument) -> None:
        def guard(session: Session) -> None:
            self._require_generation(session, document, lock_document=True)

        clean_document_for_reindex(
            tenant_id=document.ref.dataset.tenant_id,
            dataset_id=document.ref.dataset.dataset_id,
            document_ids=(document.ref.document_id,),
            doc_form=document.doc_form,
            new_session=self._session_factory,
            uploads=self._uploads,
            delete_blob=storage.delete,
            logger=logger,
            mutation_guard=guard,
        )

    def extract(self, document: IndexingDocument) -> ExtractionResult:
        if not document.processing_rule:
            raise ValueError("no process rule found")
        setting = self._sources.resolve_for_indexing(
            document.source,
            attach_document_for_updates=document.claim_id is None,
        )
        processor = IndexProcessorFactory(document.doc_form).init_index_processor()
        # Extractors persist downloaded images in an independent session.
        # Source versions are published with successful document completion.
        extraction: ExtractionResult | None = None
        created_artifacts: dict[str, CreatedUploadArtifact] = {}
        writer_ids: dict[str, str] = {}

        def begin_artifact_write(artifact: CreatedUploadArtifact) -> None:
            created_artifacts[artifact.id] = artifact
            writer_ids[artifact.id] = self._uploads.begin_created_upload_artifact_write(
                workspace_id=document.ref.dataset.tenant_id,
                artifact=artifact,
            )

        try:
            with (
                maintain_created_upload_artifact_writers(
                    workspace_id=document.ref.dataset.tenant_id,
                    upload_file_ids=lambda: tuple(writer_ids),
                    files=self._uploads,
                    logger=logger,
                ),
                self._session_factory() as session,
            ):
                extraction = processor.extract(
                    setting,
                    process_rule_mode=document.processing_rule["mode"],
                    session=session,
                    on_upload_artifact_created=begin_artifact_write,
                )
                created_artifacts.update((artifact.id, artifact) for artifact in extraction.created_upload_artifacts)
                self._uploads.finish_created_upload_artifact_writes_in_session(
                    session,
                    workspace_id=document.ref.dataset.tenant_id,
                    writer_ids=writer_ids,
                )
                session.commit()
            for text in extraction.documents:
                text.metadata.update(document_id=document.ref.document_id, dataset_id=document.ref.dataset.dataset_id)
            return extraction
        except Exception:
            self._cleanup_created_upload_artifacts_after_failure(document, tuple(created_artifacts.values()))
            raise

    def maintain_created_upload_file_writers(
        self,
        document: IndexingDocument,
        file_ids: Sequence[str],
    ) -> AbstractContextManager[None]:
        return maintain_created_upload_artifact_writers(
            workspace_id=document.ref.dataset.tenant_id,
            upload_file_ids=lambda: file_ids,
            files=self._uploads,
            logger=logger,
        )

    def cleanup_created_upload_files(
        self,
        document: IndexingDocument,
        file_ids: Sequence[str],
        *,
        discard_persisted_attempt: bool = False,
    ) -> None:
        if discard_persisted_attempt:
            self._discard_persisted_attempt(document, file_ids)
            return
        self._uploads.abandon_created_upload_artifact_writes(
            workspace_id=document.ref.dataset.tenant_id,
            upload_file_ids=file_ids,
        )
        cleanup_unreferenced_upload_files(
            workspace_id=document.ref.dataset.tenant_id,
            file_ids=file_ids,
            files=self._uploads,
            delete_blob=storage.delete,
            logger=logger,
        )

    def _discard_persisted_attempt(self, document: IndexingDocument, file_ids: Sequence[str]) -> None:
        candidate_ids = tuple(dict.fromkeys(file_ids))
        if not candidate_ids:
            return
        ref = document.ref
        with self._session_factory() as session:
            dataset = session.scalar(
                select(Dataset).where(
                    Dataset.id == ref.dataset.dataset_id,
                    Dataset.tenant_id == ref.dataset.tenant_id,
                )
            )
            if dataset is None:
                return
            segment_ids = tuple(
                session.scalars(
                    select(SegmentAttachmentBinding.segment_id)
                    .where(
                        SegmentAttachmentBinding.tenant_id == ref.dataset.tenant_id,
                        SegmentAttachmentBinding.dataset_id == ref.dataset.dataset_id,
                        SegmentAttachmentBinding.document_id == ref.document_id,
                        SegmentAttachmentBinding.attachment_id.in_(candidate_ids),
                    )
                    .distinct()
                )
            )
            segments = (
                session.scalars(
                    select(DocumentSegment).where(
                        DocumentSegment.tenant_id == ref.dataset.tenant_id,
                        DocumentSegment.dataset_id == ref.dataset.dataset_id,
                        DocumentSegment.document_id == ref.document_id,
                        DocumentSegment.id.in_(segment_ids),
                    )
                ).all()
                if segment_ids
                else []
            )
            summaries = (
                session.scalars(
                    select(DocumentSegmentSummary).where(
                        DocumentSegmentSummary.dataset_id == ref.dataset.dataset_id,
                        DocumentSegmentSummary.document_id == ref.document_id,
                        DocumentSegmentSummary.chunk_id.in_(segment_ids),
                    )
                ).all()
                if segment_ids
                else []
            )
            children = (
                session.scalars(
                    select(ChildChunk).where(
                        ChildChunk.tenant_id == ref.dataset.tenant_id,
                        ChildChunk.dataset_id == ref.dataset.dataset_id,
                        ChildChunk.document_id == ref.document_id,
                        ChildChunk.segment_id.in_(segment_ids),
                    )
                ).all()
                if segment_ids
                else []
            )
            summary_node_ids = [summary.summary_index_node_id for summary in summaries if summary.summary_index_node_id]
            body_node_ids = (
                [child.index_node_id for child in children if child.index_node_id]
                if document.doc_form == IndexStructureType.PARENT_CHILD_INDEX
                else [segment.index_node_id for segment in segments if segment.index_node_id]
            )
            body_cleanup_node_ids = list(dict.fromkeys([*summary_node_ids, *body_node_ids]))
            high_quality = dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY
            vector_type = (
                Vector.resolve_vector_type(dataset, session=session)
                if high_quality and (body_cleanup_node_ids or segment_ids)
                else None
            )
            session.expunge(dataset)

        vector = Vector(dataset, session=None, vector_type=vector_type) if vector_type is not None else None
        if vector_type is not None:
            if body_cleanup_node_ids:
                assert vector is not None
                vector.delete_by_ids(body_cleanup_node_ids)
        elif not high_quality and body_node_ids:

            def read() -> tuple[str, str | None]:
                with self._session_factory() as session:
                    row = get_dataset_keyword_table(dataset, session=session)
                    return (
                        (row.data_source_type, row.keyword_table)
                        if row
                        else (dify_config.KEYWORD_DATA_SOURCE_TYPE, None)
                    )

            def write(storage_type: str, data: str, keywords: Mapping[str, Sequence[str]]) -> None:
                with self._session_factory.begin() as session:
                    persist_keyword_table(
                        session,
                        tenant_id=ref.dataset.tenant_id,
                        dataset_id=ref.dataset.dataset_id,
                        storage_type=storage_type,
                        data=data,
                        keywords=keywords,
                    )

            Jieba(dataset).update_texts(
                [],
                read=read,
                write=write,
                lock=redis_client.lock(f"keyword_indexing_lock_{ref.dataset.dataset_id}", timeout=600),
                cleanup_session_factory=self._session_factory,
                deleted_ids=body_node_ids,
            )

        with self._session_factory.begin() as session:
            lock_segment_attachment_cleanup_dataset(
                session,
                tenant_id=ref.dataset.tenant_id,
                dataset_id=ref.dataset.dataset_id,
            )
            if vector is not None and segment_ids:
                actual_vector_type, collection_name = vector.get_routing_target()
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
                delete_unreferenced_attachment_vectors(
                    session,
                    tenant_id=ref.dataset.tenant_id,
                    dataset_id=ref.dataset.dataset_id,
                    document_ids=(ref.document_id,),
                    segment_ids=segment_ids,
                    candidate_attachment_ids=candidate_ids,
                    route=attachment_route,
                )
            session.scalar(
                select(DatasetDocument.id)
                .where(
                    DatasetDocument.tenant_id == ref.dataset.tenant_id,
                    DatasetDocument.dataset_id == ref.dataset.dataset_id,
                    DatasetDocument.id == ref.document_id,
                )
                .with_for_update()
            )
            if not segment_ids:
                artifacts = self._uploads.claim_unreferenced_files_for_cleanup_in_session(
                    session,
                    workspace_id=ref.dataset.tenant_id,
                    file_ids=candidate_ids,
                )
            else:
                artifacts = delete_document_segments_and_claim_uploads(
                    session,
                    tenant_id=ref.dataset.tenant_id,
                    dataset_id=ref.dataset.dataset_id,
                    document_ids=(ref.document_id,),
                    segment_ids=segment_ids,
                    candidate_upload_file_ids=candidate_ids,
                )
        cleanup_claimed_upload_files(
            artifacts=artifacts,
            files=self._uploads,
            delete_blob=storage.delete,
            logger=logger,
        )

    def _cleanup_created_upload_files_after_failure(self, document: IndexingDocument, file_ids: Sequence[str]) -> None:
        if not file_ids:
            return
        try:
            self.cleanup_created_upload_files(document, file_ids)
        except Exception:
            logger.exception("Failed to clean files created by document indexing: %s", document.ref)

    def _cleanup_created_upload_artifacts_after_failure(
        self,
        document: IndexingDocument,
        artifacts: Sequence[CreatedUploadArtifact],
    ) -> None:
        if not artifacts:
            return
        try:
            cleanup_created_upload_artifacts(
                workspace_id=document.ref.dataset.tenant_id,
                artifacts=artifacts,
                files=self._uploads,
                delete_blob=storage.delete,
                logger=logger,
            )
        except Exception:
            logger.exception("Failed to record files created by document indexing: %s", document.ref)

    def transform(self, document: IndexingDocument, texts: list[Document]) -> IndexingTransformResult:
        dataset, _ = self._documents.get_indexing_models(document.ref, claim_id=document.claim_id)
        user = self._documents.get_indexing_user(document.ref, claim_id=document.claim_id)
        model = None
        if dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            manager = ModelManager.for_tenant(tenant_id=dataset.tenant_id)
            if dataset.embedding_model_provider:
                model = manager.get_model_instance(
                    tenant_id=dataset.tenant_id,
                    provider=dataset.embedding_model_provider,
                    model_type=ModelType.TEXT_EMBEDDING,
                    model=dataset.embedding_model,
                )
            else:
                model = manager.get_default_model_instance(
                    tenant_id=dataset.tenant_id, model_type=ModelType.TEXT_EMBEDDING
                )
        processor = IndexProcessorFactory(document.doc_form).init_index_processor()
        created_upload_file_ids: list[str] = []
        try:
            with (
                maintain_created_upload_artifact_writers(
                    workspace_id=document.ref.dataset.tenant_id,
                    upload_file_ids=lambda: created_upload_file_ids,
                    files=self._uploads,
                    logger=logger,
                ),
                self._session_factory() as session,
            ):
                chunks = processor.transform(
                    texts,
                    user,
                    embedding_model_instance=model,
                    process_rule=document.processing_rule,
                    tenant_id=dataset.tenant_id,
                    doc_language=document.doc_language,
                    on_upload_file_created=created_upload_file_ids.append,
                    session=session,
                )
                session.commit()
        except Exception:
            self._cleanup_created_upload_files_after_failure(document, created_upload_file_ids)
            raise
        return IndexingTransformResult(
            chunks=chunks,
            created_upload_file_ids=tuple(created_upload_file_ids),
        )

    def count_tokens(self, document: IndexingDocument, chunks: list[Document]) -> list[int]:
        dataset, _ = self._documents.get_indexing_models(document.ref, claim_id=document.claim_id)
        return calculate_segment_token_counts(dataset=dataset, documents=chunks)

    def ensure_admission(self, document: IndexingDocument, chunks: list[Document]) -> None:
        dataset, _ = self._documents.get_indexing_models(document.ref, claim_id=document.claim_id)
        with self._session_factory() as session:
            VectorSpaceAdmissionService().ensure_document_can_be_indexed(
                dataset=dataset,
                document_id=document.ref.document_id,
                doc_form=document.doc_form,
                documents=chunks,
                include_summaries=document.need_summary,
                session=session,
            )

    def load(self, document: IndexingDocument, chunks: list[Document]) -> None:
        dataset, _ = self._documents.get_indexing_models(document.ref, claim_id=document.claim_id)
        app = cast(LocalProxy[Flask], current_app)._get_current_object()
        if dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            groups: list[list[Document]] = [[] for _ in range(10)]
            for chunk in chunks:
                groups[int(helper.generate_text_hash(chunk.page_content), 16) % 10].append(chunk)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(
                        propagate_context(self._process_chunk),
                        app,
                        document,
                        group,
                        False,
                    )
                    for group in groups
                    if group
                ]
                for future in futures:
                    future.result()
        elif (
            dataset.indexing_technique == IndexTechniqueType.ECONOMY
            and document.doc_form != IndexStructureType.PARENT_CHILD_INDEX
        ):
            # A future propagates keyword failures; Thread.join() used to hide
            # them and incorrectly let the document become completed.
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(propagate_context(self._process_chunk), app, document, chunks, True).result()

    def _process_chunk(
        self, app: Flask, indexing_document: IndexingDocument, chunks: list[Document], keywords: bool
    ) -> None:
        with app.app_context():
            ref = indexing_document.ref
            self.check_paused(ref)
            recorded_route = None if keywords else self._record_vector_route_before_write(indexing_document)
            created_vector: Vector | None = None
            created_node_ids: list[str] = []
            child_vector_route: ChildVectorRoute | None = None
            # Dataset SHARE is the parent-generation barrier. Configuration
            # transitions cannot clean/rebuild the representation while this
            # worker publishes into the representation it captured at claim.
            with self._session_factory.begin() as mutation_session:
                dataset, document = self._require_generation(
                    mutation_session,
                    indexing_document,
                    lock_document=not keywords,
                )
                if keywords:
                    # Keyword persistence locks the exact Document in its own
                    # short transaction; the outer Dataset SHARE keeps the
                    # parent generation stable across extraction/storage I/O.
                    Jieba(dataset).update_texts(
                        chunks,
                        read=lambda: self._segments.get_keyword_table(ref.dataset),
                        write=lambda storage_type, data, selected: self._segments.save_keyword_table(
                            ref.dataset,
                            storage_type=storage_type,
                            data=data,
                            keywords=selected,
                            indexing_document=indexing_document,
                        ),
                        lock=redis_client.lock(f"keyword_indexing_lock_{dataset.id}", timeout=600),
                        before_write=lambda: self._documents.renew_indexing_document(indexing_document),
                        cleanup_session_factory=self._session_factory,
                    )
                elif dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                    vector_type = (
                        recorded_route.vector_type
                        if recorded_route is not None
                        else Vector.resolve_vector_type(dataset, session=mutation_session)
                    )
                    attachments = (
                        [attachment for chunk in chunks for attachment in chunk.attachments or []]
                        if dataset.is_multimodal
                        else []
                    )
                    files = self._uploads.get_files(
                        workspace_id=ref.dataset.tenant_id,
                        file_ids=[attachment.metadata["doc_id"] for attachment in attachments],
                    )
                    vector = Vector(dataset, session=None, vector_type=vector_type)
                    if recorded_route is not None and vector.get_routing_target() != (
                        recorded_route.vector_type,
                        recorded_route.collection_name,
                    ):
                        raise DatasetIndexingGenerationLostError(dataset.id)
                    created_vector = vector
                    if document.doc_form == IndexStructureType.PARENT_CHILD_INDEX:
                        actual_vector_type, collection_name = vector.get_routing_target()
                        child_vector_route = ChildVectorRoute.capture(
                            tenant_id=dataset.tenant_id,
                            dataset_id=dataset.id,
                            vector_type=actual_vector_type,
                            collection_name=collection_name,
                            routing_index_struct=dataset.index_struct,
                            collection_binding_id=dataset.collection_binding_id,
                            indexing_technique=str(dataset.indexing_technique),
                            doc_form=str(document.doc_form),
                        )
                        for chunk in chunks:
                            if chunk.children:
                                child_documents = [
                                    Document.model_validate(child.model_dump()) for child in chunk.children
                                ]
                                vector.create(child_documents)
                                created_node_ids.extend(str(child.metadata["doc_id"]) for child in child_documents)
                    else:
                        vector.create(chunks)
                        created_node_ids.extend(str(chunk.metadata["doc_id"]) for chunk in chunks)
                    if attachments:
                        actual_vector_type, collection_name = vector.get_routing_target()
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

                        def publish(
                            bound_refs: frozenset[tuple[str, str]],
                            before_write: Callable[[Sequence[str]], None],
                            after_write: Callable[[Sequence[str]], None],
                        ) -> None:
                            bound_ids = {attachment_id for _, attachment_id in bound_refs}
                            bound_attachments = [
                                attachment for attachment in attachments if attachment.metadata["doc_id"] in bound_ids
                            ]
                            if bound_attachments:
                                vector.create_multimodal(
                                    bound_attachments,
                                    upload_files={
                                        file_id: file for file_id, file in files.items() if file_id in bound_ids
                                    },
                                    before_write=before_write,
                                    after_write=after_write,
                                )

                        publish_bound_attachment_vectors(
                            mutation_session,
                            tenant_id=ref.dataset.tenant_id,
                            dataset_id=ref.dataset.dataset_id,
                            attachment_refs=[
                                (ref.document_id, str(attachment.metadata["doc_id"])) for attachment in attachments
                            ],
                            route=attachment_route,
                            publish=publish,
                            store=self._attachment_vectors,
                        )
                if indexing_document.claim_id is not None:
                    refresh_locked_indexing_claim(
                        mutation_session,
                        ref,
                        indexing_request_id=indexing_document.indexing_request_id,
                        claim_id=indexing_document.claim_id,
                        allowed_statuses=(IndexingStatus.INDEXING,),
                    )
            try:
                self.check_paused(ref)
                self._segments.complete_indexing_segments(
                    ref,
                    [chunk.metadata["doc_id"] for chunk in chunks],
                    claim_id=indexing_document.claim_id,
                    dataset_generation_id=indexing_document.dataset_generation_id,
                    child_vector_route=child_vector_route,
                )
            except (DocumentIsDeletedPausedError, DocumentIsPausedError):
                if created_vector is not None and created_node_ids:
                    try:
                        created_vector.delete_by_ids(list(dict.fromkeys(created_node_ids)))
                    except Exception:
                        logger.exception("Failed to clean vector nodes after indexing ownership ended")
                raise


def build_document_indexing_service(
    *,
    session_factory: sessionmaker[Session],
    enforce_vector_space_admission: bool = False,
) -> DocumentIndexingService:
    """Assemble a fresh indexing use case from explicit database dependencies."""
    documents = SQLAlchemyDocumentRepository(session_factory=session_factory)
    segments = SQLAlchemySegmentRepository(session_factory=session_factory)
    uploads = SQLAlchemyKnowledgeUploadRepository(session_factory=session_factory)
    attachment_vectors = SQLAlchemyAttachmentVectorStore(session_factory=session_factory)
    credentials = build_data_source_credentials(database_client=session_factory)
    website_jobs = build_website_service(database_client=session_factory)
    sources = CompositeStoredSourceResolver(
        adapters={
            DataSourceType.UPLOAD_FILE: FileSourceAdapter(uploads=uploads),
            DataSourceType.NOTION_IMPORT: NotionSourceResolver(
                actor_credentials=credentials.actor, stored_credentials=credentials.stored
            ),
            DataSourceType.WEBSITE_CRAWL: WebsiteSourceAdapter(
                jobs=website_jobs,
                migrations=SQLAlchemyWebsiteSourceRepository(session_factory=session_factory),
            ),
        }
    )
    return DocumentIndexingService(
        documents=documents,
        segments=segments,
        backend=IndexingExecutionAdapter(
            session_factory=session_factory,
            documents=documents,
            segments=segments,
            uploads=uploads,
            attachment_vectors=attachment_vectors,
            sources=sources,
        ),
        enforce_vector_space_admission=enforce_vector_space_admission,
    )

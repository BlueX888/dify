import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event, Thread, local
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import delete, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import services.knowledge.dataset_service as dataset_service_module
import services.knowledge.indexing.adapters.execution as indexing_execution_adapter_module
import services.knowledge.segments.indexing as segment_indexing_module
import services.knowledge.summaries.adapters as summary_adapter_module
import tasks.regenerate_summary_index_task as summary_task_module
import tasks.remove_document_from_index_task as remove_document_task_module
import tasks.segment_state_outbox_task as segment_state_task_module
from core.entities.document_task import DocumentIndexingClaimLostError
from core.rag.extractor.entity.extract_setting import ExtractionResult
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.index_processor.index_processor_base import SummaryIndexSettingDict
from core.rag.models.document import Document as IndexDocument
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from enums.account import TenantAccountRole
from graphon.model_runtime.entities.llm_entities import LLMUsage
from models.account import Account, Tenant, TenantAccountJoin
from models.dataset import (
    ChildChunk,
    Dataset,
    DatasetCollectionBinding,
    DatasetKeywordTable,
    DatasetProcessRule,
    Document,
    DocumentPipelineExecutionLog,
    DocumentSegment,
    DocumentSegmentSummary,
    Pipeline,
)
from models.enums import (
    DatasetRuntimeMode,
    DataSourceType,
    DocumentCreatedFrom,
    IndexingStatus,
    SegmentStatus,
    SummaryStatus,
)
from models.model import (
    ChildVectorCleanupOutbox,
    DatasetIndexingOutbox,
    DocumentIndexingOutbox,
    SegmentStateOutbox,
    SummaryVectorCleanupOutbox,
)
from models.workflow import Workflow, WorkflowType
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.child_vector_cleanup_repository import SQLAlchemyChildVectorCleanupRepository
from repositories.knowledge.dataset_indexing_outbox_repository import (
    DatasetIndexingWritersActiveError,
    SQLAlchemyDatasetIndexingOutboxRepository,
    create_dataset_indexing_intents,
    lock_dataset_indexing_execution,
    lock_dataset_indexing_generation_shared,
)
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    SQLAlchemyDocumentIndexingOutboxRepository,
    create_document_indexing_intents,
    create_pipeline_document_indexing_intents,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository, require_indexing_document
from repositories.knowledge.keyword_table_repository import lock_dataset_keyword_table
from repositories.knowledge.pipeline_indexing_repository import protect_pipeline_indexing_claim
from repositories.knowledge.segment_repository import SegmentIndexingSnapshot, SQLAlchemySegmentRepository
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.entities.knowledge_entities.rag_pipeline_entities import KnowledgeConfiguration, RetrievalSetting
from services.knowledge.dataset_service import DatasetService, DocumentService, ExternalKnowledgeTemplateLookup
from services.knowledge.entities.datasets import (
    DatasetIndexingGenerationLostError,
    DatasetIndexingJob,
    DatasetIndexingRepresentation,
)
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.knowledge_entities import KnowledgeConfig
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.adapters.execution import IndexingExecutionAdapter
from services.knowledge.indexing.adapters.sources import CompositeStoredSourceResolver
from services.knowledge.indexing.estimate import StoredSource
from services.knowledge.indexing.execution import (
    DocumentIndexingService,
    IndexingDocument,
    IndexingTransformResult,
)
from services.knowledge.segments.indexing import SegmentIndexingGateway
from services.knowledge.summaries.adapters import SummaryIndexAdapter
from services.rag_pipeline.rag_pipeline_dsl_service import RagPipelineDslService


class _WorkerRole(local):
    role: str | None

    def __init__(self) -> None:
        self.role = None


@dataclass(frozen=True, slots=True)
class _EmbeddingModelConfigurationBoundary:
    provider: str
    model_name: str


class _ConfigurationModelManagerBoundary:
    def __init__(self, *, tenant_id: str) -> None:
        self._tenant_id = tenant_id

    def get_model_instance(
        self,
        *,
        tenant_id: str,
        provider: str,
        model_type: object,
        model: str,
    ) -> _EmbeddingModelConfigurationBoundary:
        del model_type
        assert tenant_id == self._tenant_id
        return _EmbeddingModelConfigurationBoundary(provider=provider, model_name=model)


class _UnexpectedExternalTemplateLookup(ExternalKnowledgeTemplateLookup):
    def require_template(self, tenant_id: str, template_id: str, *, session: Session) -> None:
        raise AssertionError(
            f"internal dataset must not resolve external template {tenant_id}/{template_id}: {session}"
        )


class _DocumentStatusRedisBoundary:
    def __init__(self) -> None:
        self.cached: list[tuple[str, int, int]] = []
        self.deleted: list[str] = []

    @staticmethod
    def get(_key: str) -> None:
        return None

    def setex(self, key: str, ttl: int, value: int) -> None:
        self.cached.append((key, ttl, value))

    def delete(self, key: str) -> None:
        self.deleted.append(key)


class _RetryLockBoundary:
    def __init__(self) -> None:
        self.released = False

    @staticmethod
    def acquire(*, blocking: bool) -> bool:
        assert blocking is False
        return True

    def release(self) -> None:
        self.released = True


class _RetryRedisBoundary:
    def __init__(self) -> None:
        self.locks: list[_RetryLockBoundary] = []

    def lock(self, _key: str, *, timeout: int, thread_local: bool) -> _RetryLockBoundary:
        assert timeout == 600
        assert thread_local is False
        lock = _RetryLockBoundary()
        self.locks.append(lock)
        return lock


class _SessionFactoryBoundary:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def create_session(self) -> Session:
        return self._sessions()


class _RemoveDocumentProcessorBoundary:
    def __init__(self, *, clean_started: Event, allow_clean_return: Event) -> None:
        self._clean_started = clean_started
        self._allow_clean_return = allow_clean_return
        self.cleaned: list[tuple[str, ...]] = []

    def clean(
        self,
        _dataset: Dataset,
        node_ids: list[str],
        *,
        with_keywords: bool,
        delete_child_chunks: bool,
        session: Session,
    ) -> None:
        assert with_keywords is True
        assert delete_child_chunks is False
        assert session.in_transaction()
        self.cleaned.append(tuple(node_ids))
        self._clean_started.set()
        assert self._allow_clean_return.wait(timeout=10)


class _RemoveDocumentProcessorFactoryBoundary:
    def __init__(self, processor: _RemoveDocumentProcessorBoundary) -> None:
        self._processor = processor

    def __call__(self, doc_form: str) -> "_RemoveDocumentProcessorFactoryBoundary":
        assert doc_form == IndexStructureType.PARAGRAPH_INDEX
        return self

    def init_index_processor(self) -> _RemoveDocumentProcessorBoundary:
        return self._processor


class _SegmentStateProcessorBoundary:
    def __init__(self) -> None:
        self.loads: list[tuple[str | None, tuple[IndexDocument, ...]]] = []

    def load(
        self,
        dataset: Dataset,
        documents: list[IndexDocument],
        *,
        multimodal_documents: list[IndexDocument],
        session: Session,
    ) -> None:
        del session
        assert multimodal_documents == []
        self.loads.append((dataset.indexing_request_id, tuple(documents)))


class _SegmentStateProcessorFactoryBoundary:
    def __init__(self, processor: _SegmentStateProcessorBoundary) -> None:
        self._processor = processor

    def __call__(
        self,
        index_type: str,
        *,
        attachment_vectors: object,
        child_vector_routes: object,
    ) -> "_SegmentStateProcessorFactoryBoundary":
        del attachment_vectors, child_vector_routes
        assert index_type == IndexStructureType.PARAGRAPH_INDEX
        return self

    def init_index_processor(self) -> _SegmentStateProcessorBoundary:
        return self._processor


class _SegmentStateDisableProcessorFactoryBoundary:
    def __init__(self, processor: _RemoveDocumentProcessorBoundary) -> None:
        self._processor = processor

    def __call__(
        self,
        index_type: str,
        *,
        attachment_vectors: object,
        child_vector_routes: object,
    ) -> "_SegmentStateDisableProcessorFactoryBoundary":
        del attachment_vectors, child_vector_routes
        assert index_type == IndexStructureType.PARAGRAPH_INDEX
        return self

    def init_index_processor(self) -> _RemoveDocumentProcessorBoundary:
        return self._processor


class _SegmentBatchRedisBoundary:
    @staticmethod
    def get(name: str | bytes) -> Any:
        del name
        return None

    @staticmethod
    def setnx(name: str | bytes, value: Any) -> bool:
        del name, value
        return True

    @staticmethod
    def setex(name: str, time: int, value: Any) -> bool:
        del name, time, value
        return True

    @staticmethod
    def lock(name: str, timeout: int) -> AbstractContextManager[None]:
        del name, timeout
        return nullcontext()


class _SegmentBatchVectorBoundary:
    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    @staticmethod
    def get_routing_target() -> tuple[str, str]:
        return "qdrant", "collection-batch-current"

    def create(self, documents: Sequence[IndexDocument]) -> None:
        self.batches.append(tuple(document.page_content for document in documents))


class _SegmentBatchVectorConstructorBoundary:
    def __init__(self, vector: _SegmentBatchVectorBoundary, *, tenant_id: str) -> None:
        self._vector = vector
        self._tenant_id = tenant_id

    def __call__(
        self,
        dataset: Dataset,
        *,
        session: Session | None,
        vector_type: str | None,
    ) -> _SegmentBatchVectorBoundary:
        assert dataset.tenant_id == self._tenant_id
        assert session is None
        assert vector_type == "qdrant"
        return self._vector


class _SegmentBatchKeywordBoundary:
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def update_texts(
        self,
        texts: Sequence[IndexDocument],
        *,
        read: Callable[[], tuple[str, str | None]],
        write: Callable[[str, str, Mapping[str, Sequence[str]]], None],
        lock: AbstractContextManager[object],
        cleanup_session_factory: Callable[[], Session],
        keywords_list: Sequence[Sequence[str] | None] | None,
        replace_existing: bool,
    ) -> None:
        del cleanup_session_factory
        assert self._dataset.indexing_technique == IndexTechniqueType.ECONOMY
        assert keywords_list is not None
        assert replace_existing is False
        with lock:
            _storage_type, current = read()
            assert current is None
            selected = {
                str(document.metadata["doc_id"]): tuple(keywords or ())
                for document, keywords in zip(texts, keywords_list, strict=True)
            }
            write("database", json.dumps({"__data__": selected}), selected)


class _SegmentBatchKeywordConstructorBoundary:
    def __call__(self, dataset: Dataset) -> _SegmentBatchKeywordBoundary:
        return _SegmentBatchKeywordBoundary(dataset)


class _SegmentStateVectorBoundary:
    def __init__(self) -> None:
        self.added: list[IndexDocument] = []
        self.deleted: list[tuple[str, ...]] = []

    @staticmethod
    def get_routing_target() -> tuple[str, str]:
        return "qdrant", "collection-g1"

    def add_texts(
        self,
        documents: list[IndexDocument],
        *,
        duplicate_check: bool,
        before_write: Callable[[], None],
    ) -> None:
        assert duplicate_check is False
        assert all(document.metadata.get("is_summary") is True for document in documents)
        before_write()
        self.added.extend(documents)

    def delete_by_ids(self, node_ids: list[str]) -> None:
        self.deleted.append(tuple(node_ids))


class _SegmentStateVectorConstructorBoundary:
    def __init__(self, vector: _SegmentStateVectorBoundary) -> None:
        self._vector = vector
        self.generations: list[str | None] = []

    def __call__(self, dataset: Dataset, *, session: Session) -> _SegmentStateVectorBoundary:
        del session
        self.generations.append(dataset.indexing_request_id)
        return self._vector


class _SummaryEmbeddingBoundary:
    def __init__(self) -> None:
        self.contents: list[list[str]] = []

    def get_text_embedding_num_tokens(self, contents: list[str]) -> list[int]:
        self.contents.append(contents)
        return [7 for _content in contents]


class _SummaryTenantModelsBoundary:
    def __init__(self, *, tenant_id: str, embedding: _SummaryEmbeddingBoundary) -> None:
        self._tenant_id = tenant_id
        self._embedding = embedding

    def get_model_instance(
        self,
        *,
        tenant_id: str,
        provider: str,
        model_type: object,
        model: str,
    ) -> _SummaryEmbeddingBoundary:
        del model_type
        assert (tenant_id, provider, model) == (
            self._tenant_id,
            "embedding-provider",
            "embedding-model",
        )
        return self._embedding


class _SummaryVectorBoundary:
    def __init__(self) -> None:
        self.added: list[IndexDocument] = []
        self.deleted: list[list[str]] = []

    def get_routing_target(self) -> tuple[str, str]:
        return "qdrant", "summary-collection-g1"

    def add_texts(
        self,
        documents: list[IndexDocument],
        *,
        duplicate_check: bool,
        before_write: Callable[[], None],
    ) -> None:
        assert duplicate_check is False
        before_write()
        self.added.extend(documents)

    def delete_by_ids(self, node_ids: list[str]) -> None:
        self.deleted.append(node_ids)


class _SummaryVectorConstructorBoundary:
    def __init__(self, vector: _SummaryVectorBoundary) -> None:
        self._vector = vector

    @staticmethod
    def resolve_vector_type(_dataset: Dataset, *, session: Session) -> str:
        del session
        return "qdrant"

    def __call__(
        self,
        _dataset: Dataset,
        *,
        session: Session | None,
        vector_type: str,
    ) -> _SummaryVectorBoundary:
        assert session is None
        assert vector_type == "qdrant"
        return self._vector


class _RestartSegments:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def save_for_indexing(
        self,
        document: IndexingDocument,
        chunks: list[IndexDocument],
        token_counts: list[int],
        *,
        created_upload_file_ids: Sequence[str] = (),
    ) -> None:
        assert document.dataset_generation_id is not None
        assert len(chunks) == len(token_counts) == 1
        assert tuple(created_upload_file_ids) == ()
        self._events.append("save")

    def resume_indexing(self, document: IndexingDocument) -> tuple[list[IndexDocument], int]:
        raise AssertionError(f"restart must not resume partial chunks: {document.ref}")


class _RestartBackend:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def describe_error(self, error: Exception) -> str:
        return str(error)

    def check_paused(self, ref: DocumentRef) -> None:
        self._events.append(f"check:{ref.document_id}")

    def clean_for_reindex(self, document: IndexingDocument) -> None:
        self._events.append("clean")

    def extract(self, document: IndexingDocument) -> ExtractionResult:
        self._events.append("extract")
        return ExtractionResult(documents=[IndexDocument(page_content="restarted", metadata={})])

    def maintain_created_upload_file_writers(
        self,
        document: IndexingDocument,
        file_ids: Sequence[str],
    ) -> AbstractContextManager[None]:
        assert tuple(file_ids) == ()
        return nullcontext()

    def cleanup_created_upload_files(
        self,
        document: IndexingDocument,
        file_ids: Sequence[str],
        *,
        discard_persisted_attempt: bool = False,
    ) -> None:
        raise AssertionError(
            "successful restart must not clean artifacts: "
            f"{document.ref}, {tuple(file_ids)}, {discard_persisted_attempt}"
        )

    def transform(self, document: IndexingDocument, texts: list[IndexDocument]) -> IndexingTransformResult:
        self._events.append("transform")
        return IndexingTransformResult(chunks=texts)

    def ensure_admission(self, document: IndexingDocument, chunks: list[IndexDocument]) -> None:
        raise AssertionError(f"recovery must retain its existing admission policy: {document.ref}, {chunks}")

    def count_tokens(self, document: IndexingDocument, chunks: list[IndexDocument]) -> list[int]:
        return [len(chunk.page_content.split()) for chunk in chunks]

    def load(self, document: IndexingDocument, chunks: list[IndexDocument]) -> None:
        assert document.dataset_generation_id is not None
        assert [chunk.page_content for chunk in chunks] == ["restarted"]
        self._events.append("load")


def _repeatable_read_sessions(mysql_session: Session) -> tuple[Engine, sessionmaker[Session]]:
    engine = cast(Engine, mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ"))
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _join_if_started(thread: Thread) -> None:
    if thread.ident is not None:
        thread.join(timeout=10)


def _is_dataset_lock(statement: str, *, shared: bool) -> bool:
    normalized = " ".join(statement.lower().replace("`", "").split())
    if " from datasets " not in normalized:
        return False
    if shared:
        return " lock in share mode" in normalized or " for share" in normalized
    return " for update" in normalized


def _representation(*, collection: str) -> DatasetIndexingRepresentation:
    return DatasetIndexingRepresentation(
        indexing_technique="high_quality",
        index_struct=json.dumps({"type": "qdrant", "vector_store": {"class_prefix": collection}}),
        collection_binding_id=None,
        embedding_model="embedding-model",
        embedding_model_provider="embedding-provider",
        keyword_number=10,
        is_multimodal=False,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        summary_index_setting=None,
    )


def _economy_representation(*, keyword_number: int) -> DatasetIndexingRepresentation:
    return DatasetIndexingRepresentation(
        indexing_technique="economy",
        index_struct=None,
        collection_binding_id=None,
        embedding_model=None,
        embedding_model_provider=None,
        keyword_number=keyword_number,
        is_multimodal=False,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        summary_index_setting=None,
    )


def _model_representation(
    *,
    provider: str,
    model: str,
    binding_id: str,
) -> DatasetIndexingRepresentation:
    return DatasetIndexingRepresentation(
        indexing_technique="high_quality",
        index_struct=None,
        collection_binding_id=binding_id,
        embedding_model=model,
        embedding_model_provider=provider,
        keyword_number=None,
        is_multimodal=False,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        summary_index_setting=None,
    )


def _create_generation(
    sessions: sessionmaker[Session],
    *,
    tenant_id: str,
    dataset_id: str,
    source_collection: str,
    target_collection: str,
) -> str:
    job = DatasetIndexingJob(
        DatasetRef(tenant_id, dataset_id),
        _representation(collection=source_collection),
        _representation(collection=target_collection),
    )
    with sessions.begin() as session:
        (intent_id,) = create_dataset_indexing_intents(session, job)
    return intent_id


def _seed_dataset(sessions: sessionmaker[Session]) -> tuple[str, str, str]:
    tenant_id = str(uuid4())
    dataset_id = str(uuid4())
    actor_id = str(uuid4())
    with sessions.begin() as session:
        session.add(
            Dataset(
                id=dataset_id,
                tenant_id=tenant_id,
                name=f"generation-race-{uuid4()}",
                data_source_type=DataSourceType.UPLOAD_FILE,
                indexing_technique="high_quality",
                index_struct=json.dumps({"type": "qdrant", "vector_store": {"class_prefix": "collection-initial"}}),
                created_by=actor_id,
            )
        )
    return tenant_id, dataset_id, actor_id


def _seed_duplicate_request(
    sessions: sessionmaker[Session],
    *,
    tenant_id: str,
    dataset_id: str,
    actor_id: str,
) -> tuple[str, str, str]:
    document_id = str(uuid4())
    segment_id = str(uuid4())
    node_id = str(uuid4())
    with sessions.begin() as session:
        request = DocumentIndexingOutbox(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            action="duplicate",
            document_ids=json.dumps([document_id]),
            actor_id=actor_id,
        )
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch=str(uuid4()),
            name="duplicate.txt",
            created_from=DocumentCreatedFrom.WEB,
            created_by=actor_id,
            indexing_status=IndexingStatus.WAITING,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            enabled=True,
        )
        document.indexing_request_id = request.id
        segment = DocumentSegment(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            position=1,
            content="duplicate vector payload",
            word_count=3,
            tokens=3,
            created_by=actor_id,
            index_node_id=node_id,
            index_node_hash=f"hash-{node_id}",
            status=SegmentStatus.INDEXING,
            enabled=False,
        )
        segment.id = segment_id
        session.add_all([request, document, segment])
        session.flush()
        request_id = request.id
    return request_id, document_id, node_id


def _seed_document_request(
    sessions: sessionmaker[Session],
    *,
    tenant_id: str,
    dataset_id: str,
    actor_id: str,
) -> tuple[str, str]:
    document_id = str(uuid4())
    with sessions.begin() as session:
        session.add(
            Document(
                id=document_id,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                position=1,
                data_source_type=DataSourceType.UPLOAD_FILE,
                batch=str(uuid4()),
                name="generation-restart.txt",
                created_from=DocumentCreatedFrom.WEB,
                created_by=actor_id,
                indexing_status=IndexingStatus.WAITING,
                doc_form=IndexStructureType.PARAGRAPH_INDEX,
                enabled=True,
            )
        )
        session.flush()
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(
                dataset=DatasetRef(tenant_id, dataset_id),
                duplicated=(document_id,),
            ),
        )
    return request_id, document_id


def _seed_summary_segment(
    sessions: sessionmaker[Session],
    *,
    tenant_id: str,
    dataset_id: str,
    actor_id: str,
) -> tuple[str, str]:
    document_id = str(uuid4())
    segment_id = str(uuid4())
    with sessions.begin() as session:
        session.add(
            Document(
                id=document_id,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                position=1,
                data_source_type=DataSourceType.UPLOAD_FILE,
                batch=str(uuid4()),
                name="summary-generation.txt",
                created_from=DocumentCreatedFrom.WEB,
                created_by=actor_id,
                indexing_status=IndexingStatus.COMPLETED,
                doc_form=IndexStructureType.PARAGRAPH_INDEX,
                enabled=True,
                need_summary=True,
            )
        )
        segment = DocumentSegment(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            position=1,
            content="summary generation source",
            word_count=3,
            tokens=3,
            created_by=actor_id,
            index_node_id=str(uuid4()),
            index_node_hash=str(uuid4()),
            status=SegmentStatus.COMPLETED,
            enabled=True,
        )
        segment.id = segment_id
        session.add(segment)
    return document_id, segment_id


def _delete_fixture(sessions: sessionmaker[Session], *, tenant_id: str, dataset_id: str) -> None:
    with sessions.begin() as session:
        session.execute(delete(DatasetKeywordTable).where(DatasetKeywordTable.dataset_id == dataset_id))
        session.execute(
            delete(SegmentStateOutbox).where(
                SegmentStateOutbox.tenant_id == tenant_id,
                SegmentStateOutbox.dataset_id == dataset_id,
            )
        )
        session.execute(
            delete(SummaryVectorCleanupOutbox).where(
                SummaryVectorCleanupOutbox.tenant_id == tenant_id,
                SummaryVectorCleanupOutbox.dataset_id == dataset_id,
            )
        )
        session.execute(delete(DocumentSegmentSummary).where(DocumentSegmentSummary.dataset_id == dataset_id))
        session.execute(
            delete(DocumentSegment).where(
                DocumentSegment.tenant_id == tenant_id,
                DocumentSegment.dataset_id == dataset_id,
            )
        )
        session.execute(
            delete(DocumentIndexingOutbox).where(
                DocumentIndexingOutbox.tenant_id == tenant_id,
                DocumentIndexingOutbox.dataset_id == dataset_id,
            )
        )
        document_ids = select(Document.id).where(
            Document.tenant_id == tenant_id,
            Document.dataset_id == dataset_id,
        )
        session.execute(
            delete(DocumentPipelineExecutionLog).where(DocumentPipelineExecutionLog.document_id.in_(document_ids))
        )
        session.execute(delete(Document).where(Document.tenant_id == tenant_id, Document.dataset_id == dataset_id))
        session.execute(
            delete(DatasetIndexingOutbox).where(
                DatasetIndexingOutbox.tenant_id == tenant_id,
                DatasetIndexingOutbox.dataset_id == dataset_id,
            )
        )
        session.execute(delete(Dataset).where(Dataset.tenant_id == tenant_id, Dataset.id == dataset_id))
        session.execute(delete(Workflow).where(Workflow.tenant_id == tenant_id))
        session.execute(delete(Pipeline).where(Pipeline.tenant_id == tenant_id))


def test_expired_redelivery_observes_owner_renewal_after_real_dataset_lock_wait(
    mysql_session: Session,
) -> None:
    """A queued takeover must use a locking current read after the Dataset-X wait."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, _actor_id = _seed_dataset(sessions)
    intent_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-a",
        target_collection="collection-b",
    )
    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sessions)
    owner = repository.claim(intent_id)
    assert owner is not None
    expired_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=1)
    with sessions.begin() as session:
        row = session.get_one(DatasetIndexingOutbox, intent_id)
        row.claim_expires_at = expired_at

    contender_attempted = Event()
    contender_finished = Event()
    contender_results: list[object] = []
    contender_errors: list[BaseException] = []
    worker = _WorkerRole()

    def observe_contender_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "contender" and _is_dataset_lock(statement, shared=False):
            contender_attempted.set()

    def attempt_takeover() -> None:
        try:
            worker.role = "contender"
            contender_results.append(repository.claim(intent_id))
        except BaseException as error:
            contender_errors.append(error)
        finally:
            contender_finished.set()

    contender = Thread(target=attempt_takeover)
    event.listen(engine, "before_cursor_execute", observe_contender_lock)
    try:
        with sessions.begin() as execution_session:
            dataset = lock_dataset_indexing_execution(
                execution_session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                generation_id=owner.generation_id,
                intent_id=owner.id,
                claim_id=owner.claim_id,
            )
            assert dataset is not None
            contender.start()
            assert contender_attempted.wait(timeout=10)
            # The test does not use a Python mutex. The contender is queued on
            # this transaction's real InnoDB Dataset row lock.
            assert not contender_finished.wait(timeout=0.25)
            repository.refresh_locked_claim(execution_session, owner.id, claim_id=owner.claim_id)

        _join_if_started(contender)
        assert not contender.is_alive()
        assert contender_errors == []
        assert contender_results == [None]
        with sessions() as verifier:
            renewed = verifier.get_one(DatasetIndexingOutbox, intent_id)
            assert renewed.claim_id == owner.claim_id
            assert renewed.claim_expires_at is not None
            assert renewed.claim_expires_at > expired_at
    finally:
        _join_if_started(contender)
        event.remove(engine, "before_cursor_execute", observe_contender_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_successor_generation_handoffs_partial_route_after_repeatable_read_snapshot(
    mysql_session: Session,
) -> None:
    """A Dataset-X config write must refresh physical evidence committed after its RR snapshot."""

    _engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, _actor_id = _seed_dataset(sessions)
    route_a = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    route_b = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="collection-b",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    with sessions.begin() as session:
        dataset = session.get_one(Dataset, dataset_id)
        dataset.indexing_applied_representation = (
            _representation(collection="collection-a")
            .model_copy(update={"vector_route": route_a, "cleanup_routes": ()})
            .model_dump_json()
        )

    intent_b_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-a",
        target_collection="collection-b",
    )
    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sessions)
    intent_b = repository.claim(intent_b_id)
    assert intent_b is not None

    try:
        with sessions.begin() as config_session:
            # These ordinary reads establish an old REPEATABLE READ snapshot in
            # the same transaction that will later create generation C.
            snapshot_dataset = config_session.scalar(select(Dataset).where(Dataset.id == dataset_id))
            snapshot_b = config_session.scalar(
                select(DatasetIndexingOutbox).where(DatasetIndexingOutbox.id == intent_b_id)
            )
            assert snapshot_dataset is not None
            assert snapshot_dataset.indexing_request_id == intent_b.generation_id
            assert snapshot_b is not None
            snapshot_target = DatasetIndexingRepresentation.model_validate_json(snapshot_b.target_representation)
            assert snapshot_target.vector_route is None
            assert snapshot_target.cleanup_routes == ()

            # The old worker records only the physical target it actually
            # reached. This commit happens after config_session's snapshot.
            with sessions.begin() as worker_session:
                locked_dataset = lock_dataset_indexing_execution(
                    worker_session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    generation_id=intent_b.generation_id,
                    intent_id=intent_b.id,
                    claim_id=intent_b.claim_id,
                )
                assert locked_dataset is not None
                recorded_target = repository.record_target_route_locked(
                    worker_session,
                    intent_id=intent_b.id,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    generation_id=intent_b.generation_id,
                    claim_id=intent_b.claim_id,
                    route=route_b,
                )
                assert recorded_target.vector_route == route_b

            (intent_c_id,) = create_dataset_indexing_intents(
                config_session,
                DatasetIndexingJob(
                    DatasetRef(tenant_id, dataset_id),
                    _representation(collection="collection-b"),
                    _representation(collection="collection-c"),
                ),
            )
            refreshed_b_target = DatasetIndexingRepresentation.model_validate_json(snapshot_b.target_representation)
            intent_c = config_session.get_one(DatasetIndexingOutbox, intent_c_id)
            source_c = DatasetIndexingRepresentation.model_validate_json(intent_c.source_representation)
            target_c = DatasetIndexingRepresentation.model_validate_json(intent_c.target_representation)

            assert refreshed_b_target.vector_route == route_b
            assert source_c.vector_route == route_a
            assert {(route.vector_type, route.collection_name) for route in source_c.cleanup_routes} == {
                (route_a.vector_type, route_a.collection_name),
                (route_b.vector_type, route_b.collection_name),
            }
            # Generation C has not performed external I/O, so it must not
            # manufacture physical target evidence merely from its config.
            assert target_c.vector_route is None
            assert target_c.cleanup_routes == ()
    finally:
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_dsl_dataset_configuration_commits_settings_and_generation_intent_atomically(
    mysql_session: Session,
) -> None:
    """The shared direct/confirm import boundary must commit config B with its durable intent."""

    _engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    old_generation_id = str(uuid4())
    with sessions.begin() as seed:
        tenant = Tenant(name=f"dsl-generation-{uuid4()}")
        tenant.id = tenant_id
        account = Account(name="DSL importer", email=f"dsl-{uuid4()}@example.com")
        account.id = actor_id
        seed.add_all(
            [
                tenant,
                account,
                TenantAccountJoin(
                    tenant_id=tenant_id,
                    account_id=actor_id,
                    current=True,
                    role=TenantAccountRole.OWNER,
                ),
            ]
        )
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.chunk_structure = IndexStructureType.PARAGRAPH_INDEX
        dataset.indexing_technique = "economy"
        dataset.keyword_number = 10
        dataset.index_struct = None
        dataset.embedding_model = None
        dataset.embedding_model_provider = None
        dataset.collection_binding_id = None
        dataset.indexing_applied_representation = _economy_representation(keyword_number=10).model_dump_json()
        dataset.indexing_request_id = old_generation_id

    configuration = KnowledgeConfiguration(
        chunk_structure=IndexStructureType.PARAGRAPH_INDEX,
        indexing_technique="economy",
        keyword_number=20,
        retrieval_model=RetrievalSetting(
            search_method=RetrievalMethod.KEYWORD_SEARCH,
            top_k=3,
        ),
    )
    intent_id: str | None = None
    try:
        with sessions() as import_session:
            dataset = import_session.get_one(Dataset, dataset_id)
            account = import_session.get_one(Account, actor_id)
            tenant = import_session.get_one(Tenant, tenant_id)
            account.set_current_tenant_with_session(tenant, session=import_session)
            service = RagPipelineDslService(import_session)
            service._apply_dataset_configuration(
                dataset=dataset,
                knowledge_configuration=configuration,
                account=account,
                has_published=True,
            )
            (intent_id,) = service.pending_dataset_indexing_intent_ids
            assert dataset.keyword_number == 20
            assert dataset.indexing_request_id is not None
            assert dataset.indexing_request_id != old_generation_id

            # A different real Session cannot observe either half before the
            # import caller commits its single transaction.
            with sessions() as observer:
                old_dataset = observer.get_one(Dataset, dataset_id)
                assert old_dataset.keyword_number == 10
                assert old_dataset.indexing_request_id == old_generation_id
                assert observer.get(DatasetIndexingOutbox, intent_id) is None
            import_session.commit()

        repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sessions)
        intent = repository.claim(intent_id)
        assert intent is not None
        assert intent.source == _economy_representation(keyword_number=10)
        assert intent.target == _economy_representation(keyword_number=20)
        with sessions() as verify:
            current = verify.get_one(Dataset, dataset_id)
            assert current.keyword_number == 20
            assert current.indexing_request_id == intent.generation_id
            assert verify.get_one(DatasetIndexingOutbox, intent_id).generation_id == intent.generation_id

        # A writer carrying the pre-import generation is fenced by the same
        # locking current read used around actual index mutations.
        with sessions.begin() as old_writer:
            assert (
                lock_dataset_indexing_generation_shared(
                    old_writer,
                    dataset_id=dataset_id,
                    generation_id=old_generation_id,
                )
                is None
            )
    finally:
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(TenantAccountJoin).where(
                    TenantAccountJoin.tenant_id == tenant_id,
                    TenantAccountJoin.account_id == actor_id,
                )
            )
            cleanup.execute(delete(Account).where(Account.id == actor_id))
            cleanup.execute(delete(Tenant).where(Tenant.id == tenant_id))


def test_stale_rag_configuration_refreshes_after_waiting_for_a_new_generation(
    mysql_session: Session,
) -> None:
    """A stale RR identity-map row must not overwrite the configuration that wins the Dataset lock first."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    old_generation_id = str(uuid4())
    with sessions.begin() as seed:
        tenant = Tenant(name=f"stale-rag-generation-{uuid4()}")
        tenant.id = tenant_id
        account = Account(name="Stale RAG importer", email=f"stale-rag-{uuid4()}@example.com")
        account.id = actor_id
        seed.add_all(
            [
                tenant,
                account,
                TenantAccountJoin(
                    tenant_id=tenant_id,
                    account_id=actor_id,
                    current=True,
                    role=TenantAccountRole.OWNER,
                ),
            ]
        )
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.chunk_structure = IndexStructureType.PARAGRAPH_INDEX
        dataset.indexing_technique = "economy"
        dataset.keyword_number = 10
        dataset.index_struct = None
        dataset.embedding_model = None
        dataset.embedding_model_provider = None
        dataset.collection_binding_id = None
        dataset.indexing_applied_representation = _economy_representation(keyword_number=10).model_dump_json()
        dataset.indexing_request_id = old_generation_id

    configuration_b = KnowledgeConfiguration(
        chunk_structure=IndexStructureType.PARAGRAPH_INDEX,
        indexing_technique="economy",
        keyword_number=20,
        retrieval_model=RetrievalSetting(
            search_method=RetrievalMethod.KEYWORD_SEARCH,
            top_k=3,
        ),
    )
    configuration_c = configuration_b.model_copy(update={"keyword_number": 10})
    stale_loaded = Event()
    begin_stale_update = Event()
    stale_lock_attempted = Event()
    stale_finished = Event()
    stale_intent_ids: list[tuple[str, ...]] = []
    stale_errors: list[BaseException] = []
    worker = _WorkerRole()

    def observe_stale_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "stale-config" and _is_dataset_lock(statement, shared=False):
            stale_lock_attempted.set()

    def apply_stale_configuration() -> None:
        try:
            with sessions() as stale_session:
                dataset = stale_session.get_one(Dataset, dataset_id)
                account = stale_session.get_one(Account, actor_id)
                tenant = stale_session.get_one(Tenant, tenant_id)
                account.set_current_tenant_with_session(tenant, session=stale_session)
                assert dataset.keyword_number == 10
                assert dataset.indexing_request_id == old_generation_id
                stale_loaded.set()
                assert begin_stale_update.wait(timeout=10)
                worker.role = "stale-config"
                stale_intent_ids.append(
                    DatasetService.update_rag_pipeline_dataset_settings(
                        dataset,
                        configuration_c,
                        has_published=True,
                        session=stale_session,
                        account=account,
                    )
                )
                stale_session.commit()
        except BaseException as error:
            stale_errors.append(error)
        finally:
            stale_finished.set()

    stale_writer = Thread(target=apply_stale_configuration)
    event.listen(engine, "before_cursor_execute", observe_stale_lock)
    try:
        stale_writer.start()
        assert stale_loaded.wait(timeout=10)

        with sessions() as writer_b:
            dataset = writer_b.get_one(Dataset, dataset_id)
            account = writer_b.get_one(Account, actor_id)
            tenant = writer_b.get_one(Tenant, tenant_id)
            account.set_current_tenant_with_session(tenant, session=writer_b)
            (intent_b_id,) = DatasetService.update_rag_pipeline_dataset_settings(
                dataset,
                configuration_b,
                has_published=True,
                session=writer_b,
                account=account,
            )
            row_b = writer_b.get_one(DatasetIndexingOutbox, intent_b_id)
            target_b = DatasetIndexingRepresentation.model_validate_json(row_b.target_representation)
            assert target_b == _economy_representation(keyword_number=20)

            begin_stale_update.set()
            assert stale_lock_attempted.wait(timeout=10)
            # The stale updater is queued on this transaction's real InnoDB
            # Dataset row lock; Events only make that database wait observable.
            assert not stale_finished.wait(timeout=0.25)
            writer_b.commit()

        _join_if_started(stale_writer)
        assert not stale_writer.is_alive()
        assert stale_errors == []
        assert len(stale_intent_ids) == 1
        (intent_c_id,) = stale_intent_ids[0]

        with sessions() as verify:
            dataset = verify.get_one(Dataset, dataset_id)
            row_b = verify.get_one(DatasetIndexingOutbox, intent_b_id)
            row_c = verify.get_one(DatasetIndexingOutbox, intent_c_id)
            source_c = DatasetIndexingRepresentation.model_validate_json(row_c.source_representation)
            target_c = DatasetIndexingRepresentation.model_validate_json(row_c.target_representation)

            assert dataset.keyword_number == 10
            assert dataset.indexing_technique == "economy"
            assert dataset.embedding_model is None
            assert dataset.embedding_model_provider is None
            assert dataset.collection_binding_id is None
            assert dataset.indexing_request_id == row_c.generation_id
            assert row_b.generation_id != row_c.generation_id
            assert source_c == _economy_representation(keyword_number=10)
            assert target_c == _economy_representation(keyword_number=10)
    finally:
        begin_stale_update.set()
        _join_if_started(stale_writer)
        event.remove(engine, "before_cursor_execute", observe_stale_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(TenantAccountJoin).where(
                    TenantAccountJoin.tenant_id == tenant_id,
                    TenantAccountJoin.account_id == actor_id,
                )
            )
            cleanup.execute(delete(Account).where(Account.id == actor_id))
            cleanup.execute(delete(Tenant).where(Tenant.id == tenant_id))


def test_stale_dataset_patch_refreshes_model_and_binding_after_real_lock_wait(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public PATCH owner must refresh stale A after B commits before creating successor C."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    binding_a_id = str(uuid4())
    binding_b_id = str(uuid4())
    old_generation_id = str(uuid4())
    with sessions.begin() as seed:
        tenant = Tenant(name=f"stale-patch-generation-{uuid4()}")
        tenant.id = tenant_id
        account = Account(name="Stale dataset editor", email=f"stale-patch-{uuid4()}@example.com")
        account.id = actor_id
        binding_a = DatasetCollectionBinding(
            provider_name="provider-a",
            model_name="model-a",
            type="dataset",
            collection_name=f"collection-{uuid4()}",
        )
        binding_a.id = binding_a_id
        binding_b = DatasetCollectionBinding(
            provider_name="provider-b",
            model_name="model-b",
            type="dataset",
            collection_name=f"collection-{uuid4()}",
        )
        binding_b.id = binding_b_id
        seed.add_all(
            [
                tenant,
                account,
                TenantAccountJoin(
                    tenant_id=tenant_id,
                    account_id=actor_id,
                    current=True,
                    role=TenantAccountRole.OWNER,
                ),
                binding_a,
                binding_b,
            ]
        )
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.chunk_structure = IndexStructureType.PARAGRAPH_INDEX
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.index_struct = None
        dataset.embedding_model_provider = "provider-a"
        dataset.embedding_model = "model-a"
        dataset.collection_binding_id = binding_a_id
        dataset.keyword_number = None
        dataset.indexing_applied_representation = _model_representation(
            provider="provider-a",
            model="model-a",
            binding_id=binding_a_id,
        ).model_dump_json()
        dataset.indexing_request_id = old_generation_id

    with sessions() as identity_session:
        current_actor = identity_session.get_one(Account, actor_id)
        current_tenant = identity_session.get_one(Tenant, tenant_id)
        current_actor.set_current_tenant_with_session(current_tenant, session=identity_session)

    manager = _ConfigurationModelManagerBoundary(tenant_id=tenant_id)
    dispatched: list[str] = []

    def model_manager_for_tenant(*, tenant_id: str) -> _ConfigurationModelManagerBoundary:
        assert tenant_id == current_actor.current_tenant_id
        return manager

    monkeypatch.setattr(dataset_service_module, "current_user", current_actor)
    monkeypatch.setattr(
        dataset_service_module.ModelManager,
        "for_tenant",
        staticmethod(model_manager_for_tenant),
    )
    monkeypatch.setattr(dataset_service_module, "enqueue_dataset_indexing_intent", dispatched.append)

    data_a = {
        "indexing_technique": IndexTechniqueType.HIGH_QUALITY,
        "embedding_model_provider": "provider-a",
        "embedding_model": "model-a",
    }
    data_b = {
        "indexing_technique": IndexTechniqueType.HIGH_QUALITY,
        "embedding_model_provider": "provider-b",
        "embedding_model": "model-b",
    }
    stale_loaded = Event()
    begin_stale_patch = Event()
    stale_lock_attempted = Event()
    stale_finished = Event()
    stale_errors: list[BaseException] = []
    worker = _WorkerRole()
    external_templates = _UnexpectedExternalTemplateLookup()

    def observe_stale_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "stale-patch" and _is_dataset_lock(statement, shared=False):
            stale_lock_attempted.set()

    def apply_stale_patch() -> None:
        try:
            with sessions() as stale_session:
                stale_dataset = stale_session.get_one(Dataset, dataset_id)
                account = stale_session.get_one(Account, actor_id)
                tenant = stale_session.get_one(Tenant, tenant_id)
                account.set_current_tenant_with_session(tenant, session=stale_session)
                assert stale_dataset.embedding_model == "model-a"
                assert stale_dataset.collection_binding_id == binding_a_id
                assert stale_dataset.indexing_request_id == old_generation_id
                stale_loaded.set()
                assert begin_stale_patch.wait(timeout=10)
                worker.role = "stale-patch"
                with DatasetService.update_dataset(
                    dataset_id,
                    dict(data_a),
                    account,
                    session=stale_session,
                    external_templates=external_templates,
                ) as updated:
                    assert updated.embedding_model == "model-a"
                    assert updated.embedding_model_provider == "provider-a"
                    assert updated.collection_binding_id == binding_a_id
        except BaseException as error:
            stale_errors.append(error)
        finally:
            stale_finished.set()

    stale_writer = Thread(target=apply_stale_patch)
    event.listen(engine, "before_cursor_execute", observe_stale_lock)
    try:
        stale_writer.start()
        assert stale_loaded.wait(timeout=10)

        with sessions() as writer_b:
            account = writer_b.get_one(Account, actor_id)
            tenant = writer_b.get_one(Tenant, tenant_id)
            account.set_current_tenant_with_session(tenant, session=writer_b)
            with DatasetService.update_dataset(
                dataset_id,
                dict(data_b),
                account,
                session=writer_b,
                external_templates=external_templates,
            ) as updated:
                assert updated.embedding_model == "model-b"
                assert updated.embedding_model_provider == "provider-b"
                assert updated.collection_binding_id == binding_b_id
                begin_stale_patch.set()
                assert stale_lock_attempted.wait(timeout=10)
                # The stale public PATCH is blocked only by this real InnoDB
                # Dataset row lock. Exiting the context commits B and releases it.
                assert not stale_finished.wait(timeout=0.25)

        _join_if_started(stale_writer)
        assert not stale_writer.is_alive()
        assert stale_errors == []
        assert len(dispatched) == 2
        intent_b_id, intent_c_id = dispatched
        assert intent_b_id != intent_c_id

        repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sessions)
        assert repository.claim(intent_b_id) is None
        intent_c = repository.claim(intent_c_id)
        assert intent_c is not None
        expected_a = _model_representation(provider="provider-a", model="model-a", binding_id=binding_a_id)
        assert intent_c.source == expected_a
        assert intent_c.target == expected_a
        with sessions() as verify:
            dataset = verify.get_one(Dataset, dataset_id)
            assert dataset.embedding_model == "model-a"
            assert dataset.embedding_model_provider == "provider-a"
            assert dataset.collection_binding_id == binding_a_id
            assert dataset.indexing_request_id == intent_c.generation_id
    finally:
        begin_stale_patch.set()
        _join_if_started(stale_writer)
        event.remove(engine, "before_cursor_execute", observe_stale_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(TenantAccountJoin).where(
                    TenantAccountJoin.tenant_id == tenant_id,
                    TenantAccountJoin.account_id == actor_id,
                )
            )
            cleanup.execute(delete(Account).where(Account.id == actor_id))
            cleanup.execute(delete(Tenant).where(Tenant.id == tenant_id))
            cleanup.execute(delete(DatasetCollectionBinding).where(DatasetCollectionBinding.id == binding_a_id))
            cleanup.execute(delete(DatasetCollectionBinding).where(DatasetCollectionBinding.id == binding_b_id))


def test_first_document_initialization_rejects_configuration_installed_while_waiting_for_dataset_lock(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale first-document request must refresh the Dataset after PATCH B wins its InnoDB lock."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    binding_b_id = str(uuid4())
    with sessions.begin() as seed:
        tenant = Tenant(name=f"first-document-patch-{uuid4()}")
        tenant.id = tenant_id
        account = Account(name="First document actor", email=f"first-document-{uuid4()}@example.com")
        account.id = actor_id
        binding_b = DatasetCollectionBinding(
            provider_name="provider-b",
            model_name="model-b",
            type="dataset",
            collection_name=f"collection-{uuid4()}",
        )
        binding_b.id = binding_b_id
        seed.add_all(
            [
                tenant,
                account,
                TenantAccountJoin(
                    tenant_id=tenant_id,
                    account_id=actor_id,
                    current=True,
                    role=TenantAccountRole.OWNER,
                ),
                binding_b,
            ]
        )
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.data_source_type = None
        dataset.indexing_technique = None
        dataset.index_struct = None
        dataset.embedding_model_provider = None
        dataset.embedding_model = None
        dataset.collection_binding_id = None
        dataset.indexing_applied_representation = None
        dataset.indexing_request_id = None

    with sessions() as identity_session:
        current_actor = identity_session.get_one(Account, actor_id)
        current_tenant = identity_session.get_one(Tenant, tenant_id)
        current_actor.set_current_tenant_with_session(current_tenant, session=identity_session)

    manager = _ConfigurationModelManagerBoundary(tenant_id=tenant_id)
    dispatched: list[str] = []

    def model_manager_for_tenant(*, tenant_id: str) -> _ConfigurationModelManagerBoundary:
        assert tenant_id == current_actor.current_tenant_id
        return manager

    monkeypatch.setattr(dataset_service_module, "current_user", current_actor)
    monkeypatch.setattr(
        dataset_service_module.ModelManager,
        "for_tenant",
        staticmethod(model_manager_for_tenant),
    )
    monkeypatch.setattr(dataset_service_module, "enqueue_dataset_indexing_intent", dispatched.append)

    prepared_a = KnowledgeConfig.model_validate(
        {
            "indexing_technique": "high_quality",
            "embedding_model_provider": "provider-a",
            "embedding_model": "model-a",
            "process_rule": {"mode": "automatic"},
            "data_source": {
                "info_list": {
                    "data_source_type": "upload_file",
                    "file_info_list": {"file_ids": [str(uuid4())]},
                }
            },
        }
    )
    patch_b = {
        "indexing_technique": IndexTechniqueType.HIGH_QUALITY,
        "embedding_model_provider": "provider-b",
        "embedding_model": "model-b",
    }
    stale_loaded = Event()
    begin_creation = Event()
    creation_lock_attempted = Event()
    creation_finished = Event()
    creation_errors: list[BaseException] = []
    worker = _WorkerRole()
    external_templates = _UnexpectedExternalTemplateLookup()

    def observe_creation_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "first-document" and _is_dataset_lock(statement, shared=False):
            creation_lock_attempted.set()

    def create_from_stale_configuration() -> None:
        try:
            with sessions() as creation_session:
                stale_dataset = creation_session.get_one(Dataset, dataset_id)
                account = creation_session.get_one(Account, actor_id)
                tenant = creation_session.get_one(Tenant, tenant_id)
                account.set_current_tenant_with_session(tenant, session=creation_session)
                assert stale_dataset.indexing_technique is None
                # These dirty prepared-A values reproduce the dangerous
                # pre-lock identity-map state. The locking current read must
                # refresh them without first flushing A over committed B.
                stale_dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
                stale_dataset.embedding_model_provider = "provider-a"
                stale_dataset.embedding_model = "model-a"
                stale_dataset.collection_binding_id = "binding-a"
                stale_loaded.set()
                assert begin_creation.wait(timeout=10)
                worker.role = "first-document"
                with pytest.raises(ValueError, match="Prepared embedding provider differs"):
                    DocumentService.save_prepared_documents(
                        stale_dataset,
                        prepared_a,
                        account,
                        session=creation_session,
                    )
                creation_session.rollback()
        except BaseException as error:
            creation_errors.append(error)
        finally:
            creation_finished.set()

    creator = Thread(target=create_from_stale_configuration)
    event.listen(engine, "before_cursor_execute", observe_creation_lock)
    try:
        creator.start()
        assert stale_loaded.wait(timeout=10)
        with sessions() as patch_session:
            account = patch_session.get_one(Account, actor_id)
            tenant = patch_session.get_one(Tenant, tenant_id)
            account.set_current_tenant_with_session(tenant, session=patch_session)
            with DatasetService.update_dataset(
                dataset_id,
                dict(patch_b),
                account,
                session=patch_session,
                external_templates=external_templates,
            ) as updated:
                assert updated.embedding_model_provider == "provider-b"
                assert updated.embedding_model == "model-b"
                assert updated.collection_binding_id == binding_b_id
                begin_creation.set()
                assert creation_lock_attempted.wait(timeout=10)
                assert not creation_finished.wait(timeout=0.25)

        _join_if_started(creator)
        assert not creator.is_alive()
        assert creation_errors == []
        assert len(dispatched) == 1
        with sessions() as verify:
            dataset = verify.get_one(Dataset, dataset_id)
            intent = verify.get_one(DatasetIndexingOutbox, dispatched[0])
            assert dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY
            assert dataset.embedding_model_provider == "provider-b"
            assert dataset.embedding_model == "model-b"
            assert dataset.collection_binding_id == binding_b_id
            assert dataset.indexing_request_id == intent.generation_id
            assert verify.scalar(select(Document.id).where(Document.dataset_id == dataset_id).limit(1)) is None
            assert (
                verify.scalar(select(DatasetProcessRule.id).where(DatasetProcessRule.dataset_id == dataset_id).limit(1))
                is None
            )
    finally:
        begin_creation.set()
        _join_if_started(creator)
        event.remove(engine, "before_cursor_execute", observe_creation_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(TenantAccountJoin).where(
                    TenantAccountJoin.tenant_id == tenant_id,
                    TenantAccountJoin.account_id == actor_id,
                )
            )
            cleanup.execute(delete(Account).where(Account.id == actor_id))
            cleanup.execute(delete(Tenant).where(Tenant.id == tenant_id))
            cleanup.execute(delete(DatasetCollectionBinding).where(DatasetCollectionBinding.id == binding_b_id))


def test_document_enable_intent_survives_parent_generation_switch_and_enables_summary(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Body and summary enable stay durable while the parent generation waits for the document commit."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    document_id = str(uuid4())
    segment_id = str(uuid4())
    old_generation_id = str(uuid4())
    source_g0 = _representation(collection="collection-g0")
    target_g1 = _representation(collection="collection-g1")
    with sessions.begin() as seed:
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.chunk_structure = IndexStructureType.PARAGRAPH_INDEX
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.index_struct = source_g0.index_struct
        dataset.embedding_model = source_g0.embedding_model
        dataset.embedding_model_provider = source_g0.embedding_model_provider
        dataset.collection_binding_id = source_g0.collection_binding_id
        dataset.keyword_number = source_g0.keyword_number
        dataset.indexing_applied_representation = source_g0.model_dump_json()
        dataset.indexing_request_id = old_generation_id
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch=str(uuid4()),
            name="enable-generation.txt",
            created_from=DocumentCreatedFrom.WEB,
            created_by=actor_id,
            indexing_status=IndexingStatus.COMPLETED,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            enabled=False,
        )
        document.completed_at = datetime.now(UTC).replace(tzinfo=None)
        segment = DocumentSegment(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            position=1,
            content="body enabled before parent generation switch",
            word_count=6,
            tokens=6,
            created_by=actor_id,
            index_node_id=f"body-{uuid4()}",
            index_node_hash=str(uuid4()),
            status=SegmentStatus.COMPLETED,
            enabled=False,
        )
        segment.id = segment_id
        summary = DocumentSegmentSummary(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_id=segment_id,
            summary_content="retained summary",
            summary_index_node_id=f"summary-{uuid4()}",
            summary_index_node_hash=str(uuid4()),
            status=SummaryStatus.COMPLETED,
            enabled=False,
        )
        seed.add_all([document, segment, summary])
        seed.flush()
        summary_id = summary.id
        initial_summary_version = summary.version

    actor = Account(name="Document enabler", email=f"enable-{uuid4()}@example.com")
    actor.id = actor_id
    model_manager = _ConfigurationModelManagerBoundary(tenant_id=tenant_id)

    def model_manager_for_tenant(*, tenant_id: str) -> _ConfigurationModelManagerBoundary:
        assert tenant_id == model_manager._tenant_id
        return model_manager

    monkeypatch.setattr(
        dataset_service_module.ModelManager,
        "for_tenant",
        staticmethod(model_manager_for_tenant),
    )
    redis = _DocumentStatusRedisBoundary()
    dispatched_documents: list[str] = []
    monkeypatch.setattr(dataset_service_module, "redis_client", redis)
    monkeypatch.setattr(dataset_service_module.add_document_to_index_task, "delay", dispatched_documents.append)

    enable_intent_inserted = Event()
    allow_enable_commit = Event()
    enable_finished = Event()
    generation_lock_attempted = Event()
    generation_finished = Event()
    enable_errors: list[BaseException] = []
    generation_errors: list[BaseException] = []
    generation_intent_ids: list[str] = []
    worker = _WorkerRole()

    def pause_enable_before_commit(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        normalized = " ".join(statement.lower().replace("`", "").split())
        if worker.role == "document-enable" and "insert into segment_state_outbox" in normalized:
            enable_intent_inserted.set()
            assert allow_enable_commit.wait(timeout=10)

    def observe_generation_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "generation-switch" and _is_dataset_lock(statement, shared=False):
            generation_lock_attempted.set()

    def enable_document() -> None:
        try:
            worker.role = "document-enable"
            with sessions() as enable_session:
                dataset = enable_session.get_one(Dataset, dataset_id)
                DocumentService.batch_update_document_status(
                    dataset,
                    [document_id],
                    "enable",
                    actor,
                    enable_session,
                )
        except BaseException as error:
            enable_errors.append(error)
        finally:
            enable_finished.set()

    def switch_generation() -> None:
        try:
            worker.role = "generation-switch"
            with sessions.begin() as generation_session:
                dataset = generation_session.scalar(
                    select(Dataset)
                    .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                assert dataset is not None
                dataset.index_struct = target_g1.index_struct
                dataset.embedding_model = target_g1.embedding_model
                dataset.embedding_model_provider = target_g1.embedding_model_provider
                dataset.collection_binding_id = target_g1.collection_binding_id
                (intent_id,) = create_dataset_indexing_intents(
                    generation_session,
                    DatasetIndexingJob(DatasetRef(tenant_id, dataset_id), source_g0, target_g1),
                )
                generation_intent_ids.append(intent_id)
        except BaseException as error:
            generation_errors.append(error)
        finally:
            generation_finished.set()

    enable_worker = Thread(target=enable_document)
    generation_worker = Thread(target=switch_generation)
    event.listen(engine, "after_cursor_execute", pause_enable_before_commit)
    event.listen(engine, "before_cursor_execute", observe_generation_lock)
    try:
        enable_worker.start()
        assert enable_intent_inserted.wait(timeout=10)
        generation_worker.start()
        assert generation_lock_attempted.wait(timeout=10)
        # The generation switch is queued on the actual Dataset X lock held by
        # the document-enable transaction whose durable intent is not yet committed.
        assert not generation_finished.wait(timeout=0.25)
        allow_enable_commit.set()
        _join_if_started(enable_worker)
        _join_if_started(generation_worker)
        assert not enable_worker.is_alive()
        assert not generation_worker.is_alive()
        assert enable_errors == []
        assert generation_errors == []
        assert dispatched_documents == [document_id]
        (generation_intent_id,) = generation_intent_ids

        with sessions() as pending:
            dataset = pending.get_one(Dataset, dataset_id)
            generation = pending.get_one(DatasetIndexingOutbox, generation_intent_id)
            segment = pending.get_one(DocumentSegment, segment_id)
            intent = pending.get_one(SegmentStateOutbox, segment.indexing_request_id)
            assert dataset.indexing_request_id == generation.generation_id
            assert segment.enabled is True
            assert intent.action == "enable"
            assert intent.summary_id == summary_id
            assert intent.summary_version == initial_summary_version
            segment_intent_id = intent.id
            generation_id = generation.generation_id

        processor = _SegmentStateProcessorBoundary()
        vector = _SegmentStateVectorBoundary()
        vector_constructor = _SegmentStateVectorConstructorBoundary(vector)
        monkeypatch.setattr(
            segment_state_task_module,
            "IndexProcessorFactory",
            _SegmentStateProcessorFactoryBoundary(processor),
        )
        monkeypatch.setattr(segment_state_task_module, "Vector", vector_constructor)

        assert segment_state_task_module.process_segment_state_intent(segment_intent_id, new_session=sessions) is True
        assert [(generation, len(documents)) for generation, documents in processor.loads] == [(generation_id, 1)]
        assert vector_constructor.generations
        assert set(vector_constructor.generations) == {generation_id}
        assert len(vector.added) == 1
        assert vector.added[0].metadata["original_chunk_id"] == segment_id

        with sessions() as verify:
            document = verify.get_one(Document, document_id)
            segment = verify.get_one(DocumentSegment, segment_id)
            summary = verify.get_one(DocumentSegmentSummary, summary_id)
            assert document.enabled is True
            assert segment.enabled is True
            assert segment.indexing_request_id is None
            assert verify.get(SegmentStateOutbox, segment_intent_id) is None
            assert summary.enabled is True
            assert summary.version == initial_summary_version + 1
            assert summary.vector_route is not None
            route = VectorRoute.model_validate_json(summary.vector_route)
            assert (route.vector_type, route.collection_name) == ("qdrant", "collection-g1")
            assert (
                verify.scalar(
                    select(SummaryVectorCleanupOutbox.id).where(
                        SummaryVectorCleanupOutbox.node_id == summary.summary_index_node_id,
                        SummaryVectorCleanupOutbox.vector_route_hash == summary.vector_route_hash,
                    )
                )
                is None
            )
    finally:
        allow_enable_commit.set()
        _join_if_started(enable_worker)
        _join_if_started(generation_worker)
        event.remove(engine, "after_cursor_execute", pause_enable_before_commit)
        event.remove(engine, "before_cursor_execute", observe_generation_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_current_enable_waits_for_remove_and_persists_new_segment_intent(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remove already doing I/O commits first; the waiting enable then becomes the durable desired state."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    document_id = str(uuid4())
    segment_id = str(uuid4())
    summary_route = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="summary-before-remove",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    with sessions.begin() as seed:
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.chunk_structure = IndexStructureType.PARAGRAPH_INDEX
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch=str(uuid4()),
            name="remove-then-enable.txt",
            created_from=DocumentCreatedFrom.WEB,
            created_by=actor_id,
            indexing_status=IndexingStatus.COMPLETED,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            enabled=False,
            disabled_by=actor_id,
        )
        document.completed_at = datetime.now(UTC).replace(tzinfo=None)
        segment = DocumentSegment(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            position=1,
            content="indexed before remove",
            word_count=3,
            tokens=3,
            created_by=actor_id,
            index_node_id=f"body-{uuid4()}",
            index_node_hash=str(uuid4()),
            status=SegmentStatus.COMPLETED,
            enabled=True,
        )
        segment.id = segment_id
        summary = DocumentSegmentSummary(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_id=segment_id,
            summary_content="summary before remove",
            summary_index_node_id=f"summary-{uuid4()}",
            summary_index_node_hash=str(uuid4()),
            vector_route=summary_route.serialized,
            vector_route_hash=summary_route.key,
            status=SummaryStatus.COMPLETED,
            enabled=True,
        )
        seed.add_all([document, segment, summary])
        seed.flush()
        summary_id = summary.id
        initial_summary_version = summary.version

    actor = Account(name="Current enabler", email=f"current-enable-{uuid4()}@example.com")
    actor.id = actor_id
    model_manager = _ConfigurationModelManagerBoundary(tenant_id=tenant_id)

    def model_manager_for_tenant(*, tenant_id: str) -> _ConfigurationModelManagerBoundary:
        assert tenant_id == model_manager._tenant_id
        return model_manager

    monkeypatch.setattr(
        dataset_service_module.ModelManager,
        "for_tenant",
        staticmethod(model_manager_for_tenant),
    )
    redis = _DocumentStatusRedisBoundary()
    clean_started = Event()
    allow_clean_return = Event()
    processor = _RemoveDocumentProcessorBoundary(
        clean_started=clean_started,
        allow_clean_return=allow_clean_return,
    )
    dispatched_documents: list[str] = []
    monkeypatch.setattr(remove_document_task_module, "session_factory", _SessionFactoryBoundary(sessions))
    monkeypatch.setattr(
        remove_document_task_module,
        "IndexProcessorFactory",
        _RemoveDocumentProcessorFactoryBoundary(processor),
    )
    monkeypatch.setattr(remove_document_task_module, "redis_client", redis)
    monkeypatch.setattr(dataset_service_module, "redis_client", redis)
    monkeypatch.setattr(dataset_service_module.add_document_to_index_task, "delay", dispatched_documents.append)

    remove_finished = Event()
    enable_lock_attempted = Event()
    enable_finished = Event()
    remove_results: list[bool] = []
    remove_errors: list[BaseException] = []
    enable_errors: list[BaseException] = []
    worker = _WorkerRole()

    def observe_enable_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "current-enable" and _is_dataset_lock(statement, shared=False):
            enable_lock_attempted.set()

    def remove_document() -> None:
        try:
            worker.role = "remove"
            remove_results.append(remove_document_task_module.remove_document_from_index_task.run(document_id))
        except BaseException as error:
            remove_errors.append(error)
        finally:
            remove_finished.set()

    def enable_document() -> None:
        try:
            worker.role = "current-enable"
            with sessions() as enable_session:
                dataset = enable_session.get_one(Dataset, dataset_id)
                DocumentService.batch_update_document_status(
                    dataset,
                    [document_id],
                    "enable",
                    actor,
                    enable_session,
                )
        except BaseException as error:
            enable_errors.append(error)
        finally:
            enable_finished.set()

    remove_worker = Thread(target=remove_document)
    enable_worker = Thread(target=enable_document)
    event.listen(engine, "before_cursor_execute", observe_enable_lock)
    try:
        remove_worker.start()
        assert clean_started.wait(timeout=10)
        enable_worker.start()
        assert enable_lock_attempted.wait(timeout=10)
        # Remove holds Dataset SHARE and Document UPDATE through its external
        # clean. The public enable's Dataset UPDATE lock therefore queues in
        # InnoDB before it can observe or mutate the Document row.
        assert not enable_finished.wait(timeout=0.25)
        allow_clean_return.set()
        _join_if_started(remove_worker)
        _join_if_started(enable_worker)
        assert not remove_worker.is_alive()
        assert not enable_worker.is_alive()
        assert remove_errors == []
        assert enable_errors == []
        assert remove_results == [True]
        assert dispatched_documents == [document_id]
        assert len(processor.cleaned) == 1

        with sessions() as verify:
            document = verify.get_one(Document, document_id)
            segment = verify.get_one(DocumentSegment, segment_id)
            summary = verify.get_one(DocumentSegmentSummary, summary_id)
            intent = verify.get_one(SegmentStateOutbox, segment.indexing_request_id)
            assert document.enabled is True
            assert document.archived is False
            assert segment.enabled is True
            assert intent.action == "enable"
            assert intent.summary_id == summary.id
            assert intent.summary_version == summary.version
            assert summary.enabled is False
            assert summary.version == initial_summary_version + 1

        # A late duplicate delivery must see the current enabled Document and
        # perform no additional vector or database cleanup.
        assert remove_document_task_module.remove_document_from_index_task.run(document_id) is False
        assert len(processor.cleaned) == 1
    finally:
        allow_clean_return.set()
        _join_if_started(remove_worker)
        _join_if_started(enable_worker)
        event.remove(engine, "before_cursor_execute", observe_enable_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_durable_document_disable_worker_serializes_with_new_enable_intent(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broker-lost disable remains durable, then a waiting enable replaces it after cleanup commits."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    document_id = str(uuid4())
    segment_id = str(uuid4())
    summary_route = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="summary-before-durable-disable",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    with sessions.begin() as seed:
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.chunk_structure = IndexStructureType.PARAGRAPH_INDEX
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.embedding_model = "embedding-model"
        dataset.embedding_model_provider = "embedding-provider"
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch=str(uuid4()),
            name="durable-disable-enable.txt",
            created_from=DocumentCreatedFrom.WEB,
            created_by=actor_id,
            indexing_status=IndexingStatus.COMPLETED,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            enabled=True,
        )
        document.completed_at = datetime.now(UTC).replace(tzinfo=None)
        segment = DocumentSegment(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            position=1,
            content="body before durable disable",
            word_count=4,
            tokens=4,
            created_by=actor_id,
            index_node_id=f"body-{uuid4()}",
            index_node_hash=str(uuid4()),
            status=SegmentStatus.COMPLETED,
            enabled=True,
        )
        segment.id = segment_id
        summary = DocumentSegmentSummary(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_id=segment_id,
            summary_content="summary before durable disable",
            summary_index_node_id=f"summary-{uuid4()}",
            summary_index_node_hash=str(uuid4()),
            vector_route=summary_route.serialized,
            vector_route_hash=summary_route.key,
            status=SummaryStatus.COMPLETED,
            enabled=True,
        )
        seed.add_all([document, segment, summary])
        seed.flush()
        summary_id = summary.id

    actor = Account(name="Disable and enable actor", email=f"disable-enable-{uuid4()}@example.com")
    actor.id = actor_id
    redis = _DocumentStatusRedisBoundary()
    model_manager = _ConfigurationModelManagerBoundary(tenant_id=tenant_id)

    def model_manager_for_tenant(*, tenant_id: str) -> _ConfigurationModelManagerBoundary:
        assert tenant_id == model_manager._tenant_id
        return model_manager

    def reject_disable_publication(_intent_ids: Sequence[str]) -> None:
        raise OSError("broker unavailable")

    dispatched_documents: list[str] = []
    monkeypatch.setattr(dataset_service_module, "redis_client", redis)
    monkeypatch.setattr(
        dataset_service_module.ModelManager,
        "for_tenant",
        staticmethod(model_manager_for_tenant),
    )
    monkeypatch.setattr(dataset_service_module, "publish_segment_state_intents", reject_disable_publication)
    monkeypatch.setattr(dataset_service_module.add_document_to_index_task, "delay", dispatched_documents.append)

    with sessions() as disable_session:
        dataset = disable_session.get_one(Dataset, dataset_id)
        DocumentService.batch_update_document_status(
            dataset,
            [document_id],
            "disable",
            actor,
            disable_session,
        )

    with sessions() as pending:
        document = pending.get_one(Document, document_id)
        segment = pending.get_one(DocumentSegment, segment_id)
        disable_intent = pending.get_one(SegmentStateOutbox, segment.indexing_request_id)
        assert document.enabled is False
        assert segment.enabled is False
        assert disable_intent.action == "disable"
        disable_intent_id = disable_intent.id

    clean_started = Event()
    allow_clean_return = Event()
    processor = _RemoveDocumentProcessorBoundary(
        clean_started=clean_started,
        allow_clean_return=allow_clean_return,
    )
    monkeypatch.setattr(
        segment_state_task_module,
        "IndexProcessorFactory",
        _SegmentStateDisableProcessorFactoryBoundary(processor),
    )

    disable_finished = Event()
    enable_lock_attempted = Event()
    enable_finished = Event()
    disable_results: list[bool] = []
    disable_errors: list[BaseException] = []
    enable_errors: list[BaseException] = []
    worker = _WorkerRole()

    def observe_enable_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "enable-after-disable" and _is_dataset_lock(statement, shared=False):
            enable_lock_attempted.set()

    def process_disable() -> None:
        try:
            worker.role = "disable-worker"
            disable_results.append(
                segment_state_task_module.process_segment_state_intent(disable_intent_id, new_session=sessions)
            )
        except BaseException as error:
            disable_errors.append(error)
        finally:
            disable_finished.set()

    def enable_document() -> None:
        try:
            worker.role = "enable-after-disable"
            with sessions() as enable_session:
                dataset = enable_session.get_one(Dataset, dataset_id)
                DocumentService.batch_update_document_status(
                    dataset,
                    [document_id],
                    "enable",
                    actor,
                    enable_session,
                )
        except BaseException as error:
            enable_errors.append(error)
        finally:
            enable_finished.set()

    disable_worker = Thread(target=process_disable)
    enable_worker = Thread(target=enable_document)
    event.listen(engine, "before_cursor_execute", observe_enable_lock)
    try:
        disable_worker.start()
        assert clean_started.wait(timeout=10)
        enable_worker.start()
        assert enable_lock_attempted.wait(timeout=10)
        # The disable worker retains Dataset SHARE and exact Document/Segment
        # ownership across body cleanup. The new command waits on Dataset X.
        assert not enable_finished.wait(timeout=0.25)
        allow_clean_return.set()
        _join_if_started(disable_worker)
        _join_if_started(enable_worker)
        assert not disable_worker.is_alive()
        assert not enable_worker.is_alive()
        assert disable_errors == []
        assert enable_errors == []
        assert disable_results == [True]
        assert dispatched_documents == [document_id]
        assert len(processor.cleaned) == 1

        with sessions() as verify:
            document = verify.get_one(Document, document_id)
            segment = verify.get_one(DocumentSegment, segment_id)
            summary = verify.get_one(DocumentSegmentSummary, summary_id)
            enable_intent = verify.get_one(SegmentStateOutbox, segment.indexing_request_id)
            assert document.enabled is True
            assert segment.enabled is True
            assert enable_intent.id != disable_intent_id
            assert enable_intent.action == "enable"
            assert enable_intent.summary_id == summary.id
            assert enable_intent.summary_version == summary.version
            assert summary.enabled is False
            assert verify.get(SegmentStateOutbox, disable_intent_id) is None

        assert segment_state_task_module.process_segment_state_intent(disable_intent_id, new_session=sessions) is False
        assert len(processor.cleaned) == 1
    finally:
        allow_clean_return.set()
        _join_if_started(disable_worker)
        _join_if_started(enable_worker)
        event.remove(engine, "before_cursor_execute", observe_enable_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_segment_batch_publication_filters_member_changed_after_materialization(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed member edit between batch snapshots and publication excludes only that member."""

    _engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    generation_intent_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-initial",
        target_collection="collection-batch-current",
    )
    document_id = str(uuid4())
    first_segment_id = str(uuid4())
    second_segment_id = str(uuid4())
    with sessions.begin() as seed:
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.chunk_structure = IndexStructureType.PARAGRAPH_INDEX
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch=str(uuid4()),
            name="batch-current-guard.txt",
            created_from=DocumentCreatedFrom.WEB,
            created_by=actor_id,
            indexing_status=IndexingStatus.COMPLETED,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            enabled=True,
            archived=False,
        )
        document.completed_at = datetime.now(UTC).replace(tzinfo=None)
        segments = []
        for position, (segment_id, content) in enumerate(
            ((first_segment_id, "first materialized"), (second_segment_id, "second materialized")),
            start=1,
        ):
            segment = DocumentSegment(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                position=position,
                content=content,
                word_count=2,
                tokens=2,
                created_by=actor_id,
                index_node_id=f"node-{segment_id}",
                index_node_hash=f"hash-{segment_id}",
                status=SegmentStatus.COMPLETED,
                enabled=True,
            )
            segment.id = segment_id
            segments.append(segment)
        seed.add_all([document, *segments])

    route_recorded = Event()
    writer_committed = Event()
    writer_errors: list[BaseException] = []
    vector = _SegmentBatchVectorBoundary()

    def edit_second_member() -> None:
        try:
            assert route_recorded.wait(timeout=10)
            with sessions.begin() as writer:
                second = writer.scalar(
                    select(DocumentSegment)
                    .where(
                        DocumentSegment.tenant_id == tenant_id,
                        DocumentSegment.dataset_id == dataset_id,
                        DocumentSegment.document_id == document_id,
                        DocumentSegment.id == second_segment_id,
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                assert second is not None
                second.content = "second committed after batch read"
                second.word_count = 5
                second.tokens = 5
                second.index_node_id = f"node-current-{second_segment_id}"
                second.index_node_hash = f"hash-current-{second_segment_id}"
        except BaseException as error:
            writer_errors.append(error)
        finally:
            writer_committed.set()

    writer = Thread(target=edit_second_member)
    monkeypatch.setattr(
        segment_indexing_module,
        "Vector",
        _SegmentBatchVectorConstructorBoundary(vector, tenant_id=tenant_id),
    )
    repository = SQLAlchemySegmentRepository(session_factory=sessions)
    record_batch_route = repository.record_batch_indexing_route

    def record_route_then_edit(
        document_ref: DocumentRef,
        snapshots: Sequence[SegmentIndexingSnapshot],
        route: VectorRoute,
    ) -> tuple[SegmentIndexingSnapshot, ...]:
        current = record_batch_route(document_ref, snapshots, route)
        route_recorded.set()
        assert writer_committed.wait(timeout=10)
        return current

    monkeypatch.setattr(repository, "record_batch_indexing_route", record_route_then_edit)
    gateway = SegmentIndexingGateway(
        segments=repository,
        uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=sessions),
        attachment_vectors=SQLAlchemyAttachmentVectorStore(session_factory=sessions),
        child_vector_cleanups=SQLAlchemyChildVectorCleanupRepository(session_factory=sessions),
        redis=_SegmentBatchRedisBoundary(),
        new_session=sessions,
    )
    document_ref = DatasetRef(tenant_id, dataset_id).document(document_id)
    try:
        writer.start()
        gateway.create_many(
            document_ref,
            segment_ids=(first_segment_id, second_segment_id),
            keywords_list=None,
        )
        _join_if_started(writer)

        assert not writer.is_alive()
        assert writer_errors == []
        assert vector.batches == [("first materialized",)]
        with sessions() as verify:
            second = verify.get_one(DocumentSegment, second_segment_id)
            assert (
                second.content,
                second.tokens,
                second.index_node_id,
                second.index_node_hash,
            ) == (
                "second committed after batch read",
                5,
                f"node-current-{second_segment_id}",
                f"hash-current-{second_segment_id}",
            )
            target = DatasetIndexingRepresentation.model_validate_json(
                verify.get_one(DatasetIndexingOutbox, generation_intent_id).target_representation
            )
            assert [(route.vector_type, route.collection_name) for route in target.cleanup_routes] == [
                ("qdrant", "collection-batch-current")
            ]
    finally:
        route_recorded.set()
        writer_committed.set()
        _join_if_started(writer)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_economy_segment_batch_persists_keywords_in_publication_transaction(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Economy publication must update locked Segment rows through its existing MySQL transaction."""

    _engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    document_id = str(uuid4())
    first_segment_id = str(uuid4())
    second_segment_id = str(uuid4())
    with sessions.begin() as seed:
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.indexing_technique = IndexTechniqueType.ECONOMY
        dataset.index_struct = None
        dataset.keyword_number = 10
        dataset.chunk_structure = IndexStructureType.PARAGRAPH_INDEX
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch=str(uuid4()),
            name="economy-batch-keywords.txt",
            created_from=DocumentCreatedFrom.WEB,
            created_by=actor_id,
            indexing_status=IndexingStatus.COMPLETED,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            enabled=True,
            archived=False,
        )
        document.completed_at = datetime.now(UTC).replace(tzinfo=None)
        segments = []
        for position, (segment_id, content) in enumerate(
            ((first_segment_id, "first economy"), (second_segment_id, "second economy")),
            start=1,
        ):
            segment = DocumentSegment(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                position=position,
                content=content,
                word_count=2,
                tokens=0,
                created_by=actor_id,
                index_node_id=f"node-{segment_id}",
                index_node_hash=f"hash-{segment_id}",
                status=SegmentStatus.COMPLETED,
                enabled=True,
            )
            segment.id = segment_id
            segments.append(segment)
        seed.add_all([document, *segments])

    monkeypatch.setattr(segment_indexing_module, "Jieba", _SegmentBatchKeywordConstructorBoundary())
    repository = SQLAlchemySegmentRepository(session_factory=sessions)
    gateway = SegmentIndexingGateway(
        segments=repository,
        uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=sessions),
        attachment_vectors=SQLAlchemyAttachmentVectorStore(session_factory=sessions),
        child_vector_cleanups=SQLAlchemyChildVectorCleanupRepository(session_factory=sessions),
        redis=_SegmentBatchRedisBoundary(),
        new_session=sessions,
    )
    try:
        gateway.create_many(
            DatasetRef(tenant_id, dataset_id).document(document_id),
            segment_ids=(first_segment_id, second_segment_id),
            keywords_list=(("first-keyword",), ("second-keyword",)),
        )

        with sessions() as verify:
            keyword_table = verify.scalar(
                select(DatasetKeywordTable).where(DatasetKeywordTable.dataset_id == dataset_id)
            )
            assert keyword_table is not None
            assert keyword_table.data_source_type == "database"
            assert json.loads(keyword_table.keyword_table) == {
                "__data__": {
                    f"node-{first_segment_id}": ["first-keyword"],
                    f"node-{second_segment_id}": ["second-keyword"],
                }
            }
            assert verify.get_one(DocumentSegment, first_segment_id).keywords == ["first-keyword"]
            assert verify.get_one(DocumentSegment, second_segment_id).keywords == ["second-keyword"]
    finally:
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_successor_generation_inherits_route_published_by_an_ordinary_document_writer(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document writer must durably record B before C can supersede the pending rebuild."""

    _engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    route_a = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    route_b = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="collection-b",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    with sessions.begin() as session:
        dataset = session.get_one(Dataset, dataset_id)
        dataset.indexing_applied_representation = (
            _representation(collection="collection-a")
            .model_copy(update={"vector_route": route_a, "cleanup_routes": ()})
            .model_dump_json()
        )

    intent_b_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-a",
        target_collection="collection-b",
    )
    with sessions() as session:
        generation_b = session.get_one(DatasetIndexingOutbox, intent_b_id).generation_id
    request_id, document_id = _seed_document_request(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        actor_id=actor_id,
    )
    documents = SQLAlchemyDocumentRepository(session_factory=sessions)
    (document_claim,) = documents.claim_indexing_request(
        request_id,
        expected_action=DocumentIndexingAction.DUPLICATE.value,
        processing_started_at=datetime.now(UTC).replace(tzinfo=None),
        admission_error=lambda _count: None,
    )
    assert document_claim.dataset_generation_id == generation_b

    class CurrentRouteVector:
        def __init__(self, dataset: Dataset, *, session: Session | None, vector_type: str) -> None:
            assert dataset.id == dataset_id
            assert session is None
            assert vector_type == route_b.vector_type

        @staticmethod
        def resolve_vector_type(dataset: Dataset, *, session: Session) -> str:
            assert dataset.id == dataset_id
            assert session.in_transaction()
            return route_b.vector_type

        @staticmethod
        def get_routing_target() -> tuple[str, str]:
            return route_b.vector_type, route_b.collection_name

    monkeypatch.setattr(indexing_execution_adapter_module, "Vector", CurrentRouteVector)
    adapter = IndexingExecutionAdapter(
        session_factory=sessions,
        documents=documents,
        segments=SQLAlchemySegmentRepository(session_factory=sessions),
        uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=sessions),
        attachment_vectors=SQLAlchemyAttachmentVectorStore(session_factory=sessions),
        sources=CompositeStoredSourceResolver(adapters={}),
    )
    indexing_document = IndexingDocument(
        source=StoredSource(
            DatasetRef(tenant_id, dataset_id).document(document_id),
            "upload_file",
            {},
            IndexStructureType.PARAGRAPH_INDEX,
        ),
        processing_rule={},
        doc_language="English",
        need_summary=False,
        claim_id=document_claim.claim_id,
        indexing_request_id=request_id,
        dataset_generation_id=generation_b,
    )

    try:
        assert adapter._record_vector_route_before_write(indexing_document) == route_b

        with sessions() as verifier:
            recorded_b = DatasetIndexingRepresentation.model_validate_json(
                verifier.get_one(DatasetIndexingOutbox, intent_b_id).target_representation
            )
            assert recorded_b.vector_route is None
            assert recorded_b.cleanup_routes == (route_b,)

        with pytest.raises(DocumentIndexingClaimLostError):
            adapter._record_vector_route_before_write(
                IndexingDocument(
                    source=indexing_document.source,
                    processing_rule=indexing_document.processing_rule,
                    doc_language=indexing_document.doc_language,
                    need_summary=indexing_document.need_summary,
                    claim_id=str(uuid4()),
                    indexing_request_id=request_id,
                    dataset_generation_id=generation_b,
                )
            )

        intent_c_id = _create_generation(
            sessions,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            source_collection="collection-b",
            target_collection="collection-c",
        )
        with sessions() as verifier:
            intent_c = verifier.get_one(DatasetIndexingOutbox, intent_c_id)
            source_c = DatasetIndexingRepresentation.model_validate_json(intent_c.source_representation)
            target_c = DatasetIndexingRepresentation.model_validate_json(intent_c.target_representation)
            assert source_c.vector_route == route_a
            assert {route.key for route in source_c.cleanup_routes} == {route_a.key, route_b.key}
            assert target_c.vector_route is None
            assert target_c.cleanup_routes == ()
    finally:
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_duplicate_publication_and_rebuild_serialize_to_a_consistent_terminal_index(
    mysql_session: Session,
) -> None:
    """The rebuild X lock must wait for the duplicate writer's actual SHARE-guarded publication."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    intent_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-a",
        target_collection="collection-b",
    )
    rebuilds = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sessions)
    rebuild_intent = rebuilds.claim(intent_id)
    assert rebuild_intent is not None
    request_id, document_id, node_id = _seed_duplicate_request(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        actor_id=actor_id,
    )
    documents = SQLAlchemyDocumentRepository(session_factory=sessions)
    (document_claim,) = documents.claim_indexing_request(
        request_id,
        expected_action="duplicate",
        processing_started_at=datetime.now(UTC).replace(tzinfo=None),
        admission_error=lambda _count: None,
    )
    vectors: set[str] = set()
    writer_locked = Event()
    allow_publication = Event()
    publication_finished = Event()
    rebuild_attempted = Event()
    rebuild_finished = Event()
    writer_errors: list[BaseException] = []
    rebuild_errors: list[BaseException] = []
    worker = _WorkerRole()

    def observe_rebuild_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "rebuild" and _is_dataset_lock(statement, shared=False):
            rebuild_attempted.set()

    def publish_duplicate() -> None:
        try:
            with sessions.begin() as session:
                dataset = lock_dataset_indexing_generation_shared(
                    session,
                    dataset_id=dataset_id,
                    generation_id=rebuild_intent.generation_id,
                )
                assert dataset is not None
                assert dataset.tenant_id == tenant_id
                document = require_indexing_document(
                    session,
                    document_claim.ref,
                    claim_id=document_claim.claim_id,
                    lock=True,
                )
                assert document.indexing_dataset_generation_id == rebuild_intent.generation_id
                segment = session.scalar(
                    select(DocumentSegment)
                    .where(
                        DocumentSegment.tenant_id == tenant_id,
                        DocumentSegment.dataset_id == dataset_id,
                        DocumentSegment.document_id == document_id,
                        DocumentSegment.index_node_id == node_id,
                    )
                    .with_for_update()
                )
                assert segment is not None
                writer_locked.set()
                assert allow_publication.wait(timeout=10)
                vectors.add(node_id)
                segment.status = SegmentStatus.COMPLETED
                segment.enabled = True
            publication_finished.set()
        except BaseException as error:
            writer_errors.append(error)
            publication_finished.set()

    def attempt_rebuild() -> None:
        try:
            worker.role = "rebuild"
            with sessions.begin() as session:
                lock_dataset_indexing_execution(
                    session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    generation_id=rebuild_intent.generation_id,
                    intent_id=rebuild_intent.id,
                    claim_id=rebuild_intent.claim_id,
                    require_quiescent=True,
                )
        except BaseException as error:
            rebuild_errors.append(error)
        finally:
            rebuild_finished.set()

    writer = Thread(target=publish_duplicate)
    rebuild = Thread(target=attempt_rebuild)
    event.listen(engine, "before_cursor_execute", observe_rebuild_lock)
    try:
        writer.start()
        assert writer_locked.wait(timeout=10)
        rebuild.start()
        assert rebuild_attempted.wait(timeout=10)
        # No Python mutex protects the rebuild. Its only blocker is the real
        # InnoDB SHARE lock held by the duplicate publication transaction.
        assert not rebuild_finished.wait(timeout=0.25)
        allow_publication.set()
        _join_if_started(writer)
        _join_if_started(rebuild)

        assert not writer.is_alive()
        assert not rebuild.is_alive()
        assert writer_errors == []
        assert len(rebuild_errors) == 1
        assert isinstance(rebuild_errors[0], DatasetIndexingWritersActiveError)
        assert publication_finished.is_set()
        # A quiescence rejection must not clean an index written by the current
        # document generation.
        assert vectors == {node_id}

        assert rebuilds.release(rebuild_intent)
        documents.complete_indexing(
            document_claim.ref,
            claim_id=document_claim.claim_id,
            tokens=3,
            latency=0.01,
        )

        retry_intent = rebuilds.claim(intent_id)
        assert retry_intent is not None
        with sessions.begin() as session:
            dataset = lock_dataset_indexing_execution(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                generation_id=retry_intent.generation_id,
                intent_id=retry_intent.id,
                claim_id=retry_intent.claim_id,
                require_quiescent=True,
            )
            assert dataset is not None
            vectors.clear()
            vectors.update(
                node_id
                for node_id in session.scalars(
                    select(DocumentSegment.index_node_id).where(
                        DocumentSegment.tenant_id == tenant_id,
                        DocumentSegment.dataset_id == dataset_id,
                        DocumentSegment.status == SegmentStatus.COMPLETED,
                        DocumentSegment.enabled.is_(True),
                        DocumentSegment.index_node_id.is_not(None),
                    )
                )
                if node_id is not None
            )
        assert rebuilds.mark_vector_completed(retry_intent)
        assert rebuilds.complete(retry_intent)

        with sessions() as verifier:
            document = verifier.get_one(Document, document_id)
            segment = verifier.scalar(
                select(DocumentSegment).where(
                    DocumentSegment.document_id == document_id,
                    DocumentSegment.index_node_id == node_id,
                )
            )
            assert document.indexing_status == IndexingStatus.COMPLETED
            assert document.indexing_request_id is None
            assert document.indexing_claim_id is None
            assert document.indexing_dataset_generation_id is None
            assert segment is not None
            assert segment.status == SegmentStatus.COMPLETED
            assert segment.enabled is True
            assert vectors == {node_id}
            assert verifier.get(DocumentIndexingOutbox, request_id) is None
            assert verifier.get(DatasetIndexingOutbox, intent_id) is None
    finally:
        allow_publication.set()
        _join_if_started(writer)
        _join_if_started(rebuild)
        event.remove(engine, "before_cursor_execute", observe_rebuild_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_pipeline_user_retry_waits_for_current_workflow_and_pins_the_committed_snapshot(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user retry waiting on Pipeline-X must bind the freshly committed workflow, not its RR snapshot."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    pipeline_id = str(uuid4())
    workflow_v1_id = str(uuid4())
    workflow_v2_id = str(uuid4())
    workflow_v3_id = str(uuid4())
    document_id = str(uuid4())
    source_log_id: str
    with sessions.begin() as seed:
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.runtime_mode = DatasetRuntimeMode.RAG_PIPELINE
        dataset.pipeline_id = pipeline_id
        for workflow_id, version in (
            (workflow_v1_id, "1"),
            (workflow_v2_id, "2"),
            (workflow_v3_id, "3"),
        ):
            seed.add(
                Workflow(
                    id=workflow_id,
                    tenant_id=tenant_id,
                    app_id=pipeline_id,
                    type=WorkflowType.RAG_PIPELINE,
                    version=version,
                    marked_name="",
                    marked_comment="",
                    graph=json.dumps({"nodes": []}),
                    features=json.dumps({}),
                    created_by=actor_id,
                    created_at=datetime.now(UTC).replace(tzinfo=None),
                    updated_by=None,
                    updated_at=datetime.now(UTC).replace(tzinfo=None),
                    environment_variables=[],
                    conversation_variables=[],
                    rag_pipeline_variables=[],
                )
            )
        pipeline = Pipeline(
            tenant_id=tenant_id,
            name="User retry workflow snapshot",
            description="",
            workflow_id=workflow_v1_id,
            is_public=False,
            is_published=True,
            created_by=actor_id,
        )
        pipeline.id = pipeline_id
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch=str(uuid4()),
            name="user-retry.txt",
            created_from=DocumentCreatedFrom.WEB,
            created_by=actor_id,
            indexing_status=IndexingStatus.ERROR,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            enabled=True,
        )
        source_log = DocumentPipelineExecutionLog(
            pipeline_id=pipeline_id,
            document_id=document_id,
            datasource_type=DataSourceType.UPLOAD_FILE,
            datasource_info=json.dumps({"upload_file_id": str(uuid4())}),
            datasource_node_id="source-v1",
            input_data={"query": "original"},
            created_by=actor_id,
            workflow_id=workflow_v1_id,
            original_document_id=None,
        )
        seed.add_all([pipeline, document, source_log])
        seed.flush()
        source_log_id = source_log.id

    retry_redis = _RetryRedisBoundary()
    dispatched: list[str] = []
    monkeypatch.setattr(dataset_service_module, "redis_client", retry_redis)
    monkeypatch.setattr(dataset_service_module, "enqueue_document_indexing_intent", dispatched.append)

    stale_loaded = Event()
    begin_retry = Event()
    pipeline_lock_attempted = Event()
    retry_finished = Event()
    retry_errors: list[BaseException] = []
    worker = _WorkerRole()

    def observe_pipeline_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        normalized = " ".join(statement.lower().replace("`", "").split())
        if (
            worker.role == "pipeline-user-retry"
            and " from pipelines " in normalized
            and (" lock in share mode" in normalized or " for share" in normalized)
        ):
            pipeline_lock_attempted.set()

    def retry_from_stale_snapshot() -> None:
        try:
            worker.role = "pipeline-user-retry"
            with sessions() as retry_session:
                stale_document = retry_session.get_one(Document, document_id)
                assert stale_document.indexing_status == IndexingStatus.ERROR
                stale_loaded.set()
                assert begin_retry.wait(timeout=10)
                DocumentService.retry_document(
                    dataset_id,
                    [stale_document],
                    retry_session,
                    actor_id=actor_id,
                )
        except BaseException as error:
            retry_errors.append(error)
        finally:
            retry_finished.set()

    retry_thread = Thread(target=retry_from_stale_snapshot)
    event.listen(engine, "before_cursor_execute", observe_pipeline_lock)
    try:
        retry_thread.start()
        assert stale_loaded.wait(timeout=10)
        with sessions.begin() as publisher:
            pipeline = publisher.scalar(select(Pipeline).where(Pipeline.id == pipeline_id).with_for_update())
            assert pipeline is not None
            pipeline.workflow_id = workflow_v2_id
            publisher.flush()
            begin_retry.set()
            assert pipeline_lock_attempted.wait(timeout=10)
            assert not retry_finished.wait(timeout=0.25)

        _join_if_started(retry_thread)
        assert not retry_thread.is_alive()
        assert retry_errors == []
        assert len(dispatched) == 1

        with sessions.begin() as publisher:
            publisher.get_one(Pipeline, pipeline_id).workflow_id = workflow_v3_id

        with sessions() as verify:
            document = verify.get_one(Document, document_id)
            request = verify.get_one(DocumentIndexingOutbox, dispatched[0])
            assert document.indexing_status == IndexingStatus.WAITING
            assert document.indexing_request_id == request.id
            assert request.pipeline_execution_log_id is not None
            retry_log = verify.get_one(DocumentPipelineExecutionLog, request.pipeline_execution_log_id)
            assert retry_log.id != source_log_id
            assert retry_log.workflow_id == workflow_v2_id
            assert retry_log.original_document_id is None
            assert retry_log.datasource_node_id == "source-v1"
            assert retry_log.input_data == {"query": "original"}
            assert verify.get_one(Pipeline, pipeline_id).workflow_id == workflow_v3_id
    finally:
        begin_retry.set()
        _join_if_started(retry_thread)
        event.remove(engine, "before_cursor_execute", observe_pipeline_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_published_pipeline_claim_serializes_with_rebuild_and_recovers_under_successor(
    mysql_session: Session,
) -> None:
    """A published-pipeline RETRY owner is fenced at the real mutation boundary and remains replayable."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    first_intent_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-initial",
        target_collection="collection-g0",
    )
    with sessions() as verifier:
        first_generation_id = verifier.get_one(DatasetIndexingOutbox, first_intent_id).generation_id

    pipeline_id = str(uuid4())
    workflow_v1_id = str(uuid4())
    workflow_v2_id = str(uuid4())
    document_id = str(uuid4())
    document_batch = str(uuid4())
    execution_log_id: str
    with sessions.begin() as session:
        dataset = session.get_one(Dataset, dataset_id)
        dataset.pipeline_id = pipeline_id
        for workflow_id, version in ((workflow_v1_id, "1"), (workflow_v2_id, "2")):
            session.add(
                Workflow(
                    id=workflow_id,
                    tenant_id=tenant_id,
                    app_id=pipeline_id,
                    type=WorkflowType.RAG_PIPELINE,
                    version=version,
                    marked_name="",
                    marked_comment="",
                    graph=json.dumps({"nodes": []}),
                    features=json.dumps({}),
                    created_by=actor_id,
                    created_at=datetime.now(UTC).replace(tzinfo=None),
                    updated_by=None,
                    updated_at=datetime.now(UTC).replace(tzinfo=None),
                    environment_variables=[],
                    conversation_variables=[],
                    rag_pipeline_variables=[],
                )
            )
        pipeline = Pipeline(
            tenant_id=tenant_id,
            name="Pinned replay pipeline",
            description="",
            workflow_id=workflow_v1_id,
            is_public=False,
            is_published=True,
            created_by=actor_id,
        )
        pipeline.id = pipeline_id
        session.add(pipeline)
        session.add(
            Document(
                id=document_id,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                position=1,
                data_source_type=DataSourceType.UPLOAD_FILE,
                batch=document_batch,
                name="published-pipeline.txt",
                created_from=DocumentCreatedFrom.WEB,
                created_by=actor_id,
                indexing_status=IndexingStatus.WAITING,
                doc_form=IndexStructureType.PARAGRAPH_INDEX,
                enabled=True,
            )
        )
        session.flush()
        execution_log = DocumentPipelineExecutionLog(
            pipeline_id=pipeline_id,
            document_id=document_id,
            datasource_type=DataSourceType.UPLOAD_FILE,
            datasource_info=json.dumps({}),
            datasource_node_id="source-node",
            input_data={},
            created_by=actor_id,
            workflow_id=workflow_v1_id,
            original_document_id=None,
        )
        session.add(execution_log)
        session.flush()
        execution_log_id = execution_log.id
        request_ids = create_pipeline_document_indexing_intents(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            pipeline_id=pipeline_id,
            pipeline_execution_log_ids={document_id: execution_log_id},
            actor_id=actor_id,
        )
        request_id = request_ids[document_id]

    with sessions.begin() as session:
        session.get_one(Pipeline, pipeline_id).workflow_id = workflow_v2_id

    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sessions)
    stale_v2_delivery = outbox.get_pipeline_intent(
        request_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        pipeline_id=pipeline_id,
        document_id=document_id,
        actor_id=actor_id,
        workflow_id=workflow_v2_id,
        datasource_type=str(DataSourceType.UPLOAD_FILE),
        datasource_info={},
        datasource_node_id="source-node",
        batch=document_batch,
        original_document_id=None,
    )
    assert stale_v2_delivery is None
    pinned_v1_delivery = outbox.get_pipeline_intent(
        request_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        pipeline_id=pipeline_id,
        document_id=document_id,
        actor_id=actor_id,
        workflow_id=workflow_v1_id,
        datasource_type=str(DataSourceType.UPLOAD_FILE),
        datasource_info={},
        datasource_node_id="source-node",
        batch=document_batch,
        original_document_id=None,
    )
    assert pinned_v1_delivery is not None
    assert pinned_v1_delivery.pipeline_execution_log_id == execution_log_id

    documents = SQLAlchemyDocumentRepository(session_factory=sessions)
    (old_claim,) = documents.claim_indexing_request(
        request_id,
        expected_action=DocumentIndexingAction.RETRY.value,
        processing_started_at=datetime.now(UTC).replace(tzinfo=None),
        admission_error=lambda _count: None,
    )
    assert old_claim.dataset_generation_id == first_generation_id

    guard_entered = Event()
    allow_guard_commit = Event()
    rebuild_attempted = Event()
    rebuild_finished = Event()
    writer_errors: list[BaseException] = []
    rebuild_errors: list[BaseException] = []
    successor_generation_ids: list[str] = []
    worker = _WorkerRole()

    def observe_rebuild_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "pipeline-rebuild" and _is_dataset_lock(statement, shared=False):
            rebuild_attempted.set()

    def hold_pipeline_mutation_guard() -> None:
        try:
            worker.role = "pipeline-writer"
            with sessions.begin() as session:
                with protect_pipeline_indexing_claim(
                    session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    indexing_request_id=request_id,
                    claim_id=old_claim.claim_id,
                ):
                    guard_entered.set()
                    assert allow_guard_commit.wait(timeout=10)
        except BaseException as error:
            writer_errors.append(error)
            guard_entered.set()

    def switch_generation() -> None:
        try:
            worker.role = "pipeline-rebuild"
            intent_id = _create_generation(
                sessions,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                source_collection="collection-g0",
                target_collection="collection-g1",
            )
            with sessions() as verifier:
                successor_generation_ids.append(verifier.get_one(DatasetIndexingOutbox, intent_id).generation_id)
        except BaseException as error:
            rebuild_errors.append(error)
        finally:
            rebuild_finished.set()

    writer = Thread(target=hold_pipeline_mutation_guard)
    rebuild = Thread(target=switch_generation)
    event.listen(engine, "before_cursor_execute", observe_rebuild_lock)
    try:
        writer.start()
        assert guard_entered.wait(timeout=10)
        assert writer_errors == []
        rebuild.start()
        assert rebuild_attempted.wait(timeout=10)
        assert not rebuild_finished.wait(timeout=0.25)

        allow_guard_commit.set()
        _join_if_started(writer)
        _join_if_started(rebuild)
        assert not writer.is_alive()
        assert not rebuild.is_alive()
        assert writer_errors == []
        assert rebuild_errors == []
        assert len(successor_generation_ids) == 1
        successor_generation_id = successor_generation_ids[0]

        with sessions.begin() as session:
            with pytest.raises(DatasetIndexingGenerationLostError):
                with protect_pipeline_indexing_claim(
                    session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    indexing_request_id=request_id,
                    claim_id=old_claim.claim_id,
                ):
                    raise AssertionError("the stale pipeline owner must not reach vector publication")

        documents.release_dataset_generation_owner(
            old_claim.ref,
            indexing_request_id=request_id,
            claim_id=old_claim.claim_id,
            dataset_generation_id=first_generation_id,
        )
        with sessions() as verifier:
            released = verifier.get_one(Document, document_id)
            request = verifier.get_one(DocumentIndexingOutbox, request_id)
            assert released.indexing_status == IndexingStatus.WAITING
            assert released.indexing_request_id == request_id
            assert released.indexing_claim_id is None
            assert released.indexing_claim_expires_at is None
            assert released.indexing_dataset_generation_id is None
            assert request.action == DocumentIndexingAction.RETRY.value
            assert request.actor_id == actor_id
            assert request.pipeline_execution_log_id == execution_log_id
            execution_log = verifier.get_one(DocumentPipelineExecutionLog, execution_log_id)
            assert execution_log.workflow_id == workflow_v1_id
            assert execution_log.original_document_id is None
            assert verifier.get_one(Pipeline, pipeline_id).workflow_id == workflow_v2_id
            assert (
                verifier.scalar(
                    select(DocumentPipelineExecutionLog.id).where(
                        DocumentPipelineExecutionLog.document_id == document_id,
                        DocumentPipelineExecutionLog.pipeline_id == pipeline_id,
                    )
                )
                == execution_log_id
            )

        (successor_claim,) = documents.claim_indexing_request(
            request_id,
            expected_action=DocumentIndexingAction.RETRY.value,
            processing_started_at=datetime.now(UTC).replace(tzinfo=None),
            admission_error=lambda _count: None,
        )
        assert successor_claim.recovery_required is True
        assert successor_claim.dataset_generation_id == successor_generation_id
        assert successor_claim.claim_id != old_claim.claim_id
    finally:
        allow_guard_commit.set()
        _join_if_started(writer)
        _join_if_started(rebuild)
        event.remove(engine, "before_cursor_execute", observe_rebuild_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_lost_document_generation_restarts_under_successor_and_completes(mysql_session: Session) -> None:
    """A G0 worker waiting on G1 must release its exact owner and fully restart under G1."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    first_intent_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-initial",
        target_collection="collection-g0",
    )
    with sessions() as verifier:
        first_generation_id = verifier.get_one(DatasetIndexingOutbox, first_intent_id).generation_id
    request_id, document_id = _seed_document_request(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        actor_id=actor_id,
    )
    documents = SQLAlchemyDocumentRepository(session_factory=sessions)
    (old_claim,) = documents.claim_indexing_request(
        request_id,
        expected_action=DocumentIndexingAction.DUPLICATE.value,
        processing_started_at=datetime.now(UTC).replace(tzinfo=None),
        admission_error=lambda _count: None,
    )
    assert old_claim.dataset_generation_id == first_generation_id
    documents.mark_splitting(old_claim.ref, claim_id=old_claim.claim_id)

    switch_locked = Event()
    allow_switch_commit = Event()
    old_guard_attempted = Event()
    old_guard_finished = Event()
    switch_errors: list[BaseException] = []
    old_guard_errors: list[BaseException] = []
    old_guard_results: list[bool] = []
    successor_generation_ids: list[str] = []
    worker = _WorkerRole()

    def observe_old_guard(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "old-document" and _is_dataset_lock(statement, shared=True):
            old_guard_attempted.set()

    def switch_generation() -> None:
        try:
            job = DatasetIndexingJob(
                DatasetRef(tenant_id, dataset_id),
                _representation(collection="collection-g0"),
                _representation(collection="collection-g1"),
            )
            with sessions.begin() as session:
                (intent_id,) = create_dataset_indexing_intents(session, job)
                successor = session.get_one(DatasetIndexingOutbox, intent_id)
                successor_generation_ids.append(successor.generation_id)
                switch_locked.set()
                assert allow_switch_commit.wait(timeout=10)
        except BaseException as error:
            switch_errors.append(error)
            switch_locked.set()

    def attempt_old_guard() -> None:
        try:
            assert switch_locked.wait(timeout=10)
            worker.role = "old-document"
            with sessions.begin() as session:
                owned = lock_dataset_indexing_generation_shared(
                    session,
                    dataset_id=dataset_id,
                    generation_id=first_generation_id,
                )
                old_guard_results.append(owned is not None)
        except BaseException as error:
            old_guard_errors.append(error)
        finally:
            old_guard_finished.set()

    switcher = Thread(target=switch_generation)
    old_writer = Thread(target=attempt_old_guard)
    event.listen(engine, "before_cursor_execute", observe_old_guard)
    try:
        switcher.start()
        assert switch_locked.wait(timeout=10)
        old_writer.start()
        assert old_guard_attempted.wait(timeout=10)
        assert not old_guard_finished.wait(timeout=0.25)
        allow_switch_commit.set()
        _join_if_started(switcher)
        _join_if_started(old_writer)

        assert not switcher.is_alive()
        assert not old_writer.is_alive()
        assert switch_errors == []
        assert old_guard_errors == []
        assert old_guard_results == [False]
        assert len(successor_generation_ids) == 1
        successor_generation_id = successor_generation_ids[0]

        documents.release_dataset_generation_owner(
            old_claim.ref,
            indexing_request_id=request_id,
            claim_id=old_claim.claim_id,
            dataset_generation_id=first_generation_id,
        )
        with sessions() as verifier:
            released = verifier.get_one(Document, document_id)
            assert released.indexing_status == IndexingStatus.WAITING
            assert released.indexing_request_id == request_id
            assert released.indexing_claim_id is None
            assert released.indexing_claim_expires_at is None
            assert released.indexing_dataset_generation_id is None
            assert released.processing_started_at is not None
            assert released.parsing_completed_at is None
            assert released.cleaning_completed_at is None
            assert released.splitting_completed_at is None
            assert released.completed_at is None
            assert released.stopped_at is None
            assert released.error is None

        (successor_claim,) = documents.claim_indexing_request(
            request_id,
            expected_action=DocumentIndexingAction.DUPLICATE.value,
            processing_started_at=datetime.now(UTC).replace(tzinfo=None),
            admission_error=lambda _count: None,
        )
        assert successor_claim.recovery_required is True
        assert successor_claim.dataset_generation_id == successor_generation_id
        assert successor_claim.claim_id != old_claim.claim_id

        restart_events: list[str] = []
        service = DocumentIndexingService(
            documents=documents,
            segments=_RestartSegments(restart_events),
            backend=_RestartBackend(restart_events),
        )
        service.restart_claimed(successor_claim.ref, claim_id=successor_claim.claim_id)

        assert restart_events.index("clean") < restart_events.index("extract")
        assert restart_events.index("extract") < restart_events.index("save")
        assert restart_events.index("save") < restart_events.index("load")
        with sessions() as verifier:
            completed = verifier.get_one(Document, document_id)
            assert completed.indexing_status == IndexingStatus.COMPLETED
            assert completed.indexing_request_id is None
            assert completed.indexing_claim_id is None
            assert completed.indexing_claim_expires_at is None
            assert completed.indexing_dataset_generation_id is None
            assert completed.error is None
            assert completed.tokens == 1
            assert verifier.get(DocumentIndexingOutbox, request_id) is None
    finally:
        allow_switch_commit.set()
        _join_if_started(switcher)
        _join_if_started(old_writer)
        event.remove(engine, "before_cursor_execute", observe_old_guard)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


@pytest.mark.parametrize("first_writer", ["edit", "disable"])
def test_manual_summary_edit_and_disable_serialize_without_reviving_pending_enable(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    first_writer: str,
) -> None:
    """Both real MySQL lock orders leave the enable intent stale and the summary disabled."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    document_id = str(uuid4())
    segment_id = str(uuid4())
    route = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="summary-manual-edit-disable",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    with sessions.begin() as seed:
        dataset = seed.get_one(Dataset, dataset_id)
        dataset.chunk_structure = IndexStructureType.PARAGRAPH_INDEX
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch=str(uuid4()),
            name="manual-summary-edit-disable.txt",
            created_from=DocumentCreatedFrom.WEB,
            created_by=actor_id,
            indexing_status=IndexingStatus.COMPLETED,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            enabled=True,
        )
        segment = DocumentSegment(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            position=1,
            content="body",
            word_count=1,
            tokens=1,
            created_by=actor_id,
            index_node_id=f"body-{uuid4()}",
            index_node_hash=str(uuid4()),
            status=SegmentStatus.COMPLETED,
            enabled=True,
        )
        segment.id = segment_id
        summary = DocumentSegmentSummary(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_id=segment_id,
            summary_content="summary v1",
            summary_index_node_id=f"summary-{uuid4()}",
            summary_index_node_hash=str(uuid4()),
            vector_route=route.serialized,
            vector_route_hash=route.key,
            status=SummaryStatus.COMPLETED,
            enabled=False,
        )
        seed.add_all([document, segment, summary])
        seed.flush()
        initial_version = summary.version
        summary_id = summary.id
        node_id = summary.summary_index_node_id
        intent = SegmentStateOutbox(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            action="enable",
            actor_id=actor_id,
            summary_id=summary_id,
            summary_version=initial_version,
        )
        segment.indexing_request_id = intent.id
        seed.add(intent)
        intent_id = intent.id

    repository = SQLAlchemySegmentRepository(session_factory=sessions)
    segment_ref = DatasetRef(tenant_id, dataset_id).document(document_id).segment(segment_id)
    with sessions() as dataset_session:
        detached_dataset = dataset_session.get_one(Dataset, dataset_id)
    monkeypatch.setattr(summary_adapter_module, "session_factory", _SessionFactoryBoundary(sessions))

    first_locked = Event()
    allow_first_commit = Event()
    second_summary_lock_attempted = Event()
    edit_finished = Event()
    disable_finished = Event()
    edit_errors: list[BaseException] = []
    disable_errors: list[BaseException] = []
    inherited_intent_ids: list[str | None] = []
    worker = _WorkerRole()
    second_writer = "disable" if first_writer == "edit" else "edit"

    def is_update_lock(statement: str, table: str) -> bool:
        normalized = " ".join(statement.lower().replace("`", "").split())
        return f" from {table} " in normalized and " for update" in normalized

    def hold_first_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        expected_table = "document_segments" if first_writer == "edit" else "document_segment_summaries"
        if worker.role == first_writer and is_update_lock(statement, expected_table):
            first_locked.set()
            assert allow_first_commit.wait(timeout=10)

    def observe_second_summary_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == second_writer and is_update_lock(statement, "document_segment_summaries"):
            second_summary_lock_attempted.set()

    def edit_summary() -> None:
        try:
            worker.role = "edit"
            result = repository.save_summary(segment_ref, "manual summary v2")
            inherited_intent_ids.append(result.inherited_enable_intent_id)
        except BaseException as error:
            edit_errors.append(error)
        finally:
            edit_finished.set()

    def disable_summary() -> None:
        try:
            worker.role = "disable"
            SummaryIndexAdapter.disable_summaries_for_segments(
                detached_dataset,
                segment_ids=[segment_id],
                disabled_by=actor_id,
            )
        except BaseException as error:
            disable_errors.append(error)
        finally:
            disable_finished.set()

    edit_thread = Thread(target=edit_summary)
    disable_thread = Thread(target=disable_summary)
    first_thread = edit_thread if first_writer == "edit" else disable_thread
    second_thread = disable_thread if first_writer == "edit" else edit_thread
    second_finished = disable_finished if first_writer == "edit" else edit_finished
    event.listen(engine, "after_cursor_execute", hold_first_lock)
    event.listen(engine, "before_cursor_execute", observe_second_summary_lock)
    try:
        first_thread.start()
        assert first_locked.wait(timeout=10)
        second_thread.start()
        assert second_summary_lock_attempted.wait(timeout=10)
        assert not second_finished.wait(timeout=0.25)
        allow_first_commit.set()
        _join_if_started(first_thread)
        _join_if_started(second_thread)
        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert edit_errors == []
        assert disable_errors == []
        assert inherited_intent_ids == ([intent_id] if first_writer == "edit" else [None])

        with sessions() as verify:
            summary = verify.get_one(DocumentSegmentSummary, summary_id)
            intent = verify.get_one(SegmentStateOutbox, intent_id)
            cleanup = verify.scalar(
                select(SummaryVectorCleanupOutbox).where(
                    SummaryVectorCleanupOutbox.tenant_id == tenant_id,
                    SummaryVectorCleanupOutbox.dataset_id == dataset_id,
                    SummaryVectorCleanupOutbox.node_id == node_id,
                    SummaryVectorCleanupOutbox.vector_route_hash == route.key,
                )
            )
            assert summary.summary_content == "manual summary v2"
            assert summary.enabled is False
            assert summary.version == initial_version + 2
            assert summary.summary_index_node_id == node_id
            assert summary.vector_route_hash == route.key
            assert intent.summary_id == summary_id
            assert intent.summary_version == (initial_version + 1 if first_writer == "edit" else initial_version)
            assert cleanup is not None
            assert cleanup.writer_finished_at is not None
    finally:
        allow_first_commit.set()
        _join_if_started(first_thread)
        _join_if_started(second_thread)
        event.remove(engine, "after_cursor_execute", hold_first_lock)
        event.remove(engine, "before_cursor_execute", observe_second_summary_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


@pytest.mark.parametrize("outcome", ["content", "error"])
def test_old_summary_generation_cannot_mutate_completed_successor(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    """G0 reserve/content/error paths must all reject G1 after a real Dataset-lock wait."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    first_intent_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-initial",
        target_collection="summary-g0",
    )
    with sessions() as verifier:
        first_generation_id = verifier.get_one(DatasetIndexingOutbox, first_intent_id).generation_id
    _document_id, segment_id = _seed_summary_segment(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        actor_id=actor_id,
    )
    with sessions() as loader:
        dataset = loader.get_one(Dataset, dataset_id)
        segment = loader.get_one(DocumentSegment, segment_id)
    assert SummaryIndexAdapter.batch_create_summary_records(
        [segment],
        dataset,
        indexing_request_id=first_generation_id,
    )

    model_reached = Event()
    successor_locked = Event()
    allow_successor_commit = Event()
    old_guard_attempted = Event()
    old_finished = Event()
    successor_errors: list[BaseException] = []
    old_errors: list[BaseException] = []
    old_results: list[DocumentSegmentSummary | None] = []
    successor_states: list[tuple[str, int, str | None, str, str | None, str | None, str | None, int | None, bool]] = []
    worker = _WorkerRole()

    def observe_old_guard(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "old-summary" and _is_dataset_lock(statement, shared=True):
            old_guard_attempted.set()

    def generate_old_content(
        _segment: DocumentSegment,
        _dataset: Dataset,
        _setting: SummaryIndexSettingDict,
        *,
        session: Session,
    ) -> tuple[str, LLMUsage]:
        del session
        model_reached.set()
        assert successor_locked.wait(timeout=10)
        worker.role = "old-summary"
        if outcome == "error":
            raise RuntimeError("old G0 model failure")
        return "old G0 content", LLMUsage.empty_usage()

    monkeypatch.setattr(SummaryIndexAdapter, "generate_summary_for_segment", generate_old_content)

    def complete_successor() -> None:
        try:
            assert model_reached.wait(timeout=10)
            job = DatasetIndexingJob(
                DatasetRef(tenant_id, dataset_id),
                _representation(collection="summary-g0"),
                _representation(collection="summary-g1"),
            )
            with sessions.begin() as session:
                create_dataset_indexing_intents(session, job)
                summary = session.scalar(
                    select(DocumentSegmentSummary)
                    .where(
                        DocumentSegmentSummary.dataset_id == dataset_id,
                        DocumentSegmentSummary.chunk_id == segment_id,
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                assert summary is not None
                summary.version += 1
                summary.summary_content = "completed G1 content"
                summary.status = SummaryStatus.COMPLETED
                summary.error = None
                summary.summary_index_node_id = "g1-node"
                summary.summary_index_node_hash = "g1-hash"
                summary.tokens = 7
                summary.enabled = True
                session.flush()
                successor_states.append(
                    (
                        summary.id,
                        summary.version,
                        summary.summary_content,
                        str(summary.status),
                        summary.error,
                        summary.summary_index_node_id,
                        summary.summary_index_node_hash,
                        summary.tokens,
                        summary.enabled,
                    )
                )
                successor_locked.set()
                assert allow_successor_commit.wait(timeout=10)
        except BaseException as error:
            successor_errors.append(error)
            successor_locked.set()

    def finish_old_generation() -> None:
        try:
            with sessions() as session:
                old_results.append(
                    SummaryIndexAdapter.generate_and_vectorize_summary(
                        segment,
                        dataset,
                        {"enable": True},
                        session=session,
                        indexing_request_id=first_generation_id,
                    )
                )
        except BaseException as error:
            old_errors.append(error)
        finally:
            old_finished.set()

    old_worker = Thread(target=finish_old_generation)
    successor = Thread(target=complete_successor)
    event.listen(engine, "before_cursor_execute", observe_old_guard)
    try:
        old_worker.start()
        assert model_reached.wait(timeout=10)
        successor.start()
        assert successor_locked.wait(timeout=10)
        assert old_guard_attempted.wait(timeout=10)
        assert not old_finished.wait(timeout=0.25)
        allow_successor_commit.set()
        _join_if_started(successor)
        _join_if_started(old_worker)

        assert not successor.is_alive()
        assert not old_worker.is_alive()
        assert successor_errors == []
        assert old_errors == []
        assert old_results == [None]
        assert len(successor_states) == 1
        expected_successor_state = successor_states[0]

        assert not SummaryIndexAdapter.batch_create_summary_records(
            [segment],
            dataset,
            indexing_request_id=first_generation_id,
        )
        with sessions() as verifier:
            summary = verifier.scalar(
                select(DocumentSegmentSummary).where(
                    DocumentSegmentSummary.dataset_id == dataset_id,
                    DocumentSegmentSummary.chunk_id == segment_id,
                )
            )
            assert summary is not None
            assert (
                summary.id,
                summary.version,
                summary.summary_content,
                str(summary.status),
                summary.error,
                summary.summary_index_node_id,
                summary.summary_index_node_hash,
                summary.tokens,
                summary.enabled,
            ) == expected_successor_state
    finally:
        allow_successor_commit.set()
        _join_if_started(successor)
        _join_if_started(old_worker)
        event.remove(engine, "before_cursor_execute", observe_old_guard)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_successor_summary_rebuild_converges_row_left_generating_by_old_generation(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G1 must take over an unfinished G0 summary before its rebuild intent can be acknowledged."""

    _engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    intent_g0_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-initial",
        target_collection="summary-g0",
    )
    with sessions() as verifier:
        generation_g0 = verifier.get_one(DatasetIndexingOutbox, intent_g0_id).generation_id
    _document_id, segment_id = _seed_summary_segment(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        actor_id=actor_id,
    )

    monkeypatch.setattr(summary_task_module.session_factory, "create_session", sessions)
    embedding = _SummaryEmbeddingBoundary()
    tenant_models = _SummaryTenantModelsBoundary(tenant_id=tenant_id, embedding=embedding)

    class _ModelManagerBoundary:
        @staticmethod
        def for_tenant(*, tenant_id: str) -> _SummaryTenantModelsBoundary:
            assert tenant_id == tenant_models._tenant_id
            return tenant_models

    vectors = _SummaryVectorBoundary()
    monkeypatch.setattr(summary_adapter_module, "ModelManager", _ModelManagerBoundary)
    monkeypatch.setattr(summary_adapter_module, "Vector", _SummaryVectorConstructorBoundary(vectors))

    old_llm_started = Event()
    release_old_llm = Event()
    generated_contents: list[str] = []

    def generate(
        _segment: DocumentSegment,
        _dataset: Dataset,
        setting: SummaryIndexSettingDict,
        *,
        session: Session,
    ) -> tuple[str, LLMUsage]:
        assert setting.get("enable") is True
        # Match the production model boundary: the GENERATING reservation is
        # committed and no Dataset lock remains while the LLM is running.
        session.commit()
        if not old_llm_started.is_set():
            old_llm_started.set()
            assert release_old_llm.wait(timeout=10)
            generated_contents.append("old G0 summary")
            return "old G0 summary", LLMUsage.empty_usage()
        generated_contents.append("completed G1 summary")
        return "completed G1 summary", LLMUsage.empty_usage()

    monkeypatch.setattr(SummaryIndexAdapter, "generate_summary_for_segment", generate)
    old_results: list[DocumentSegmentSummary | None] = []
    old_errors: list[BaseException] = []

    def run_old_generation() -> None:
        try:
            with sessions() as session:
                dataset = session.get_one(Dataset, dataset_id)
                segment = session.get_one(DocumentSegment, segment_id)
                old_results.append(
                    SummaryIndexAdapter.generate_and_vectorize_summary(
                        segment,
                        dataset,
                        {"enable": True},
                        session=session,
                        indexing_request_id=generation_g0,
                    )
                )
        except BaseException as error:
            old_errors.append(error)

    old_worker = Thread(target=run_old_generation, name="mysql-old-summary-generation", daemon=True)
    old_worker.start()
    assert old_llm_started.wait(timeout=10)

    rebuilds = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sessions)
    intent_g1_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="summary-g0",
        target_collection="summary-g1",
    )
    intent_g1 = rebuilds.claim(intent_g1_id)
    assert intent_g1 is not None
    with sessions.begin() as configure:
        dataset = configure.get_one(Dataset, dataset_id)
        assert dataset.indexing_request_id == intent_g1.generation_id
        dataset.indexing_technique = "high_quality"
        dataset.embedding_model = "embedding-model"
        dataset.embedding_model_provider = "embedding-provider"
        dataset.summary_index_setting = {"enable": True}

    successor_completed = False
    try:
        rebuilt = summary_task_module.regenerate_summary_index_task.run(
            dataset_id,
            regenerate_reason="embedding_model_changed",
            regenerate_vectors_only=True,
            indexing_request_id=intent_g1.generation_id,
            indexing_intent_id=intent_g1.id,
            indexing_claim_id=intent_g1.claim_id,
            tenant_id=tenant_id,
        )
        assert rebuilt is True
        assert rebuilds.mark_vector_completed(intent_g1)
        assert rebuilds.complete(intent_g1)

        with sessions() as verifier:
            assert verifier.get(DatasetIndexingOutbox, intent_g1_id) is None
            summary = verifier.scalar(
                select(DocumentSegmentSummary).where(
                    DocumentSegmentSummary.dataset_id == dataset_id,
                    DocumentSegmentSummary.chunk_id == segment_id,
                )
            )
            assert summary is not None
            expected_state = (
                summary.id,
                summary.version,
                summary.summary_content,
                summary.summary_index_node_id,
                summary.summary_index_node_hash,
                summary.tokens,
                summary.status,
                summary.error,
            )
            assert summary.summary_content == "completed G1 summary"
            assert summary.summary_index_node_id is not None
            assert summary.status == SummaryStatus.COMPLETED
            assert summary.error is None
        successor_completed = True
    finally:
        release_old_llm.set()
        _join_if_started(old_worker)
        if not successor_completed:
            _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)

    try:
        assert not old_worker.is_alive()
        assert old_errors == []
        assert old_results == [None]
        assert generated_contents == ["completed G1 summary", "old G0 summary"]
        assert [document.page_content for document in vectors.added] == ["completed G1 summary"]
        assert embedding.contents == [["completed G1 summary"]]
        with sessions() as verifier:
            summary = verifier.scalar(
                select(DocumentSegmentSummary).where(
                    DocumentSegmentSummary.dataset_id == dataset_id,
                    DocumentSegmentSummary.chunk_id == segment_id,
                )
            )
            assert summary is not None
            assert (
                summary.id,
                summary.version,
                summary.summary_content,
                summary.summary_index_node_id,
                summary.summary_index_node_hash,
                summary.tokens,
                summary.status,
                summary.error,
            ) == expected_state
    finally:
        release_old_llm.set()
        _join_if_started(old_worker)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_summary_cleanup_delete_blocks_same_route_publication(mysql_session: Session) -> None:
    """Cleanup's Dataset UPDATE guard must cover reference-check through physical deletion."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, _actor_id = _seed_dataset(sessions)
    route_a = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="summary-collection-a",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="summary",
    )
    route_b = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="summary-collection-b",
        collection_binding_id="binding-b",
        indexing_technique="high_quality",
        doc_form="summary",
    )
    summary_id = str(uuid4())
    node_id = str(uuid4())
    with sessions.begin() as session:
        summary = DocumentSegmentSummary(
            dataset_id=dataset_id,
            document_id=str(uuid4()),
            chunk_id=str(uuid4()),
            summary_content="summary",
            summary_index_node_id=node_id,
            summary_index_node_hash="hash-b",
            vector_route=route_b.serialized,
            vector_route_hash=route_b.key,
            status=SummaryStatus.COMPLETED,
            enabled=True,
        )
        summary.id = summary_id
        cleanup = SummaryVectorCleanupOutbox(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=summary.document_id,
            segment_id=summary.chunk_id,
            summary_id=summary.id,
            summary_version=summary.version,
            node_id=node_id,
            writer_id=str(uuid4()),
            vector_route=route_a.serialized,
            vector_route_hash=route_a.key,
            writer_finished_at=datetime.now(UTC).replace(tzinfo=None),
        )
        session.add_all([summary, cleanup])
        session.flush()
        cleanup_id = cleanup.id
    repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sessions)
    claim = repository.claim_pending_cleanup(cleanup_id)
    assert claim is not None

    delete_started = Event()
    allow_delete = Event()
    cleanup_finished = Event()
    writer_attempted = Event()
    writer_acquired = Event()
    writer_finished = Event()
    cleanup_errors: list[BaseException] = []
    writer_errors: list[BaseException] = []
    order: list[str] = []
    worker = _WorkerRole()

    def observe_writer_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "writer" and _is_dataset_lock(statement, shared=True):
            writer_attempted.set()

    def cleanup_old_route() -> None:
        try:
            with repository.guard_cleanup_delete(claim) as referenced:
                assert referenced is False
                order.append("delete-started")
                delete_started.set()
                assert allow_delete.wait(timeout=10)
                order.append("delete-finished")
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            cleanup_finished.set()

    def publish_same_route() -> None:
        try:
            assert delete_started.wait(timeout=10)
            worker.role = "writer"
            with sessions.begin() as session:
                dataset = lock_dataset_indexing_generation_shared(
                    session,
                    dataset_id=dataset_id,
                    generation_id=None,
                )
                assert dataset is not None
                writer_acquired.set()
                current = session.scalar(
                    select(DocumentSegmentSummary)
                    .where(DocumentSegmentSummary.id == summary_id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                assert current is not None
                order.append("writer-published")
                current.vector_route = route_a.serialized
                current.vector_route_hash = route_a.key
        except BaseException as error:
            writer_errors.append(error)
        finally:
            writer_finished.set()

    cleanup_thread = Thread(target=cleanup_old_route)
    writer_thread = Thread(target=publish_same_route)
    event.listen(engine, "before_cursor_execute", observe_writer_lock)
    try:
        cleanup_thread.start()
        assert delete_started.wait(timeout=10)
        writer_thread.start()
        assert writer_attempted.wait(timeout=10)
        assert not writer_acquired.wait(timeout=0.25)

        allow_delete.set()
        _join_if_started(cleanup_thread)
        _join_if_started(writer_thread)

        assert not cleanup_thread.is_alive()
        assert not writer_thread.is_alive()
        assert cleanup_errors == []
        assert writer_errors == []
        assert cleanup_finished.is_set()
        assert writer_finished.is_set()
        assert order.index("delete-finished") < order.index("writer-published")
        with sessions() as session:
            current = session.get_one(DocumentSegmentSummary, summary_id)
            assert current.vector_route_hash == route_a.key
            assert session.get(SummaryVectorCleanupOutbox, cleanup_id) is None
    finally:
        allow_delete.set()
        _join_if_started(cleanup_thread)
        _join_if_started(writer_thread)
        event.remove(engine, "before_cursor_execute", observe_writer_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_referenced_summary_cleanup_ack_precedes_detach_handoff(mysql_session: Session) -> None:
    """A detach queued after the final reference check must persist a fresh cleanup receipt."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    document_id, segment_id = _seed_summary_segment(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        actor_id=actor_id,
    )
    route = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="summary-atomic-ack",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    node_id = str(uuid4())
    with sessions.begin() as seed:
        summary = DocumentSegmentSummary(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_id=segment_id,
            summary_content="referenced summary",
            summary_index_node_id=node_id,
            summary_index_node_hash="summary-hash",
            vector_route=route.serialized,
            vector_route_hash=route.key,
            status=SummaryStatus.COMPLETED,
            enabled=True,
        )
        cleanup = SummaryVectorCleanupOutbox(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            summary_id=summary.id,
            summary_version=summary.version,
            node_id=node_id,
            writer_id=str(uuid4()),
            vector_route=route.serialized,
            vector_route_hash=route.key,
            writer_finished_at=datetime.now(UTC).replace(tzinfo=None),
        )
        seed.add_all([summary, cleanup])
        seed.flush()
        summary_id = summary.id
        cleanup_id = cleanup.id

    repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sessions)
    claim = repository.claim_pending_cleanup(cleanup_id)
    assert claim is not None
    guard_checked = Event()
    allow_guard_exit = Event()
    cleanup_finished = Event()
    handoff_attempted = Event()
    handoff_finished = Event()
    cleanup_errors: list[BaseException] = []
    handoff_errors: list[BaseException] = []
    vectors = {node_id}
    worker = _WorkerRole()

    def observe_handoff_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "summary-handoff" and _is_dataset_lock(statement, shared=False):
            handoff_attempted.set()

    def acknowledge_referenced_cleanup() -> None:
        try:
            with repository.guard_cleanup_delete(claim) as referenced:
                assert referenced is True
                guard_checked.set()
                assert allow_guard_exit.wait(timeout=10)
                # A referenced vector is retained; normal context exit ACKs the
                # old finished receipt in this same locking transaction.
                assert node_id in vectors
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            cleanup_finished.set()

    def detach_summary() -> None:
        try:
            assert guard_checked.wait(timeout=10)
            worker.role = "summary-handoff"
            with sessions.begin() as session:
                dataset = session.scalar(
                    select(Dataset)
                    .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                assert dataset is not None
                current = session.scalar(
                    select(DocumentSegmentSummary)
                    .where(
                        DocumentSegmentSummary.id == summary_id,
                        DocumentSegmentSummary.dataset_id == dataset_id,
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                assert current is not None
                SQLAlchemySummaryVectorCleanupRepository.add_terminal_cleanups(
                    session,
                    tenant_id=tenant_id,
                    summaries=(current,),
                )
                current.enabled = False
        except BaseException as error:
            handoff_errors.append(error)
        finally:
            handoff_finished.set()

    cleanup_thread = Thread(target=acknowledge_referenced_cleanup)
    handoff_thread = Thread(target=detach_summary)
    event.listen(engine, "before_cursor_execute", observe_handoff_lock)
    try:
        cleanup_thread.start()
        assert guard_checked.wait(timeout=10)
        handoff_thread.start()
        assert handoff_attempted.wait(timeout=10)
        assert not handoff_finished.wait(timeout=0.25)

        allow_guard_exit.set()
        _join_if_started(cleanup_thread)
        _join_if_started(handoff_thread)

        assert not cleanup_thread.is_alive()
        assert not handoff_thread.is_alive()
        assert cleanup_errors == []
        assert handoff_errors == []
        assert cleanup_finished.is_set()
        assert handoff_finished.is_set()
        assert vectors == {node_id}
        with sessions() as verify:
            current = verify.get_one(DocumentSegmentSummary, summary_id)
            assert current.enabled is False
            receipts = verify.scalars(
                select(SummaryVectorCleanupOutbox).where(
                    SummaryVectorCleanupOutbox.tenant_id == tenant_id,
                    SummaryVectorCleanupOutbox.node_id == node_id,
                    SummaryVectorCleanupOutbox.vector_route_hash == route.key,
                )
            ).all()
            assert len(receipts) == 1
            fresh = receipts[0]
            assert fresh.id != cleanup_id
            assert fresh.writer_finished_at is not None
            assert fresh.cleanup_started_at is None
            assert fresh.claim_id is None
    finally:
        allow_guard_exit.set()
        _join_if_started(cleanup_thread)
        _join_if_started(handoff_thread)
        event.remove(engine, "before_cursor_execute", observe_handoff_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_referenced_child_cleanup_ack_precedes_detach_handoff(mysql_session: Session) -> None:
    """A child detach must wait for referenced cleanup ACK, then create a successor receipt."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, actor_id = _seed_dataset(sessions)
    document_id, segment_id = _seed_summary_segment(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        actor_id=actor_id,
    )
    ref = DatasetRef(tenant_id, dataset_id).document(document_id).segment(segment_id)
    route = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name="child-atomic-ack",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form=IndexStructureType.PARENT_CHILD_INDEX,
    )
    node_id = str(uuid4())
    with sessions.begin() as seed:
        child = ChildChunk(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            position=1,
            content="referenced child",
            word_count=2,
            created_by=actor_id,
            index_node_id=node_id,
            index_node_hash="child-hash",
            vector_route=route.serialized,
            vector_route_hash=route.key,
        )
        cleanup = ChildVectorCleanupOutbox(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            child_id=child.id,
            node_id=node_id,
            writer_id=str(uuid4()),
            vector_route=route.serialized,
            vector_route_hash=route.key,
            writer_finished_at=datetime.now(UTC).replace(tzinfo=None),
        )
        seed.add_all([child, cleanup])
        seed.flush()
        child_id = child.id
        cleanup_id = cleanup.id

    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sessions)
    claim = repository.claim_pending_cleanup(cleanup_id)
    assert claim is not None
    guard_checked = Event()
    allow_guard_exit = Event()
    cleanup_finished = Event()
    handoff_attempted = Event()
    handoff_finished = Event()
    cleanup_errors: list[BaseException] = []
    handoff_errors: list[BaseException] = []
    vectors = {node_id}
    worker = _WorkerRole()

    def observe_child_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        sql = " ".join(statement.lower().replace("`", "").split())
        if worker.role == "child-handoff" and " from child_chunks " in sql and " for update" in sql:
            handoff_attempted.set()

    def acknowledge_referenced_cleanup() -> None:
        try:
            with repository.guard_cleanup_delete(claim) as referenced:
                assert referenced is True
                guard_checked.set()
                assert allow_guard_exit.wait(timeout=10)
                assert node_id in vectors
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            cleanup_finished.set()

    def detach_child() -> None:
        try:
            assert guard_checked.wait(timeout=10)
            worker.role = "child-handoff"
            with sessions.begin() as session:
                dataset = session.scalar(
                    select(Dataset)
                    .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
                    .with_for_update(read=True)
                    .execution_options(populate_existing=True)
                )
                assert dataset is not None
                document = session.scalar(
                    select(Document)
                    .where(
                        Document.id == document_id,
                        Document.tenant_id == tenant_id,
                        Document.dataset_id == dataset_id,
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                assert document is not None
                segment = session.scalar(
                    select(DocumentSegment)
                    .where(
                        DocumentSegment.id == segment_id,
                        DocumentSegment.tenant_id == tenant_id,
                        DocumentSegment.dataset_id == dataset_id,
                        DocumentSegment.document_id == document_id,
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                assert segment is not None
                current = session.scalar(
                    select(ChildChunk)
                    .where(
                        ChildChunk.id == child_id,
                        ChildChunk.tenant_id == tenant_id,
                        ChildChunk.dataset_id == dataset_id,
                        ChildChunk.document_id == document_id,
                        ChildChunk.segment_id == segment_id,
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                assert current is not None
                SQLAlchemyChildVectorCleanupRepository.add_terminal_cleanups(
                    session,
                    ref=ref,
                    obsolete_nodes=((current.id, node_id, route),),
                )
                session.delete(current)
        except BaseException as error:
            handoff_errors.append(error)
        finally:
            handoff_finished.set()

    cleanup_thread = Thread(target=acknowledge_referenced_cleanup)
    handoff_thread = Thread(target=detach_child)
    event.listen(engine, "before_cursor_execute", observe_child_lock)
    try:
        cleanup_thread.start()
        assert guard_checked.wait(timeout=10)
        handoff_thread.start()
        assert handoff_attempted.wait(timeout=10)
        assert not handoff_finished.wait(timeout=0.25)

        allow_guard_exit.set()
        _join_if_started(cleanup_thread)
        _join_if_started(handoff_thread)

        assert not cleanup_thread.is_alive()
        assert not handoff_thread.is_alive()
        assert cleanup_errors == []
        assert handoff_errors == []
        assert cleanup_finished.is_set()
        assert handoff_finished.is_set()
        assert vectors == {node_id}
        with sessions() as verify:
            assert verify.get(ChildChunk, child_id) is None
            receipts = verify.scalars(
                select(ChildVectorCleanupOutbox).where(
                    ChildVectorCleanupOutbox.tenant_id == tenant_id,
                    ChildVectorCleanupOutbox.node_id == node_id,
                    ChildVectorCleanupOutbox.vector_route_hash == route.key,
                )
            ).all()
            assert len(receipts) == 1
            fresh = receipts[0]
            assert fresh.id != cleanup_id
            assert fresh.child_id == child_id
            assert fresh.writer_finished_at is not None
            assert fresh.cleanup_started_at is None
            assert fresh.claim_id is None
    finally:
        allow_guard_exit.set()
        _join_if_started(cleanup_thread)
        _join_if_started(handoff_thread)
        event.remove(engine, "before_cursor_execute", observe_child_lock)
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(ChildVectorCleanupOutbox).where(
                    ChildVectorCleanupOutbox.tenant_id == tenant_id,
                    ChildVectorCleanupOutbox.dataset_id == dataset_id,
                )
            )
            cleanup.execute(
                delete(ChildChunk).where(
                    ChildChunk.tenant_id == tenant_id,
                    ChildChunk.dataset_id == dataset_id,
                )
            )
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_writer_waiting_behind_generation_switch_cannot_publish_to_the_successor(
    mysql_session: Session,
) -> None:
    """A locking current read must reject G1 after waiting for the G2 configuration commit."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, _actor_id = _seed_dataset(sessions)
    first_intent_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-initial",
        target_collection="collection-a",
    )
    with sessions() as verifier:
        first_intent = verifier.get_one(DatasetIndexingOutbox, first_intent_id)
        first_generation = first_intent.generation_id

    switch_locked = Event()
    allow_switch_commit = Event()
    writer_attempted = Event()
    writer_finished = Event()
    switch_errors: list[BaseException] = []
    writer_errors: list[BaseException] = []
    writer_observed_generation: list[str | None] = []
    vectors: set[str] = set()
    worker = _WorkerRole()

    def observe_writer_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "writer" and _is_dataset_lock(statement, shared=True):
            writer_attempted.set()

    def switch_generation() -> None:
        try:
            job = DatasetIndexingJob(
                DatasetRef(tenant_id, dataset_id),
                _representation(collection="collection-a"),
                _representation(collection="collection-b"),
            )
            with sessions.begin() as session:
                create_dataset_indexing_intents(session, job)
                switch_locked.set()
                assert allow_switch_commit.wait(timeout=10)
        except BaseException as error:
            switch_errors.append(error)
            switch_locked.set()

    def publish_old_generation() -> None:
        try:
            assert switch_locked.wait(timeout=10)
            worker.role = "writer"
            with sessions.begin() as session:
                dataset = lock_dataset_indexing_generation_shared(
                    session,
                    dataset_id=dataset_id,
                    generation_id=first_generation,
                )
                writer_observed_generation.append(dataset.indexing_request_id if dataset is not None else None)
                if dataset is not None:
                    vectors.add("old-generation-node")
        except BaseException as error:
            writer_errors.append(error)
        finally:
            writer_finished.set()

    switcher = Thread(target=switch_generation)
    writer = Thread(target=publish_old_generation)
    event.listen(engine, "before_cursor_execute", observe_writer_lock)
    try:
        switcher.start()
        assert switch_locked.wait(timeout=10)
        writer.start()
        assert writer_attempted.wait(timeout=10)
        # The writer has issued its SELECT ... FOR SHARE. It cannot return while
        # the configuration transaction owns the Dataset's InnoDB X lock.
        assert not writer_finished.wait(timeout=0.25)
        allow_switch_commit.set()
        _join_if_started(switcher)
        _join_if_started(writer)

        assert not switcher.is_alive()
        assert not writer.is_alive()
        assert switch_errors == []
        assert writer_errors == []
        assert writer_observed_generation == [None]
        assert vectors == set()
        with sessions() as verifier:
            dataset = verifier.get_one(Dataset, dataset_id)
            assert dataset.indexing_request_id is not None
            assert dataset.indexing_request_id != first_generation
    finally:
        allow_switch_commit.set()
        _join_if_started(switcher)
        _join_if_started(writer)
        event.remove(engine, "before_cursor_execute", observe_writer_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)


def test_keyword_reader_does_not_relock_dataset_behind_a_queued_rebuild(mysql_session: Session) -> None:
    """A standalone keyword current-read must not join the Dataset lock queue."""

    engine, sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, _actor_id = _seed_dataset(sessions)
    intent_id = _create_generation(
        sessions,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        source_collection="collection-initial",
        target_collection="collection-keyword",
    )
    rebuilds = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sessions)
    rebuild_intent = rebuilds.claim(intent_id)
    assert rebuild_intent is not None
    keyword_payload = json.dumps({"__data__": {"table": {"alpha": ["node-1"]}}})
    with sessions.begin() as session:
        session.add(
            DatasetKeywordTable(
                dataset_id=dataset_id,
                keyword_table=keyword_payload,
                data_source_type="database",
            )
        )

    publisher_locked = Event()
    allow_publisher_commit = Event()
    publisher_finished = Event()
    rebuild_attempted = Event()
    rebuild_acquired = Event()
    rebuild_finished = Event()
    keyword_finished = Event()
    publisher_errors: list[BaseException] = []
    rebuild_errors: list[BaseException] = []
    keyword_errors: list[BaseException] = []
    keyword_results: list[str] = []
    worker = _WorkerRole()

    def observe_rebuild_lock(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if worker.role == "rebuild" and _is_dataset_lock(statement, shared=False):
            rebuild_attempted.set()

    def hold_publication_guard() -> None:
        try:
            with sessions.begin() as session:
                dataset = lock_dataset_indexing_generation_shared(
                    session,
                    dataset_id=dataset_id,
                    generation_id=rebuild_intent.generation_id,
                )
                assert dataset is not None
                assert dataset.tenant_id == tenant_id
                publisher_locked.set()
                assert allow_publisher_commit.wait(timeout=10)
        except BaseException as error:
            publisher_errors.append(error)
            publisher_locked.set()
        finally:
            publisher_finished.set()

    def acquire_rebuild_guard() -> None:
        try:
            worker.role = "rebuild"
            with sessions.begin() as session:
                dataset = lock_dataset_indexing_execution(
                    session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    generation_id=rebuild_intent.generation_id,
                    intent_id=rebuild_intent.id,
                    claim_id=rebuild_intent.claim_id,
                    require_quiescent=True,
                )
                assert dataset is not None
                rebuild_acquired.set()
        except BaseException as error:
            rebuild_errors.append(error)
        finally:
            rebuild_finished.set()

    def read_keyword_row() -> None:
        try:
            with sessions.begin() as session:
                row = lock_dataset_keyword_table(
                    session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    read=True,
                )
                assert row is not None
                keyword_results.append(row.keyword_table)
        except BaseException as error:
            keyword_errors.append(error)
        finally:
            keyword_finished.set()

    publisher = Thread(target=hold_publication_guard)
    rebuild = Thread(target=acquire_rebuild_guard)
    keyword_reader = Thread(target=read_keyword_row)
    event.listen(engine, "before_cursor_execute", observe_rebuild_lock)
    try:
        publisher.start()
        assert publisher_locked.wait(timeout=10)
        rebuild.start()
        assert rebuild_attempted.wait(timeout=10)
        assert not rebuild_finished.wait(timeout=0.25)

        keyword_reader.start()
        # Session A still owns Dataset SHARE and Session B has already queued
        # Dataset FOR UPDATE. Session C must finish without requesting another
        # Dataset lock; otherwise it waits behind B while B waits behind A.
        assert keyword_finished.wait(timeout=10)
        assert keyword_errors == []
        assert keyword_results == [keyword_payload]
        assert not rebuild_finished.is_set()

        allow_publisher_commit.set()
        _join_if_started(publisher)
        _join_if_started(rebuild)
        _join_if_started(keyword_reader)

        assert not publisher.is_alive()
        assert not rebuild.is_alive()
        assert not keyword_reader.is_alive()
        assert publisher_errors == []
        assert rebuild_errors == []
        assert publisher_finished.is_set()
        assert rebuild_acquired.is_set()
    finally:
        allow_publisher_commit.set()
        _join_if_started(publisher)
        _join_if_started(rebuild)
        _join_if_started(keyword_reader)
        event.remove(engine, "before_cursor_execute", observe_rebuild_lock)
        _delete_fixture(sessions, tenant_id=tenant_id, dataset_id=dataset_id)

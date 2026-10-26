from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from threading import Event, Thread
from typing import Literal

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

import services.knowledge.indexing.adapters.execution as indexing_execution_adapter_module
import services.knowledge.summaries.adapters as summary_adapter_module
import tasks.dataset_indexing_outbox_task as task_module
import tasks.regenerate_summary_index_task as summary_task_module
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from core.rag.index_processor.index_processor_base import SummaryIndexSettingDict
from core.rag.models.document import Document as IndexDocument
from graphon.model_runtime.entities.llm_entities import LLMUsage
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import SegmentStatus, SummaryStatus
from models.model import DatasetIndexingOutbox, SummaryVectorCleanupOutbox
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.dataset_indexing_outbox_repository import (
    DatasetIndexingIntent,
    SQLAlchemyDatasetIndexingOutboxRepository,
    create_dataset_indexing_intents,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from repositories.knowledge.segment_repository import SQLAlchemySegmentRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.entities.datasets import DatasetIndexingJob, DatasetIndexingRepresentation
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.adapters.execution import IndexingExecutionAdapter
from services.knowledge.indexing.adapters.sources import CompositeStoredSourceResolver
from services.knowledge.indexing.estimate import StoredSource
from services.knowledge.indexing.execution import IndexingDocument
from tests.unit_tests.model_factories import make_dataset, make_document


def _usage() -> LLMUsage:
    return LLMUsage(
        prompt_tokens=3,
        prompt_unit_price=Decimal(0),
        prompt_price_unit=Decimal(0),
        prompt_price=Decimal(0),
        completion_tokens=4,
        completion_unit_price=Decimal(0),
        completion_price_unit=Decimal(0),
        completion_price=Decimal(0),
        total_tokens=7,
        total_price=Decimal(0),
        currency="USD",
        latency=0.1,
    )


def _route(vector_type: str, collection_name: str) -> VectorRoute:
    return VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type=vector_type,
        collection_name=collection_name,
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="text_model",
    )


class _EmbeddingModelBoundary:
    def __init__(self) -> None:
        self.contents: list[list[str]] = []

    def get_text_embedding_num_tokens(self, contents: list[str]) -> list[int]:
        self.contents.append(contents)
        return [7 for _content in contents]


class _TenantModelManagerBoundary:
    def __init__(self, embedding: _EmbeddingModelBoundary) -> None:
        self._embedding = embedding

    def get_model_instance(
        self,
        *,
        tenant_id: str,
        provider: str,
        model_type: object,
        model: str,
    ) -> _EmbeddingModelBoundary:
        del model_type
        assert (tenant_id, provider, model) == ("tenant-1", "provider", "text-embedding-model")
        return self._embedding


class _VectorBoundary:
    def __init__(self) -> None:
        self.added: list[IndexDocument] = []
        self.deleted: list[list[str]] = []

    def get_routing_target(self) -> tuple[str, str]:
        return "qdrant", "summary-collection"

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


class _VectorConstructorBoundary:
    def __init__(self, vector: _VectorBoundary) -> None:
        self._vector = vector

    @staticmethod
    def resolve_vector_type(_dataset: Dataset, *, session: Session) -> str:
        del session
        return "qdrant"

    def __call__(
        self,
        dataset: Dataset,
        *,
        session: Session | None,
        vector_type: str,
    ) -> _VectorBoundary:
        assert (dataset.tenant_id, dataset.id, vector_type) == ("tenant-1", "dataset-1", "qdrant")
        assert session is None
        return self._vector


def _create_intents(
    session_factory: sessionmaker[Session],
    *,
    action: Literal["add", "remove", "update"],
) -> tuple[str, ...]:
    economy = DatasetIndexingRepresentation(
        indexing_technique="economy",
        index_struct=None,
        collection_binding_id=None,
        embedding_model=None,
        embedding_model_provider=None,
        keyword_number=10,
        is_multimodal=False,
        doc_form="text_model",
        summary_index_setting=None,
    )
    high_quality = economy.model_copy(
        update={
            "indexing_technique": "high_quality",
            "embedding_model": "text-embedding-model",
            "embedding_model_provider": "provider",
            "summary_index_setting": {"enable": action == "update"},
        }
    )
    source, target = {
        "add": (economy, high_quality),
        "remove": (high_quality, economy),
        "update": (high_quality, high_quality),
    }[action]
    with session_factory.begin() as session:
        if session.get(Dataset, "dataset-1") is None:
            session.add(make_dataset(dataset_id="dataset-1", tenant_id="tenant-1"))
            session.flush()
        return create_dataset_indexing_intents(
            session,
            DatasetIndexingJob(DatasetRef("tenant-1", "dataset-1"), source, target),
        )


def _use_repository(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
) -> SQLAlchemyDatasetIndexingOutboxRepository:
    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=session_factory)
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    return repository


def test_periodic_drain_recovers_fast_path_broker_failure(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (intent_id,) = _create_intents(sqlite_session_factory, action="remove")
    _use_repository(monkeypatch, sqlite_session_factory)

    def reject(_intent_id: str) -> None:
        raise OSError("broker unavailable")

    monkeypatch.setattr(task_module.process_dataset_indexing_intent_task, "delay", reject)
    task_module.enqueue_dataset_indexing_intent(intent_id)
    with sqlite_session_factory() as session:
        assert session.get(DatasetIndexingOutbox, intent_id) is not None

    executed: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        task_module.deal_dataset_vector_index_task,
        "run",
        lambda dataset_id, action, generation_id, **_kwargs: (
            executed.append((dataset_id, action, generation_id)) or True
        ),
    )
    monkeypatch.setattr(
        task_module.process_dataset_indexing_intent_task,
        "delay",
        task_module.process_dataset_indexing_intent_task.run,
    )

    task_module.drain_dataset_indexing_outbox_task.run()

    assert len(executed) == 1
    assert executed[0][:2] == ("dataset-1", "remove")
    with sqlite_session_factory() as session:
        assert session.get(DatasetIndexingOutbox, intent_id) is None


def test_execution_failure_releases_claim_for_a_later_drain(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (intent_id,) = _create_intents(sqlite_session_factory, action="add")
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    monkeypatch.setattr(
        task_module.deal_dataset_vector_index_task,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("broker unavailable")),
    )

    with pytest.raises(OSError, match="broker unavailable"):
        task_module.process_dataset_indexing_intent_task.run(intent_id)

    with sqlite_session_factory() as session:
        row = session.get(DatasetIndexingOutbox, intent_id)
        assert row is not None
        assert row.claim_id is None
    assert repository.list_pending_intent_ids(limit=10) == (intent_id,)


def test_claim_preserves_route_receipt_when_dataset_handoff_is_not_proven(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    (intent_id,) = _create_intents(sqlite_session_factory, action="update")
    with sqlite_session_factory.begin() as session:
        session.execute(delete(Dataset).where(Dataset.id == "dataset-1"))

    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert repository.claim(intent_id) is None

    with sqlite_session_factory() as session:
        assert session.get(DatasetIndexingOutbox, intent_id) is not None


def test_superseded_generation_is_discarded_without_publication(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (old_intent_id,) = _create_intents(sqlite_session_factory, action="add")
    (new_intent_id,) = _create_intents(sqlite_session_factory, action="remove")
    _use_repository(monkeypatch, sqlite_session_factory)
    executed: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        task_module.deal_dataset_vector_index_task,
        "run",
        lambda dataset_id, action, generation_id, **_kwargs: (
            executed.append((dataset_id, action, generation_id)) or True
        ),
    )

    task_module.process_dataset_indexing_intent_task.run(old_intent_id)
    task_module.process_dataset_indexing_intent_task.run(new_intent_id)

    assert len(executed) == 1
    assert executed[0][:2] == ("dataset-1", "remove")
    with sqlite_session_factory() as session:
        assert session.scalar(select(DatasetIndexingOutbox.id).where(DatasetIndexingOutbox.id == old_intent_id)) is None
        dataset = session.get_one(Dataset, "dataset-1")
        assert dataset.indexing_request_id == executed[0][2]


def test_unstarted_target_is_not_misreported_as_applied_cleanup_route(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route_a = _route("qdrant", "collection-a")
    route_b = _route("weaviate", "collection-b")
    route_c = _route("milvus", "collection-c")

    def representation(route: VectorRoute) -> DatasetIndexingRepresentation:
        return DatasetIndexingRepresentation(
            indexing_technique="high_quality",
            index_struct=None,
            collection_binding_id=None,
            embedding_model="embedding",
            embedding_model_provider="provider",
            keyword_number=10,
            is_multimodal=False,
            doc_form="text_model",
            summary_index_setting=None,
            vector_route=route,
            cleanup_routes=(route,),
        )

    with sqlite_session_factory.begin() as session:
        session.add(make_dataset(dataset_id="dataset-1", tenant_id="tenant-1"))
        create_dataset_indexing_intents(
            session,
            DatasetIndexingJob(
                DatasetRef("tenant-1", "dataset-1"),
                representation(route_a),
                representation(route_b),
            ),
        )
        (new_intent_id,) = create_dataset_indexing_intents(
            session,
            DatasetIndexingJob(
                DatasetRef("tenant-1", "dataset-1"),
                representation(route_b),
                representation(route_c),
            ),
        )

    with sqlite_session_factory() as session:
        row = session.get_one(DatasetIndexingOutbox, new_intent_id)
        source = DatasetIndexingRepresentation.model_validate_json(row.source_representation)
        assert {(route.vector_type, route.collection_name) for route in source.cleanup_routes} == {
            ("qdrant", "collection-a"),
        }


def test_intent_creation_refreshes_applied_receipt_from_locked_dataset_owner(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route_a = _route("qdrant", "collection-a")
    route_b = _route("qdrant", "collection-b")
    route_c = _route("qdrant", "collection-c")

    def representation(route: VectorRoute) -> DatasetIndexingRepresentation:
        return DatasetIndexingRepresentation(
            indexing_technique="high_quality",
            index_struct=route.index_struct,
            collection_binding_id=None,
            embedding_model="embedding",
            embedding_model_provider="provider",
            keyword_number=10,
            is_multimodal=False,
            doc_form="text_model",
            summary_index_setting=None,
            vector_route=route,
        )

    applied_a = representation(route_a)
    applied_b = representation(route_b)
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_applied_representation = applied_a.model_dump_json()
        session.add(dataset)

    configuration_session = sqlite_session_factory()
    try:
        stale_dataset = configuration_session.get_one(Dataset, "dataset-1")
        configuration_session.commit()
        with sqlite_session_factory.begin() as worker_session:
            current_dataset = worker_session.get_one(Dataset, "dataset-1")
            current_dataset.indexing_applied_representation = applied_b.model_dump_json()
        assert stale_dataset.indexing_applied_representation == applied_a.model_dump_json()

        (intent_id,) = create_dataset_indexing_intents(
            configuration_session,
            DatasetIndexingJob(
                DatasetRef("tenant-1", "dataset-1"),
                applied_a,
                representation(route_c),
            ),
        )
        configuration_session.commit()
    finally:
        configuration_session.close()

    with sqlite_session_factory() as session:
        row = session.get_one(DatasetIndexingOutbox, intent_id)
        source = DatasetIndexingRepresentation.model_validate_json(row.source_representation)
        assert source.vector_route == route_b
        assert source.cleanup_routes == (route_b,)


def test_new_generation_inherits_partial_target_route_before_supersede(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route_a = _route("qdrant", "collection-a")
    route_b = _route("weaviate", "collection-b")
    route_c = _route("milvus", "collection-c")

    def representation(route: VectorRoute | None) -> DatasetIndexingRepresentation:
        return DatasetIndexingRepresentation(
            indexing_technique="high_quality",
            index_struct=None,
            collection_binding_id=None,
            embedding_model="embedding",
            embedding_model_provider="provider",
            keyword_number=10,
            is_multimodal=False,
            doc_form="text_model",
            summary_index_setting=None,
            vector_route=route,
            cleanup_routes=(route,) if route is not None else (),
        )

    with sqlite_session_factory.begin() as session:
        session.add(make_dataset(dataset_id="dataset-1", tenant_id="tenant-1"))
        (first_intent_id,) = create_dataset_indexing_intents(
            session,
            DatasetIndexingJob(
                DatasetRef("tenant-1", "dataset-1"),
                representation(route_a),
                representation(None),
            ),
        )
    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sqlite_session_factory)
    first_intent = repository.claim(first_intent_id)
    assert first_intent is not None
    with sqlite_session_factory.begin() as session:
        session.get_one(Dataset, "dataset-1", with_for_update=True)
        repository.record_target_route_locked(
            session,
            intent_id=first_intent.id,
            tenant_id=first_intent.tenant_id,
            dataset_id=first_intent.dataset_id,
            generation_id=first_intent.generation_id,
            claim_id=first_intent.claim_id,
            route=route_b,
        )
        retried_target = repository.record_target_route_locked(
            session,
            intent_id=first_intent.id,
            tenant_id=first_intent.tenant_id,
            dataset_id=first_intent.dataset_id,
            generation_id=first_intent.generation_id,
            claim_id=first_intent.claim_id,
            route=route_b,
        )
        assert retried_target.cleanup_routes == (route_b,)
    with sqlite_session_factory.begin() as session:
        (new_intent_id,) = create_dataset_indexing_intents(
            session,
            DatasetIndexingJob(
                DatasetRef("tenant-1", "dataset-1"),
                representation(route_a),
                representation(route_c),
            ),
        )

    with sqlite_session_factory() as session:
        row = session.get_one(DatasetIndexingOutbox, new_intent_id)
        source = DatasetIndexingRepresentation.model_validate_json(row.source_representation)
        assert {(route.vector_type, route.collection_name) for route in source.cleanup_routes} == {
            ("qdrant", "collection-a"),
            ("weaviate", "collection-b"),
        }
        dataset = session.get_one(Dataset, "dataset-1")
        assert dataset.indexing_applied_representation is not None
        applied = DatasetIndexingRepresentation.model_validate_json(dataset.indexing_applied_representation)
        assert applied.vector_route == route_a


def test_completed_generation_document_writer_reuses_applied_physical_route(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route("qdrant", "collection-applied")
    applied = DatasetIndexingRepresentation(
        indexing_technique="high_quality",
        index_struct=route.index_struct,
        collection_binding_id=None,
        embedding_model="embedding",
        embedding_model_provider="provider",
        keyword_number=10,
        is_multimodal=False,
        doc_form="text_model",
        summary_index_setting=None,
        vector_route=route,
    )
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.indexing_request_id = "generation-applied"
        dataset.indexing_applied_representation = applied.model_dump_json()
        document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        document.indexing_request_id = "document-request"
        document.indexing_claim_id = "document-claim"
        document.indexing_claim_expires_at = datetime(2099, 1, 1)
        document.indexing_dataset_generation_id = dataset.indexing_request_id
        session.add_all([dataset, document])

    class AppliedRouteVector:
        def __init__(self, dataset: Dataset, *, session: Session | None, vector_type: str) -> None:
            assert dataset.id == "dataset-1"
            assert session is None
            assert vector_type == "qdrant"

        @staticmethod
        def resolve_vector_type(dataset: Dataset, *, session: Session) -> str:
            assert dataset.id == "dataset-1"
            assert session.in_transaction()
            return "qdrant"

        @staticmethod
        def get_routing_target() -> tuple[str, str]:
            return "qdrant", "collection-applied"

    monkeypatch.setattr(indexing_execution_adapter_module, "Vector", AppliedRouteVector)
    adapter = IndexingExecutionAdapter(
        session_factory=sqlite_session_factory,
        documents=SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory),
        segments=SQLAlchemySegmentRepository(session_factory=sqlite_session_factory),
        uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory),
        attachment_vectors=SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory),
        sources=CompositeStoredSourceResolver(adapters={}),
    )
    indexing_document = IndexingDocument(
        source=StoredSource(
            DatasetRef("tenant-1", "dataset-1").document("document-1"),
            "upload_file",
            {},
            "text_model",
        ),
        processing_rule={},
        doc_language="English",
        need_summary=False,
        claim_id="document-claim",
        indexing_request_id="document-request",
        dataset_generation_id="generation-applied",
    )

    assert adapter._record_vector_route_before_write(indexing_document) == route

    with sqlite_session_factory() as session:
        assert session.scalar(select(DatasetIndexingOutbox.id)) is None
        assert session.get_one(Document, "document-1").indexing_claim_id == "document-claim"


def test_embedding_update_runs_ordered_vector_and_summary_phases_before_ack(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_ids = _create_intents(sqlite_session_factory, action="update")

    assert len(intent_ids) == 1
    with sqlite_session_factory() as session:
        rows = session.scalars(select(DatasetIndexingOutbox).where(DatasetIndexingOutbox.id.in_(intent_ids))).all()
        assert {row.operation for row in rows} == {"rebuild"}
        assert len({row.generation_id for row in rows}) == 1
        dataset = session.get_one(Dataset, "dataset-1")
        assert dataset.indexing_request_id == rows[0].generation_id

    _use_repository(monkeypatch, sqlite_session_factory)
    vectors: list[tuple[str, str, str]] = []
    summaries: list[tuple[str, str, bool, str]] = []
    monkeypatch.setattr(
        task_module.deal_dataset_vector_index_task,
        "run",
        lambda dataset_id, action, generation_id, **_kwargs: (
            vectors.append((dataset_id, action, generation_id)) or True
        ),
    )
    monkeypatch.setattr(
        task_module.regenerate_summary_index_task,
        "run",
        lambda dataset_id, *, regenerate_reason, regenerate_vectors_only, indexing_request_id, **_kwargs: (
            summaries.append((dataset_id, regenerate_reason, regenerate_vectors_only, indexing_request_id)) or True
        ),
    )

    for intent_id in intent_ids:
        task_module.process_dataset_indexing_intent_task.run(intent_id)

    assert len(vectors) == 1
    assert vectors[0][:2] == ("dataset-1", "update")
    assert summaries == [("dataset-1", "embedding_model_changed", True, vectors[0][2])]
    with sqlite_session_factory() as session:
        assert session.get(DatasetIndexingOutbox, intent_ids[0]) is None


def test_add_to_high_quality_restores_only_enabled_retained_summaries_before_ack(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (intent_id,) = _create_intents(sqlite_session_factory, action="add")
    summary_route = VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="summary-collection",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    with sqlite_session_factory.begin() as session:
        document = make_document(
            document_id="document-1",
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="text_model",
        )
        document.indexing_status = "completed"
        document.enabled = True
        document.archived = False
        document.need_summary = True
        enabled_segment = DocumentSegment(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id=document.id,
            position=1,
            content="enabled segment",
            word_count=2,
            tokens=2,
            created_by="account-1",
            index_node_id="segment-node-1",
            index_node_hash="segment-hash-1",
            status=SegmentStatus.COMPLETED,
        )
        enabled_segment.id = "segment-1"
        disabled_segment = DocumentSegment(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id=document.id,
            position=2,
            content="disabled segment",
            word_count=2,
            tokens=2,
            created_by="account-1",
            index_node_id="segment-node-2",
            index_node_hash="segment-hash-2",
            status=SegmentStatus.COMPLETED,
        )
        disabled_segment.id = "segment-2"
        missing_segment = DocumentSegment(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id=document.id,
            position=3,
            content="missing summary segment",
            word_count=3,
            tokens=3,
            created_by="account-1",
            index_node_id="segment-node-3",
            index_node_hash="segment-hash-3",
            status=SegmentStatus.COMPLETED,
        )
        missing_segment.id = "segment-3"
        pending_segment = DocumentSegment(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id=document.id,
            position=4,
            content="pending summary segment",
            word_count=3,
            tokens=3,
            created_by="account-1",
            index_node_id="segment-node-4",
            index_node_hash="segment-hash-4",
            status=SegmentStatus.COMPLETED,
        )
        pending_segment.id = "segment-4"
        enabled_summary = DocumentSegmentSummary(
            dataset_id="dataset-1",
            document_id=document.id,
            chunk_id=enabled_segment.id,
            summary_content="enabled retained summary",
            summary_index_node_id="old-enabled-node",
            summary_index_node_hash="old-enabled-hash",
            vector_route=summary_route.serialized,
            vector_route_hash=summary_route.key,
            tokens=3,
            status=SummaryStatus.COMPLETED,
            enabled=True,
        )
        enabled_summary.id = "summary-1"
        disabled_summary = DocumentSegmentSummary(
            dataset_id="dataset-1",
            document_id=document.id,
            chunk_id=disabled_segment.id,
            summary_content="disabled retained summary",
            summary_index_node_id="old-disabled-node",
            summary_index_node_hash="old-disabled-hash",
            vector_route=summary_route.serialized,
            vector_route_hash=summary_route.key,
            tokens=5,
            status=SummaryStatus.COMPLETED,
            enabled=False,
        )
        disabled_summary.id = "summary-2"
        pending_summary = DocumentSegmentSummary(
            dataset_id="dataset-1",
            document_id=document.id,
            chunk_id=pending_segment.id,
            summary_content=None,
            status=SummaryStatus.GENERATING,
            enabled=True,
        )
        pending_summary.id = "summary-4"
        session.add_all(
            [
                document,
                enabled_segment,
                disabled_segment,
                missing_segment,
                pending_segment,
                enabled_summary,
                disabled_summary,
                pending_summary,
            ]
        )

    _use_repository(monkeypatch, sqlite_session_factory)

    vector_phases: list[tuple[str, str, str]] = []

    def persist_high_quality_target(
        dataset_id: str,
        action: str,
        generation_id: str,
        **_kwargs: object,
    ) -> bool:
        vector_phases.append((dataset_id, action, generation_id))
        with sqlite_session_factory.begin() as session:
            dataset = session.get_one(Dataset, dataset_id)
            assert dataset.indexing_request_id == generation_id
            dataset.indexing_technique = "high_quality"
            dataset.embedding_model = "text-embedding-model"
            dataset.embedding_model_provider = "provider"
            dataset.summary_index_setting = {"enable": False}
        return True

    monkeypatch.setattr(task_module.deal_dataset_vector_index_task, "run", persist_high_quality_target)
    assert summary_task_module.session_factory is summary_adapter_module.session_factory
    monkeypatch.setattr(summary_task_module.session_factory, "create_session", sqlite_session_factory)

    embedding = _EmbeddingModelBoundary()
    manager = _TenantModelManagerBoundary(embedding)

    class _ModelManagerBoundary:
        @staticmethod
        def for_tenant(*, tenant_id: str) -> _TenantModelManagerBoundary:
            assert tenant_id == "tenant-1"
            return manager

    vector = _VectorBoundary()
    monkeypatch.setattr(summary_adapter_module, "ModelManager", _ModelManagerBoundary)
    monkeypatch.setattr(summary_adapter_module, "Vector", _VectorConstructorBoundary(vector))

    def vector_for_route(route: VectorRoute) -> _VectorBoundary:
        assert route == summary_route
        return vector

    monkeypatch.setattr(summary_adapter_module, "vector_for_route", vector_for_route)

    task_module.process_dataset_indexing_intent_task.run(intent_id)

    assert len(vector_phases) == 1
    assert vector_phases[0][:2] == ("dataset-1", "add")
    assert embedding.contents == [["enabled retained summary"]]
    assert len(vector.added) == 1
    published = vector.added[0]
    assert published.page_content == "enabled retained summary"
    assert published.metadata["dataset_id"] == "dataset-1"
    assert published.metadata["document_id"] == "document-1"
    assert published.metadata["original_chunk_id"] == "segment-1"
    assert published.metadata["is_summary"] is True
    assert vector.deleted == [["old-enabled-node"]]

    with sqlite_session_factory() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        assert dataset.summary_index_setting == {"enable": False}
        refreshed_enabled = session.get_one(DocumentSegmentSummary, "summary-1")
        assert refreshed_enabled.enabled is True
        # Embedding generations are owned by the parent intent and durable
        # vector writer; retained summary text keeps its content revision.
        assert refreshed_enabled.version == 1
        assert refreshed_enabled.status == SummaryStatus.COMPLETED
        assert refreshed_enabled.summary_content == "enabled retained summary"
        assert refreshed_enabled.summary_index_node_id not in (None, "old-enabled-node")
        assert refreshed_enabled.summary_index_node_hash is not None
        assert refreshed_enabled.tokens == 7
        assert refreshed_enabled.vector_route == summary_route.serialized
        assert refreshed_enabled.vector_route_hash == summary_route.key

        refreshed_disabled = session.get_one(DocumentSegmentSummary, "summary-2")
        assert refreshed_disabled.enabled is False
        assert refreshed_disabled.version == 1
        assert refreshed_disabled.status == SummaryStatus.COMPLETED
        assert refreshed_disabled.summary_index_node_id == "old-disabled-node"
        assert refreshed_disabled.summary_index_node_hash == "old-disabled-hash"
        assert refreshed_disabled.tokens == 5

        assert (
            session.scalar(select(DocumentSegmentSummary.id).where(DocumentSegmentSummary.chunk_id == "segment-3"))
            is None
        )
        refreshed_pending = session.get_one(DocumentSegmentSummary, "summary-4")
        assert refreshed_pending.version == 1
        assert refreshed_pending.summary_content is None
        assert refreshed_pending.status == SummaryStatus.GENERATING

        assert session.get(DatasetIndexingOutbox, intent_id) is None
        assert session.scalar(select(SummaryVectorCleanupOutbox.id)) is None


def test_auto_enabled_rebuild_generates_only_requested_missing_and_pending_before_ack(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (intent_id,) = _create_intents(sqlite_session_factory, action="update")

    def make_segment(segment_id: str, position: int, *, document_id: str = "document-1") -> DocumentSegment:
        segment = DocumentSegment(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id=document_id,
            position=position,
            content=f"segment content {position}",
            word_count=3,
            tokens=3,
            created_by="account-1",
            index_node_id=f"segment-node-{position}",
            index_node_hash=f"segment-hash-{position}",
            status=SegmentStatus.COMPLETED,
        )
        segment.id = segment_id
        return segment

    with sqlite_session_factory.begin() as session:
        document = make_document(
            document_id="document-1",
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="text_model",
        )
        document.indexing_status = "completed"
        document.enabled = True
        document.archived = False
        document.need_summary = True
        opted_out_document = make_document(
            document_id="document-no-summary",
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="text_model",
        )
        opted_out_document.indexing_status = "completed"
        opted_out_document.enabled = True
        opted_out_document.archived = False
        opted_out_document.need_summary = False
        retained_segment = make_segment("segment-retained", 1)
        pending_segment = make_segment("segment-pending", 2)
        missing_segment = make_segment("segment-missing", 3)
        disabled_segment = make_segment("segment-disabled", 4)
        opted_out_retained_segment = make_segment("segment-opted-out-retained", 1, document_id=opted_out_document.id)
        opted_out_pending_segment = make_segment("segment-opted-out-pending", 2, document_id=opted_out_document.id)
        opted_out_missing_segment = make_segment("segment-opted-out-missing", 3, document_id=opted_out_document.id)
        retained = DocumentSegmentSummary(
            dataset_id="dataset-1",
            document_id=document.id,
            chunk_id=retained_segment.id,
            summary_content="retained text",
            status=SummaryStatus.COMPLETED,
            enabled=True,
        )
        retained.id = "summary-retained"
        pending = DocumentSegmentSummary(
            dataset_id="dataset-1",
            document_id=document.id,
            chunk_id=pending_segment.id,
            summary_content=None,
            status=SummaryStatus.GENERATING,
            enabled=True,
        )
        pending.id = "summary-pending"
        disabled = DocumentSegmentSummary(
            dataset_id="dataset-1",
            document_id=document.id,
            chunk_id=disabled_segment.id,
            summary_content=None,
            status=SummaryStatus.GENERATING,
            enabled=False,
        )
        disabled.id = "summary-disabled"
        opted_out_retained = DocumentSegmentSummary(
            dataset_id="dataset-1",
            document_id=opted_out_document.id,
            chunk_id=opted_out_retained_segment.id,
            summary_content="opted-out retained text",
            status=SummaryStatus.COMPLETED,
            enabled=True,
        )
        opted_out_retained.id = "summary-opted-out-retained"
        opted_out_pending = DocumentSegmentSummary(
            dataset_id="dataset-1",
            document_id=opted_out_document.id,
            chunk_id=opted_out_pending_segment.id,
            summary_content=None,
            status=SummaryStatus.GENERATING,
            enabled=True,
        )
        opted_out_pending.id = "summary-opted-out-pending"
        session.add_all(
            [
                document,
                opted_out_document,
                retained_segment,
                pending_segment,
                missing_segment,
                disabled_segment,
                opted_out_retained_segment,
                opted_out_pending_segment,
                opted_out_missing_segment,
                retained,
                pending,
                disabled,
                opted_out_retained,
                opted_out_pending,
            ]
        )

    _use_repository(monkeypatch, sqlite_session_factory)

    def persist_target(dataset_id: str, _action: str, generation_id: str, **_kwargs: object) -> bool:
        with sqlite_session_factory.begin() as session:
            dataset = session.get_one(Dataset, dataset_id)
            assert dataset.indexing_request_id == generation_id
            dataset.indexing_technique = "high_quality"
            dataset.embedding_model = "text-embedding-model"
            dataset.embedding_model_provider = "provider"
            dataset.summary_index_setting = {"enable": True}
        return True

    monkeypatch.setattr(task_module.deal_dataset_vector_index_task, "run", persist_target)
    monkeypatch.setattr(summary_task_module.session_factory, "create_session", sqlite_session_factory)
    embedding = _EmbeddingModelBoundary()
    manager = _TenantModelManagerBoundary(embedding)

    class _ModelManagerBoundary:
        @staticmethod
        def for_tenant(*, tenant_id: str) -> _TenantModelManagerBoundary:
            assert tenant_id == "tenant-1"
            return manager

    vector = _VectorBoundary()
    monkeypatch.setattr(summary_adapter_module, "ModelManager", _ModelManagerBoundary)
    monkeypatch.setattr(summary_adapter_module, "Vector", _VectorConstructorBoundary(vector))
    generated_segments: list[str] = []

    def generate(
        segment: DocumentSegment,
        dataset: Dataset,
        setting: SummaryIndexSettingDict,
        *,
        session: Session,
    ) -> tuple[str, LLMUsage]:
        assert dataset.indexing_request_id is not None
        assert setting.get("enable") is True
        session.commit()
        generated_segments.append(segment.id)
        return f"generated:{segment.id}", _usage()

    monkeypatch.setattr(summary_adapter_module.SummaryIndexAdapter, "generate_summary_for_segment", generate)

    task_module.process_dataset_indexing_intent_task.run(intent_id)

    assert generated_segments == ["segment-pending", "segment-missing"]
    assert [document.page_content for document in vector.added] == [
        "retained text",
        "generated:segment-pending",
        "generated:segment-missing",
        "opted-out retained text",
    ]
    assert embedding.contents == [
        ["retained text"],
        ["generated:segment-pending"],
        ["generated:segment-missing"],
        ["opted-out retained text"],
    ]
    with sqlite_session_factory() as session:
        assert session.get(DatasetIndexingOutbox, intent_id) is None
        retained = session.get_one(DocumentSegmentSummary, "summary-retained")
        assert (retained.version, retained.summary_content, retained.status) == (
            1,
            "retained text",
            SummaryStatus.COMPLETED,
        )
        pending = session.get_one(DocumentSegmentSummary, "summary-pending")
        assert (pending.version, pending.summary_content, pending.status) == (
            2,
            "generated:segment-pending",
            SummaryStatus.COMPLETED,
        )
        missing = session.scalar(
            select(DocumentSegmentSummary).where(DocumentSegmentSummary.chunk_id == "segment-missing")
        )
        assert missing is not None
        assert (missing.version, missing.summary_content, missing.status) == (
            1,
            "generated:segment-missing",
            SummaryStatus.COMPLETED,
        )
        disabled = session.get_one(DocumentSegmentSummary, "summary-disabled")
        assert (disabled.enabled, disabled.version, disabled.summary_content, disabled.status) == (
            False,
            1,
            None,
            SummaryStatus.GENERATING,
        )
        opted_out_retained = session.get_one(DocumentSegmentSummary, "summary-opted-out-retained")
        assert (
            opted_out_retained.version,
            opted_out_retained.summary_content,
            opted_out_retained.status,
        ) == (1, "opted-out retained text", SummaryStatus.COMPLETED)
        opted_out_pending = session.get_one(DocumentSegmentSummary, "summary-opted-out-pending")
        assert (
            opted_out_pending.version,
            opted_out_pending.summary_content,
            opted_out_pending.status,
        ) == (1, None, SummaryStatus.GENERATING)
        assert (
            session.scalar(
                select(DocumentSegmentSummary.id).where(DocumentSegmentSummary.chunk_id == "segment-opted-out-missing")
            )
            is None
        )


def test_new_generation_takes_over_summary_blocked_in_old_llm_before_ack(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_request_id = "generation-0"
        dataset.indexing_technique = "high_quality"
        dataset.embedding_model = "text-embedding-model"
        dataset.embedding_model_provider = "provider"
        dataset.summary_index_setting = {"enable": True}
        document = make_document(
            document_id="document-1",
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="text_model",
        )
        document.indexing_status = "completed"
        document.enabled = True
        document.archived = False
        document.need_summary = True
        segment = DocumentSegment(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id=document.id,
            position=1,
            content="segment content",
            word_count=2,
            tokens=2,
            created_by="account-1",
            index_node_id="segment-node",
            index_node_hash="segment-hash",
            status=SegmentStatus.COMPLETED,
        )
        segment.id = "segment-1"
        session.add_all([dataset, document, segment])

    monkeypatch.setattr(summary_task_module.session_factory, "create_session", sqlite_session_factory)
    embedding = _EmbeddingModelBoundary()
    manager = _TenantModelManagerBoundary(embedding)

    class _ModelManagerBoundary:
        @staticmethod
        def for_tenant(*, tenant_id: str) -> _TenantModelManagerBoundary:
            assert tenant_id == "tenant-1"
            return manager

    vector = _VectorBoundary()
    monkeypatch.setattr(summary_adapter_module, "ModelManager", _ModelManagerBoundary)
    monkeypatch.setattr(summary_adapter_module, "Vector", _VectorConstructorBoundary(vector))
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
        session.commit()
        if not old_llm_started.is_set():
            old_llm_started.set()
            assert release_old_llm.wait(timeout=5)
            generated_contents.append("old generation content")
            return "old generation content", _usage()
        generated_contents.append("new generation content")
        return "new generation content", _usage()

    monkeypatch.setattr(summary_adapter_module.SummaryIndexAdapter, "generate_summary_for_segment", generate)
    old_results: list[DocumentSegmentSummary | None] = []
    old_errors: list[BaseException] = []

    def run_old_generation() -> None:
        try:
            with sqlite_session_factory() as session:
                dataset = session.get_one(Dataset, "dataset-1")
                segment = session.get_one(DocumentSegment, "segment-1")
                old_results.append(
                    summary_adapter_module.SummaryIndexAdapter.generate_and_vectorize_summary(
                        segment,
                        dataset,
                        {"enable": True},
                        session=session,
                        indexing_request_id="generation-0",
                    )
                )
        except BaseException as error:
            old_errors.append(error)

    old_thread = Thread(target=run_old_generation, name="old-summary-generation", daemon=True)
    old_thread.start()
    assert old_llm_started.wait(timeout=5)

    (intent_id,) = _create_intents(sqlite_session_factory, action="update")
    _use_repository(monkeypatch, sqlite_session_factory)

    def persist_target(dataset_id: str, _action: str, generation_id: str, **_kwargs: object) -> bool:
        with sqlite_session_factory.begin() as session:
            dataset = session.get_one(Dataset, dataset_id)
            assert dataset.indexing_request_id == generation_id
            dataset.indexing_technique = "high_quality"
            dataset.embedding_model = "text-embedding-model"
            dataset.embedding_model_provider = "provider"
            dataset.summary_index_setting = {"enable": True}
        return True

    monkeypatch.setattr(task_module.deal_dataset_vector_index_task, "run", persist_target)

    try:
        task_module.process_dataset_indexing_intent_task.run(intent_id)
        with sqlite_session_factory() as session:
            assert session.get(DatasetIndexingOutbox, intent_id) is None
            summary = session.scalar(
                select(DocumentSegmentSummary).where(DocumentSegmentSummary.chunk_id == "segment-1")
            )
            assert summary is not None
            assert (summary.version, summary.summary_content, summary.status, summary.error) == (
                2,
                "new generation content",
                SummaryStatus.COMPLETED,
                None,
            )
            published_node_id = summary.summary_index_node_id
            assert published_node_id is not None
    finally:
        release_old_llm.set()
        old_thread.join(timeout=5)

    assert not old_thread.is_alive()
    assert old_errors == []
    assert old_results == [None]
    assert generated_contents == ["new generation content", "old generation content"]
    assert [document.page_content for document in vector.added] == ["new generation content"]
    with sqlite_session_factory() as session:
        summary = session.scalar(select(DocumentSegmentSummary).where(DocumentSegmentSummary.chunk_id == "segment-1"))
        assert summary is not None
        assert (
            summary.version,
            summary.summary_content,
            summary.summary_index_node_id,
            summary.status,
            summary.error,
        ) == (
            2,
            "new generation content",
            published_node_id,
            SummaryStatus.COMPLETED,
            None,
        )


def test_crash_before_vector_phase_checkpoint_rebuilds_vector_again(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (intent_id,) = _create_intents(sqlite_session_factory, action="update")
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    phases: list[str] = []
    monkeypatch.setattr(
        task_module.deal_dataset_vector_index_task,
        "run",
        lambda *_args, **_kwargs: phases.append("vector") or True,
    )
    monkeypatch.setattr(
        task_module.regenerate_summary_index_task,
        "run",
        lambda *_args, **_kwargs: phases.append("summary") or True,
    )
    original_mark_vector_completed = repository.mark_vector_completed
    checkpoint_attempts = 0

    def crash_once(intent: DatasetIndexingIntent) -> bool:
        nonlocal checkpoint_attempts
        checkpoint_attempts += 1
        if checkpoint_attempts == 1:
            raise OSError("crash before vector checkpoint")
        return original_mark_vector_completed(intent)

    monkeypatch.setattr(repository, "mark_vector_completed", crash_once)

    with pytest.raises(OSError, match="crash before vector checkpoint"):
        task_module.process_dataset_indexing_intent_task.run(intent_id)
    task_module.process_dataset_indexing_intent_task.run(intent_id)

    assert phases == ["vector", "vector", "summary"]
    with sqlite_session_factory() as session:
        assert session.get(DatasetIndexingOutbox, intent_id) is None


def test_summary_failure_after_vector_checkpoint_retries_only_summary(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (intent_id,) = _create_intents(sqlite_session_factory, action="update")
    _use_repository(monkeypatch, sqlite_session_factory)
    phases: list[str] = []
    monkeypatch.setattr(
        task_module.deal_dataset_vector_index_task,
        "run",
        lambda *_args, **_kwargs: phases.append("vector") or True,
    )
    summary_attempts = 0

    def summary(*_args, **_kwargs) -> bool:
        nonlocal summary_attempts
        summary_attempts += 1
        phases.append("summary")
        if summary_attempts == 1:
            raise OSError("summary unavailable")
        return True

    monkeypatch.setattr(task_module.regenerate_summary_index_task, "run", summary)

    with pytest.raises(OSError, match="summary unavailable"):
        task_module.process_dataset_indexing_intent_task.run(intent_id)
    with sqlite_session_factory() as session:
        row = session.get_one(DatasetIndexingOutbox, intent_id)
        assert row.vector_completed_at is not None
        assert row.claim_id is None

    task_module.process_dataset_indexing_intent_task.run(intent_id)

    assert phases == ["vector", "summary", "summary"]
    with sqlite_session_factory() as session:
        assert session.get(DatasetIndexingOutbox, intent_id) is None


def test_duplicate_delivery_cannot_enter_execution_while_claim_is_active(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    (intent_id,) = _create_intents(sqlite_session_factory, action="add")
    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sqlite_session_factory)

    first = repository.claim(intent_id)
    second = repository.claim(intent_id)

    assert first is not None
    assert second is None


def test_false_summary_result_retains_intent_for_retry(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (intent_id,) = _create_intents(sqlite_session_factory, action="update")
    _use_repository(monkeypatch, sqlite_session_factory)
    monkeypatch.setattr(task_module.deal_dataset_vector_index_task, "run", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(task_module.regenerate_summary_index_task, "run", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="Summary rebuild did not complete"):
        task_module.process_dataset_indexing_intent_task.run(intent_id)

    with sqlite_session_factory() as session:
        row = session.get_one(DatasetIndexingOutbox, intent_id)
        assert row.vector_completed_at is not None
        assert row.claim_id is None

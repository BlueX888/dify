from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime

import pytest
from sqlalchemy import update
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.orm import Session, sessionmaker

import tasks.dataset_indexing_outbox_task as outbox_task_module
import tasks.deal_dataset_vector_index_task as vector_task_module
import tasks.regenerate_summary_index_task as summary_task_module
from core.rag.models.document import Document as IndexDocument
from models.dataset import Dataset, DocumentSegment, DocumentSegmentSummary
from models.dataset import Document as DatasetDocument
from models.enums import IndexingStatus, SegmentStatus, SummaryStatus
from repositories.knowledge.dataset_indexing_outbox_repository import (
    DatasetIndexingIntent,
    DatasetIndexingWritersActiveError,
    SQLAlchemyDatasetIndexingOutboxRepository,
    create_dataset_indexing_intents,
    dataset_indexing_generation_lock_statement,
    lock_dataset_indexing_execution,
)
from repositories.knowledge.document_indexing_outbox_repository import create_document_indexing_intents
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.knowledge.entities.datasets import DatasetIndexingJob, DatasetIndexingRepresentation
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.vector_route import VectorRoute
from tests.unit_tests.model_factories import make_dataset, make_document


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


class _IndexProcessorStub:
    def __init__(
        self,
        *,
        clean: Callable[..., None],
        load: Callable[..., None],
    ) -> None:
        self._clean = clean
        self._load = load

    def clean(self, *args: object, **kwargs: object) -> None:
        self._clean(*args, **kwargs)

    def load(self, *args: object, **kwargs: object) -> None:
        self._load(*args, **kwargs)


class _IndexProcessorFactoryStub:
    def __init__(self, processor: _IndexProcessorStub) -> None:
        self._processor = processor

    def init_index_processor(self) -> _IndexProcessorStub:
        return self._processor


class _ResolvedTargetVector:
    def __init__(self, _dataset: Dataset, *, session: Session) -> None:
        del session

    def get_routing_target(self) -> tuple[str, str]:
        return "qdrant", "target-collection"


@pytest.fixture(autouse=True)
def _resolve_target_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vector_task_module, "Vector", _ResolvedTargetVector)


def _persist_newer_generation(session_factory: sessionmaker[Session]) -> None:
    with session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_request_id = "new-generation"
        dataset.indexing_technique = "high_quality"
        session.add(dataset)


def _claim_rebuild(
    session_factory: sessionmaker[Session],
    *,
    action: str,
) -> tuple[SQLAlchemyDatasetIndexingOutboxRepository, DatasetIndexingIntent]:
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
    high_quality = economy.model_copy(update={"indexing_technique": "high_quality"})
    source, target = {
        "add": (economy, high_quality),
        "remove": (high_quality, economy),
        "update": (high_quality, high_quality),
    }[action]
    with session_factory.begin() as session:
        (intent_id,) = create_dataset_indexing_intents(
            session,
            DatasetIndexingJob(DatasetRef("tenant-1", "dataset-1"), source, target),
        )
    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=session_factory)
    intent = repository.claim(intent_id)
    assert intent is not None
    return repository, intent


def _run_rebuild(intent: DatasetIndexingIntent, action: str) -> bool:
    return vector_task_module.deal_dataset_vector_index_task.run(
        intent.dataset_id,
        action,
        intent.generation_id,
        indexing_intent_id=intent.id,
        indexing_claim_id=intent.claim_id,
        tenant_id=intent.tenant_id,
        source_representation=intent.source.model_dump_json(),
        target_representation=intent.target.model_dump_json(),
    )


@pytest.mark.parametrize("request_id", ["old-generation", None])
def test_superseded_or_legacy_vector_worker_never_constructs_an_index_processor(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    request_id: str | None,
) -> None:
    _persist_newer_generation(sqlite_session_factory)
    monkeypatch.setattr(vector_task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(
        vector_task_module,
        "IndexProcessorFactory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stale vector worker ran")),
    )

    vector_task_module.deal_dataset_vector_index_task.run("dataset-1", "remove", request_id)

    with sqlite_session_factory() as session:
        assert session.get_one(Dataset, "dataset-1").indexing_request_id == "new-generation"


@pytest.mark.parametrize("request_id", ["old-generation", None])
def test_superseded_or_legacy_summary_worker_never_reads_or_writes_summary_rows(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    request_id: str | None,
) -> None:
    _persist_newer_generation(sqlite_session_factory)
    monkeypatch.setattr(summary_task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(
        summary_task_module.SummaryIndexAdapter,
        "vectorize_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stale summary worker ran")),
    )

    summary_task_module.regenerate_summary_index_task.run(
        "dataset-1",
        regenerate_reason="embedding_model_changed",
        regenerate_vectors_only=True,
        indexing_request_id=request_id,
    )

    with sqlite_session_factory() as session:
        assert session.get_one(Dataset, "dataset-1").indexing_request_id == "new-generation"


def test_legacy_generation_messages_republish_current_execution_instead_of_mutating(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(make_dataset(dataset_id="dataset-1", tenant_id="tenant-1"))
    repository, intent = _claim_rebuild(sqlite_session_factory, action="update")
    intent_id = intent.id
    generation_id = intent.generation_id
    monkeypatch.setattr(vector_task_module, "_repository", lambda: repository)
    monkeypatch.setattr(
        summary_task_module,
        "SQLAlchemyDatasetIndexingOutboxRepository",
        lambda **_kwargs: repository,
    )
    republished: list[str] = []
    monkeypatch.setattr(outbox_task_module, "enqueue_dataset_indexing_intent", republished.append)
    monkeypatch.setattr(
        vector_task_module,
        "IndexProcessorFactory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy vector message performed I/O")),
    )

    assert not vector_task_module.deal_dataset_vector_index_task.run("dataset-1", "update", generation_id)
    assert not summary_task_module.regenerate_summary_index_task.run(
        "dataset-1",
        regenerate_reason="embedding_model_changed",
        regenerate_vectors_only=True,
        indexing_request_id=generation_id,
    )

    assert republished == [intent_id, intent_id]


@pytest.mark.parametrize("action", ["add", "update"])
def test_generation_change_after_worker_start_fences_later_vector_io(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_request_id = "generation-1"
        dataset.indexing_technique = "high_quality"
        document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        document.indexing_status = "completed"
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=1,
            content="content",
            word_count=1,
            tokens=1,
            created_by="account-1",
            index_node_id="node-1",
            index_node_hash="hash-1",
            status=SegmentStatus.COMPLETED,
        )
        session.add_all([dataset, document, segment])

    external_calls: list[str] = []
    processor = _IndexProcessorStub(
        clean=lambda *_args, **_kwargs: external_calls.append("clean"),
        load=lambda *_args, **_kwargs: external_calls.append("load"),
    )
    monkeypatch.setattr(
        vector_task_module,
        "IndexProcessorFactory",
        lambda *_args, **_kwargs: _IndexProcessorFactoryStub(processor),
    )
    repository, intent = _claim_rebuild(sqlite_session_factory, action=action)
    monkeypatch.setattr(vector_task_module, "_repository", lambda: repository)

    task_session = sqlite_session_factory()
    original_commit = task_session.commit
    commit_count = 0

    def commit_then_publish_new_generation() -> None:
        nonlocal commit_count
        original_commit()
        commit_count += 1
        if commit_count == 1:
            with sqlite_session_factory.begin() as config_session:
                config_session.execute(
                    update(Dataset).where(Dataset.id == "dataset-1").values(indexing_request_id="generation-2")
                )

    monkeypatch.setattr(task_session, "commit", commit_then_publish_new_generation)
    monkeypatch.setattr(vector_task_module.session_factory, "create_session", lambda: task_session)

    _run_rebuild(intent, action)

    assert commit_count == 1
    # The first commit records target-route evidence only. A superseding
    # generation between that receipt and publication fences all vector I/O.
    assert external_calls == []
    with sqlite_session_factory() as session:
        assert session.get_one(Dataset, "dataset-1").indexing_request_id == "generation-2"


def test_partial_b_cleanup_and_current_c_publication_preserve_same_node_identity(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_a = _route("qdrant", "collection-a")
    route_b = _route("qdrant", "collection-b")
    route_c = _route("qdrant", "collection-c")

    def representation(route: VectorRoute | None) -> DatasetIndexingRepresentation:
        return DatasetIndexingRepresentation(
            indexing_technique="high_quality",
            index_struct=route.index_struct if route is not None else None,
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
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_technique = "high_quality"
        dataset.embedding_model = "embedding"
        dataset.embedding_model_provider = "provider"
        document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        document.indexing_status = "completed"
        session.add_all(
            [
                dataset,
                document,
                DocumentSegment(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_id=document.id,
                    position=1,
                    content="content",
                    word_count=1,
                    tokens=1,
                    created_by="account-1",
                    index_node_id="node-1",
                    index_node_hash="hash-1",
                    status=SegmentStatus.COMPLETED,
                ),
            ]
        )
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
    with sqlite_session_factory.begin() as session:
        (current_intent_id,) = create_dataset_indexing_intents(
            session,
            DatasetIndexingJob(
                DatasetRef("tenant-1", "dataset-1"),
                representation(route_a),
                representation(None),
            ),
        )
    current_intent = repository.claim(current_intent_id)
    assert current_intent is not None

    remote_nodes = {("collection-a", "node-1"), ("collection-b", "node-1")}

    def route_name(dataset: Dataset) -> str:
        assert dataset.index_struct_dict is not None
        return str(dataset.index_struct_dict["vector_store"]["class_prefix"])

    def clean(dataset: Dataset, _node_ids: object, **_kwargs: object) -> None:
        collection = route_name(dataset)
        remote_nodes.difference_update({item for item in remote_nodes if item[0] == collection})

    def load(dataset: Dataset, documents: list[IndexDocument], **_kwargs: object) -> None:
        collection = route_name(dataset)
        remote_nodes.update((collection, str(document.metadata["doc_id"])) for document in documents)

    class CurrentTargetVector:
        def __init__(self, _dataset: Dataset, *, session: Session) -> None:
            del session

        def get_routing_target(self) -> tuple[str, str]:
            return "qdrant", "collection-c"

    class PhysicalCleanupVector:
        def __init__(self, route: VectorRoute) -> None:
            self._route = route

        def delete(self) -> None:
            remote_nodes.difference_update({item for item in remote_nodes if item[0] == self._route.collection_name})

    processor = _IndexProcessorStub(clean=clean, load=load)
    monkeypatch.setattr(vector_task_module, "Vector", CurrentTargetVector)
    monkeypatch.setattr(vector_task_module, "vector_for_route", PhysicalCleanupVector)
    monkeypatch.setattr(
        vector_task_module,
        "IndexProcessorFactory",
        lambda *_args, **_kwargs: _IndexProcessorFactoryStub(processor),
    )
    monkeypatch.setattr(vector_task_module, "_repository", lambda: repository)
    monkeypatch.setattr(vector_task_module.session_factory, "create_session", sqlite_session_factory)

    assert _run_rebuild(current_intent, "update")
    assert remote_nodes == {("collection-c", "node-1")}
    with sqlite_session_factory() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        assert dataset.indexing_applied_representation is not None
        applied = DatasetIndexingRepresentation.model_validate_json(dataset.indexing_applied_representation)
        assert applied.vector_route == route_c


def test_new_generation_takes_over_only_documents_with_dataset_rebuild_markers(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_technique = "high_quality"
        old_rebuild = make_document(
            document_id="old-rebuild",
            dataset_id=dataset.id,
            tenant_id=dataset.tenant_id,
        )
        old_rebuild.indexing_status = "indexing"
        old_rebuild.dataset_indexing_request_id = "old-generation"
        ordinary = make_document(
            document_id="ordinary-indexing",
            dataset_id=dataset.id,
            tenant_id=dataset.tenant_id,
        )
        ordinary.indexing_status = "indexing"
        ordinary.dataset_indexing_request_id = None
        disabled = make_document(
            document_id="disabled-rebuild",
            dataset_id=dataset.id,
            tenant_id=dataset.tenant_id,
        )
        disabled.indexing_status = "indexing"
        disabled.dataset_indexing_request_id = "old-generation"
        disabled.enabled = False
        session.add_all(
            [
                dataset,
                old_rebuild,
                ordinary,
                disabled,
                DocumentSegment(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_id=old_rebuild.id,
                    position=1,
                    content="old rebuild content",
                    word_count=3,
                    tokens=3,
                    created_by="account-1",
                    index_node_id="old-node",
                    index_node_hash="old-hash",
                    status=SegmentStatus.COMPLETED,
                ),
                DocumentSegment(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_id=ordinary.id,
                    position=1,
                    content="ordinary content",
                    word_count=2,
                    tokens=2,
                    created_by="account-1",
                    index_node_id="ordinary-node",
                    index_node_hash="ordinary-hash",
                    status=SegmentStatus.COMPLETED,
                ),
            ]
        )

    loaded_document_ids: list[str] = []

    def load(_dataset: Dataset, documents: list[IndexDocument], **_kwargs: object) -> None:
        loaded_document_ids.extend(document.metadata["document_id"] for document in documents)

    processor = _IndexProcessorStub(clean=lambda *_args, **_kwargs: None, load=load)
    monkeypatch.setattr(
        vector_task_module,
        "IndexProcessorFactory",
        lambda *_args, **_kwargs: _IndexProcessorFactoryStub(processor),
    )
    repository, intent = _claim_rebuild(sqlite_session_factory, action="add")
    monkeypatch.setattr(vector_task_module, "_repository", lambda: repository)
    monkeypatch.setattr(vector_task_module.session_factory, "create_session", sqlite_session_factory)

    assert _run_rebuild(intent, "add")

    assert loaded_document_ids == ["old-rebuild"]
    with sqlite_session_factory() as session:
        old_rebuild = session.get_one(DatasetDocument, "old-rebuild")
        ordinary = session.get_one(DatasetDocument, "ordinary-indexing")
        disabled = session.get_one(DatasetDocument, "disabled-rebuild")
        assert old_rebuild.indexing_status == "completed"
        assert old_rebuild.dataset_indexing_request_id is None
        assert ordinary.indexing_status == "indexing"
        assert ordinary.dataset_indexing_request_id is None
        assert disabled.indexing_status == "completed"
        assert disabled.dataset_indexing_request_id is None


def test_ordinary_request_after_batch_marking_invalidates_dataset_rebuild_owner(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_technique = "high_quality"
        document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        document.indexing_status = "completed"
        session.add_all(
            [
                dataset,
                document,
                DocumentSegment(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_id=document.id,
                    position=1,
                    content="content",
                    word_count=1,
                    tokens=1,
                    created_by="account-1",
                    index_node_id="node-1",
                    index_node_hash="hash-1",
                    status=SegmentStatus.COMPLETED,
                ),
            ]
        )

    external_calls: list[str] = []
    processor = _IndexProcessorStub(
        clean=lambda *_args, **_kwargs: external_calls.append("clean"),
        load=lambda *_args, **_kwargs: external_calls.append("load"),
    )
    monkeypatch.setattr(
        vector_task_module,
        "IndexProcessorFactory",
        lambda *_args, **_kwargs: _IndexProcessorFactoryStub(processor),
    )
    repository, intent = _claim_rebuild(sqlite_session_factory, action="add")
    monkeypatch.setattr(vector_task_module, "_repository", lambda: repository)
    task_session = sqlite_session_factory()
    original_commit = task_session.commit
    commit_count = 0

    def commit_then_create_ordinary_request() -> None:
        nonlocal commit_count
        original_commit()
        commit_count += 1
        if commit_count == 1:
            with sqlite_session_factory.begin() as ordinary_session:
                ordinary_session.execute(
                    update(DatasetDocument)
                    .where(DatasetDocument.id == "document-1")
                    .values(indexing_request_id="ordinary-request", dataset_indexing_request_id=None)
                )

    monkeypatch.setattr(task_session, "commit", commit_then_create_ordinary_request)
    monkeypatch.setattr(vector_task_module.session_factory, "create_session", lambda: task_session)

    with pytest.raises(DatasetIndexingWritersActiveError):
        _run_rebuild(intent, "add")

    # The request lands between target-route preparation and the destructive
    # rebuild phase. The second Dataset-X quiescence check must stop before
    # cleaning or loading so the ordinary worker can finish first.
    assert external_calls == []
    with sqlite_session_factory() as session:
        document = session.get_one(DatasetDocument, "document-1")
        assert document.indexing_request_id == "ordinary-request"
        assert document.dataset_indexing_request_id is None


def test_generation_guards_compile_as_locking_current_reads() -> None:
    update_statement = dataset_indexing_generation_lock_statement(
        dataset_id="dataset-1",
        generation_id="generation-1",
    )
    shared_statement = dataset_indexing_generation_lock_statement(
        dataset_id="dataset-1",
        generation_id="generation-1",
        read=True,
    )

    assert "FOR UPDATE" in str(update_statement.compile(dialect=postgresql.dialect()))
    assert "FOR SHARE" in str(shared_statement.compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in str(update_statement.compile(dialect=mysql.dialect()))
    assert "LOCK IN SHARE MODE" in str(shared_statement.compile(dialect=mysql.dialect()))


def test_paused_document_request_does_not_permanently_block_dataset_rebuild(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        document.indexing_request_id = "paused-request"
        document.indexing_claim_id = "expired-claim"
        document.indexing_claim_expires_at = datetime(2000, 1, 1)
        document.indexing_dataset_generation_id = "old-generation"
        document.is_paused = True
        session.add_all([dataset, document])
    _repository, intent = _claim_rebuild(sqlite_session_factory, action="add")

    with sqlite_session_factory.begin() as session:
        locked = lock_dataset_indexing_execution(
            session,
            tenant_id=intent.tenant_id,
            dataset_id=intent.dataset_id,
            generation_id=intent.generation_id,
            intent_id=intent.id,
            claim_id=intent.claim_id,
            require_quiescent=True,
        )
        assert locked is not None
        persisted = session.get_one(DatasetDocument, "document-1")
        assert persisted.indexing_request_id == "paused-request"
        assert persisted.indexing_status == IndexingStatus.WAITING
        assert persisted.indexing_claim_id is None
        assert persisted.indexing_claim_expires_at is None
        assert persisted.indexing_dataset_generation_id is None
        assert persisted.processing_started_at is not None
        assert persisted.parsing_completed_at is None
        assert persisted.cleaning_completed_at is None
        assert persisted.splitting_completed_at is None
        assert persisted.completed_at is None
        assert persisted.stopped_at is None
        assert persisted.error is None


def test_rebuild_quiescence_hands_stale_document_owner_back_to_its_request(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id)
        document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        session.add_all([dataset, document])
        session.flush()
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, created=(document.id,)),
        )

    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (old_claim,) = documents.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )
    with sqlite_session_factory.begin() as session:
        persisted = session.get_one(DatasetDocument, old_claim.ref.document_id)
        persisted.indexing_status = IndexingStatus.SPLITTING
        persisted.parsing_completed_at = datetime(2026, 1, 1, 0, 1)
        persisted.splitting_completed_at = datetime(2026, 1, 1, 0, 2)

    _repository, rebuild = _claim_rebuild(sqlite_session_factory, action="add")
    session = sqlite_session_factory()
    try:
        with pytest.raises(DatasetIndexingWritersActiveError):
            lock_dataset_indexing_execution(
                session,
                tenant_id=rebuild.tenant_id,
                dataset_id=rebuild.dataset_id,
                generation_id=rebuild.generation_id,
                intent_id=rebuild.id,
                claim_id=rebuild.claim_id,
                require_quiescent=True,
            )
        # Production commits this restart handoff before releasing the rebuild
        # claim, even though the active durable request keeps the barrier closed.
        session.commit()
    finally:
        session.close()

    with sqlite_session_factory() as read_session:
        released = read_session.get_one(DatasetDocument, old_claim.ref.document_id)
        assert released.indexing_status == IndexingStatus.WAITING
        assert released.indexing_request_id == request_id
        assert released.indexing_claim_id is None
        assert released.indexing_claim_expires_at is None
        assert released.indexing_dataset_generation_id is None
        assert released.processing_started_at == datetime(2026, 1, 1)
        assert released.parsing_completed_at is None
        assert released.splitting_completed_at is None

    (successor,) = documents.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 2),
        admission_error=lambda _count: None,
    )
    assert successor.recovery_required
    assert successor.dataset_generation_id == rebuild.generation_id


def test_current_unexpired_document_owner_blocks_dataset_rebuild(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        session.add_all([dataset, document])
    _repository, intent = _claim_rebuild(sqlite_session_factory, action="add")
    with sqlite_session_factory.begin() as session:
        document = session.get_one(DatasetDocument, "document-1")
        document.indexing_claim_id = "active-claim"
        document.indexing_claim_expires_at = datetime(2100, 1, 1)
        document.indexing_dataset_generation_id = intent.generation_id

    with sqlite_session_factory.begin() as session:
        with pytest.raises(DatasetIndexingWritersActiveError):
            lock_dataset_indexing_execution(
                session,
                tenant_id=intent.tenant_id,
                dataset_id=intent.dataset_id,
                generation_id=intent.generation_id,
                intent_id=intent.id,
                claim_id=intent.claim_id,
                require_quiescent=True,
            )


def test_high_quality_to_economy_materializes_keyword_target_before_success(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route("qdrant", "old-collection")
    source = DatasetIndexingRepresentation(
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
    target = source.model_copy(
        update={
            "indexing_technique": "economy",
            "embedding_model": None,
            "embedding_model_provider": None,
            "vector_route": None,
            "cleanup_routes": (),
        }
    )
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_technique = "economy"
        document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        document.indexing_status = "completed"
        session.add_all(
            [
                dataset,
                document,
                DocumentSegment(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_id=document.id,
                    position=1,
                    content="keyword content",
                    word_count=2,
                    tokens=2,
                    created_by="account-1",
                    index_node_id="node-1",
                    index_node_hash="hash-1",
                    status=SegmentStatus.COMPLETED,
                ),
            ]
        )
        (intent_id,) = create_dataset_indexing_intents(
            session,
            DatasetIndexingJob(DatasetRef("tenant-1", "dataset-1"), source, target),
        )
    repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sqlite_session_factory)
    intent = repository.claim(intent_id)
    assert intent is not None
    monkeypatch.setattr(vector_task_module, "_repository", lambda: repository)
    monkeypatch.setattr(vector_task_module.session_factory, "create_session", sqlite_session_factory)
    calls: list[str] = []
    processor = _IndexProcessorStub(
        clean=lambda *_args, **_kwargs: calls.append("clean-vector"),
        load=lambda *_args, **_kwargs: calls.append("load-vector"),
    )
    monkeypatch.setattr(
        vector_task_module,
        "IndexProcessorFactory",
        lambda *_args, **_kwargs: _IndexProcessorFactoryStub(processor),
    )

    class PhysicalCleanupVector:
        @staticmethod
        def delete() -> None:
            calls.append("clean-vector")

    monkeypatch.setattr(vector_task_module, "vector_for_route", lambda _route: PhysicalCleanupVector())

    class KeywordIndex:
        def __init__(self, _dataset: Dataset) -> None:
            pass

        def delete(self, *, session: Session) -> None:
            del session
            calls.append("delete-keywords")

        def add_texts(self, documents: list[IndexDocument], session: Session) -> None:
            del session
            assert [document.page_content for document in documents] == ["keyword content"]
            calls.append("load-keywords")

    monkeypatch.setattr(vector_task_module, "Jieba", KeywordIndex)

    assert _run_rebuild(intent, "remove")
    assert calls == ["clean-vector", "load-keywords"]


def test_summary_task_releases_outer_guard_before_adapter_writer_starts(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_request_id = "generation-1"
        dataset.indexing_technique = "high_quality"
        document = make_document(
            document_id="document-1",
            dataset_id=dataset.id,
            tenant_id=dataset.tenant_id,
            doc_form="text_model",
        )
        document.indexing_status = "completed"
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=1,
            content="content",
            word_count=1,
            tokens=1,
            created_by="account-1",
            index_node_id="node-1",
            index_node_hash="hash-1",
            status=SegmentStatus.COMPLETED,
        )
        session.add_all([dataset, document, segment])
        session.flush()
        session.add(
            DocumentSegmentSummary(
                dataset_id=dataset.id,
                document_id=document.id,
                chunk_id=segment.id,
                summary_content="summary",
                status=SummaryStatus.COMPLETED,
            )
        )

    monkeypatch.setattr(summary_task_module.session_factory, "create_session", sqlite_session_factory)
    original_guard = summary_task_module._lock_current_dataset_generation
    active_guards = 0

    _repository, intent = _claim_rebuild(sqlite_session_factory, action="update")

    @contextmanager
    def observed_guard(
        dataset_id: str,
        request_id: str | None,
        *,
        indexing_intent_id: str | None = None,
        indexing_claim_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Generator[Dataset | None]:
        nonlocal active_guards
        with original_guard(
            dataset_id,
            request_id,
            indexing_intent_id=indexing_intent_id,
            indexing_claim_id=indexing_claim_id,
            tenant_id=tenant_id,
        ) as current_dataset:
            active_guards += 1
            try:
                yield current_dataset
            finally:
                active_guards -= 1

    vectorized: list[str] = []

    def vectorize(_summary, _segment, dataset: Dataset, **_kwargs: object) -> bool:
        assert active_guards == 0
        assert dataset.indexing_request_id == intent.generation_id
        vectorized.append(dataset.id)
        return True

    monkeypatch.setattr(summary_task_module, "_lock_current_dataset_generation", observed_guard)
    monkeypatch.setattr(summary_task_module.SummaryIndexAdapter, "vectorize_summary", vectorize)

    summary_task_module.regenerate_summary_index_task.run(
        "dataset-1",
        regenerate_reason="embedding_model_changed",
        regenerate_vectors_only=True,
        indexing_request_id=intent.generation_id,
        indexing_intent_id=intent.id,
        indexing_claim_id=intent.claim_id,
        tenant_id=intent.tenant_id,
    )

    assert vectorized == ["dataset-1"]
    assert active_guards == 0


@pytest.mark.parametrize("invalidated_owner", ["document", "segment"])
def test_summary_convergence_revalidates_current_document_and_segment_eligibility(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    invalidated_owner: str,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_request_id = "generation-1"
        dataset.indexing_technique = "high_quality"
        dataset.summary_index_setting = {"enable": True}
        document = make_document(
            document_id="document-1",
            dataset_id=dataset.id,
            tenant_id=dataset.tenant_id,
            doc_form="text_model",
        )
        document.indexing_status = "completed"
        document.enabled = True
        document.archived = False
        document.need_summary = True
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=1,
            content="content",
            word_count=1,
            tokens=1,
            created_by="account-1",
            index_node_id="node-1",
            index_node_hash="hash-1",
            status=SegmentStatus.COMPLETED,
            enabled=True,
        )
        segment.id = "segment-1"
        summary = DocumentSegmentSummary(
            dataset_id=dataset.id,
            document_id=document.id,
            chunk_id=segment.id,
            summary_content="retained summary",
            status=SummaryStatus.COMPLETED,
            enabled=True,
        )
        summary.id = "summary-1"
        session.add_all([dataset, document, segment, summary])

    monkeypatch.setattr(summary_task_module.session_factory, "create_session", sqlite_session_factory)
    original_guard = summary_task_module._lock_current_dataset_generation
    _repository, intent = _claim_rebuild(sqlite_session_factory, action="update")
    guard_calls = 0

    @contextmanager
    def invalidate_before_current_read(
        dataset_id: str,
        request_id: str | None,
        *,
        indexing_intent_id: str | None = None,
        indexing_claim_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Generator[Dataset | None]:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            with sqlite_session_factory.begin() as concurrent:
                if invalidated_owner == "document":
                    concurrent.get_one(DatasetDocument, "document-1").archived = True
                else:
                    concurrent.get_one(DocumentSegment, "segment-1").enabled = False
        with original_guard(
            dataset_id,
            request_id,
            indexing_intent_id=indexing_intent_id,
            indexing_claim_id=indexing_claim_id,
            tenant_id=tenant_id,
        ) as current_dataset:
            yield current_dataset

    monkeypatch.setattr(summary_task_module, "_lock_current_dataset_generation", invalidate_before_current_read)
    monkeypatch.setattr(
        summary_task_module.SummaryIndexAdapter,
        "vectorize_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ineligible vector write")),
    )
    monkeypatch.setattr(
        summary_task_module.SummaryIndexAdapter,
        "generate_and_vectorize_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ineligible LLM write")),
    )

    assert summary_task_module.regenerate_summary_index_task.run(
        "dataset-1",
        regenerate_reason="embedding_model_changed",
        regenerate_vectors_only=True,
        indexing_request_id=intent.generation_id,
        indexing_intent_id=intent.id,
        indexing_claim_id=intent.claim_id,
        tenant_id=intent.tenant_id,
    )

    with sqlite_session_factory() as session:
        summary = session.get_one(DocumentSegmentSummary, "summary-1")
        assert (summary.version, summary.summary_content, summary.status) == (
            1,
            "retained summary",
            SummaryStatus.COMPLETED,
        )


def test_summary_convergence_generates_from_current_locked_segment_content(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.indexing_request_id = "generation-1"
        dataset.indexing_technique = "high_quality"
        dataset.summary_index_setting = {"enable": True}
        document = make_document(
            document_id="document-1",
            dataset_id=dataset.id,
            tenant_id=dataset.tenant_id,
            doc_form="text_model",
        )
        document.indexing_status = "completed"
        document.enabled = True
        document.archived = False
        document.need_summary = True
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=1,
            content="stale snapshot content",
            word_count=3,
            tokens=3,
            created_by="account-1",
            index_node_id="node-1",
            index_node_hash="hash-1",
            status=SegmentStatus.COMPLETED,
            enabled=True,
        )
        segment.id = "segment-1"
        session.add_all([dataset, document, segment])

    monkeypatch.setattr(summary_task_module.session_factory, "create_session", sqlite_session_factory)
    original_guard = summary_task_module._lock_current_dataset_generation
    _repository, intent = _claim_rebuild(sqlite_session_factory, action="update")
    guard_calls = 0

    @contextmanager
    def edit_before_current_read(
        dataset_id: str,
        request_id: str | None,
        *,
        indexing_intent_id: str | None = None,
        indexing_claim_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Generator[Dataset | None]:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            with sqlite_session_factory.begin() as concurrent:
                current_segment = concurrent.get_one(DocumentSegment, "segment-1")
                current_segment.content = "current locked content"
                current_segment.word_count = 3
                current_segment.tokens = 3
        with original_guard(
            dataset_id,
            request_id,
            indexing_intent_id=indexing_intent_id,
            indexing_claim_id=indexing_claim_id,
            tenant_id=tenant_id,
        ) as current_dataset:
            yield current_dataset

    generated_from: list[str] = []

    def generate(current_segment: DocumentSegment, *_args: object, **_kwargs: object) -> DocumentSegmentSummary:
        generated_from.append(current_segment.content)
        return DocumentSegmentSummary(
            dataset_id=current_segment.dataset_id,
            document_id=current_segment.document_id,
            chunk_id=current_segment.id,
            summary_content="generated summary",
        )

    monkeypatch.setattr(summary_task_module, "_lock_current_dataset_generation", edit_before_current_read)
    monkeypatch.setattr(summary_task_module.SummaryIndexAdapter, "generate_and_vectorize_summary", generate)
    monkeypatch.setattr(
        summary_task_module.SummaryIndexAdapter,
        "vectorize_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected retained summary")),
    )

    assert summary_task_module.regenerate_summary_index_task.run(
        "dataset-1",
        regenerate_reason="embedding_model_changed",
        regenerate_vectors_only=True,
        indexing_request_id=intent.generation_id,
        indexing_intent_id=intent.id,
        indexing_claim_id=intent.claim_id,
        tenant_id=intent.tenant_id,
    )
    assert generated_from == ["current locked content"]

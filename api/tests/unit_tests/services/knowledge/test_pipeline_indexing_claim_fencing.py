from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.index_processor import index_processor as index_processor_module
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from core.rag.index_processor.index_processor import IndexProcessor
from core.rag.index_processor.processor.paragraph_index_processor import ParagraphIndexProcessor
from models import Dataset, Tenant
from models.dataset import Document
from models.enums import IndexingStatus
from models.model import DatasetIndexingOutbox, DocumentIndexingOutbox
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from repositories.knowledge.pipeline_indexing_repository import (
    build_pipeline_indexing_guard_from_ids,
    build_pipeline_indexing_route_recorder_from_ids,
    protect_pipeline_indexing_claim,
)
from services.knowledge.entities.datasets import DatasetIndexingGenerationLostError, DatasetIndexingRepresentation
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError, DocumentIsDeletedPausedError
from tests.unit_tests.model_factories import make_dataset, make_document, make_tenant


def _persist_claimed_document(
    session: Session,
    *,
    indexing_request_id: str = "request-new",
    claim_id: str = "claim-new",
) -> tuple[Dataset, Document]:
    tenant: Tenant = make_tenant(tenant_id="tenant-1")
    dataset: Dataset = make_dataset(dataset_id="dataset-1", tenant_id=tenant.id)
    document: Document = make_document(
        document_id="document-1",
        dataset_id=dataset.id,
        tenant_id=tenant.id,
    )
    document.indexing_status = IndexingStatus.PARSING
    document.indexing_request_id = indexing_request_id
    document.indexing_claim_id = claim_id
    document.indexing_claim_expires_at = datetime(2099, 1, 1)
    session.add_all([tenant, dataset, document])
    session.commit()
    return dataset, document


def test_stale_rag_worker_cannot_clean_index_fail_or_complete_successor(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document = _persist_claimed_document(
        sqlite_session,
        indexing_request_id="request-old",
        claim_id="claim-old",
    )
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(dataset.tenant_id, dataset.id).document(document.id)

    def takeover_after_stale_worker_releases_snapshot_lock(_session: Session) -> None:
        with sqlite_session_factory.begin() as takeover_session:
            successor = takeover_session.get_one(Document, document.id)
            successor.indexing_request_id = "request-new"
            successor.indexing_claim_id = "claim-new"
            successor.indexing_claim_expires_at = datetime(2099, 1, 1)
            successor.indexing_status = IndexingStatus.PARSING
        repository.complete_indexing(ref, claim_id="claim-new", tokens=7, latency=2.0)

    # The initial guard is valid. As soon as its metadata snapshot transaction
    # commits, a successor takes over and completes before the stale worker
    # reaches the actual vector/keyword cleanup boundary.
    event.listen(
        sqlite_session,
        "after_commit",
        takeover_after_stale_worker_releases_snapshot_lock,
        once=True,
    )

    class ProcessorCapture:
        clean_calls = 0
        index_calls = 0

        def clean(self, *_args: object, **_kwargs: object) -> None:
            type(self).clean_calls += 1

        def index(self, *_args: object, **_kwargs: object) -> None:
            type(self).index_calls += 1

    processor = ProcessorCapture()

    class ProcessorFactoryStub:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def init_index_processor() -> ProcessorCapture:
            return processor

    class VectorMustNotRun:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("stale pipeline owner constructed a vector writer")

        @staticmethod
        def resolve_vector_type(*_args: object, **_kwargs: object) -> str:
            raise AssertionError("stale pipeline owner resolved a vector writer")

    monkeypatch.setattr(index_processor_module, "IndexProcessorFactory", ProcessorFactoryStub)
    monkeypatch.setattr(index_processor_module, "Vector", VectorMustNotRun)

    with pytest.raises(DocumentIndexingClaimLostError):
        IndexProcessor(indexing_guard_factory=build_pipeline_indexing_guard_from_ids).index_and_clean(
            dataset_id=dataset.id,
            document_id=document.id,
            original_document_id=document.id,
            chunks={},
            batch="retry",
            session=sqlite_session,
            indexing_request_id="request-old",
            claim_id="claim-old",
        )

    assert ProcessorCapture.clean_calls == 0
    assert ProcessorCapture.index_calls == 0

    # Even graph-failure and success callbacks arriving after the successor
    # completed must not change the successor's published result.
    with pytest.raises(DocumentIndexingClaimLostError):
        repository.mark_failed(
            workspace_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            error="stale failure",
            indexing_request_id="request-old",
            claim_id="claim-old",
        )
    with pytest.raises(DocumentIndexingClaimLostError):
        repository.complete_indexing(
            ref,
            claim_id="claim-old",
            tokens=1,
            latency=1.0,
        )

    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document.id)
    assert persisted.indexing_status == IndexingStatus.COMPLETED
    assert persisted.error is None
    assert persisted.tokens == 7


def test_parent_generation_change_releases_pipeline_claim_without_writing_error(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset, document = _persist_claimed_document(sqlite_session)
    dataset.indexing_request_id = "dataset-generation-2"
    document.indexing_dataset_generation_id = "dataset-generation-1"
    sqlite_session.commit()
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)

    repository.mark_failed(
        workspace_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        error="stale pipeline failure",
        indexing_request_id="request-new",
        claim_id="claim-new",
    )

    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document.id)
    assert persisted.indexing_status == IndexingStatus.WAITING
    assert persisted.error is None
    assert persisted.indexing_request_id == "request-new"
    assert persisted.indexing_claim_id is None
    assert persisted.indexing_claim_expires_at is None
    assert persisted.indexing_dataset_generation_id is None
    assert persisted.processing_started_at is not None


def test_pipeline_graph_failure_during_pause_releases_claim_without_acknowledging_request(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset, document = _persist_claimed_document(sqlite_session)
    document.indexing_status = IndexingStatus.SPLITTING
    document.is_paused = True
    request = DocumentIndexingOutbox(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        action="retry",
        document_ids='["document-1"]',
        actor_id="actor-1",
    )
    request.id = "request-new"
    sqlite_session.add(request)
    sqlite_session.commit()
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)

    repository.mark_failed(
        workspace_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        error="paused graph failure",
        indexing_request_id=request.id,
        claim_id="claim-new",
    )

    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document.id)
        assert persisted.indexing_status == IndexingStatus.SPLITTING
        assert persisted.error is None
        assert persisted.indexing_request_id == request.id
        assert persisted.indexing_claim_id is None
        assert persisted.indexing_claim_expires_at is None
        assert persisted.indexing_dataset_generation_id is None
        assert session.get(DocumentIndexingOutbox, request.id) is not None


def test_pipeline_writer_records_actual_route_for_pending_dataset_generation(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset, document = _persist_claimed_document(sqlite_session)
    generation_id = "dataset-generation"
    route_a = VectorRoute.capture(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="text_model",
    )
    representation = DatasetIndexingRepresentation(
        indexing_technique="high_quality",
        index_struct=route_a.index_struct,
        collection_binding_id=None,
        embedding_model="embedding",
        embedding_model_provider="provider",
        keyword_number=10,
        is_multimodal=False,
        doc_form="text_model",
        summary_index_setting=None,
        vector_route=route_a,
    )
    dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
    dataset.indexing_request_id = generation_id
    dataset.indexing_applied_representation = representation.model_dump_json()
    document.indexing_dataset_generation_id = generation_id
    intent = DatasetIndexingOutbox(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        generation_id=generation_id,
        operation="rebuild",
        action="update",
        source_representation=representation.model_dump_json(),
        target_representation=representation.model_copy(update={"vector_route": None}).model_dump_json(),
    )
    sqlite_session.add(intent)
    sqlite_session.commit()

    recorder = build_pipeline_indexing_route_recorder_from_ids(
        sqlite_session,
        dataset_id=dataset.id,
        document_id=document.id,
        indexing_request_id="request-new",
        claim_id="claim-new",
        new_session=sqlite_session_factory,
    )
    assert recorder is not None
    recorder("qdrant", "collection-b", "text_model")

    with sqlite_session_factory() as observer:
        target = DatasetIndexingRepresentation.model_validate_json(
            observer.get_one(DatasetIndexingOutbox, intent.id).target_representation
        )
        assert target.vector_route is None
        assert [(route.vector_type, route.collection_name) for route in target.cleanup_routes] == [
            ("qdrant", "collection-b")
        ]


def test_pipeline_writer_with_legacy_null_generation_records_no_route_receipt(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset, document = _persist_claimed_document(sqlite_session)
    dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
    assert dataset.indexing_request_id is None
    assert document.indexing_dataset_generation_id is None
    sqlite_session.commit()

    recorder = build_pipeline_indexing_route_recorder_from_ids(
        sqlite_session,
        dataset_id=dataset.id,
        document_id=document.id,
        indexing_request_id="request-new",
        claim_id="claim-new",
        new_session=sqlite_session_factory,
    )
    assert recorder is not None
    recorder("qdrant", "collection-bootstrap", "text_model")

    with sqlite_session_factory() as observer:
        assert observer.scalars(select(DatasetIndexingOutbox)).all() == []
        persisted = observer.get_one(Dataset, dataset.id)
        assert persisted.indexing_request_id is None


def test_pipeline_route_recorder_rejects_dataset_outside_the_claimed_tenant(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset, document = _persist_claimed_document(sqlite_session)
    recorder = build_pipeline_indexing_route_recorder_from_ids(
        sqlite_session,
        dataset_id=dataset.id,
        document_id=document.id,
        indexing_request_id="request-new",
        claim_id="claim-new",
        new_session=sqlite_session_factory,
    )
    assert recorder is not None

    sqlite_session.add(make_tenant(tenant_id="tenant-2"))
    dataset.tenant_id = "tenant-2"
    sqlite_session.commit()

    with pytest.raises(DocumentIsDeletedPausedError):
        recorder("qdrant", "collection-foreign", "text_model")


def test_pipeline_route_recorder_rejects_expired_nonnull_dataset_generation(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset, document = _persist_claimed_document(sqlite_session)
    dataset.indexing_request_id = "dataset-generation-old"
    document.indexing_dataset_generation_id = "dataset-generation-old"
    sqlite_session.commit()
    recorder = build_pipeline_indexing_route_recorder_from_ids(
        sqlite_session,
        dataset_id=dataset.id,
        document_id=document.id,
        indexing_request_id="request-new",
        claim_id="claim-new",
        new_session=sqlite_session_factory,
    )
    assert recorder is not None

    dataset.indexing_request_id = "dataset-generation-current"
    sqlite_session.commit()

    with pytest.raises(DatasetIndexingGenerationLostError):
        recorder("qdrant", "collection-stale", "text_model")


def test_claim_lost_after_segment_commit_blocks_actual_vector_write(
    sqlite_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_calls = 0

    @contextmanager
    def lost_claim() -> Generator[None]:
        raise DocumentIndexingClaimLostError()
        yield

    def mutation_guard(_session: Session) -> AbstractContextManager[None]:
        nonlocal guard_calls
        guard_calls += 1
        return nullcontext() if guard_calls == 1 else lost_claim()

    dataset, document = _persist_claimed_document(sqlite_session)
    dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
    dataset.is_multimodal = False
    sqlite_session.commit()

    class DocumentStoreCapture:
        add_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            pass

        def add_documents(self, **_kwargs: object) -> None:
            type(self).add_calls += 1

    class VectorMustNotRun:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("stale publisher constructed a vector writer")

    monkeypatch.setattr(ParagraphIndexProcessor, "_get_content_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "core.rag.index_processor.processor.paragraph_index_processor.calculate_segment_token_counts",
        lambda **_kwargs: [1],
    )
    monkeypatch.setattr(
        "core.rag.index_processor.processor.paragraph_index_processor.DatasetDocumentStore",
        DocumentStoreCapture,
    )
    monkeypatch.setattr(
        "core.rag.index_processor.processor.paragraph_index_processor.Vector",
        VectorMustNotRun,
    )

    with pytest.raises(DocumentIndexingClaimLostError):
        ParagraphIndexProcessor().index(
            dataset,
            document,
            ["new content"],
            sqlite_session,
            mutation_guard=mutation_guard,
        )

    assert DocumentStoreCapture.add_calls == 1


def test_claim_lost_after_segment_commit_blocks_actual_keyword_write(
    sqlite_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_calls = 0

    @contextmanager
    def lost_claim() -> Generator[None]:
        raise DocumentIndexingClaimLostError()
        yield

    def mutation_guard(_session: Session) -> AbstractContextManager[None]:
        nonlocal guard_calls
        guard_calls += 1
        return nullcontext() if guard_calls == 1 else lost_claim()

    dataset, document = _persist_claimed_document(sqlite_session)
    dataset.indexing_technique = IndexTechniqueType.ECONOMY
    dataset.is_multimodal = False
    sqlite_session.commit()

    class DocumentStoreCapture:
        add_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            pass

        def add_documents(self, **_kwargs: object) -> None:
            type(self).add_calls += 1

    class KeywordMustNotRun:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("stale publisher constructed a keyword writer")

    monkeypatch.setattr(ParagraphIndexProcessor, "_get_content_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "core.rag.index_processor.processor.paragraph_index_processor.calculate_segment_token_counts",
        lambda **_kwargs: [1],
    )
    monkeypatch.setattr(
        "core.rag.index_processor.processor.paragraph_index_processor.DatasetDocumentStore",
        DocumentStoreCapture,
    )
    monkeypatch.setattr(
        "core.rag.index_processor.processor.paragraph_index_processor.Keyword",
        KeywordMustNotRun,
    )

    with pytest.raises(DocumentIndexingClaimLostError):
        ParagraphIndexProcessor().index(
            dataset,
            document,
            ["new content"],
            sqlite_session,
            mutation_guard=mutation_guard,
        )

    assert DocumentStoreCapture.add_calls == 1


def test_guard_refreshes_same_claim_after_lease_expires_while_row_is_locked(sqlite_session: Session) -> None:
    dataset, document = _persist_claimed_document(sqlite_session)
    ref = DatasetRef(dataset.tenant_id, dataset.id).document(document.id)

    with protect_pipeline_indexing_claim(
        sqlite_session,
        tenant_id=ref.dataset.tenant_id,
        dataset_id=ref.dataset.dataset_id,
        document_id=ref.document_id,
        indexing_request_id="request-new",
        claim_id="claim-new",
    ):
        # Simulate an external vector operation outliving the nominal lease.
        # The continuously held row lock still makes this worker the owner.
        document.indexing_claim_expires_at = datetime(2000, 1, 1)
    sqlite_session.commit()

    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document.id)
    assert persisted.indexing_claim_id == "claim-new"
    assert persisted.indexing_claim_expires_at is not None
    assert persisted.indexing_claim_expires_at > datetime.now(UTC).replace(tzinfo=None)

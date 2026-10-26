"""SQLite-backed tests for :mod:`services.knowledge.summaries.adapters`."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from threading import Event
from time import monotonic
from types import SimpleNamespace
from typing import override
from unittest.mock import MagicMock

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

import services.knowledge.summaries.adapters as summary_module
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.embedding.embedding_base import Embeddings
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.index_processor.index_processor_base import SummaryIndexSettingDict
from core.rag.models.document import Document as IndexDocument
from graphon.model_runtime.entities.llm_entities import LLMUsage
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import DataSourceType, DocumentCreatedFrom, SegmentStatus, SummaryStatus
from models.model import DatasetIndexingOutbox
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository
from services.knowledge.entities.datasets import DatasetIndexingRepresentation
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.summaries import (
    SummaryVectorCleanupClaim,
    SummaryVectorInput,
    SummaryVectorResult,
    SummaryVectorWriteIntent,
)
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.summaries.adapters import SummaryIndexAdapter

TENANT_ID = "tenant-1"
OTHER_TENANT_ID = "tenant-2"


class RenewalBlocker(SQLAlchemySummaryVectorCleanupRepository):
    def __init__(self, *, started: Event, unblock: Event) -> None:
        self.started = started
        self.unblock = unblock

    @override
    def renew_write(self, intent: SummaryVectorWriteIntent) -> None:
        del intent
        self.started.set()
        self.unblock.wait(timeout=5)


class EmbeddingsProbe(Embeddings):
    def __init__(
        self,
        *,
        response: list[list[float]] | None = None,
        embed: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.response = response or [[0.1]]
        self.embed = embed
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self.embed is not None:
            return self.embed(texts)
        return self.response

    def embed_multimodal_documents(self, multimodel_documents: list[dict[str, object]]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_multimodal_query(self, multimodel_document: dict[str, object]) -> list[float]:
        raise NotImplementedError


class VectorProcessorProbe(BaseVector):
    def __init__(self, *, create_error: Exception | None = None) -> None:
        super().__init__("summary-collection")
        self.create_error = create_error
        self.created: list[tuple[list[IndexDocument], list[list[float]]]] = []
        self.deleted: list[list[str]] = []

    def get_type(self) -> str:
        return "qdrant"

    def create(self, texts: list[IndexDocument], embeddings: list[list[float]], **_kwargs: object) -> list[str] | None:
        self.created.append((texts, embeddings))
        if self.create_error is not None:
            raise self.create_error
        return None

    def add_texts(
        self, documents: list[IndexDocument], embeddings: list[list[float]], **kwargs: object
    ) -> list[str] | None:
        return self.create(documents, embeddings, **kwargs)

    def text_exists(self, id: str) -> bool:
        del id
        return False

    def delete_by_ids(self, ids: list[str]) -> None:
        self.deleted.append(ids)

    def delete_by_metadata_field(self, key: str, value: str) -> None:
        raise NotImplementedError

    def search_by_vector(self, query_vector: list[float], **kwargs: object) -> list[IndexDocument]:
        raise NotImplementedError

    def search_by_full_text(self, query: str, **kwargs: object) -> list[IndexDocument]:
        raise NotImplementedError

    def delete(self) -> None:
        raise NotImplementedError


class SleepProbe:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _route(collection_name: str = "summary-collection") -> VectorRoute:
    return VectorRoute.capture(
        tenant_id=TENANT_ID,
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name=collection_name,
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )


def test_writer_heartbeat_shutdown_does_not_wait_for_blocked_renewal() -> None:
    from services.knowledge.summaries.application import _maintain_write_claim

    started = Event()
    unblock = Event()
    store = RenewalBlocker(started=started, unblock=unblock)
    intent = SummaryVectorWriteIntent("cleanup-1", TENANT_ID, "dataset-1", "node-1", "writer-1", "claim-1", _route())
    before = monotonic()
    with _maintain_write_claim(store, intent, interval_seconds=0.001):
        assert started.wait(timeout=1)
    elapsed = monotonic() - before
    unblock.set()

    assert elapsed < 1


@pytest.fixture(autouse=True)
def _install_sqlite_factory(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    """Route every service-owned session through the per-test SQLite database."""

    monkeypatch.setattr(summary_module.session_factory, "create_session", sqlite_session_factory)


def _persist_dataset(
    session: Session,
    *,
    dataset_id: str = "dataset-1",
    tenant_id: str = TENANT_ID,
    indexing_technique: IndexTechniqueType = IndexTechniqueType.HIGH_QUALITY,
) -> Dataset:
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name=f"Dataset {dataset_id}",
        description="",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=indexing_technique,
        created_by="account-1",
        embedding_model_provider="openai",
        embedding_model="text-embedding",
        chunk_structure=IndexStructureType.PARAGRAPH_INDEX,
    )
    session.add(dataset)
    session.commit()
    return dataset


def _persist_document(
    session: Session,
    dataset: Dataset,
    *,
    document_id: str = "doc-1",
    doc_form: IndexStructureType = IndexStructureType.PARAGRAPH_INDEX,
    doc_language: str | None = "en",
) -> Document:
    document = Document(
        id=document_id,
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="batch-1",
        name=f"Document {document_id}",
        created_from=DocumentCreatedFrom.WEB,
        created_by="account-1",
        indexing_status="completed",
        enabled=True,
        archived=False,
        doc_form=doc_form,
        doc_language=doc_language,
    )
    session.add(document)
    session.commit()
    return document


def _persist_segment(
    session: Session,
    dataset: Dataset,
    document: Document,
    *,
    segment_id: str = "seg-1",
    position: int = 1,
    content: str = "hello world",
    enabled: bool = True,
    status: SegmentStatus = SegmentStatus.COMPLETED,
) -> DocumentSegment:
    segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=position,
        content=content,
        word_count=len(content.split()),
        tokens=2,
        created_by="account-1",
        enabled=enabled,
        status=status,
    )
    segment.id = segment_id
    session.add(segment)
    session.commit()
    return segment


def _persist_summary(
    session: Session,
    dataset: Dataset,
    document: Document,
    segment: DocumentSegment,
    *,
    summary_id: str = "sum-1",
    content: str | None = "summary",
    node_id: str | None = None,
    status: SummaryStatus = SummaryStatus.GENERATING,
    enabled: bool = True,
    created_at: datetime | None = None,
) -> DocumentSegmentSummary:
    route = _route() if node_id is not None else None
    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=segment.id,
        summary_content=content,
        summary_index_node_id=node_id,
        vector_route=route.serialized if route is not None else None,
        vector_route_hash=route.key if route is not None else None,
        status=status,
        enabled=enabled,
    )
    summary.id = summary_id
    if created_at is not None:
        summary.created_at = created_at
        summary.updated_at = created_at
    session.add(summary)
    session.commit()
    return summary


def _graph(session: Session) -> tuple[Dataset, Document, DocumentSegment]:
    dataset = _persist_dataset(session)
    document = _persist_document(session, dataset)
    segment = _persist_segment(session, dataset, document)
    return dataset, document, segment


def _usage(*, total: int = 10) -> LLMUsage:
    return LLMUsage(
        prompt_tokens=3,
        prompt_unit_price=Decimal(0),
        prompt_price_unit=Decimal(0),
        prompt_price=Decimal(0),
        completion_tokens=total - 3,
        completion_unit_price=Decimal(0),
        completion_price_unit=Decimal(0),
        completion_price=Decimal(0),
        total_tokens=total,
        total_price=Decimal(0),
        currency="USD",
        latency=0.1,
    )


def _install_summary_generator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    content: str = "generated summary",
    usage: LLMUsage | None = None,
) -> MagicMock:
    generate = MagicMock(return_value=(content, usage or _usage()))
    paragraph_module = SimpleNamespace(ParagraphIndexProcessor=SimpleNamespace(generate_summary_from_inputs=generate))
    monkeypatch.setitem(
        sys.modules,
        "core.rag.index_processor.processor.paragraph_index_processor",
        paragraph_module,
    )
    return generate


def _install_vector_dependencies(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setattr(summary_module.uuid, "uuid4", MagicMock(return_value="node-1"))
    monkeypatch.setattr(summary_module.helper, "generate_text_hash", MagicMock(return_value="hash-1"))
    embedding_model = MagicMock(name="embedding_model")
    embedding_model.get_text_embedding_num_tokens.return_value = [5]
    manager = MagicMock(name="model_manager")
    manager.get_model_instance.return_value = embedding_model
    monkeypatch.setattr(summary_module.ModelManager, "for_tenant", MagicMock(return_value=manager))
    vector = MagicMock(name="vector")
    vector.get_routing_target.return_value = ("qdrant", "summary-collection")
    monkeypatch.setattr(summary_module, "Vector", MagicMock(return_value=vector))
    monkeypatch.setattr(summary_module, "vector_for_route", lambda _route: vector)
    return vector


class TestGenerateAndCreate:
    def test_generate_uses_document_language(self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset, _, segment = _graph(sqlite_session)
        usage = _usage()
        generate = _install_summary_generator(monkeypatch, usage=usage)

        def call_model(**_kwargs: object) -> tuple[str, LLMUsage]:
            assert not sqlite_session.in_transaction()
            return "generated summary", usage

        generate.side_effect = call_model

        content, result_usage = SummaryIndexAdapter.generate_summary_for_segment(
            segment, dataset, {"enable": True}, session=sqlite_session
        )

        assert content == "generated summary"
        assert result_usage is usage
        assert generate.call_args.kwargs["document_language"] == "en"
        assert callable(generate.call_args.kwargs["image_loader"])
        assert not sqlite_session.in_transaction()

    def test_generate_allows_missing_document_but_rejects_empty_summary(
        self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = _persist_dataset(sqlite_session)
        orphan_document = _persist_document(sqlite_session, dataset)
        segment = _persist_segment(sqlite_session, dataset, orphan_document)
        sqlite_session.delete(orphan_document)
        sqlite_session.commit()
        _install_summary_generator(monkeypatch, content="")

        with pytest.raises(ValueError, match="Generated summary is empty"):
            SummaryIndexAdapter.generate_summary_for_segment(segment, dataset, {"enable": True}, session=sqlite_session)

    def test_create_updates_only_matching_dataset_and_reenables(self, sqlite_session: Session) -> None:
        dataset, document, segment = _graph(sqlite_session)
        existing = _persist_summary(
            sqlite_session, dataset, document, segment, content="old", node_id="old-node", enabled=False
        )
        other_dataset = _persist_dataset(sqlite_session, dataset_id="dataset-2")
        other_document = _persist_document(sqlite_session, other_dataset, document_id="doc-2")
        decoy = _persist_summary(
            sqlite_session,
            other_dataset,
            other_document,
            segment,
            summary_id="decoy",
            content="decoy",
        )
        existing.disabled_at = datetime(2026, 1, 1)
        existing.disabled_by = "account-2"
        sqlite_session.commit()

        result = SummaryIndexAdapter.create_summary_record(
            segment, dataset, "new", status=SummaryStatus.GENERATING, session=sqlite_session
        )

        assert result.id == existing.id
        assert (result.summary_content, result.status, result.enabled) == (
            "new",
            SummaryStatus.GENERATING,
            True,
        )
        assert (result.disabled_at, result.disabled_by, result.error) == (None, None, None)
        assert sqlite_session.scalar(select(func.count()).select_from(DocumentSegmentSummary)) == 2
        sqlite_session.refresh(decoy)
        assert decoy.summary_content == "decoy"

    def test_create_persists_new_record_in_current_transaction(self, sqlite_session: Session) -> None:
        dataset, document, segment = _graph(sqlite_session)

        result = SummaryIndexAdapter.create_summary_record(
            segment, dataset, "new", status=SummaryStatus.NOT_STARTED, session=sqlite_session
        )

        assert result.id is not None
        stored = sqlite_session.get(DocumentSegmentSummary, result.id)
        assert stored is result
        assert stored is not None
        assert (stored.dataset_id, stored.document_id, stored.chunk_id, stored.summary_content) == (
            dataset.id,
            document.id,
            segment.id,
            "new",
        )


class TestVectorizeSummary:
    def test_skips_economy_dataset(self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset = _persist_dataset(sqlite_session, indexing_technique=IndexTechniqueType.ECONOMY)
        document = _persist_document(sqlite_session, dataset)
        segment = _persist_segment(sqlite_session, dataset, document)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        vector_class = MagicMock()
        monkeypatch.setattr(summary_module, "Vector", vector_class)

        SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        vector_class.assert_not_called()

    def test_rejects_blank_content(self, sqlite_session: Session) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, content=" ")

        with pytest.raises(ValueError, match="Summary content is empty"):
            SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

    def test_segment_disabled_before_vector_publication_discards_unsubmitted_writer(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, content="current summary")
        vector = _install_vector_dependencies(monkeypatch)
        original_lock = summary_module._lock_summary_dataset_owner
        owner_locks = 0

        def disable_before_publication(
            session: Session,
            *,
            ref: DatasetRef,
            indexing_request_id: str | None,
            indexing_intent_id: str | None,
            indexing_claim_id: str | None,
        ) -> Dataset | None:
            nonlocal owner_locks
            owner_locks += 1
            if owner_locks == 2:
                with sqlite_session_factory.begin() as concurrent:
                    concurrent.get_one(DocumentSegment, segment.id).enabled = False
            return original_lock(
                session,
                ref=ref,
                indexing_request_id=indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
            )

        monkeypatch.setattr(summary_module, "_lock_summary_dataset_owner", disable_before_publication)

        assert not SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)
        vector.add_texts.assert_not_called()
        with sqlite_session_factory() as observer:
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            assert (stored.status, stored.error, stored.summary_index_node_id) == (
                SummaryStatus.GENERATING,
                None,
                None,
            )
            assert observer.scalar(select(func.count()).select_from(SummaryVectorCleanupOutbox)) == 0

    def test_provided_session_releases_transaction_before_every_vector_attempt(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        vector = _install_vector_dependencies(monkeypatch)
        node_ids = iter(["node-1", "node-2"])

        def next_node_id() -> str:
            return next(node_ids)

        monkeypatch.setattr(summary_module.uuid, "uuid4", next_node_id)

        def write_vector(*_args: object, **kwargs: object) -> None:
            assert not sqlite_session.in_transaction()
            before_write = kwargs["before_write"]
            assert callable(before_write)
            before_write()
            if vector.add_texts.call_count == 1:
                raise ConnectionError("connection timeout")

        vector.add_texts.side_effect = write_vector
        sleep = MagicMock()
        monkeypatch.setattr(summary_module.time, "sleep", sleep)

        SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        assert vector.add_texts.call_count == 2
        sleep.assert_called_once_with(2.0)
        assert (summary.status, summary.summary_index_node_id, summary.summary_index_node_hash, summary.tokens) == (
            SummaryStatus.COMPLETED,
            "node-2",
            "hash-1",
            5,
        )
        with sqlite_session_factory() as observer:
            uncertain = observer.scalar(
                select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "node-1")
            )
            assert uncertain is not None
            assert uncertain.writer_finished_at is None

    def test_embedding_failure_discards_writer_after_cleanup_takeover_and_unblocks_dataset_cleanup(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from core.rag.datasource.vdb.vector_factory import Vector
        from models.model import SummaryVectorCleanupOutbox
        from repositories.knowledge.resource_cleanup_repository import (
            ResourceCleanupIntent,
            ResourceCleanupType,
            SQLAlchemyResourceCleanupRepository,
        )
        from repositories.knowledge.summary_vector_cleanup_repository import (
            SQLAlchemySummaryVectorCleanupRepository,
        )
        from services.knowledge.summaries.application import SummaryVectorWriteNotSubmittedError

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        _install_vector_dependencies(monkeypatch)
        vector = Vector.__new__(Vector)
        takeover_repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
        takeover_claims = []

        def fail_after_cleanup_takeover(_texts: list[str]) -> list[list[float]]:
            with sqlite_session_factory.begin() as takeover_session:
                row = takeover_session.scalar(select(SummaryVectorCleanupOutbox))
                assert row is not None
                row.claim_expires_at = datetime(2000, 1, 1)
                cleanup_id = row.id
            cleanup_claim = takeover_repository.claim_pending_cleanup(cleanup_id)
            assert cleanup_claim is not None
            assert cleanup_claim.writer_finished is False
            with sqlite_session_factory() as observer:
                claimed_row = observer.get_one(SummaryVectorCleanupOutbox, cleanup_id)
                assert claimed_row.cleanup_started_at is not None
            takeover_claims.append(cleanup_claim)
            raise ValueError("embedding model is not configured")

        embeddings = EmbeddingsProbe(embed=fail_after_cleanup_takeover)
        processor = VectorProcessorProbe()
        vector._embeddings = embeddings
        vector._vector_processor = processor

        with pytest.raises(SummaryVectorWriteNotSubmittedError, match="not configured"):
            SummaryIndexAdapter.vectorize_summary(
                summary,
                segment,
                dataset,
                session=sqlite_session,
                vector_factory=lambda: vector,
            )

        assert embeddings.calls == [[summary.summary_content]]
        assert processor.created == []
        assert processor.deleted == []
        assert len(takeover_claims) == 1
        with sqlite_session_factory() as observer:
            cleanup = observer.scalar(select(SummaryVectorCleanupOutbox))
            assert cleanup is not None
            # A cleanup that already started is itself durable fencing evidence.
            # Proven non-submission makes it terminal instead of deleting it
            # underneath that cleaner; it no longer blocks dataset cleanup.
            assert cleanup.writer_finished_at is not None
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            assert stored.status == SummaryStatus.ERROR

        parent_intent = ResourceCleanupIntent(
            id="dataset-cleanup-1",
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            resource_type=ResourceCleanupType.DATASET,
            resource_id=dataset.id,
            doc_form=None,
            source_file_id=None,
            indexing_technique=dataset.indexing_technique,
            index_struct=dataset.index_struct,
            collection_binding_id=dataset.collection_binding_id,
            pipeline_id=dataset.pipeline_id,
        )
        cleanup_repository = SQLAlchemyResourceCleanupRepository(session_factory=sqlite_session_factory)
        assert cleanup_repository.has_unfinished_summary_writers(parent_intent) is False

    def test_submit_timeout_retains_unfinished_writers_for_reconciliation(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from core.rag.datasource.vdb.vector_factory import Vector
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        _install_vector_dependencies(monkeypatch)
        vector = Vector.__new__(Vector)
        embeddings = EmbeddingsProbe(response=[[0.1]])
        processor = VectorProcessorProbe(create_error=TimeoutError("vector submission timeout"))
        vector._embeddings = embeddings
        vector._vector_processor = processor
        node_ids = iter(["node-1", "node-2", "node-3"])

        def next_node_id() -> str:
            return next(node_ids)

        monkeypatch.setattr(summary_module.uuid, "uuid4", next_node_id)
        sleep = SleepProbe()
        monkeypatch.setattr(summary_module.time, "sleep", sleep)

        with pytest.raises(TimeoutError, match="submission timeout"):
            SummaryIndexAdapter.vectorize_summary(
                summary,
                segment,
                dataset,
                session=sqlite_session,
                vector_factory=lambda: vector,
            )

        assert len(embeddings.calls) == 3
        assert len(processor.created) == 3
        assert processor.deleted == []
        assert sleep.calls == [2.0, 4.0]
        with sqlite_session_factory() as observer:
            pending = observer.scalars(
                select(SummaryVectorCleanupOutbox).order_by(SummaryVectorCleanupOutbox.node_id)
            ).all()
            assert [row.node_id for row in pending] == ["node-1", "node-2", "node-3"]
            assert all(row.writer_finished_at is None for row in pending)

    def test_service_owned_session_updates_persisted_record(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        _install_vector_dependencies(monkeypatch)

        SummaryIndexAdapter.vectorize_summary(summary, segment, dataset)

        with sqlite_session_factory() as observer:
            stored = observer.get(DocumentSegmentSummary, summary.id)
            assert stored is not None
            assert (stored.status, stored.summary_index_node_id, stored.summary_index_node_hash, stored.tokens) == (
                SummaryStatus.COMPLETED,
                "node-1",
                "hash-1",
                5,
            )
            assert stored.vector_route_hash == _route().key
            assert VectorRoute.model_validate_json(stored.vector_route or "") == _route()

    def test_ordinary_writer_captures_dataset_snapshot_generation(
        self,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        dataset.indexing_request_id = "generation-1"
        dataset.indexing_applied_representation = DatasetIndexingRepresentation(
            indexing_technique="high_quality",
            index_struct=_route().index_struct,
            collection_binding_id=None,
            embedding_model=dataset.embedding_model,
            embedding_model_provider=dataset.embedding_model_provider,
            keyword_number=dataset.keyword_number,
            is_multimodal=dataset.is_multimodal,
            doc_form="text_model",
            summary_index_setting=dataset.summary_index_setting,
            vector_route=_route(),
        ).model_dump_json()
        sqlite_session.commit()
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        vector = _install_vector_dependencies(monkeypatch)
        original_lock = summary_module._lock_summary_dataset_owner
        captured_generations: list[str | None] = []

        def record_lock(
            session: Session,
            *,
            ref: DatasetRef,
            indexing_request_id: str | None,
            indexing_intent_id: str | None,
            indexing_claim_id: str | None,
        ) -> Dataset | None:
            captured_generations.append(indexing_request_id)
            return original_lock(
                session,
                ref=ref,
                indexing_request_id=indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
            )

        monkeypatch.setattr(summary_module, "_lock_summary_dataset_owner", record_lock)

        assert (
            SummaryIndexAdapter.vectorize_summary(
                summary,
                segment,
                dataset,
                session=sqlite_session,
                indexing_request_id="generation-1",
            )
            is True
        )

        assert captured_generations == ["generation-1", "generation-1"]
        vector.add_texts.assert_called_once()

    def test_stale_dataset_snapshot_cannot_publish_summary_vector(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        dataset.indexing_request_id = "generation-1"
        sqlite_session.commit()
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        vector = _install_vector_dependencies(monkeypatch)
        with sqlite_session_factory.begin() as concurrent:
            current = concurrent.get_one(Dataset, dataset.id)
            current.indexing_request_id = "generation-2"

        with pytest.raises(LookupError, match="Summary segment no longer exists"):
            SummaryIndexAdapter.vectorize_summary(
                summary,
                segment,
                dataset,
                session=sqlite_session,
                indexing_request_id="generation-1",
            )

        vector.add_texts.assert_not_called()
        with sqlite_session_factory() as observer:
            assert observer.scalar(select(func.count()).select_from(SummaryVectorCleanupOutbox)) == 0

    def test_pending_generation_summary_writer_records_actual_route_before_submission(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        generation_id = "generation-pending"
        applied_route = _route("summary-collection-a")
        target_route = _route()
        source = DatasetIndexingRepresentation(
            indexing_technique="high_quality",
            index_struct=applied_route.index_struct,
            collection_binding_id=None,
            embedding_model=dataset.embedding_model,
            embedding_model_provider=dataset.embedding_model_provider,
            keyword_number=dataset.keyword_number,
            is_multimodal=dataset.is_multimodal,
            doc_form="text_model",
            summary_index_setting=dataset.summary_index_setting,
            vector_route=applied_route,
        )
        target = source.model_copy(update={"vector_route": None, "cleanup_routes": ()})
        dataset.indexing_request_id = generation_id
        dataset.indexing_applied_representation = source.model_dump_json()
        intent = DatasetIndexingOutbox(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            generation_id=generation_id,
            operation="rebuild",
            action="update",
            source_representation=source.model_dump_json(),
            target_representation=target.model_dump_json(),
        )
        sqlite_session.add(intent)
        sqlite_session.commit()
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        _install_vector_dependencies(monkeypatch)

        assert SummaryIndexAdapter.vectorize_summary(
            summary,
            segment,
            dataset,
            session=sqlite_session,
            indexing_request_id=generation_id,
        )

        with sqlite_session_factory() as observer:
            stored = DatasetIndexingRepresentation.model_validate_json(
                observer.get_one(DatasetIndexingOutbox, intent.id).target_representation
            )
            assert stored.vector_route is None
            assert stored.cleanup_routes == (target_route,)

    def test_service_owned_session_does_not_fall_back_from_summary_id(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        stored = _persist_summary(sqlite_session, dataset, document, segment, summary_id="stored-id")
        detached = DocumentSegmentSummary(
            dataset_id=dataset.id,
            document_id=document.id,
            chunk_id=segment.id,
            summary_content="replacement",
        )
        detached.id = "missing-id"
        _install_vector_dependencies(monkeypatch)

        persisted = SummaryIndexAdapter.vectorize_summary(detached, segment, dataset)

        assert persisted is False
        with sqlite_session_factory() as observer:
            refreshed = observer.get(DocumentSegmentSummary, stored.id)
            assert refreshed is not None
            assert refreshed.summary_content == "summary"
            assert refreshed.status == SummaryStatus.GENERATING
            assert observer.get(DocumentSegmentSummary, detached.id) is None

    def test_service_owned_session_does_not_recreate_deleted_record(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, _, segment = _graph(sqlite_session)
        detached = DocumentSegmentSummary(
            dataset_id=dataset.id,
            document_id=segment.document_id,
            chunk_id=segment.id,
            summary_content="new summary",
        )
        detached.id = "new-summary"
        _install_vector_dependencies(monkeypatch)

        persisted = SummaryIndexAdapter.vectorize_summary(detached, segment, dataset)

        assert persisted is False
        with sqlite_session_factory() as observer:
            stored = observer.get(DocumentSegmentSummary, detached.id)
            assert stored is None

    def test_completion_does_not_resurrect_summary_cleared_during_vector_write(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        vector = _install_vector_dependencies(monkeypatch)

        def clear_summary(*_args: object, **_kwargs: object) -> None:
            with sqlite_session_factory.begin() as concurrent:
                current = concurrent.get(DocumentSegmentSummary, summary.id)
                assert current is not None
                concurrent.delete(current)

        vector.add_texts.side_effect = clear_summary

        persisted = SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        assert persisted is False
        assert vector.delete_by_ids.call_args_list[-1].args == (["node-1"],)
        with sqlite_session_factory() as observer:
            assert observer.get(DocumentSegmentSummary, summary.id) is None

    def test_completion_does_not_overwrite_a_newer_summary_version(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, content="old")
        vector = _install_vector_dependencies(monkeypatch)

        def replace_summary(*_args: object, **_kwargs: object) -> None:
            with sqlite_session_factory.begin() as concurrent:
                current = concurrent.get(DocumentSegmentSummary, summary.id)
                assert current is not None
                current.version += 1
                current.summary_content = "new"
                current.summary_index_node_id = "new-node"
                current.status = SummaryStatus.COMPLETED

        vector.add_texts.side_effect = replace_summary

        persisted = SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        assert persisted is False
        assert vector.delete_by_ids.call_args_list[-1].args == (["node-1"],)
        with sqlite_session_factory() as observer:
            current = observer.get(DocumentSegmentSummary, summary.id)
            assert current is not None
            assert (current.version, current.summary_content, current.summary_index_node_id) == (2, "new", "new-node")

    def test_completion_failure_reclaims_confirmed_unpublished_vector(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        vector = _install_vector_dependencies(monkeypatch)

        def completion_failed(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("completion write failed")

        monkeypatch.setattr(
            summary_module.SQLAlchemySummaryVectorCleanupRepository,
            "completed",
            completion_failed,
        )

        with pytest.raises(RuntimeError, match="completion write failed"):
            SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        vector.delete_by_ids.assert_called_once_with(["node-1"])
        with sqlite_session_factory() as observer:
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            assert stored.status == SummaryStatus.ERROR
            assert stored.summary_index_node_id is None

    def test_failed_immediate_compensation_leaves_claimable_durable_cleanup(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        vector = _install_vector_dependencies(monkeypatch)
        vector.delete_by_ids.side_effect = RuntimeError("vector delete unavailable")

        def completion_failed(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("completion write failed")

        monkeypatch.setattr(
            summary_module.SQLAlchemySummaryVectorCleanupRepository,
            "completed",
            completion_failed,
        )

        with pytest.raises(RuntimeError, match="completion write failed"):
            SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        with sqlite_session_factory() as observer:
            cleanup = observer.scalar(
                select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "node-1")
            )
            assert cleanup is not None
            assert cleanup.claim_id is None
            assert cleanup.claim_expires_at is None

    def test_completion_commit_then_exception_keeps_published_vector(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, node_id="old-node")
        vector = _install_vector_dependencies(monkeypatch)
        original_completed = summary_module.SQLAlchemySummaryVectorCleanupRepository.completed

        def commit_then_raise(
            repository: SQLAlchemySummaryVectorCleanupRepository,
            command: SummaryVectorInput,
            result: SummaryVectorResult,
            claim: SummaryVectorCleanupClaim,
        ) -> None:
            assert original_completed(repository, command, result, claim).published is True
            raise RuntimeError("commit result was lost")

        monkeypatch.setattr(summary_module.SQLAlchemySummaryVectorCleanupRepository, "completed", commit_then_raise)

        persisted = SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        assert persisted is True
        vector.delete_by_ids.assert_not_called()
        with sqlite_session_factory() as observer:
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            assert (stored.status, stored.summary_index_node_id, stored.summary_index_node_hash, stored.tokens) == (
                SummaryStatus.COMPLETED,
                "node-1",
                "hash-1",
                5,
            )
            cleanup = observer.scalar(
                select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "old-node")
            )
            assert cleanup is not None
            assert cleanup.writer_finished_at is not None

    def test_previous_node_cleanup_uses_locked_row_and_retries_delete_failure(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, node_id="stale-caller-node")
        vector = _install_vector_dependencies(monkeypatch)

        def replace_previous_node(*_args: object, **_kwargs: object) -> None:
            with sqlite_session_factory.begin() as concurrent:
                current = concurrent.get_one(DocumentSegmentSummary, summary.id)
                current.summary_index_node_id = "actual-previous-node"

        vector.add_texts.side_effect = replace_previous_node
        vector.delete_by_ids.side_effect = RuntimeError("vector delete unavailable")

        persisted = SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        assert persisted is True
        vector.delete_by_ids.assert_called_once_with(["actual-previous-node"])
        with sqlite_session_factory() as observer:
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            assert stored.summary_index_node_id == "node-1"
            cleanup = observer.scalar(
                select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "actual-previous-node")
            )
            assert cleanup is not None
            assert cleanup.vector_route_hash == _route().key
            assert VectorRoute.model_validate_json(cleanup.vector_route or "") == _route()
            assert cleanup.writer_finished_at is not None
            assert cleanup.claim_id is None
            assert cleanup.claim_expires_at is None
            assert (
                observer.scalar(
                    select(func.count())
                    .select_from(SummaryVectorCleanupOutbox)
                    .where(SummaryVectorCleanupOutbox.node_id == "stale-caller-node")
                )
                == 0
            )

    def test_previous_node_cleanup_is_acked_only_after_successful_delete(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, node_id="old-node")
        vector = _install_vector_dependencies(monkeypatch)

        persisted = SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        assert persisted is True
        vector.delete_by_ids.assert_called_once_with(["old-node"])
        with sqlite_session_factory() as observer:
            assert observer.scalar(select(func.count()).select_from(SummaryVectorCleanupOutbox)) == 0

    def test_unknown_completion_outcome_keeps_durable_cleanup_without_deleting(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        vector = _install_vector_dependencies(monkeypatch)

        failed_calls: list[str] = []

        def completion_unknown(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("commit result was lost")

        def publication_unknown(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("database unavailable")

        def record_failure(_repository: object, _command: object, error: str) -> bool:
            failed_calls.append(error)
            return True

        monkeypatch.setattr(
            summary_module.SQLAlchemySummaryVectorCleanupRepository,
            "completed",
            completion_unknown,
        )
        monkeypatch.setattr(
            summary_module.SQLAlchemySummaryVectorCleanupRepository,
            "is_published",
            publication_unknown,
        )
        monkeypatch.setattr(summary_module.SQLAlchemySummaryVectorCleanupRepository, "failed", record_failure)

        with pytest.raises(RuntimeError, match="commit result was lost"):
            SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        vector.delete_by_ids.assert_not_called()
        assert failed_calls == []
        with sqlite_session_factory() as observer:
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            cleanup = observer.scalar(
                select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "node-1")
            )
            assert stored.status == SummaryStatus.GENERATING
            assert stored.summary_index_node_id is None
            assert cleanup is not None

    def test_cleanup_probe_protects_node_referenced_by_newer_generation(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        from repositories.knowledge.summary_vector_cleanup_repository import (
            SQLAlchemySummaryVectorCleanupRepository,
        )
        from services.knowledge.entities.resource_scope import DatasetRef
        from services.knowledge.entities.summaries import SummaryVectorInput, SummaryVectorResult

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        command = SummaryVectorInput(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id).segment(segment.id),
            summary.id,
            summary.version,
            summary.summary_content or "",
            None,
        )
        result = SummaryVectorResult("node-1", "hash-1", 5)
        repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
        route = _route()
        intent = repository.begin_write(command, result, route)
        assert repository.list_pending_cleanup_ids(limit=10) == ()
        claim = repository.finish_write(intent)

        with sqlite_session_factory.begin() as concurrent:
            current = concurrent.get_one(DocumentSegmentSummary, summary.id)
            current.version += 1
            current.summary_index_node_id = result.node_id
            current.vector_route = route.serialized
            current.vector_route_hash = route.key
            current.status = SummaryStatus.COMPLETED

        with repository.guard_cleanup_delete(claim) as referenced:
            assert referenced is True
        assert repository.list_pending_cleanup_ids(limit=10) == ()

    def test_uncertain_publication_confirmation_requires_exact_physical_route(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        from repositories.knowledge.summary_vector_cleanup_repository import (
            SQLAlchemySummaryVectorCleanupRepository,
        )
        from services.knowledge.entities.resource_scope import DatasetRef
        from services.knowledge.entities.summaries import (
            SummaryVectorCleanupClaim,
            SummaryVectorInput,
            SummaryVectorResult,
        )

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(
            sqlite_session,
            dataset,
            document,
            segment,
            node_id="node-1",
            status=SummaryStatus.COMPLETED,
        )
        route_b = _route("summary-collection-b")
        summary.summary_index_node_hash = "hash-1"
        summary.vector_route = route_b.serialized
        summary.vector_route_hash = route_b.key
        summary.tokens = 5
        sqlite_session.commit()
        command = SummaryVectorInput(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id).segment(segment.id),
            summary.id,
            summary.version,
            summary.summary_content or "",
            None,
        )
        result = SummaryVectorResult("node-1", "hash-1", 5)
        repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
        claim_a = SummaryVectorCleanupClaim(
            "cleanup-a",
            dataset.tenant_id,
            dataset.id,
            result.node_id,
            "claim-a",
            True,
            _route("summary-collection-a"),
        )
        claim_b = SummaryVectorCleanupClaim(
            "cleanup-b",
            dataset.tenant_id,
            dataset.id,
            result.node_id,
            "claim-b",
            True,
            route_b,
        )

        assert repository.is_published(command, result, claim_a) is False
        assert repository.is_published(command, result, claim_b) is True

    def test_cleanup_cannot_ack_late_writer_after_delete_snapshot(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox
        from repositories.knowledge.summary_vector_cleanup_repository import (
            SQLAlchemySummaryVectorCleanupRepository,
            SummaryVectorCleanupClaimLostError,
        )
        from services.knowledge.entities.resource_scope import DatasetRef
        from services.knowledge.entities.summaries import SummaryVectorInput, SummaryVectorResult

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        command = SummaryVectorInput(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id).segment(segment.id),
            summary.id,
            summary.version,
            summary.summary_content or "",
            None,
        )
        result = SummaryVectorResult("late-node", "hash-1", 5)
        repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
        writer = repository.begin_write(command, result, _route())
        vector_nodes: set[str] = set()

        with sqlite_session_factory.begin() as concurrent:
            row = concurrent.get_one(SummaryVectorCleanupOutbox, writer.id)
            row.claim_expires_at = datetime(2000, 1, 1)

        cleanup = repository.claim_pending_cleanup(writer.id)
        assert cleanup is not None
        assert cleanup.writer_finished is False
        with repository.guard_cleanup_delete(cleanup) as referenced:
            assert referenced is False
            vector_nodes.discard(result.node_id)

        # The cleanup delete is still running after its lease expires. That
        # stale lease must never let the late writer reclaim publication rights.
        with sqlite_session_factory.begin() as concurrent:
            row = concurrent.get_one(SummaryVectorCleanupOutbox, writer.id)
            row.claim_expires_at = datetime(2000, 1, 1)

        # The vector delete happened while the external write was still in
        # flight. Its late return records completion but cannot steal the active
        # cleanup claim.
        vector_nodes.add(result.node_id)
        with pytest.raises(SummaryVectorCleanupClaimLostError, match="another worker"):
            repository.finish_write(writer)

        # Even though the database now says the writer finished, this cleanup
        # pass observed it unfinished before deleting and therefore cannot ACK.
        assert repository.release_cleanup(cleanup) is True
        assert vector_nodes == {result.node_id}
        next_cleanup = repository.claim_pending_cleanup(writer.id)
        assert next_cleanup is not None
        assert next_cleanup.writer_finished is True
        with repository.guard_cleanup_delete(next_cleanup) as referenced:
            assert referenced is False
            vector_nodes.discard(result.node_id)

        with sqlite_session_factory() as observer:
            assert observer.get(SummaryVectorCleanupOutbox, writer.id) is None
        assert vector_nodes == set()

    @pytest.mark.parametrize("writer_outcome", ["late_finish", "discard_unsubmitted"])
    def test_terminal_handoff_preserves_unfinished_summary_writer_until_safe_ack(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        writer_outcome: str,
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox
        from repositories.knowledge.summary_vector_cleanup_repository import (
            SummaryVectorCleanupClaimLostError,
        )

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(
            sqlite_session,
            dataset,
            document,
            segment,
            node_id="shared-node",
            status=SummaryStatus.COMPLETED,
        )
        route = _route()
        command = SummaryVectorInput(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id).segment(segment.id),
            summary.id,
            summary.version,
            summary.summary_content or "",
            summary.summary_index_node_id,
        )
        result = SummaryVectorResult("shared-node", "hash-1", 5)
        repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
        writer = repository.begin_write(command, result, route)

        with sqlite_session_factory.begin() as terminal:
            current = terminal.get_one(DocumentSegmentSummary, summary.id)
            repository.add_terminal_cleanups(terminal, tenant_id=dataset.tenant_id, summaries=[current])
            current.enabled = False
            current.version += 1
        with sqlite_session_factory() as observer:
            handed_off = observer.get_one(SummaryVectorCleanupOutbox, writer.id)
            cleanup_started_at = handed_off.cleanup_started_at
            assert handed_off.writer_id == writer.writer_id
            assert handed_off.writer_finished_at is None
            assert cleanup_started_at is not None
            assert handed_off.claim_id is None
            assert handed_off.vector_route == route.serialized
            assert handed_off.vector_route_hash == route.key

        # Repeated resource termination updates ownership metadata without
        # reopening publication or inventing remote completion.
        with sqlite_session_factory.begin() as terminal:
            current = terminal.get_one(DocumentSegmentSummary, summary.id)
            repository.add_terminal_cleanups(terminal, tenant_id=dataset.tenant_id, summaries=[current])
        with sqlite_session_factory() as observer:
            handed_off = observer.get_one(SummaryVectorCleanupOutbox, writer.id)
            assert handed_off.writer_id == writer.writer_id
            assert handed_off.writer_finished_at is None
            assert handed_off.cleanup_started_at == cleanup_started_at

        vector_nodes = {result.node_id}
        first_cleanup = repository.claim_pending_cleanup(writer.id)
        assert first_cleanup is not None
        assert first_cleanup.writer_finished is False
        with repository.guard_cleanup_delete(first_cleanup) as referenced:
            assert referenced is False
            vector_nodes.discard(result.node_id)
        assert repository.release_cleanup(first_cleanup) is True
        with sqlite_session_factory() as observer:
            assert observer.get(SummaryVectorCleanupOutbox, writer.id) is not None

        if writer_outcome == "late_finish":
            vector_nodes.add(result.node_id)
            with pytest.raises(SummaryVectorCleanupClaimLostError, match="another worker"):
                repository.finish_write(writer)
        else:
            repository.discard_unsubmitted_write(writer)

        final_cleanup = repository.claim_pending_cleanup(writer.id)
        assert final_cleanup is not None
        assert final_cleanup.writer_finished is True
        with repository.guard_cleanup_delete(final_cleanup) as referenced:
            assert referenced is False
            vector_nodes.discard(result.node_id)

        with sqlite_session_factory() as observer:
            assert observer.get(SummaryVectorCleanupOutbox, writer.id) is None
        assert vector_nodes == set()

    def test_writer_registration_rejects_deleted_dataset(
        self,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        from models.model import SummaryVectorCleanupOutbox
        from repositories.knowledge.summary_vector_cleanup_repository import (
            SQLAlchemySummaryVectorCleanupRepository,
        )
        from services.knowledge.entities.resource_scope import DatasetRef
        from services.knowledge.entities.summaries import SummaryVectorInput, SummaryVectorResult

        command = SummaryVectorInput(
            DatasetRef("deleted-tenant", "deleted-dataset").document("deleted-document").segment("deleted-segment"),
            "deleted-summary",
            1,
            "summary",
            None,
        )
        repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
        route = VectorRoute.capture(
            tenant_id="deleted-tenant",
            dataset_id="deleted-dataset",
            vector_type="qdrant",
            collection_name="deleted-collection",
            collection_binding_id=None,
            indexing_technique="high_quality",
            doc_form="summary",
        )

        with pytest.raises(LookupError, match="dataset no longer exists"):
            repository.begin_write(command, SummaryVectorResult("node-1", "hash-1", 5), route)

        with sqlite_session_factory() as observer:
            assert observer.scalar(select(func.count()).select_from(SummaryVectorCleanupOutbox)) == 0

    def test_failure_does_not_mark_a_newer_summary_version_as_error(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, content="old")
        vector = _install_vector_dependencies(monkeypatch)

        def replace_then_fail(*_args: object, **kwargs: object) -> None:
            before_write = kwargs["before_write"]
            assert callable(before_write)
            before_write()
            with sqlite_session_factory.begin() as concurrent:
                current = concurrent.get(DocumentSegmentSummary, summary.id)
                assert current is not None
                current.version += 1
                current.summary_content = "new"
                current.status = SummaryStatus.COMPLETED
                current.error = None
            raise RuntimeError("old vector failed")

        vector.add_texts.side_effect = replace_then_fail

        with pytest.raises(RuntimeError, match="old vector failed"):
            SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        with sqlite_session_factory() as observer:
            current = observer.get(DocumentSegmentSummary, summary.id)
            assert current is not None
            assert (current.version, current.summary_content, current.status, current.error) == (
                2,
                "new",
                SummaryStatus.COMPLETED,
                None,
            )

    def test_late_failure_does_not_overwrite_new_node_at_same_summary_version(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        from repositories.knowledge.summary_vector_cleanup_repository import (
            SQLAlchemySummaryVectorCleanupRepository,
        )
        from services.knowledge.entities.resource_scope import DatasetRef
        from services.knowledge.entities.summaries import SummaryVectorInput

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(
            sqlite_session,
            dataset,
            document,
            segment,
            content="summary",
            node_id="old-node",
            status=SummaryStatus.GENERATING,
        )
        command = SummaryVectorInput(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id).segment(segment.id),
            summary.id,
            summary.version,
            summary.summary_content or "",
            "old-node",
        )
        with sqlite_session_factory.begin() as concurrent:
            current = concurrent.get_one(DocumentSegmentSummary, summary.id)
            current.summary_index_node_id = "new-node"
            current.summary_index_node_hash = "new-hash"
            current.status = SummaryStatus.COMPLETED
            current.error = None

        repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)

        assert repository.failed(command, "old worker failed") is False
        with sqlite_session_factory() as observer:
            current = observer.get_one(DocumentSegmentSummary, summary.id)
            assert current.version == command.version
            assert (current.summary_index_node_id, current.status, current.error) == (
                "new-node",
                SummaryStatus.COMPLETED,
                None,
            )

    def test_provided_session_is_committed_before_independent_error_persistence(
        self,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        vector = _install_vector_dependencies(monkeypatch)

        def fail_after_write_started(*_args: object, **kwargs: object) -> None:
            before_write = kwargs["before_write"]
            assert callable(before_write)
            before_write()
            raise RuntimeError("fatal vector failure")

        vector.add_texts.side_effect = fail_after_write_started
        original_create_session = summary_module.session_factory.create_session
        created_sessions = 0

        def create_session() -> Session:
            nonlocal created_sessions
            created_sessions += 1
            return original_create_session()

        monkeypatch.setattr(summary_module.session_factory, "create_session", create_session)

        with pytest.raises(RuntimeError, match="fatal vector failure"):
            SummaryIndexAdapter.vectorize_summary(summary, segment, dataset, session=sqlite_session)

        # Ownership registration and error persistence use their own bounded
        # transactions after the caller transaction closes. The uncertain
        # external write keeps its durable intent for reconciliation.
        assert created_sessions >= 3
        assert not sqlite_session.in_transaction()
        assert summary.status == SummaryStatus.ERROR
        assert summary.error == "Vectorization failed: fatal vector failure"


class TestBatchAndGeneration:
    @pytest.mark.parametrize("foreign_document", [False, True])
    def test_task_resolves_owned_document_and_releases_read_transaction(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
        foreign_document: bool,
    ) -> None:
        from tasks.generate_summary_index_task import generate_summary_index_task

        dataset, document, _ = _graph(sqlite_session)
        dataset.summary_index_setting = {"enable": True}
        document.need_summary = True
        if foreign_document:
            other = _persist_dataset(sqlite_session, dataset_id="other", tenant_id=OTHER_TENANT_ID)
            document.dataset_id, document.tenant_id = other.id, other.tenant_id
        sqlite_session.commit()
        sessions: list[Session] = []

        def create_session() -> Session:
            session = sqlite_session_factory()
            sessions.append(session)
            return session

        def generate(**_kwargs: object) -> list[DocumentSegmentSummary]:
            assert sessions
            assert all(not session.in_transaction() for session in sessions)
            generated.append(True)
            return []

        generated: list[bool] = []
        monkeypatch.setattr(summary_module.session_factory, "create_session", create_session)
        monkeypatch.setattr(SummaryIndexAdapter, "generate_summaries_for_document", generate)

        generate_summary_index_task(dataset.id, document.id)

        assert len(generated) == (0 if foreign_document else 1)

    @pytest.mark.parametrize(("task_generation", "expected_calls"), [(None, []), ("generation-1", ["generation-1"])])
    def test_task_requires_the_generation_captured_at_dispatch(
        self,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
        task_generation: str | None,
        expected_calls: list[str],
    ) -> None:
        from tasks.generate_summary_index_task import generate_summary_index_task

        dataset, document, _ = _graph(sqlite_session)
        dataset.indexing_request_id = "generation-1"
        dataset.summary_index_setting = {"enable": True}
        document.need_summary = True
        sqlite_session.commit()
        calls: list[str] = []

        def generate(
            *,
            dataset: Dataset,
            document: Document,
            summary_index_setting: SummaryIndexSettingDict,
            segment_ids: list[str] | None,
            only_parent_chunks: bool,
            indexing_request_id: str | None,
        ) -> list[DocumentSegmentSummary]:
            del document, summary_index_setting, segment_ids, only_parent_chunks
            assert dataset.id == "dataset-1"
            assert indexing_request_id is not None
            calls.append(indexing_request_id)
            return []

        monkeypatch.setattr(SummaryIndexAdapter, "generate_summaries_for_document", generate)

        generate_summary_index_task(dataset.id, document.id, indexing_request_id=task_generation)

        assert calls == expected_calls

    def test_batch_create_no_segments_is_noop(self, sqlite_session: Session) -> None:
        dataset = _persist_dataset(sqlite_session)

        SummaryIndexAdapter.batch_create_summary_records([], dataset, indexing_request_id=None)

        assert sqlite_session.scalar(select(func.count()).select_from(DocumentSegmentSummary)) == 0

    def test_batch_create_inserts_updates_reenables_and_scopes(
        self, sqlite_session: Session, sqlite_session_factory: sessionmaker[Session]
    ) -> None:
        dataset, document, first = _graph(sqlite_session)
        second = _persist_segment(sqlite_session, dataset, document, segment_id="seg-2", position=2)
        existing = _persist_summary(sqlite_session, dataset, document, first, enabled=False)
        existing.error = "old"
        existing.disabled_by = "account-2"
        sqlite_session.commit()

        SummaryIndexAdapter.batch_create_summary_records(
            [first, second], dataset, status=SummaryStatus.NOT_STARTED, indexing_request_id=None
        )

        with sqlite_session_factory() as observer:
            rows = observer.scalars(
                select(DocumentSegmentSummary)
                .where(DocumentSegmentSummary.dataset_id == dataset.id)
                .order_by(DocumentSegmentSummary.chunk_id)
            ).all()
            assert [row.chunk_id for row in rows] == [first.id, second.id]
            assert all(row.status == SummaryStatus.NOT_STARTED for row in rows)
            assert all(row.enabled and row.error is None for row in rows)

    def test_generate_and_vectorize_creates_commits_and_returns_row(
        self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset, _, segment = _graph(sqlite_session)
        monkeypatch.setattr(
            SummaryIndexAdapter,
            "generate_summary_for_segment",
            lambda *_args, **_kwargs: ("generated", _usage()),
        )

        def vectorize(
            record: DocumentSegmentSummary,
            _segment: DocumentSegment,
            _dataset: Dataset,
            session: Session | None = None,
            *,
            indexing_request_id: str | None = None,
            indexing_intent_id: str | None = None,
            indexing_claim_id: str | None = None,
        ) -> bool:
            assert indexing_request_id is None
            assert indexing_intent_id is None
            assert indexing_claim_id is None
            record.status = SummaryStatus.COMPLETED
            record.summary_index_node_id = "node-1"
            sqlite_session.flush()
            return True

        monkeypatch.setattr(SummaryIndexAdapter, "vectorize_summary", vectorize)

        result = SummaryIndexAdapter.generate_and_vectorize_summary(
            segment, dataset, {"enable": True}, session=sqlite_session, indexing_request_id=None
        )

        assert (result.summary_content, result.status, result.summary_index_node_id) == (
            "generated",
            SummaryStatus.COMPLETED,
            "node-1",
        )

    def test_generation_handoff_rejects_document_archived_before_reservation(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(
            sqlite_session,
            dataset,
            document,
            segment,
            content="current summary",
            status=SummaryStatus.COMPLETED,
        )
        with sqlite_session_factory.begin() as concurrent:
            concurrent.get_one(Document, document.id).archived = True

        model_inputs: list[str] = []

        def generate(
            current_segment: DocumentSegment,
            _dataset: Dataset,
            _setting: SummaryIndexSettingDict,
            *,
            session: Session,
        ) -> tuple[str, LLMUsage]:
            del session
            model_inputs.append(current_segment.content)
            return "must not publish", _usage()

        monkeypatch.setattr(SummaryIndexAdapter, "generate_summary_for_segment", generate)

        result = SummaryIndexAdapter.generate_and_vectorize_summary(
            segment,
            dataset,
            {"enable": True},
            session=sqlite_session,
            indexing_request_id=None,
        )

        assert result is None
        assert model_inputs == []
        with sqlite_session_factory() as observer:
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            assert (stored.version, stored.summary_content, stored.status, stored.error) == (
                1,
                "current summary",
                SummaryStatus.COMPLETED,
                None,
            )

    def test_segment_disabled_during_llm_cannot_publish_content_or_error(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(
            sqlite_session,
            dataset,
            document,
            segment,
            content="current summary",
            status=SummaryStatus.COMPLETED,
        )

        def disable_during_llm(
            current_segment: DocumentSegment,
            _dataset: Dataset,
            _setting: SummaryIndexSettingDict,
            *,
            session: Session,
        ) -> tuple[str, LLMUsage]:
            session.commit()
            with sqlite_session_factory.begin() as concurrent:
                concurrent.get_one(DocumentSegment, current_segment.id).enabled = False
            return "stale generated content", _usage()

        vectorized: list[str] = []

        def vectorize(record: DocumentSegmentSummary, *_args: object, **_kwargs: object) -> bool:
            vectorized.append(record.id)
            return True

        monkeypatch.setattr(SummaryIndexAdapter, "generate_summary_for_segment", disable_during_llm)
        monkeypatch.setattr(SummaryIndexAdapter, "vectorize_summary", vectorize)

        result = SummaryIndexAdapter.generate_and_vectorize_summary(
            segment,
            dataset,
            {"enable": True},
            session=sqlite_session,
            indexing_request_id=None,
        )

        assert result is None
        assert vectorized == []
        with sqlite_session_factory() as observer:
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            assert (stored.version, stored.summary_content, stored.status, stored.error) == (
                2,
                "current summary",
                SummaryStatus.GENERATING,
                None,
            )
            assert observer.get_one(DocumentSegment, segment.id).enabled is False

    def test_old_generation_does_not_restore_a_summary_cleared_while_llm_runs(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        summary_id = summary.id

        def clear_then_finish(*_args: object, **_kwargs: object) -> tuple[str, LLMUsage]:
            with sqlite_session_factory.begin() as concurrent:
                current = concurrent.get(DocumentSegmentSummary, summary_id)
                assert current is not None
                concurrent.delete(current)
            return "stale generated content", _usage()

        monkeypatch.setattr(SummaryIndexAdapter, "generate_summary_for_segment", clear_then_finish)
        vectorized_summary_ids: list[str] = []

        def record_vectorization(record: DocumentSegmentSummary, *_args: object, **_kwargs: object) -> bool:
            vectorized_summary_ids.append(record.id)
            return True

        monkeypatch.setattr(SummaryIndexAdapter, "vectorize_summary", record_vectorization)

        result = SummaryIndexAdapter.generate_and_vectorize_summary(
            segment, dataset, {"enable": True}, session=sqlite_session, indexing_request_id=None
        )

        assert result is None
        assert vectorized_summary_ids == []
        with sqlite_session_factory() as observer:
            assert observer.get(DocumentSegmentSummary, summary_id) is None

    def test_generation_reservation_refreshes_a_stale_loaded_version(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment)
        assert summary.version == 1
        with sqlite_session_factory.begin() as concurrent:
            current = concurrent.get(DocumentSegmentSummary, summary.id)
            assert current is not None
            current.version = 2

        monkeypatch.setattr(
            SummaryIndexAdapter,
            "generate_summary_for_segment",
            lambda *_args, **_kwargs: ("generated", _usage()),
        )

        def vectorize(
            record: DocumentSegmentSummary,
            _segment: DocumentSegment,
            _dataset: Dataset,
            session: Session | None = None,
            *,
            indexing_request_id: str | None = None,
            indexing_intent_id: str | None = None,
            indexing_claim_id: str | None = None,
        ) -> bool:
            assert indexing_request_id is None
            assert indexing_intent_id is None
            assert indexing_claim_id is None
            assert record.version == 3
            return True

        monkeypatch.setattr(SummaryIndexAdapter, "vectorize_summary", vectorize)

        result = SummaryIndexAdapter.generate_and_vectorize_summary(
            segment, dataset, {"enable": True}, session=sqlite_session, indexing_request_id=None
        )

        assert result is not None
        assert result.version == 3

    def test_generate_failure_rolls_back_then_persists_error(
        self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, status=SummaryStatus.NOT_STARTED)
        monkeypatch.setattr(
            SummaryIndexAdapter,
            "generate_summary_for_segment",
            MagicMock(side_effect=RuntimeError("LLM failed")),
        )

        with pytest.raises(RuntimeError, match="LLM failed"):
            SummaryIndexAdapter.generate_and_vectorize_summary(
                segment, dataset, {"enable": True}, session=sqlite_session, indexing_request_id=None
            )

        sqlite_session.refresh(summary)
        assert (summary.status, summary.error) == (SummaryStatus.ERROR, "LLM failed")

    def test_generation_switch_before_content_cas_discards_old_model_result(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        dataset.indexing_request_id = "generation-0"
        summary = _persist_summary(sqlite_session, dataset, document, segment, content="original")
        vectorized_summary_ids: list[str] = []

        def switch_generation(
            _segment: DocumentSegment,
            _dataset: Dataset,
            _setting: SummaryIndexSettingDict,
            *,
            session: Session,
        ) -> tuple[str, LLMUsage]:
            session.commit()
            with sqlite_session_factory.begin() as concurrent:
                current_dataset = concurrent.get_one(Dataset, dataset.id)
                current_dataset.indexing_request_id = "generation-1"
            return "stale generation-0 content", _usage()

        def record_vectorization(record: DocumentSegmentSummary, *_args: object, **_kwargs: object) -> bool:
            vectorized_summary_ids.append(record.id)
            return True

        monkeypatch.setattr(SummaryIndexAdapter, "generate_summary_for_segment", switch_generation)
        monkeypatch.setattr(SummaryIndexAdapter, "vectorize_summary", record_vectorization)

        result = SummaryIndexAdapter.generate_and_vectorize_summary(
            segment,
            dataset,
            {"enable": True},
            session=sqlite_session,
            indexing_request_id="generation-0",
        )

        assert result is None
        assert vectorized_summary_ids == []
        with sqlite_session_factory() as observer:
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            assert stored.version == 2
            assert (stored.summary_content, stored.status, stored.error) == (
                "original",
                SummaryStatus.GENERATING,
                None,
            )

    def test_generation_switch_before_error_cas_discards_old_failure(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, segment = _graph(sqlite_session)
        dataset.indexing_request_id = "generation-0"
        summary = _persist_summary(sqlite_session, dataset, document, segment, content="original")

        def switch_generation_then_fail(
            _segment: DocumentSegment,
            _dataset: Dataset,
            _setting: SummaryIndexSettingDict,
            *,
            session: Session,
        ) -> tuple[str, LLMUsage]:
            session.commit()
            with sqlite_session_factory.begin() as concurrent:
                current_dataset = concurrent.get_one(Dataset, dataset.id)
                current_dataset.indexing_request_id = "generation-1"
            raise RuntimeError("old generation model failed")

        monkeypatch.setattr(SummaryIndexAdapter, "generate_summary_for_segment", switch_generation_then_fail)

        result = SummaryIndexAdapter.generate_and_vectorize_summary(
            segment,
            dataset,
            {"enable": True},
            session=sqlite_session,
            indexing_request_id="generation-0",
        )

        assert result is None
        with sqlite_session_factory() as observer:
            stored = observer.get_one(DocumentSegmentSummary, summary.id)
            assert stored.version == 2
            assert (stored.summary_content, stored.status, stored.error) == (
                "original",
                SummaryStatus.GENERATING,
                None,
            )

    def test_document_batch_reuses_captured_generation_across_segments(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset, document, first = _graph(sqlite_session)
        second = _persist_segment(sqlite_session, dataset, document, segment_id="seg-2", position=2)
        third = _persist_segment(sqlite_session, dataset, document, segment_id="seg-3", position=3)
        dataset.indexing_request_id = "generation-0"
        dataset.summary_index_setting = {"enable": True}
        sqlite_session.commit()
        original_generate = SummaryIndexAdapter.generate_and_vectorize_summary
        seen_generations: list[str | None] = []
        generation_1_state: dict[str, tuple[int, str | None, SummaryStatus, str | None]] = {}

        def generate_content(
            _segment: DocumentSegment,
            _dataset: Dataset,
            _setting: SummaryIndexSettingDict,
            *,
            session: Session,
        ) -> tuple[str, LLMUsage]:
            session.commit()
            return "generation-0 content", _usage()

        def vectorize(
            record: DocumentSegmentSummary,
            _segment: DocumentSegment,
            _dataset: Dataset,
            session: Session | None = None,
            *,
            indexing_request_id: str | None = None,
            indexing_intent_id: str | None = None,
            indexing_claim_id: str | None = None,
        ) -> bool:
            assert session is not None
            assert indexing_request_id == "generation-0"
            assert indexing_intent_id is None
            assert indexing_claim_id is None
            record.status = SummaryStatus.COMPLETED
            record.error = None
            session.flush()
            return True

        def interleave_generation_1(
            segment: DocumentSegment,
            current_dataset: Dataset,
            setting: SummaryIndexSettingDict,
            *,
            session: Session,
            indexing_request_id: str | None,
            indexing_intent_id: str | None = None,
            indexing_claim_id: str | None = None,
        ) -> DocumentSegmentSummary | None:
            seen_generations.append(indexing_request_id)
            generated = original_generate(
                segment,
                current_dataset,
                setting,
                session=session,
                indexing_request_id=indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
            )
            if len(seen_generations) == 1:
                session.commit()
                with sqlite_session_factory.begin() as concurrent:
                    current = concurrent.get_one(Dataset, dataset.id)
                    current.indexing_request_id = "generation-1"
                    summaries = concurrent.scalars(
                        select(DocumentSegmentSummary)
                        .where(DocumentSegmentSummary.dataset_id == dataset.id)
                        .order_by(DocumentSegmentSummary.chunk_id)
                    ).all()
                    for row in summaries:
                        row.version += 1
                        row.summary_content = f"generation-1:{row.chunk_id}"
                        row.status = SummaryStatus.COMPLETED
                        row.error = None
                        generation_1_state[row.id] = (row.version, row.summary_content, row.status, row.error)
            return generated

        monkeypatch.setattr(SummaryIndexAdapter, "generate_summary_for_segment", generate_content)
        monkeypatch.setattr(SummaryIndexAdapter, "vectorize_summary", vectorize)
        monkeypatch.setattr(SummaryIndexAdapter, "generate_and_vectorize_summary", interleave_generation_1)

        SummaryIndexAdapter.generate_summaries_for_document(
            dataset,
            document,
            {"enable": True},
            segment_ids=[first.id, second.id, third.id],
            indexing_request_id="generation-0",
        )

        assert seen_generations == ["generation-0", "generation-0"]
        with sqlite_session_factory() as observer:
            stored = observer.scalars(
                select(DocumentSegmentSummary)
                .where(DocumentSegmentSummary.dataset_id == dataset.id)
                .order_by(DocumentSegmentSummary.chunk_id)
            ).all()
            assert len(stored) == 3
            assert {row.id: (row.version, row.summary_content, row.status, row.error) for row in stored} == (
                generation_1_state
            )

    @pytest.mark.parametrize(
        ("technique", "enabled", "doc_form"),
        [
            (IndexTechniqueType.ECONOMY, True, IndexStructureType.PARAGRAPH_INDEX),
            (IndexTechniqueType.HIGH_QUALITY, False, IndexStructureType.PARAGRAPH_INDEX),
            (IndexTechniqueType.HIGH_QUALITY, True, IndexStructureType.QA_INDEX),
        ],
    )
    def test_generate_document_skip_conditions(
        self,
        sqlite_session: Session,
        technique: IndexTechniqueType,
        enabled: bool,
        doc_form: IndexStructureType,
    ) -> None:
        dataset = _persist_dataset(sqlite_session, indexing_technique=technique)
        document = _persist_document(sqlite_session, dataset, doc_form=doc_form)

        assert (
            SummaryIndexAdapter.generate_summaries_for_document(
                dataset, document, {"enable": enabled}, indexing_request_id=None
            )
            == []
        )

    def test_generate_document_filters_segments_and_continues_after_failure(
        self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset, document, first = _graph(sqlite_session)
        second = _persist_segment(sqlite_session, dataset, document, segment_id="seg-2", position=2)
        _persist_segment(sqlite_session, dataset, document, segment_id="disabled", position=3, enabled=False)
        generated = DocumentSegmentSummary(
            dataset_id=dataset.id,
            document_id=document.id,
            chunk_id=first.id,
            summary_content="ok",
            status=SummaryStatus.COMPLETED,
        )
        generate = MagicMock(side_effect=[generated, RuntimeError("boom")])
        monkeypatch.setattr(SummaryIndexAdapter, "generate_and_vectorize_summary", generate)

        result = SummaryIndexAdapter.generate_summaries_for_document(
            dataset,
            document,
            {"enable": True},
            segment_ids=[first.id, second.id],
            only_parent_chunks=True,
            indexing_request_id=None,
        )

        assert result == [generated]
        assert [call.args[0].id for call in generate.call_args_list] == [first.id, second.id]


class TestEnableDisableDelete:
    def test_disable_filters_ids_schedules_exact_route_cleanup_and_persists_flags(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        dataset, document, first = _graph(sqlite_session)
        second = _persist_segment(sqlite_session, dataset, document, segment_id="seg-2", position=2)
        selected = _persist_summary(sqlite_session, dataset, document, first, node_id="node-1")
        untouched = _persist_summary(sqlite_session, dataset, document, second, summary_id="sum-2", node_id="node-2")
        from models.model import SummaryVectorCleanupOutbox

        SummaryIndexAdapter.disable_summaries_for_segments(dataset, segment_ids=[first.id], disabled_by="account-2")

        with sqlite_session_factory() as observer:
            stored_selected = observer.get(DocumentSegmentSummary, selected.id)
            stored_untouched = observer.get(DocumentSegmentSummary, untouched.id)
            assert stored_selected is not None
            assert stored_untouched is not None
            assert stored_selected.enabled is False
            assert stored_selected.disabled_by == "account-2"
            assert stored_selected.disabled_at is not None
            assert stored_untouched.enabled is True
            cleanup = observer.scalar(
                select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "node-1")
            )
            assert cleanup is not None
            assert cleanup.vector_route_hash == _route().key
            assert cleanup.writer_finished_at is not None
            assert (
                observer.scalar(
                    select(func.count())
                    .select_from(SummaryVectorCleanupOutbox)
                    .where(SummaryVectorCleanupOutbox.node_id == "node-2")
                )
                == 0
            )

    def test_disable_does_not_depend_on_immediate_vector_io(self, sqlite_session: Session) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, node_id="node-1")

        SummaryIndexAdapter.disable_summaries_for_segments(dataset)

        sqlite_session.refresh(summary)
        assert summary.enabled is False
        assert (
            sqlite_session.scalar(
                select(func.count())
                .select_from(SummaryVectorCleanupOutbox)
                .where(SummaryVectorCleanupOutbox.node_id == "node-1")
            )
            == 1
        )

    def test_delete_is_dataset_and_segment_scoped(self, sqlite_session: Session) -> None:
        from models.model import SummaryVectorCleanupOutbox

        dataset, document, first = _graph(sqlite_session)
        second = _persist_segment(sqlite_session, dataset, document, segment_id="seg-2", position=2)
        deleted = _persist_summary(sqlite_session, dataset, document, first, node_id="node-1")
        kept = _persist_summary(sqlite_session, dataset, document, second, summary_id="sum-2", node_id="node-2")
        SummaryIndexAdapter.delete_summaries_for_segments(dataset, segment_ids=[first.id], session=sqlite_session)

        assert sqlite_session.get(DocumentSegmentSummary, deleted.id) is None
        assert sqlite_session.get(DocumentSegmentSummary, kept.id) is not None
        cleanup = sqlite_session.scalar(
            select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "node-1")
        )
        assert cleanup is not None
        assert cleanup.vector_route_hash == _route().key


class TestReadModels:
    def test_getters_filter_dataset_document_segments_and_enabled(self, sqlite_session: Session) -> None:
        dataset, document, first = _graph(sqlite_session)
        second = _persist_segment(sqlite_session, dataset, document, segment_id="seg-2", position=2)
        first_summary = _persist_summary(sqlite_session, dataset, document, first, status=SummaryStatus.COMPLETED)
        _persist_summary(sqlite_session, dataset, document, second, summary_id="sum-2", enabled=False)
        other_dataset = _persist_dataset(sqlite_session, dataset_id="dataset-2")
        other_document = _persist_document(sqlite_session, other_dataset, document_id="doc-2")
        other_segment = _persist_segment(sqlite_session, other_dataset, other_document, segment_id="seg-3")
        _persist_summary(sqlite_session, other_dataset, other_document, other_segment, summary_id="sum-3")

        assert SummaryIndexAdapter.get_segment_summary(first.id, dataset.id, session=sqlite_session) is first_summary
        assert SummaryIndexAdapter.get_segment_summary(second.id, dataset.id, session=sqlite_session) is None
        assert SummaryIndexAdapter.get_segments_summaries(
            [first.id, second.id, other_segment.id], dataset.id, session=sqlite_session
        ) == {first.id: first_summary}
        assert SummaryIndexAdapter.get_segments_summaries([], dataset.id, session=sqlite_session) == {}
        assert SummaryIndexAdapter.get_document_summaries(
            document.id, dataset.id, [first.id, second.id], session=sqlite_session
        ) == [first_summary]

    def test_single_document_status_uses_real_pending_rows_and_tenant_scope(self, sqlite_session: Session) -> None:
        dataset, document, segment = _graph(sqlite_session)
        summary = _persist_summary(sqlite_session, dataset, document, segment, status=SummaryStatus.GENERATING)

        assert (
            SummaryIndexAdapter.get_document_summary_index_status(
                document.id, dataset.id, TENANT_ID, session=sqlite_session
            )
            == "SUMMARIZING"
        )
        summary.status = SummaryStatus.COMPLETED
        sqlite_session.commit()
        assert (
            SummaryIndexAdapter.get_document_summary_index_status(
                document.id, dataset.id, TENANT_ID, session=sqlite_session
            )
            is None
        )
        assert (
            SummaryIndexAdapter.get_document_summary_index_status(
                document.id, dataset.id, OTHER_TENANT_ID, session=sqlite_session
            )
            is None
        )

    def test_multiple_document_status_groups_real_rows(self, sqlite_session: Session) -> None:
        dataset = _persist_dataset(sqlite_session)
        first_doc = _persist_document(sqlite_session, dataset, document_id="doc-1")
        second_doc = _persist_document(sqlite_session, dataset, document_id="doc-2")
        first_segment = _persist_segment(sqlite_session, dataset, first_doc, segment_id="seg-1")
        second_segment = _persist_segment(sqlite_session, dataset, second_doc, segment_id="seg-2")
        _persist_summary(sqlite_session, dataset, first_doc, first_segment, status=SummaryStatus.NOT_STARTED)
        _persist_summary(
            sqlite_session,
            dataset,
            second_doc,
            second_segment,
            summary_id="sum-2",
            status=SummaryStatus.COMPLETED,
        )

        result = SummaryIndexAdapter.get_documents_summary_index_status(
            [first_doc.id, second_doc.id, "missing"], dataset.id, TENANT_ID, session=sqlite_session
        )

        assert result == {first_doc.id: "SUMMARIZING", second_doc.id: None, "missing": None}
        assert (
            SummaryIndexAdapter.get_documents_summary_index_status([], dataset.id, TENANT_ID, session=sqlite_session)
            == {}
        )

    def test_status_detail_counts_and_previews(self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset, document, first = _graph(sqlite_session)
        second = _persist_segment(sqlite_session, dataset, document, segment_id="seg-2", position=2)
        created_at = datetime(2026, 1, 1)
        _persist_summary(
            sqlite_session,
            dataset,
            document,
            first,
            content="x" * 150,
            status=SummaryStatus.COMPLETED,
            created_at=created_at,
        )
        segment_service = SimpleNamespace(get_segments_by_document_and_dataset=MagicMock(return_value=[first, second]))
        monkeypatch.setitem(
            sys.modules, "services.knowledge.dataset_service", SimpleNamespace(SegmentService=segment_service)
        )

        detail = SummaryIndexAdapter.get_document_summary_status_detail(
            document.id, dataset.id, sqlite_session, tenant_id=dataset.tenant_id
        )

        assert detail["total_segments"] == 2
        assert detail["summary_status"]["completed"] == 1
        assert detail["summary_status"]["not_started"] == 1
        assert detail["summaries"][0]["summary_preview"] == "x" * 100 + "..."
        assert detail["summaries"][0]["created_at"] == int(created_at.timestamp())
        assert detail["summaries"][1]["status"] == SummaryStatus.NOT_STARTED

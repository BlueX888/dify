from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import tasks.remove_document_from_index_task as task_module
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus, SegmentStatus, SummaryStatus
from models.model import SegmentStateOutbox, SummaryVectorCleanupOutbox
from services.knowledge.entities.vector_route import VectorRoute


@dataclass
class _RedisBoundary:
    deleted: list[str] = field(default_factory=list)

    def delete(self, key: str) -> None:
        self.deleted.append(key)


@dataclass
class _ProcessorBoundary:
    cleaned: list[tuple[str, ...]] = field(default_factory=list)
    before_return: Callable[[], None] | None = None

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
        if self.before_return is not None:
            self.before_return()


class _ProcessorFactoryBoundary:
    def __init__(self, processor: _ProcessorBoundary) -> None:
        self._processor = processor

    def __call__(self, _doc_form: str) -> "_ProcessorFactoryBoundary":
        return self

    def init_index_processor(self) -> _ProcessorBoundary:
        return self._processor


def _route(dataset: Dataset) -> VectorRoute:
    return VectorRoute.capture(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="summary-collection",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )


def _seed_document(
    session: Session,
    *,
    enabled: bool,
    archived: bool = False,
) -> tuple[Dataset, Document, DocumentSegment, DocumentSegmentSummary]:
    dataset = Dataset(
        id="dataset-1",
        tenant_id="tenant-1",
        name="Dataset",
        description="",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique="high_quality",
        created_by="account-1",
    )
    document = Document(
        id="document-1",
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="batch-1",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by="account-1",
        indexing_status=IndexingStatus.COMPLETED,
        completed_at=datetime(2026, 1, 1),
        enabled=enabled,
        archived=archived,
        doc_form="text_model",
        word_count=1,
    )
    segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="body",
        word_count=1,
        tokens=1,
        created_by="account-1",
        index_node_id="body-node-1",
        index_node_hash="body-hash-1",
        # Document status is committed before this asynchronous worker updates
        # its searchable children.
        enabled=True,
        status=SegmentStatus.COMPLETED,
    )
    segment.id = "segment-1"
    route = _route(dataset)
    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=segment.id,
        summary_content="summary",
        summary_index_node_id="summary-node-1",
        summary_index_node_hash="summary-hash-1",
        vector_route=route.serialized,
        vector_route_hash=route.key,
        status=SummaryStatus.COMPLETED,
        enabled=True,
    )
    summary.id = "summary-1"
    session.add_all([dataset, document, segment, summary])
    session.flush()
    return dataset, document, segment, summary


def _bind_task(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session_factory: sessionmaker[Session],
    processor: _ProcessorBoundary,
    redis: _RedisBoundary,
) -> None:
    monkeypatch.setattr(task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(task_module, "redis_client", redis)


def test_stale_remove_after_current_enable_performs_no_io(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        _, document, segment, summary = _seed_document(session, enabled=True)
        document_id, segment_id, summary_id = document.id, segment.id, summary.id
    processor = _ProcessorBoundary()
    redis = _RedisBoundary()
    _bind_task(monkeypatch, sqlite_session_factory, processor, redis)

    assert task_module.remove_document_from_index_task.run(document_id) is False

    assert processor.cleaned == []
    assert redis.deleted == []
    with sqlite_session_factory() as observer:
        assert observer.get_one(Document, document_id).enabled is True
        assert observer.get_one(DocumentSegment, segment_id).enabled is True
        assert observer.get_one(DocumentSegmentSummary, summary_id).enabled is True
        assert observer.scalar(select(SummaryVectorCleanupOutbox.id)) is None


def test_successful_remove_commits_summary_handoff_and_segment_state_together(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        _, document, segment, summary = _seed_document(session, enabled=False)
        document.disabled_by = "account-2"
        document_id, segment_id, summary_id = document.id, segment.id, summary.id
        summary_version = summary.version
    processor = _ProcessorBoundary()
    redis = _RedisBoundary()
    _bind_task(monkeypatch, sqlite_session_factory, processor, redis)

    assert task_module.remove_document_from_index_task.run(document_id) is True

    assert processor.cleaned == [("body-node-1",)]
    assert redis.deleted == [f"document_{document_id}_indexing"]
    with sqlite_session_factory() as observer:
        current_segment = observer.get_one(DocumentSegment, segment_id)
        current_summary = observer.get_one(DocumentSegmentSummary, summary_id)
        cleanup = observer.scalar(
            select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "summary-node-1")
        )
        assert current_segment.enabled is False
        assert current_segment.disabled_by == "account-2"
        assert current_summary.enabled is False
        assert current_summary.version == summary_version + 1
        assert current_summary.disabled_by == "account-2"
        assert cleanup is not None
        assert cleanup.vector_route_hash == current_summary.vector_route_hash


def test_remove_failure_does_not_reenable_document(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        _, document, segment, summary = _seed_document(session, enabled=False)
        document_id, segment_id, summary_id = document.id, segment.id, summary.id

    def fail_summary_handoff(
        _session: Session,
        *,
        tenant_id: str,
        summaries: Sequence[DocumentSegmentSummary],
    ) -> None:
        assert tenant_id == "tenant-1"
        assert [row.id for row in summaries] == [summary_id]
        raise RuntimeError("handoff failed")

    processor = _ProcessorBoundary()
    redis = _RedisBoundary()
    _bind_task(monkeypatch, sqlite_session_factory, processor, redis)
    monkeypatch.setattr(
        task_module.SQLAlchemySummaryVectorCleanupRepository,
        "add_terminal_cleanups",
        fail_summary_handoff,
    )

    assert task_module.remove_document_from_index_task.run(document_id) is False

    assert processor.cleaned == []
    with sqlite_session_factory() as observer:
        assert observer.get_one(Document, document_id).enabled is False
        assert observer.get_one(DocumentSegment, segment_id).enabled is True
        assert observer.get_one(DocumentSegmentSummary, summary_id).enabled is True
        assert observer.scalar(select(SummaryVectorCleanupOutbox.id)) is None


@pytest.mark.parametrize("claim_expires_at", [None, datetime(2099, 1, 1)])
def test_active_document_indexing_claim_defers_remove_without_io(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    claim_expires_at: datetime | None,
) -> None:
    with sqlite_session_factory.begin() as session:
        _, document, _, _ = _seed_document(session, enabled=False)
        document.indexing_claim_id = "claim-1"
        document.indexing_claim_expires_at = claim_expires_at
        document_id = document.id
    processor = _ProcessorBoundary()
    redis = _RedisBoundary()
    _bind_task(monkeypatch, sqlite_session_factory, processor, redis)

    assert task_module.remove_document_from_index_task.run(document_id) is False

    assert processor.cleaned == []
    assert redis.deleted == []


def test_current_enable_after_serialized_remove_preserves_desired_state_and_intent(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        _, document, segment, summary = _seed_document(session, enabled=False)
        document_id, segment_id, summary_id = document.id, segment.id, summary.id
    processor = _ProcessorBoundary()
    redis = _RedisBoundary()
    _bind_task(monkeypatch, sqlite_session_factory, processor, redis)

    assert task_module.remove_document_from_index_task.run(document_id) is True

    # SQLite does not implement SELECT FOR UPDATE. Apply the state transition in
    # the order guaranteed by the production lock protocol: enable waits for the
    # remove owner to commit, then persists the desired state and durable intent.
    with sqlite_session_factory.begin() as enable:
        current_document = enable.get_one(Document, document_id)
        current_segment = enable.get_one(DocumentSegment, segment_id)
        current_summary = enable.get_one(DocumentSegmentSummary, summary_id)
        current_document.enabled = True
        current_document.archived = False
        current_segment.enabled = True
        current_summary.enabled = True
        intent = SegmentStateOutbox(
            tenant_id=current_document.tenant_id,
            dataset_id=current_document.dataset_id,
            document_id=current_document.id,
            segment_id=current_segment.id,
            action="enable",
            actor_id="account-2",
            summary_id=current_summary.id,
            summary_version=current_summary.version,
        )
        current_segment.indexing_request_id = intent.id
        enable.add(intent)
        intent_id = intent.id

    assert processor.cleaned == [("body-node-1",)]
    with sqlite_session_factory() as observer:
        current_document = observer.get_one(Document, document_id)
        current_segment = observer.get_one(DocumentSegment, segment_id)
        current_summary = observer.get_one(DocumentSegmentSummary, summary_id)
        assert current_document.enabled is True
        assert current_document.archived is False
        assert current_segment.enabled is True
        assert current_segment.indexing_request_id == intent_id
        assert current_summary.enabled is True
        assert observer.get(SegmentStateOutbox, intent_id) is not None

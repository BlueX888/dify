from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import services.knowledge.dataset_service as dataset_service_module
import tasks.segment_state_outbox_task as state_task_module
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus, SegmentStatus, SummaryStatus
from models.model import SegmentStateOutbox, SummaryVectorCleanupOutbox
from services.knowledge.dataset_service import DatasetService, DocumentService
from services.knowledge.entities.vector_route import VectorRoute
from tasks.segment_state_outbox_task import (
    drain_segment_state_outbox,
    process_segment_state_intent,
    process_segment_state_intents,
)
from tests.unit_tests.model_factories import make_account, make_tenant


@dataclass
class _RedisBoundary:
    cached: list[tuple[str, int, int]] = field(default_factory=list)

    def get(self, _key: str) -> None:
        return None

    def setex(self, key: str, ttl: int, value: int) -> None:
        self.cached.append((key, ttl, value))


@dataclass
class _AddTaskBoundary:
    document_ids: list[str] = field(default_factory=list)

    def delay(self, document_id: str) -> None:
        self.document_ids.append(document_id)


@dataclass
class _ProcessorBoundary:
    cleaned: list[tuple[str, ...]] = field(default_factory=list)
    vector_nodes: set[str] = field(default_factory=set)

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
        if node_ids:
            self.vector_nodes.difference_update(node_ids)
        else:
            self.vector_nodes.clear()


class _ProcessorFactoryBoundary:
    def __init__(self, processor: _ProcessorBoundary) -> None:
        self._processor = processor

    def __call__(self, *_args: object, **_kwargs: object) -> "_ProcessorFactoryBoundary":
        return self

    def init_index_processor(self) -> _ProcessorBoundary:
        return self._processor


def _seed_graph(session: Session) -> tuple[Dataset, Document, DocumentSegment, DocumentSegmentSummary, object]:
    tenant = make_tenant(tenant_id="tenant-1", name="Tenant")
    account = make_account(
        account_id="account-1",
        name="Account",
        email="account@example.com",
        tenant=tenant,
    )
    dataset = Dataset(
        id="dataset-1",
        tenant_id=tenant.id,
        name="Dataset",
        description="",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique="high_quality",
        created_by=account.id,
    )
    document = Document(
        id="document-1",
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="batch-1",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by=account.id,
        indexing_status=IndexingStatus.COMPLETED,
        completed_at=datetime(2026, 1, 1),
        enabled=True,
        archived=False,
        doc_form="text_model",
        word_count=1,
    )
    segment = DocumentSegment(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="body",
        word_count=1,
        tokens=1,
        created_by=account.id,
        index_node_id="body-node-1",
        index_node_hash="body-hash-1",
        enabled=True,
        status=SegmentStatus.COMPLETED,
    )
    segment.id = "segment-1"
    route = VectorRoute.capture(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="summary-collection",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    # A disabled summary may still own a Phase-2 node and must participate in
    # the newer document-disable generation.
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
        enabled=False,
        version=2,
    )
    summary.id = "summary-1"
    session.add_all([tenant, account, dataset, document, segment, summary])
    session.commit()
    return dataset, document, segment, summary, account


@pytest.mark.parametrize("action", ["disable", "archive"])
def test_broker_rejection_and_noop_retry_leave_disable_for_drainer(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    action: Literal["disable", "archive"],
) -> None:
    dataset, document, segment, summary, account = _seed_graph(sqlite_session)
    redis = _RedisBoundary()
    publish_attempts: list[tuple[str, ...]] = []

    def reject_publish(intent_ids: Sequence[str]) -> None:
        publish_attempts.append(tuple(intent_ids))
        raise OSError("broker unavailable")

    monkeypatch.setattr(dataset_service_module, "redis_client", redis)
    monkeypatch.setattr(dataset_service_module, "publish_segment_state_intents", reject_publish)

    DocumentService.batch_update_document_status(dataset, [document.id], action, account, sqlite_session)

    with sqlite_session_factory() as observer:
        current_document = observer.get_one(Document, document.id)
        current_segment = observer.get_one(DocumentSegment, segment.id)
        intent_id = current_segment.indexing_request_id
        assert intent_id is not None
        intent = observer.get_one(SegmentStateOutbox, intent_id)
        assert intent.action == "disable"
        assert current_segment.enabled is False
        assert current_document.enabled is (action == "archive")
        assert current_document.archived is (action == "archive")
        assert observer.get_one(DocumentSegmentSummary, summary.id).version == summary.version
    assert publish_attempts == [(intent_id,)]
    assert redis.cached == []

    # Repeating the same PATCH is a no-op, but the original durable intent is
    # still discoverable by the drainer.
    DocumentService.batch_update_document_status(dataset, [document.id], action, account, sqlite_session)
    assert publish_attempts == [(intent_id,)]

    processor = _ProcessorBoundary()
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    drained: list[tuple[str, ...]] = []

    def consume(intent_ids: Sequence[str]) -> None:
        drained.append(tuple(intent_ids))
        assert process_segment_state_intents(intent_ids, new_session=sqlite_session_factory) == {intent_id: True}

    drain_segment_state_outbox(new_session=sqlite_session_factory, publish=consume)

    assert drained == [(intent_id,)]
    assert processor.cleaned == [("body-node-1",)]
    with sqlite_session_factory() as observer:
        current_segment = observer.get_one(DocumentSegment, segment.id)
        current_summary = observer.get_one(DocumentSegmentSummary, summary.id)
        cleanup = observer.scalar(
            select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "summary-node-1")
        )
        assert current_segment.enabled is False
        assert current_segment.indexing_request_id is None
        assert current_summary.enabled is False
        assert current_summary.version == summary.version + 1
        assert cleanup is not None
        assert observer.get(SegmentStateOutbox, intent_id) is None


def test_later_document_enable_replaces_disable_generation(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document, segment, _summary, account = _seed_graph(sqlite_session)
    redis = _RedisBoundary()

    def reject_publish(_intent_ids: Sequence[str]) -> None:
        raise OSError("broker unavailable")

    monkeypatch.setattr(dataset_service_module, "redis_client", redis)
    monkeypatch.setattr(dataset_service_module, "publish_segment_state_intents", reject_publish)
    DocumentService.batch_update_document_status(dataset, [document.id], "disable", account, sqlite_session)
    with sqlite_session_factory() as observer:
        old_intent_id = observer.get_one(DocumentSegment, segment.id).indexing_request_id
        assert old_intent_id is not None

    add_task = _AddTaskBoundary()
    checked: list[str] = []

    def check_model(current: Dataset) -> None:
        checked.append(current.id)

    monkeypatch.setattr(dataset_service_module, "add_document_to_index_task", add_task)
    monkeypatch.setattr(DatasetService, "check_dataset_model_setting", check_model)
    DocumentService.batch_update_document_status(dataset, [document.id], "enable", account, sqlite_session)

    with sqlite_session_factory() as observer:
        current_document = observer.get_one(Document, document.id)
        current_segment = observer.get_one(DocumentSegment, segment.id)
        new_intent_id = current_segment.indexing_request_id
        assert new_intent_id is not None
        assert new_intent_id != old_intent_id
        assert current_document.enabled is True
        assert current_segment.enabled is True
        assert observer.get_one(SegmentStateOutbox, new_intent_id).action == "enable"
        assert observer.get(SegmentStateOutbox, old_intent_id) is None
    assert checked == [dataset.id]
    assert add_task.document_ids == [document.id]

    processor = _ProcessorBoundary()
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    assert process_segment_state_intent(old_intent_id, new_session=sqlite_session_factory) is False
    assert processor.cleaned == []
    with sqlite_session_factory() as observer:
        assert observer.get_one(DocumentSegment, segment.id).indexing_request_id == new_intent_id
        assert observer.get(SegmentStateOutbox, new_intent_id) is not None


def test_incomplete_archive_reconciles_error_segment_without_body_node_and_its_summary(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document, segment, summary, account = _seed_graph(sqlite_session)
    document.indexing_status = IndexingStatus.ERROR
    document.completed_at = None
    segment.status = SegmentStatus.ERROR
    segment.index_node_id = None
    sqlite_session.commit()
    published: list[tuple[str, ...]] = []

    def publish(intent_ids: Sequence[str]) -> None:
        published.append(tuple(intent_ids))

    monkeypatch.setattr(dataset_service_module, "redis_client", _RedisBoundary())
    monkeypatch.setattr(dataset_service_module, "publish_segment_state_intents", publish)

    DocumentService.batch_update_document_status(dataset, [document.id], "archive", account, sqlite_session)

    with sqlite_session_factory() as observer:
        current_document = observer.get_one(Document, document.id)
        current_segment = observer.get_one(DocumentSegment, segment.id)
        intent_id = current_segment.indexing_request_id
        assert current_document.archived is True
        assert current_segment.enabled is False
        assert intent_id is not None
        assert observer.get_one(SegmentStateOutbox, intent_id).action == "disable"
    assert published == [(intent_id,)]

    processor = _ProcessorBoundary(vector_nodes={"other-document-node"})
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    assert process_segment_state_intent(intent_id, new_session=sqlite_session_factory) is True

    assert processor.cleaned == []
    assert processor.vector_nodes == {"other-document-node"}
    with sqlite_session_factory() as observer:
        current_summary = observer.get_one(DocumentSegmentSummary, summary.id)
        cleanup = observer.scalar(
            select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "summary-node-1")
        )
        assert current_summary.enabled is False
        assert current_summary.version == summary.version + 1
        assert cleanup is not None
        assert observer.get(SegmentStateOutbox, intent_id) is None

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import override

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import tasks.add_document_to_index_task as add_task_module
import tasks.segment_state_outbox_task as state_task_module
import tasks.summary_vector_cleanup_task as summary_cleanup_task_module
from core.rag.models.document import Document as IndexDocument
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus, SegmentStatus, SummaryStatus
from models.model import DatasetIndexingOutbox, SegmentStateOutbox, SummaryVectorCleanupOutbox
from repositories.knowledge.segment_repository import SQLAlchemySegmentRepository
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository
from services.knowledge.dataset_service import DatasetService, DocumentService
from services.knowledge.entities.datasets import DatasetIndexingRepresentation
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.summaries import (
    SummaryPublicationResult,
    SummaryVectorCleanupClaim,
    SummaryVectorInput,
    SummaryVectorResult,
)
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.segments.adapters import SQLAlchemySegmentStateTransitions
from services.knowledge.summaries.adapters import SummaryIndexAdapter
from tasks.segment_state_outbox_task import process_segment_state_intent, process_segment_state_intents
from tests.unit_tests.model_factories import make_account, make_tenant


@dataclass
class _RedisBoundary:
    cached: list[tuple[str, int, int]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def get(self, _key: str) -> None:
        return None

    def setex(self, key: str, ttl: int, value: int) -> None:
        self.cached.append((key, ttl, value))

    def delete(self, key: str) -> None:
        self.deleted.append(key)


@dataclass
class _DocumentTaskBoundary:
    document_ids: list[str] = field(default_factory=list)

    def delay(self, document_id: str) -> None:
        self.document_ids.append(document_id)


@dataclass
class _ProcessorBoundary:
    loaded: list[tuple[str, ...]] = field(default_factory=list)
    cleaned: list[tuple[str, ...]] = field(default_factory=list)

    def load(
        self,
        _dataset: Dataset,
        documents: list[IndexDocument],
        *,
        multimodal_documents: list[IndexDocument],
        session: Session,
    ) -> None:
        del multimodal_documents
        assert session.in_transaction()
        self.loaded.append(tuple(document.page_content for document in documents))

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


class _ProcessorFactoryBoundary:
    def __init__(self, processor: _ProcessorBoundary) -> None:
        self._processor = processor

    def __call__(self, *_args: object, **_kwargs: object) -> "_ProcessorFactoryBoundary":
        return self

    def init_index_processor(self) -> _ProcessorBoundary:
        return self._processor


@dataclass
class _VectorBoundary:
    added: list[IndexDocument] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    delete_failures_remaining: int = 0

    def get_routing_target(self) -> tuple[str, str]:
        return "qdrant", "current-summary-collection"

    def add_texts(
        self,
        documents: list[IndexDocument],
        *,
        duplicate_check: bool,
        before_write: Callable[[], None] | None = None,
    ) -> None:
        assert duplicate_check is False
        if before_write is not None:
            before_write()
        self.added.extend(documents)

    def delete_by_ids(self, node_ids: list[str]) -> None:
        if self.delete_failures_remaining:
            self.delete_failures_remaining -= 1
            raise RuntimeError("vector delete unavailable")
        self.deleted.extend(node_ids)


class _VectorConstructorBoundary:
    def __init__(self, vector: _VectorBoundary) -> None:
        self._vector = vector

    def __call__(self, _dataset: Dataset, *, session: Session | None) -> _VectorBoundary:
        if session is not None:
            assert session.in_transaction()
        return self._vector


class _SelectiveFailureVectorBoundary(_VectorBoundary):
    def __init__(self, *, failing_content: str) -> None:
        super().__init__()
        self.failing_content = failing_content
        self.attempted: list[IndexDocument] = []

    @override
    def add_texts(
        self,
        documents: list[IndexDocument],
        *,
        duplicate_check: bool,
        before_write: Callable[[], None] | None = None,
    ) -> None:
        assert duplicate_check is False
        if before_write is not None:
            before_write()
        self.attempted.extend(documents)
        if any(document.page_content == self.failing_content for document in documents):
            raise RuntimeError("permanent summary write failure")
        self.added.extend(documents)


@dataclass
class _GenerationVectorState:
    added: list[IndexDocument] = field(default_factory=list)


class _GenerationVectorBoundary:
    def __init__(self, generation_id: str | None, state: _GenerationVectorState) -> None:
        self._generation_id = generation_id
        self._state = state

    def get_routing_target(self) -> tuple[str, str]:
        return "qdrant", f"collection-{self._generation_id or 'legacy'}"

    def add_texts(
        self,
        documents: list[IndexDocument],
        *,
        duplicate_check: bool,
        before_write: Callable[[], None] | None = None,
    ) -> None:
        assert duplicate_check is False
        if before_write is not None:
            before_write()
        self._state.added.extend(documents)


class _GenerationVectorConstructorBoundary:
    def __init__(self, state: _GenerationVectorState) -> None:
        self._state = state

    def __call__(self, dataset: Dataset, *, session: Session | None) -> _GenerationVectorBoundary:
        if session is not None:
            assert session.in_transaction()
        return _GenerationVectorBoundary(dataset.indexing_request_id, self._state)


def _indexing_representation(dataset: Dataset, route: VectorRoute) -> DatasetIndexingRepresentation:
    return DatasetIndexingRepresentation(
        indexing_technique="high_quality",
        index_struct=route.index_struct,
        collection_binding_id=None,
        embedding_model=dataset.embedding_model,
        embedding_model_provider=dataset.embedding_model_provider,
        keyword_number=dataset.keyword_number,
        is_multimodal=dataset.is_multimodal,
        doc_form="text_model",
        summary_index_setting=dataset.summary_index_setting,
        vector_route=route,
    )


@dataclass(frozen=True)
class _PendingSummaryEnable:
    tenant_id: str
    dataset_id: str
    document_id: str
    segment_id: str
    summary_id: str
    intent_id: str


def _seed_pending_summary_enable(
    sqlite_session_factory: sessionmaker[Session], *, suffix: str
) -> _PendingSummaryEnable:
    tenant_id = f"tenant-{suffix}"
    dataset_id = f"dataset-{suffix}"
    document_id = f"document-{suffix}"
    segment_id = f"segment-{suffix}"
    summary_id = f"summary-{suffix}"
    route = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name=f"summary-collection-{suffix}",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    with sqlite_session_factory.begin() as session:
        dataset = Dataset(
            id=dataset_id,
            tenant_id=tenant_id,
            name="Dataset",
            description="",
            provider="vendor",
            data_source_type=DataSourceType.UPLOAD_FILE,
            indexing_technique="high_quality",
            created_by=f"account-{suffix}",
            indexing_request_id=None,
        )
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            data_source_info="{}",
            batch=f"batch-{suffix}",
            name="Document",
            created_from=DocumentCreatedFrom.API,
            created_by=dataset.created_by,
            indexing_status=IndexingStatus.COMPLETED,
            completed_at=datetime(2026, 1, 1),
            enabled=True,
            archived=False,
            doc_form="text_model",
            word_count=2,
        )
        segment = DocumentSegment(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            position=1,
            content="body content",
            word_count=2,
            tokens=2,
            created_by=dataset.created_by,
            index_node_id=f"body-node-{suffix}",
            index_node_hash=f"body-hash-{suffix}",
            enabled=False,
            status=SegmentStatus.COMPLETED,
        )
        segment.id = segment_id
        summary = DocumentSegmentSummary(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_id=segment_id,
            summary_content="old summary",
            summary_index_node_id=f"old-summary-node-{suffix}",
            summary_index_node_hash=f"old-summary-hash-{suffix}",
            vector_route=route.serialized,
            vector_route_hash=route.key,
            status=SummaryStatus.COMPLETED,
            enabled=False,
        )
        summary.id = summary_id
        session.add_all([dataset, document, segment, summary])

    transition = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory).request(
        document_ref=DatasetRef(tenant_id, dataset_id).document(document_id),
        segment_ids=[segment_id],
        action="enable",
        actor_id=f"account-{suffix}",
    )[0]
    return _PendingSummaryEnable(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        segment_id=segment_id,
        summary_id=summary_id,
        intent_id=transition.id,
    )


def test_manual_summary_edit_advances_pending_enable_to_the_new_revision(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _seed_pending_summary_enable(sqlite_session_factory, suffix="manual-enable")
    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(owner.tenant_id, owner.dataset_id).document(owner.document_id).segment(owner.segment_id)

    write = repository.save_summary(ref, "new manual summary")
    assert write.summary is not None
    assert write.inherited_enable_intent_id == owner.intent_id
    vector = _VectorBoundary()
    with sqlite_session_factory() as session:
        dataset = session.get_one(Dataset, owner.dataset_id)
        segment = session.get_one(DocumentSegment, owner.segment_id)
        session.expunge_all()
    assert SummaryIndexAdapter.vectorize_summary(
        write.summary,
        segment,
        dataset,
        new_session=sqlite_session_factory,
        vector_factory=lambda: vector,
        indexing_request_id=dataset.indexing_request_id,
        expected_segment_state_intent_id=owner.intent_id,
    )

    with sqlite_session_factory() as observer:
        pending_intent = observer.get_one(SegmentStateOutbox, owner.intent_id)
        pending_summary = observer.get_one(DocumentSegmentSummary, owner.summary_id)
        assert (pending_intent.summary_id, pending_intent.summary_version) == (
            pending_summary.id,
            pending_summary.version,
        )
        assert pending_summary.summary_content == "new manual summary"
        assert pending_summary.status == SummaryStatus.COMPLETED
        assert pending_summary.enabled is False

    processor = _ProcessorBoundary()
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(state_task_module, "Vector", _VectorConstructorBoundary(vector))
    assert process_segment_state_intent(owner.intent_id, new_session=sqlite_session_factory) is True

    with sqlite_session_factory() as observer:
        enabled_summary = observer.get_one(DocumentSegmentSummary, owner.summary_id)
        enabled_segment = observer.get_one(DocumentSegment, owner.segment_id)
        assert enabled_summary.summary_content == "new manual summary"
        assert enabled_summary.enabled is True
        assert enabled_summary.status == SummaryStatus.COMPLETED
        assert enabled_segment.indexing_request_id is None
        assert observer.get(SegmentStateOutbox, owner.intent_id) is None


def test_old_enable_worker_cannot_ack_an_intent_advanced_by_manual_edit(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _seed_pending_summary_enable(sqlite_session_factory, suffix="old-worker")
    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(owner.tenant_id, owner.dataset_id).document(owner.document_id).segment(owner.segment_id)
    processor = _ProcessorBoundary()
    vector = _VectorBoundary()
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(state_task_module, "Vector", _VectorConstructorBoundary(vector))
    publish = state_task_module._publish_enable_summaries
    edited = False

    def publish_after_edit(
        publication: state_task_module._EnableBatchPublication,
        *,
        active_intent_ids: frozenset[str],
        new_session: sessionmaker[Session],
    ) -> frozenset[str]:
        nonlocal edited
        if not edited:
            edited = True
            write = repository.save_summary(ref, "newer manual summary")
            assert write.inherited_enable_intent_id == owner.intent_id
        return publish(publication, active_intent_ids=active_intent_ids, new_session=new_session)

    monkeypatch.setattr(state_task_module, "_publish_enable_summaries", publish_after_edit)
    with pytest.raises(RuntimeError, match=f"Summary intent {owner.intent_id} changed after publication preflight"):
        process_segment_state_intent(owner.intent_id, new_session=sqlite_session_factory)

    with sqlite_session_factory() as observer:
        retained_intent = observer.get_one(SegmentStateOutbox, owner.intent_id)
        retained_summary = observer.get_one(DocumentSegmentSummary, owner.summary_id)
        assert retained_intent.summary_version == retained_summary.version
        assert retained_summary.summary_content == "newer manual summary"
        assert retained_summary.enabled is False
        assert observer.get_one(DocumentSegment, owner.segment_id).indexing_request_id == owner.intent_id

    assert process_segment_state_intent(owner.intent_id, new_session=sqlite_session_factory) is True
    with sqlite_session_factory() as observer:
        assert observer.get_one(DocumentSegmentSummary, owner.summary_id).enabled is True
        assert observer.get(SegmentStateOutbox, owner.intent_id) is None


@pytest.mark.parametrize("disable_before_edit", [False, True], ids=["edit-then-disable", "disable-then-edit"])
def test_explicit_summary_disable_wins_over_manual_edit_and_pending_enable(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    disable_before_edit: bool,
) -> None:
    owner = _seed_pending_summary_enable(
        sqlite_session_factory,
        suffix="disable-before" if disable_before_edit else "disable-after",
    )
    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(owner.tenant_id, owner.dataset_id).document(owner.document_id).segment(owner.segment_id)

    def disable_summary() -> None:
        with sqlite_session_factory() as session:
            dataset = session.get_one(Dataset, owner.dataset_id)
            session.expunge(dataset)
        SummaryIndexAdapter.disable_summaries_for_segments(
            dataset,
            segment_ids=[owner.segment_id],
            disabled_by="explicit-actor",
        )

    if disable_before_edit:
        disable_summary()
    write = repository.save_summary(ref, "manual content after explicit choice")
    if not disable_before_edit:
        assert write.inherited_enable_intent_id == owner.intent_id
        disable_summary()
    else:
        assert write.inherited_enable_intent_id is None

    processor = _ProcessorBoundary()
    vector = _VectorBoundary()
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(state_task_module, "Vector", _VectorConstructorBoundary(vector))
    assert process_segment_state_intent(owner.intent_id, new_session=sqlite_session_factory) is True

    assert vector.added == []
    with sqlite_session_factory() as observer:
        current_summary = observer.get_one(DocumentSegmentSummary, owner.summary_id)
        assert current_summary.summary_content == "manual content after explicit choice"
        assert current_summary.enabled is False
        assert current_summary.disabled_by == "explicit-actor"
        assert observer.get(SegmentStateOutbox, owner.intent_id) is None


@pytest.mark.parametrize("explicit_summary_disable", [False, True], ids=["enable-wins", "newer-disable-wins"])
def test_document_enable_handoff_survives_dataset_generation_switch(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    explicit_summary_disable: bool,
) -> None:
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
        indexing_request_id="generation-0",
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
        enabled=False,
        archived=False,
        doc_form="text_model",
        word_count=1,
    )
    segment = DocumentSegment(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="body content",
        word_count=2,
        tokens=2,
        created_by=account.id,
        index_node_id="body-node-1",
        index_node_hash="body-hash-1",
        enabled=False,
        status=SegmentStatus.COMPLETED,
    )
    segment.id = "segment-1"
    old_route = VectorRoute.capture(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="old-summary-collection",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=segment.id,
        summary_content="retained summary",
        summary_index_node_id="summary-node-1",
        summary_index_node_hash="summary-hash-1",
        vector_route=old_route.serialized,
        vector_route_hash=old_route.key,
        status=SummaryStatus.COMPLETED,
        enabled=False,
    )
    summary.id = "summary-1"
    sqlite_session.add_all([tenant, account, dataset, document, segment, summary])
    sqlite_session.commit()
    redis = _RedisBoundary()
    dispatched = _DocumentTaskBoundary()
    monkeypatch.setattr("services.knowledge.dataset_service.redis_client", redis)
    monkeypatch.setattr("services.knowledge.dataset_service.add_document_to_index_task", dispatched)

    def accept_model_setting(_dataset: Dataset) -> None:
        pass

    monkeypatch.setattr(DatasetService, "check_dataset_model_setting", accept_model_setting)

    DocumentService.batch_update_document_status(dataset, [document.id], "enable", account, sqlite_session)

    assert dispatched.document_ids == [document.id]
    with sqlite_session_factory.begin() as switch_generation:
        switch_generation.get_one(Dataset, dataset.id).indexing_request_id = None
    with sqlite_session_factory() as observer:
        current_segment = observer.get_one(DocumentSegment, segment.id)
        intent_id = current_segment.indexing_request_id
        assert intent_id is not None
        intent = observer.get_one(SegmentStateOutbox, intent_id)
        assert (intent.summary_id, intent.summary_version) == (summary.id, summary.version)

    if explicit_summary_disable:
        SummaryIndexAdapter.disable_summaries_for_segments(
            dataset,
            segment_ids=[segment.id],
            disabled_by=account.id,
        )

    forwarded: list[tuple[str, ...]] = []
    monkeypatch.setattr(add_task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(add_task_module, "redis_client", redis)
    monkeypatch.setattr(add_task_module, "publish_segment_state_intents", lambda ids: forwarded.append(tuple(ids)))
    assert add_task_module.add_document_to_index_task.run(document.id) is False
    assert forwarded == [(intent_id,)]
    with sqlite_session_factory() as observer:
        assert observer.get(SegmentStateOutbox, intent_id) is not None

    processor = _ProcessorBoundary()
    vector = _VectorBoundary()
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(state_task_module, "Vector", _VectorConstructorBoundary(vector))

    assert process_segment_state_intent(intent_id, new_session=sqlite_session_factory) is True

    assert processor.loaded == [("body content",)]
    assert [document.page_content for document in vector.added] == (
        [] if explicit_summary_disable else ["retained summary"]
    )
    with sqlite_session_factory() as observer:
        current_segment = observer.get_one(DocumentSegment, segment.id)
        current_summary = observer.get_one(DocumentSegmentSummary, summary.id)
        assert current_segment.enabled is True
        assert current_segment.indexing_request_id is None
        assert current_summary.enabled is not explicit_summary_disable
        assert current_summary.version == 2
        assert current_summary.vector_route_hash is not None
        assert (current_summary.vector_route_hash != old_route.key) is not explicit_summary_disable
        assert observer.get(SegmentStateOutbox, intent_id) is None


def test_segment_reenable_rebuilds_failed_summary_without_previous_node(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset(
        id="dataset-summary-node-missing",
        tenant_id="tenant-summary-node-missing",
        name="Dataset",
        description="",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique="high_quality",
        created_by="account-summary-node-missing",
        indexing_request_id=None,
    )
    document = Document(
        id="document-summary-node-missing",
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="batch-summary-node-missing",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by=dataset.created_by,
        indexing_status=IndexingStatus.COMPLETED,
        completed_at=datetime(2026, 1, 1),
        enabled=True,
        archived=False,
        doc_form="text_model",
        word_count=1,
    )
    segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="body content",
        word_count=2,
        tokens=2,
        created_by=dataset.created_by,
        index_node_id="body-node-summary-node-missing",
        index_node_hash="body-hash-summary-node-missing",
        enabled=True,
        status=SegmentStatus.COMPLETED,
    )
    segment.id = "segment-summary-node-missing"
    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=segment.id,
        summary_content="retained summary after failed vectorization",
        summary_index_node_id=None,
        summary_index_node_hash=None,
        status=SummaryStatus.ERROR,
        error="previous vectorization failed",
        enabled=True,
    )
    summary.id = "summary-node-missing"
    sqlite_session.add_all([dataset, document, segment, summary])
    sqlite_session.commit()

    processor = _ProcessorBoundary()
    vector = _VectorBoundary()
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(state_task_module, "Vector", _VectorConstructorBoundary(vector))
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    document_ref = DatasetRef(dataset.tenant_id, dataset.id).document(document.id)

    disable_intent = transitions.request(
        document_ref=document_ref,
        segment_ids=[segment.id],
        action="disable",
        actor_id=dataset.created_by,
    )[0]
    assert process_segment_state_intent(disable_intent.id, new_session=sqlite_session_factory) is True
    with sqlite_session_factory() as observer:
        disabled_summary = observer.get_one(DocumentSegmentSummary, summary.id)
        assert disabled_summary.status == SummaryStatus.ERROR
        assert disabled_summary.summary_content == summary.summary_content
        assert disabled_summary.summary_index_node_id is None
        assert disabled_summary.enabled is False
        assert disabled_summary.version == 2

    enable_intent = transitions.request(
        document_ref=document_ref,
        segment_ids=[segment.id],
        action="enable",
        actor_id=dataset.created_by,
    )[0]
    with sqlite_session_factory() as observer:
        persisted_intent = observer.get_one(SegmentStateOutbox, enable_intent.id)
        assert (persisted_intent.summary_id, persisted_intent.summary_version) == (summary.id, 2)

    assert process_segment_state_intent(enable_intent.id, new_session=sqlite_session_factory) is True

    assert processor.cleaned == [(segment.index_node_id,)]
    assert processor.loaded == [(segment.content,)]
    assert [document.page_content for document in vector.added] == [summary.summary_content]
    with sqlite_session_factory() as observer:
        current_segment = observer.get_one(DocumentSegment, segment.id)
        current_summary = observer.get_one(DocumentSegmentSummary, summary.id)
        assert current_segment.enabled is True
        assert current_segment.indexing_request_id is None
        assert current_summary.status == SummaryStatus.COMPLETED
        assert current_summary.error is None
        assert current_summary.summary_index_node_id is not None
        assert current_summary.summary_index_node_hash is not None
        assert current_summary.vector_route_hash is not None
        assert current_summary.enabled is True
        assert current_summary.version == 3
        assert observer.get(SegmentStateOutbox, enable_intent.id) is None
        assert observer.scalar(select(SummaryVectorCleanupOutbox.id)) is None


def test_summary_enable_commits_successful_sibling_when_later_publication_fails(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset(
        id="dataset-summary-partial-failure",
        tenant_id="tenant-summary-partial-failure",
        name="Dataset",
        description="",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique="high_quality",
        created_by="account-summary-partial-failure",
        indexing_request_id=None,
    )
    document = Document(
        id="document-summary-partial-failure",
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="batch-summary-partial-failure",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by=dataset.created_by,
        indexing_status=IndexingStatus.COMPLETED,
        completed_at=datetime(2026, 1, 1),
        enabled=True,
        archived=False,
        doc_form="text_model",
        word_count=2,
    )
    segments = [
        DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=position,
            content=f"body content {position}",
            word_count=3,
            tokens=3,
            created_by=dataset.created_by,
            index_node_id=f"body-node-summary-partial-{position}",
            index_node_hash=f"body-hash-summary-partial-{position}",
            enabled=False,
            status=SegmentStatus.COMPLETED,
        )
        for position in (1, 2)
    ]
    for position, segment in enumerate(segments, start=1):
        segment.id = f"segment-summary-partial-{position}"
    first_summary_content = "first summary succeeds"
    failed_summary_content = "second summary fails"
    summaries = [
        DocumentSegmentSummary(
            dataset_id=dataset.id,
            document_id=document.id,
            chunk_id=segment.id,
            summary_content=content,
            summary_index_node_id=f"summary-node-partial-{position}",
            summary_index_node_hash=f"summary-hash-partial-{position}",
            status=SummaryStatus.COMPLETED,
            enabled=False,
        )
        for position, (segment, content) in enumerate(
            zip(segments, (first_summary_content, failed_summary_content), strict=True),
            start=1,
        )
    ]
    for position, summary in enumerate(summaries, start=1):
        summary.id = f"summary-partial-{position}"
    sqlite_session.add_all([dataset, document, *segments, *summaries])
    sqlite_session.commit()

    processor = _ProcessorBoundary()
    vector = _SelectiveFailureVectorBoundary(failing_content=failed_summary_content)
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(state_task_module, "Vector", _VectorConstructorBoundary(vector))
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    intent_refs = transitions.request(
        document_ref=DatasetRef(dataset.tenant_id, dataset.id).document(document.id),
        segment_ids=[segment.id for segment in segments],
        action="enable",
        actor_id=dataset.created_by,
    )
    intent_ids = tuple(intent.id for intent in intent_refs)
    assert len(intent_ids) == 2

    with pytest.raises(RuntimeError, match=f"Summary {summaries[1].id} publication remains pending"):
        process_segment_state_intents(intent_ids, new_session=sqlite_session_factory)

    assert [document.page_content for document in vector.attempted] == [
        first_summary_content,
        failed_summary_content,
    ]
    assert [document.page_content for document in vector.added] == [first_summary_content]
    with sqlite_session_factory() as observer:
        current_segments = [observer.get_one(DocumentSegment, segment.id) for segment in segments]
        current_summaries = [observer.get_one(DocumentSegmentSummary, summary.id) for summary in summaries]
        assert current_segments[0].indexing_request_id is None
        assert observer.get(SegmentStateOutbox, intent_ids[0]) is None
        assert current_summaries[0].enabled is True
        assert current_summaries[0].version == 2
        assert current_summaries[0].status == SummaryStatus.COMPLETED
        assert current_segments[1].indexing_request_id == intent_ids[1]
        assert observer.get(SegmentStateOutbox, intent_ids[1]) is not None
        assert current_summaries[1].enabled is False
        assert current_summaries[1].version == 1
        assert current_summaries[1].status == SummaryStatus.ERROR
        pending_writers = observer.scalars(
            select(SummaryVectorCleanupOutbox).where(
                SummaryVectorCleanupOutbox.tenant_id == dataset.tenant_id,
                SummaryVectorCleanupOutbox.segment_id == segments[1].id,
            )
        ).all()
        assert len(pending_writers) == 1
        assert pending_writers[0].writer_finished_at is None

    # Redelivery may continue reporting the failed item, but the acknowledged
    # sibling is absent from the owner batch and must never be published again.
    with pytest.raises(RuntimeError, match=f"Summary {summaries[1].id} publication remains pending"):
        process_segment_state_intents(intent_ids, new_session=sqlite_session_factory)
    assert [document.page_content for document in vector.attempted].count(first_summary_content) == 1
    with sqlite_session_factory() as observer:
        assert observer.get(SegmentStateOutbox, intent_ids[0]) is None
        assert observer.get_one(DocumentSegmentSummary, summaries[0].id).version == 2
        assert observer.get(SegmentStateOutbox, intent_ids[1]) is not None


def test_summary_enable_keeps_cleanup_evidence_when_vector_succeeds_before_publication_failure(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = make_tenant(tenant_id="tenant-summary-failure", name="Tenant")
    account = make_account(
        account_id="account-summary-failure",
        name="Account",
        email="summary-failure@example.com",
        tenant=tenant,
    )
    dataset = Dataset(
        id="dataset-summary-failure",
        tenant_id=tenant.id,
        name="Dataset",
        description="",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique="high_quality",
        created_by=account.id,
        indexing_request_id=None,
    )
    document = Document(
        id="document-summary-failure",
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="batch-summary-failure",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by=account.id,
        indexing_status=IndexingStatus.COMPLETED,
        completed_at=datetime(2026, 1, 1),
        enabled=False,
        archived=False,
        doc_form="text_model",
        word_count=1,
    )
    segment = DocumentSegment(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="body content",
        word_count=2,
        tokens=2,
        created_by=account.id,
        index_node_id="body-node-summary-failure",
        index_node_hash="body-hash-summary-failure",
        enabled=False,
        status=SegmentStatus.COMPLETED,
    )
    segment.id = "segment-summary-failure"
    old_route = VectorRoute.capture(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="summary-old",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    new_route = VectorRoute.capture(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="current-summary-collection",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=segment.id,
        summary_content="retained summary",
        summary_index_node_id="summary-node-failure",
        summary_index_node_hash="summary-hash-failure",
        vector_route=old_route.serialized,
        vector_route_hash=old_route.key,
        status=SummaryStatus.COMPLETED,
        enabled=False,
    )
    summary.id = "summary-failure"
    sqlite_session.add_all([tenant, account, dataset, document, segment, summary])
    sqlite_session.commit()

    redis = _RedisBoundary()
    dispatched = _DocumentTaskBoundary()
    monkeypatch.setattr("services.knowledge.dataset_service.redis_client", redis)
    monkeypatch.setattr("services.knowledge.dataset_service.add_document_to_index_task", dispatched)

    def accept_model_setting(_dataset: Dataset) -> None:
        pass

    monkeypatch.setattr(DatasetService, "check_dataset_model_setting", accept_model_setting)
    DocumentService.batch_update_document_status(dataset, [document.id], "enable", account, sqlite_session)
    with sqlite_session_factory() as observer:
        intent_id = observer.get_one(DocumentSegment, segment.id).indexing_request_id
    assert intent_id is not None

    processor = _ProcessorBoundary()
    vector = _VectorBoundary(delete_failures_remaining=1)
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(state_task_module, "Vector", _VectorConstructorBoundary(vector))

    def vector_for_route(route: VectorRoute) -> _VectorBoundary:
        assert route.key == new_route.key
        return vector

    monkeypatch.setattr("services.knowledge.summaries.adapters.vector_for_route", vector_for_route)

    def fail_summary_publication(
        _repository: SQLAlchemySummaryVectorCleanupRepository,
        _command: SummaryVectorInput,
        _result: SummaryVectorResult,
        _claim: SummaryVectorCleanupClaim,
    ) -> SummaryPublicationResult:
        raise RuntimeError("summary publication transaction failed")

    monkeypatch.setattr(SQLAlchemySummaryVectorCleanupRepository, "completed", fail_summary_publication)
    with pytest.raises(RuntimeError, match=f"Summary {summary.id} publication remains pending"):
        process_segment_state_intent(intent_id, new_session=sqlite_session_factory)

    # The body phase committed, but the exact segment intent and the successful
    # remote summary write both remain durable after the publication failure.
    with sqlite_session_factory() as observer:
        current_segment = observer.get_one(DocumentSegment, segment.id)
        current_summary = observer.get_one(DocumentSegmentSummary, summary.id)
        pending_new_route = observer.scalar(
            select(SummaryVectorCleanupOutbox).where(
                SummaryVectorCleanupOutbox.tenant_id == tenant.id,
                SummaryVectorCleanupOutbox.node_id == summary.summary_index_node_id,
                SummaryVectorCleanupOutbox.vector_route_hash == new_route.key,
            )
        )
        assert current_segment.enabled is True
        assert current_segment.indexing_request_id == intent_id
        assert current_summary.enabled is False
        assert current_summary.vector_route_hash == old_route.key
        assert pending_new_route is not None
        assert pending_new_route.writer_finished_at is not None
        cleanup_id = pending_new_route.id

    # A newer explicit disable supersedes the enable snapshot. The B-route node
    # is still independently recoverable rather than being orphaned by the skip.
    SummaryIndexAdapter.disable_summaries_for_segments(
        dataset,
        segment_ids=[segment.id],
        disabled_by=account.id,
    )
    cleanup_repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)

    def cleanup_repository_factory() -> SQLAlchemySummaryVectorCleanupRepository:
        return cleanup_repository

    vector.delete_failures_remaining = 0
    monkeypatch.setattr(summary_cleanup_task_module, "_repository", cleanup_repository_factory)
    monkeypatch.setattr(summary_cleanup_task_module, "vector_for_route", vector_for_route)
    summary_cleanup_task_module.process_summary_vector_cleanup_task.run(cleanup_id)

    assert vector.deleted == [summary.summary_index_node_id]
    with sqlite_session_factory() as observer:
        current_summary = observer.get_one(DocumentSegmentSummary, summary.id)
        assert current_summary.enabled is False
        assert current_summary.version == 2
        assert current_summary.vector_route_hash == old_route.key
        assert observer.get(SummaryVectorCleanupOutbox, cleanup_id) is None


def test_segment_disable_after_summary_publication_transfers_disabled_summary_route(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset(
        id="dataset-summary-disable-race",
        tenant_id="tenant-summary-disable-race",
        name="Dataset",
        description="",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique="high_quality",
        created_by="account-summary-disable-race",
        indexing_request_id=None,
    )
    document = Document(
        id="document-summary-disable-race",
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="batch-summary-disable-race",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by=dataset.created_by,
        indexing_status=IndexingStatus.COMPLETED,
        completed_at=datetime(2026, 1, 1),
        enabled=True,
        archived=False,
        doc_form="text_model",
        word_count=1,
    )
    segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="body content",
        word_count=2,
        tokens=2,
        created_by=dataset.created_by,
        index_node_id="body-node-summary-disable-race",
        index_node_hash="body-hash-summary-disable-race",
        enabled=False,
        status=SegmentStatus.COMPLETED,
    )
    segment.id = "segment-summary-disable-race"
    old_route = VectorRoute.capture(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="summary-route-a",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    new_route = VectorRoute.capture(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="current-summary-collection",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=segment.id,
        summary_content="retained summary",
        summary_index_node_id="summary-node-disable-race",
        summary_index_node_hash="summary-hash-disable-race",
        vector_route=old_route.serialized,
        vector_route_hash=old_route.key,
        status=SummaryStatus.COMPLETED,
        enabled=False,
    )
    summary.id = "summary-disable-race"
    sqlite_session.add_all([dataset, document, segment, summary])
    sqlite_session.commit()

    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    enable_intent = transitions.request(
        document_ref=DatasetRef(dataset.tenant_id, dataset.id).document(document.id),
        segment_ids=[segment.id],
        action="enable",
        actor_id=dataset.created_by,
    )[0]
    processor = _ProcessorBoundary()
    vector = _VectorBoundary()
    resolved_routes: list[str] = []
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(state_task_module, "Vector", _VectorConstructorBoundary(vector))

    def vector_for_route(route: VectorRoute) -> _VectorBoundary:
        resolved_routes.append(route.key)
        return vector

    monkeypatch.setattr("services.knowledge.summaries.adapters.vector_for_route", vector_for_route)
    publish_summaries = state_task_module._publish_enable_summaries
    disable_intent_ids: list[str] = []

    def publish_then_disable_segment(
        publication: state_task_module._EnableBatchPublication,
        *,
        active_intent_ids: frozenset[str],
        new_session: sessionmaker[Session],
    ) -> frozenset[str]:
        published = publish_summaries(
            publication,
            active_intent_ids=active_intent_ids,
            new_session=new_session,
        )
        disable_intents = transitions.request(
            document_ref=DatasetRef(dataset.tenant_id, dataset.id).document(document.id),
            segment_ids=[segment.id],
            action="disable",
            actor_id=dataset.created_by,
        )
        disable_intent_ids.extend(intent.id for intent in disable_intents)
        return published

    monkeypatch.setattr(state_task_module, "_publish_enable_summaries", publish_then_disable_segment)
    assert process_segment_state_intent(enable_intent.id, new_session=sqlite_session_factory) is False
    assert len(disable_intent_ids) == 1
    with sqlite_session_factory() as observer:
        current_summary = observer.get_one(DocumentSegmentSummary, summary.id)
        assert current_summary.enabled is False
        assert current_summary.version == 1
        assert current_summary.vector_route_hash == new_route.key

    monkeypatch.setattr(state_task_module, "_publish_enable_summaries", publish_summaries)
    assert process_segment_state_intent(disable_intent_ids[0], new_session=sqlite_session_factory) is True
    with sqlite_session_factory() as observer:
        current_summary = observer.get_one(DocumentSegmentSummary, summary.id)
        new_route_cleanup = observer.scalar(
            select(SummaryVectorCleanupOutbox).where(
                SummaryVectorCleanupOutbox.tenant_id == dataset.tenant_id,
                SummaryVectorCleanupOutbox.node_id == summary.summary_index_node_id,
                SummaryVectorCleanupOutbox.vector_route_hash == new_route.key,
            )
        )
        assert current_summary.enabled is False
        assert current_summary.version == 2
        assert new_route_cleanup is not None
        cleanup_id = new_route_cleanup.id

    cleanup_repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)

    def cleanup_repository_factory() -> SQLAlchemySummaryVectorCleanupRepository:
        return cleanup_repository

    monkeypatch.setattr(summary_cleanup_task_module, "_repository", cleanup_repository_factory)
    monkeypatch.setattr(summary_cleanup_task_module, "vector_for_route", vector_for_route)
    summary_cleanup_task_module.process_summary_vector_cleanup_task.run(cleanup_id)

    assert resolved_routes == [old_route.key, new_route.key]
    with sqlite_session_factory() as observer:
        assert observer.get(SummaryVectorCleanupOutbox, cleanup_id) is None


def test_segment_enable_defers_when_generation_changes_after_route_preflight(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset(
        id="dataset-route-fence",
        tenant_id="tenant-route-fence",
        name="Dataset",
        description="",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique="high_quality",
        embedding_model="embedding",
        embedding_model_provider="provider",
        created_by="account-route-fence",
        indexing_request_id="generation-0",
    )
    document = Document(
        id="document-route-fence",
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="batch-route-fence",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by="account-route-fence",
        indexing_status=IndexingStatus.COMPLETED,
        completed_at=datetime(2026, 1, 1),
        enabled=True,
        archived=False,
        doc_form="text_model",
        word_count=1,
    )
    segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="body content",
        word_count=2,
        tokens=2,
        created_by="account-route-fence",
        index_node_id="body-node-route-fence",
        index_node_hash="body-hash-route-fence",
        enabled=True,
        status=SegmentStatus.COMPLETED,
    )
    segment.id = "segment-route-fence"
    intent = SegmentStateOutbox(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        action="enable",
        actor_id="account-route-fence",
        summary_id=None,
        summary_version=0,
    )
    segment.indexing_request_id = intent.id
    route_0 = VectorRoute.capture(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="collection-generation-0",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="text_model",
    )
    representation_0 = _indexing_representation(dataset, route_0)
    generation_0 = DatasetIndexingOutbox(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        generation_id="generation-0",
        operation="rebuild",
        action="update",
        source_representation=representation_0.model_dump_json(),
        target_representation=representation_0.model_copy(update={"vector_route": None}).model_dump_json(),
    )
    sqlite_session.add_all([dataset, document, segment, intent, generation_0])
    sqlite_session.commit()

    processor = _ProcessorBoundary()
    vector_state = _GenerationVectorState()
    monkeypatch.setattr(state_task_module, "IndexProcessorFactory", _ProcessorFactoryBoundary(processor))
    monkeypatch.setattr(state_task_module, "Vector", _GenerationVectorConstructorBoundary(vector_state))
    record_route = state_task_module._record_enable_batch_route

    def record_then_switch_generation(
        key: state_task_module._IntentBatchKey,
        intent_ids: Sequence[str],
        *,
        new_session: sessionmaker[Session],
    ) -> state_task_module._EnableBatchPublication | None:
        publication = record_route(key, intent_ids, new_session=new_session)
        with sqlite_session_factory.begin() as switch:
            current_dataset = switch.get_one(Dataset, dataset.id)
            current_dataset.indexing_request_id = "generation-1"
            route_1 = VectorRoute.capture(
                tenant_id=dataset.tenant_id,
                dataset_id=dataset.id,
                vector_type="qdrant",
                collection_name="collection-generation-1",
                collection_binding_id=None,
                indexing_technique="high_quality",
                doc_form="text_model",
            )
            representation_1 = _indexing_representation(current_dataset, route_1)
            switch.add(
                DatasetIndexingOutbox(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    generation_id="generation-1",
                    operation="rebuild",
                    action="update",
                    source_representation=representation_1.model_dump_json(),
                    target_representation=representation_1.model_copy(update={"vector_route": None}).model_dump_json(),
                )
            )
        return publication

    monkeypatch.setattr(state_task_module, "_record_enable_batch_route", record_then_switch_generation)
    with pytest.raises(RuntimeError, match="changed after segment publication preflight"):
        process_segment_state_intent(intent.id, new_session=sqlite_session_factory)

    assert processor.loaded == []
    assert vector_state.added == []
    with sqlite_session_factory() as observer:
        assert observer.get_one(DocumentSegment, segment.id).indexing_request_id == intent.id
        assert observer.get(SegmentStateOutbox, intent.id) is not None

    monkeypatch.setattr(state_task_module, "_record_enable_batch_route", record_route)
    assert process_segment_state_intent(intent.id, new_session=sqlite_session_factory) is True
    assert processor.loaded == [("body content",)]
    with sqlite_session_factory() as observer:
        assert observer.get_one(Dataset, dataset.id).indexing_request_id == "generation-1"
        assert observer.get_one(DocumentSegment, segment.id).indexing_request_id is None
        assert observer.get(SegmentStateOutbox, intent.id) is None

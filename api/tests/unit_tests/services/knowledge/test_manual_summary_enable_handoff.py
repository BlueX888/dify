from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import services.knowledge.summaries.adapters as summary_module
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.models.document import Document as IndexDocument
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus, SegmentStatus, SummaryStatus
from models.model import SegmentStateOutbox, SummaryVectorCleanupOutbox
from repositories.knowledge.segment_repository import SQLAlchemySegmentRepository
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.summaries.adapters import SummaryIndexAdapter
from tests.unit_tests.model_factories import make_account, make_tenant


@dataclass
class _VectorBoundary:
    collection_name: str
    nodes: set[str]
    writes: list[str] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)

    def get_routing_target(self) -> tuple[str, str]:
        return "qdrant", self.collection_name

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
        for document in documents:
            node_id = str(document.metadata["doc_id"])
            self.nodes.add(node_id)
            self.writes.append(node_id)

    def delete_by_ids(self, node_ids: list[str]) -> None:
        for node_id in node_ids:
            self.nodes.discard(node_id)
            self.deletes.append(node_id)


def test_manual_publish_while_disabled_is_handed_off_by_later_explicit_disable(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = make_tenant(tenant_id="tenant-manual-summary", name="Tenant")
    account = make_account(
        account_id="account-manual-summary",
        name="Account",
        email="manual-summary@example.com",
        tenant=tenant,
    )
    dataset = Dataset(
        id="dataset-manual-summary",
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
        id="document-manual-summary",
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="batch-manual-summary",
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
        index_node_id="body-node",
        index_node_hash="body-hash",
        enabled=True,
        status=SegmentStatus.COMPLETED,
    )
    segment.id = "segment-manual-summary"
    route_a = VectorRoute.capture(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="summary-a",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    route_b = VectorRoute.capture(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name="summary-b",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=segment.id,
        summary_content="old summary",
        summary_index_node_id="old-summary-node",
        summary_index_node_hash="old-summary-hash",
        vector_route=route_a.serialized,
        vector_route_hash=route_a.key,
        status=SummaryStatus.COMPLETED,
        enabled=False,
    )
    summary.id = "summary-manual-summary"
    intent = SegmentStateOutbox(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        action="enable",
        actor_id=account.id,
        summary_id=summary.id,
        summary_version=summary.version,
    )
    segment.indexing_request_id = intent.id
    sqlite_session.add_all([tenant, account, dataset, document, segment, summary, intent])
    sqlite_session.commit()

    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(tenant.id, dataset.id).document(document.id).segment(segment.id)
    write = repository.save_summary(ref, "edited summary")
    assert write.summary is not None
    assert write.inherited_enable_intent_id == intent.id
    assert write.summary.enabled is False
    assert write.summary.version == 2

    vector_nodes = {"old-summary-node"}
    vector = _VectorBoundary("summary-b", vector_nodes)

    def vector_for_route(route: VectorRoute) -> Vector:
        assert route.key in {route_a.key, route_b.key}
        return cast(Vector, vector)

    monkeypatch.setattr(summary_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(summary_module, "vector_for_route", vector_for_route)
    assert SummaryIndexAdapter.vectorize_summary(
        write.summary,
        segment,
        dataset,
        new_session=sqlite_session_factory,
        vector_factory=lambda: cast(Vector, vector),
        indexing_request_id=None,
        expected_segment_state_intent_id=intent.id,
    )

    with sqlite_session_factory() as observer:
        published = observer.get_one(DocumentSegmentSummary, summary.id)
        current_intent = observer.get_one(SegmentStateOutbox, intent.id)
        assert published.enabled is False
        assert published.version == 2
        assert published.vector_route_hash == route_b.key
        assert current_intent.summary_version == 2
        published_node_id = published.summary_index_node_id
        assert published_node_id is not None
        assert vector_nodes == {published_node_id}

    SummaryIndexAdapter.disable_summaries_for_segments(
        dataset,
        segment_ids=[segment.id],
        disabled_by=account.id,
    )

    with sqlite_session_factory() as observer:
        disabled = observer.get_one(DocumentSegmentSummary, summary.id)
        stale_enable = observer.get_one(SegmentStateOutbox, intent.id)
        cleanup = observer.scalar(
            select(SummaryVectorCleanupOutbox).where(
                SummaryVectorCleanupOutbox.node_id == published_node_id,
                SummaryVectorCleanupOutbox.vector_route_hash == route_b.key,
            )
        )
        assert disabled.enabled is False
        assert disabled.version == 3
        assert stale_enable.summary_version == 2
        assert cleanup is not None
        assert cleanup.vector_route == route_b.serialized
        assert cleanup.writer_finished_at is not None
        cleanup_id = cleanup.id

    cleanup_repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
    cleanup_claim = cleanup_repository.claim_pending_cleanup(cleanup_id)
    assert cleanup_claim is not None
    assert cleanup_claim.route == route_b
    with cleanup_repository.guard_cleanup_delete(cleanup_claim) as referenced:
        assert referenced is False
        vector.delete_by_ids([cleanup_claim.node_id])

    assert vector_nodes == set()
    assert vector.deletes == ["old-summary-node", published_node_id]
    with sqlite_session_factory() as observer:
        assert observer.get(SummaryVectorCleanupOutbox, cleanup_id) is None

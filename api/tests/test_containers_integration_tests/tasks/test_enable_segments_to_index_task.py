"""Container integration coverage for legacy segment-enable task bridges."""

from collections.abc import Callable
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.rag.index_processor.constant.index_type import IndexTechniqueType
from models.dataset import DocumentSegment, DocumentSegmentSummary
from models.model import SegmentStateOutbox
from tasks.enable_segment_to_index_task import enable_segment_to_index_task
from tasks.enable_segments_to_index_task import enable_segments_to_index_task
from tasks.segment_state_outbox_task import process_segment_state_intents
from tests.test_containers_integration_tests.helpers.segment_state import (
    RecordingIndexProcessor,
    RecordingIndexProcessorFactory,
    create_graph,
    create_segment,
    create_summary,
)


def _capture_published_intents(call: Callable[[], object]) -> list[str]:
    published: list[list[str]] = []
    with patch(
        "tasks.segment_state_legacy.publish_segment_state_intents",
        side_effect=lambda intent_ids: published.append(list(intent_ids)),
    ):
        call()
    return [intent_id for batch in published for intent_id in batch]


def test_legacy_batch_enable_creates_owned_intents_and_reconciles_in_one_batch(
    db_session_with_containers: Session,
) -> None:
    graph = create_graph(db_session_with_containers)
    segments = [
        create_segment(db_session_with_containers, graph, position=position, enabled=True) for position in range(1, 4)
    ]
    foreign = create_graph(db_session_with_containers)
    foreign_segment = create_segment(db_session_with_containers, foreign, position=1, enabled=True)

    intent_ids = _capture_published_intents(
        lambda: enable_segments_to_index_task(
            [*[segment.id for segment in segments], foreign_segment.id],
            graph.dataset.id,
            graph.document.id,
        )
    )

    assert len(intent_ids) == len(segments)
    db_session_with_containers.expire_all()
    intents = db_session_with_containers.scalars(
        select(SegmentStateOutbox).where(SegmentStateOutbox.id.in_(intent_ids)).order_by(SegmentStateOutbox.segment_id)
    ).all()
    assert {intent.segment_id for intent in intents} == {segment.id for segment in segments}
    assert all(
        intent.tenant_id == graph.tenant.id
        and intent.dataset_id == graph.dataset.id
        and intent.document_id == graph.document.id
        and intent.action == "enable"
        for intent in intents
    )
    assert (
        db_session_with_containers.scalar(
            select(SegmentStateOutbox).where(SegmentStateOutbox.segment_id == foreign_segment.id)
        )
        is None
    )
    db_session_with_containers.commit()

    processor = RecordingIndexProcessor()
    with patch(
        "tasks.segment_state_outbox_task.IndexProcessorFactory",
        return_value=RecordingIndexProcessorFactory(processor),
    ):
        results = process_segment_state_intents(intent_ids)

    assert all(results.values())
    assert len(processor.loaded_documents) == 1
    indexed_documents = processor.loaded_documents[0]
    assert {document.metadata["doc_id"] for document in indexed_documents} == {
        segment.index_node_id for segment in segments
    }
    db_session_with_containers.expire_all()
    persisted = db_session_with_containers.scalars(
        select(DocumentSegment).where(DocumentSegment.id.in_([segment.id for segment in segments]))
    ).all()
    assert all(segment.indexing_request_id is None for segment in persisted)
    assert not db_session_with_containers.scalars(
        select(SegmentStateOutbox).where(SegmentStateOutbox.id.in_(intent_ids))
    ).all()


def test_legacy_single_enable_resolves_owner_and_reuses_current_intent(
    db_session_with_containers: Session,
) -> None:
    graph = create_graph(db_session_with_containers)
    segment = create_segment(db_session_with_containers, graph, position=1, enabled=True)
    current = SegmentStateOutbox(
        tenant_id=graph.tenant.id,
        dataset_id=graph.dataset.id,
        document_id=graph.document.id,
        segment_id=segment.id,
        action="enable",
        actor_id=graph.account.id,
    )
    db_session_with_containers.add(current)
    segment.indexing_request_id = current.id
    db_session_with_containers.commit()

    intent_ids = _capture_published_intents(lambda: enable_segment_to_index_task(segment.id))

    assert intent_ids == [current.id]
    processor = RecordingIndexProcessor()
    with patch(
        "tasks.segment_state_outbox_task.IndexProcessorFactory",
        return_value=RecordingIndexProcessorFactory(processor),
    ):
        assert process_segment_state_intents(intent_ids) == {current.id: True}
    assert len(processor.loaded_documents) == 1


def test_stale_legacy_enable_cannot_override_a_current_disable(
    db_session_with_containers: Session,
) -> None:
    graph = create_graph(db_session_with_containers)
    segment = create_segment(db_session_with_containers, graph, position=1, enabled=False)
    current = SegmentStateOutbox(
        tenant_id=graph.tenant.id,
        dataset_id=graph.dataset.id,
        document_id=graph.document.id,
        segment_id=segment.id,
        action="disable",
        actor_id=graph.account.id,
    )
    db_session_with_containers.add(current)
    segment.indexing_request_id = current.id
    db_session_with_containers.commit()

    intent_ids = _capture_published_intents(
        lambda: enable_segments_to_index_task([segment.id], graph.dataset.id, graph.document.id)
    )

    assert intent_ids == []
    db_session_with_containers.expire_all()
    persisted = db_session_with_containers.get_one(DocumentSegment, segment.id)
    assert persisted.enabled is False
    assert persisted.indexing_request_id == current.id
    assert db_session_with_containers.get_one(SegmentStateOutbox, current.id).action == "disable"


def test_legacy_enable_in_economy_mode_skips_retained_summary_vectors(
    db_session_with_containers: Session,
) -> None:
    graph = create_graph(db_session_with_containers, indexing_technique=IndexTechniqueType.ECONOMY)
    graph.dataset.embedding_model = None
    graph.dataset.embedding_model_provider = None
    segment = create_segment(db_session_with_containers, graph, position=1, enabled=True)
    summary = create_summary(db_session_with_containers, graph, segment, enabled=False)
    db_session_with_containers.commit()
    intent_ids = _capture_published_intents(
        lambda: enable_segments_to_index_task([segment.id], graph.dataset.id, graph.document.id)
    )

    processor = RecordingIndexProcessor()
    with (
        patch(
            "tasks.segment_state_outbox_task.IndexProcessorFactory",
            return_value=RecordingIndexProcessorFactory(processor),
        ),
        patch("tasks.segment_state_outbox_task.Vector") as vector_factory,
    ):
        results = process_segment_state_intents(intent_ids)

    assert all(results.values())
    assert len(processor.loaded_documents) == 1
    vector_factory.assert_not_called()
    db_session_with_containers.expire_all()
    assert db_session_with_containers.get_one(DocumentSegmentSummary, summary.id).enabled is False

"""Container integration coverage for the legacy batch-disable task bridge."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from models.dataset import DocumentSegment
from models.model import SegmentStateOutbox
from tasks.disable_segments_from_index_task import disable_segments_from_index_task
from tasks.segment_state_outbox_task import process_segment_state_intents
from tests.test_containers_integration_tests.helpers.segment_state import create_graph, create_segment


def _capture_published_intents(call: Callable[[], object]) -> list[str]:
    published: list[list[str]] = []
    with patch(
        "tasks.segment_state_legacy.publish_segment_state_intents",
        side_effect=lambda intent_ids: published.append(list(intent_ids)),
    ):
        call()
    return [intent_id for batch in published for intent_id in batch]


def test_legacy_batch_disable_reuses_current_intent_filters_owner_and_reconciles_once(
    db_session_with_containers: Session,
) -> None:
    graph = create_graph(db_session_with_containers)
    segments = [
        create_segment(db_session_with_containers, graph, position=position, enabled=False) for position in range(1, 4)
    ]
    current = SegmentStateOutbox(
        tenant_id=graph.tenant.id,
        dataset_id=graph.dataset.id,
        document_id=graph.document.id,
        segment_id=segments[0].id,
        action="disable",
        actor_id=graph.account.id,
    )
    db_session_with_containers.add(current)
    segments[0].indexing_request_id = current.id
    foreign = create_graph(db_session_with_containers)
    foreign_segment = create_segment(db_session_with_containers, foreign, position=1, enabled=False)
    db_session_with_containers.commit()

    intent_ids = _capture_published_intents(
        lambda: disable_segments_from_index_task(
            [*[segment.id for segment in segments], foreign_segment.id],
            graph.dataset.id,
            graph.document.id,
        )
    )

    assert len(intent_ids) == len(segments)
    assert current.id in intent_ids
    db_session_with_containers.expire_all()
    intents = db_session_with_containers.scalars(
        select(SegmentStateOutbox).where(SegmentStateOutbox.id.in_(intent_ids))
    ).all()
    assert {intent.segment_id for intent in intents} == {segment.id for segment in segments}
    assert all(
        intent.tenant_id == graph.tenant.id
        and intent.dataset_id == graph.dataset.id
        and intent.document_id == graph.document.id
        and intent.action == "disable"
        for intent in intents
    )
    assert (
        db_session_with_containers.scalar(
            select(SegmentStateOutbox).where(SegmentStateOutbox.segment_id == foreign_segment.id)
        )
        is None
    )
    db_session_with_containers.commit()

    processor = MagicMock()
    with patch("tasks.segment_state_outbox_task.IndexProcessorFactory") as processor_factory:
        processor_factory.return_value.init_index_processor.return_value = processor
        results = process_segment_state_intents(intent_ids)

    assert all(results.values())
    processor.clean.assert_called_once()
    assert set(processor.clean.call_args.args[1]) == {segment.index_node_id for segment in segments}
    db_session_with_containers.expire_all()
    persisted = db_session_with_containers.scalars(
        select(DocumentSegment).where(DocumentSegment.id.in_([segment.id for segment in segments]))
    ).all()
    assert all(segment.enabled is False and segment.indexing_request_id is None for segment in persisted)
    assert not db_session_with_containers.scalars(
        select(SegmentStateOutbox).where(SegmentStateOutbox.id.in_(intent_ids))
    ).all()


def test_stale_legacy_disable_cannot_override_a_current_enable(
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

    intent_ids = _capture_published_intents(
        lambda: disable_segments_from_index_task([segment.id], graph.dataset.id, graph.document.id)
    )

    assert intent_ids == []
    db_session_with_containers.expire_all()
    persisted = db_session_with_containers.get_one(DocumentSegment, segment.id)
    assert persisted.enabled is True
    assert persisted.indexing_request_id == current.id
    assert db_session_with_containers.get_one(SegmentStateOutbox, current.id).action == "enable"


def test_batch_disable_failure_preserves_current_intents_for_retry(
    db_session_with_containers: Session,
) -> None:
    graph = create_graph(db_session_with_containers)
    segments = [
        create_segment(db_session_with_containers, graph, position=position, enabled=False) for position in range(1, 3)
    ]
    intent_ids = _capture_published_intents(
        lambda: disable_segments_from_index_task(
            [segment.id for segment in segments], graph.dataset.id, graph.document.id
        )
    )

    processor = MagicMock()
    processor.clean.side_effect = RuntimeError("vector backend unavailable")
    with patch("tasks.segment_state_outbox_task.IndexProcessorFactory") as processor_factory:
        processor_factory.return_value.init_index_processor.return_value = processor
        with pytest.raises(RuntimeError, match="vector backend unavailable"):
            process_segment_state_intents(intent_ids)

    db_session_with_containers.expire_all()
    intents = db_session_with_containers.scalars(
        select(SegmentStateOutbox).where(SegmentStateOutbox.id.in_(intent_ids))
    ).all()
    assert len(intents) == len(segments)
    assert all(intent.last_attempted_at is not None for intent in intents)
    persisted = db_session_with_containers.scalars(
        select(DocumentSegment).where(DocumentSegment.id.in_([segment.id for segment in segments]))
    ).all()
    assert {segment.indexing_request_id for segment in persisted} == set(intent_ids)


def test_legacy_batch_disable_ignores_empty_and_missing_ids(db_session_with_containers: Session) -> None:
    graph = create_graph(db_session_with_containers)
    missing_id = "00000000-0000-0000-0000-000000000001"

    empty = _capture_published_intents(
        lambda: disable_segments_from_index_task([], graph.dataset.id, graph.document.id)
    )
    missing = _capture_published_intents(
        lambda: disable_segments_from_index_task([missing_id], graph.dataset.id, graph.document.id)
    )

    assert empty == []
    assert missing == []
    assert db_session_with_containers.scalar(select(SegmentStateOutbox.id)) is None

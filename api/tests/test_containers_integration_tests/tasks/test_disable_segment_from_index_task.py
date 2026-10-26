"""Container integration coverage for the legacy single-disable task bridge."""

from collections.abc import Callable
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from models.dataset import DocumentSegment
from models.enums import SegmentStatus
from models.model import SegmentStateOutbox
from tasks.disable_segment_from_index_task import disable_segment_from_index_task
from tasks.segment_state_outbox_task import process_segment_state_intents
from tests.test_containers_integration_tests.helpers.segment_state import (
    RecordingIndexProcessor,
    RecordingIndexProcessorFactory,
    create_graph,
    create_segment,
)


def _capture_published_intents(call: Callable[[], object]) -> list[str]:
    published: list[list[str]] = []
    with patch(
        "tasks.segment_state_legacy.publish_segment_state_intents",
        side_effect=lambda intent_ids: published.append(list(intent_ids)),
    ):
        call()
    return [intent_id for batch in published for intent_id in batch]


def test_legacy_single_disable_derives_owner_and_reconciles_current_generation(
    db_session_with_containers: Session,
) -> None:
    graph = create_graph(db_session_with_containers)
    segment = create_segment(db_session_with_containers, graph, position=1, enabled=False)

    intent_ids = _capture_published_intents(lambda: disable_segment_from_index_task(segment.id))

    assert len(intent_ids) == 1
    db_session_with_containers.expire_all()
    intent = db_session_with_containers.get_one(SegmentStateOutbox, intent_ids[0])
    assert (
        intent.tenant_id,
        intent.dataset_id,
        intent.document_id,
        intent.segment_id,
        intent.action,
    ) == (
        graph.tenant.id,
        graph.dataset.id,
        graph.document.id,
        segment.id,
        "disable",
    )
    db_session_with_containers.commit()

    processor = RecordingIndexProcessor()
    with patch(
        "tasks.segment_state_outbox_task.IndexProcessorFactory",
        return_value=RecordingIndexProcessorFactory(processor),
    ):
        assert process_segment_state_intents(intent_ids) == {intent.id: True}

    assert processor.cleaned_node_ids == [(segment.index_node_id,)]
    db_session_with_containers.expire_all()
    persisted = db_session_with_containers.get_one(DocumentSegment, segment.id)
    assert persisted.indexing_request_id is None
    assert db_session_with_containers.get(SegmentStateOutbox, intent.id) is None


def test_legacy_single_disable_ignores_missing_segment(db_session_with_containers: Session) -> None:
    create_graph(db_session_with_containers)

    intent_ids = _capture_published_intents(lambda: disable_segment_from_index_task(str(uuid4())))

    assert intent_ids == []
    assert db_session_with_containers.scalar(select(SegmentStateOutbox.id)) is None


def test_legacy_single_disable_does_not_create_intent_for_noncompleted_segment(
    db_session_with_containers: Session,
) -> None:
    graph = create_graph(db_session_with_containers)
    segment = create_segment(
        db_session_with_containers,
        graph,
        position=1,
        enabled=False,
        status=SegmentStatus.ERROR,
    )

    intent_ids = _capture_published_intents(lambda: disable_segment_from_index_task(segment.id))

    assert intent_ids == []
    db_session_with_containers.expire_all()
    persisted = db_session_with_containers.get_one(DocumentSegment, segment.id)
    assert persisted.indexing_request_id is None
    assert (
        db_session_with_containers.scalar(
            select(SegmentStateOutbox.id).where(SegmentStateOutbox.segment_id == segment.id)
        )
        is None
    )

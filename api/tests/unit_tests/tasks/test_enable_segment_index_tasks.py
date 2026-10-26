"""Compatibility tasks must bridge queued legacy deliveries to the fenced outbox."""

from unittest.mock import patch

from tasks.enable_segment_to_index_task import enable_segment_to_index_task
from tasks.enable_segments_to_index_task import enable_segments_to_index_task


def test_single_enable_bridges_to_current_state_reconciler() -> None:
    with patch("tasks.enable_segment_to_index_task.enqueue_legacy_segment_state") as enqueue:
        enable_segment_to_index_task.run("segment-1")
    enqueue.assert_called_once_with(action="enable", segment_ids=["segment-1"])


def test_batch_enable_preserves_owner_arguments_for_bridge() -> None:
    with patch("tasks.enable_segments_to_index_task.enqueue_legacy_segment_state") as enqueue:
        enable_segments_to_index_task.run(["segment-1", "segment-2"], "dataset-1", "document-1")
    enqueue.assert_called_once_with(
        action="enable",
        segment_ids=["segment-1", "segment-2"],
        dataset_id="dataset-1",
        document_id="document-1",
    )

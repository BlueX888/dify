"""Compatibility entrypoint for enable tasks queued before the durable outbox."""

from celery import shared_task

from tasks.segment_state_legacy import enqueue_legacy_segment_state


@shared_task(queue="dataset")
def enable_segment_to_index_task(segment_id: str) -> None:
    enqueue_legacy_segment_state(action="enable", segment_ids=[segment_id])

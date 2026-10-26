"""Compatibility entrypoint for disable tasks queued before the durable outbox."""

from celery import shared_task

from tasks.segment_state_legacy import enqueue_legacy_segment_state


@shared_task(queue="dataset")
def disable_segment_from_index_task(segment_id: str) -> None:
    enqueue_legacy_segment_state(action="disable", segment_ids=[segment_id])

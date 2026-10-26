"""Compatibility entrypoint for batch enable tasks queued before the durable outbox."""

from celery import shared_task

from tasks.segment_state_legacy import enqueue_legacy_segment_state


@shared_task(queue="dataset")
def enable_segments_to_index_task(segment_ids: list[str], dataset_id: str, document_id: str) -> None:
    enqueue_legacy_segment_state(
        action="enable",
        segment_ids=segment_ids,
        dataset_id=dataset_id,
        document_id=document_id,
    )

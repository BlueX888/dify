"""Bridge pre-deployment segment tasks into the durable state reconciler."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from sqlalchemy import select

from core.db.session_factory import session_factory
from models.dataset import DocumentSegment
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.segments.adapters import SQLAlchemySegmentStateTransitions
from tasks.segment_state_outbox_task import publish_segment_state_intents


def enqueue_legacy_segment_state(
    *,
    action: Literal["enable", "disable"],
    segment_ids: Sequence[str],
    dataset_id: str | None = None,
    document_id: str | None = None,
) -> None:
    """Fence an old task by replacing it with a current durable generation.

    Legacy tasks may already be queued during deployment. They must not perform
    their original unfenced vector mutation after a newer opposite request.
    """

    if not segment_ids:
        return
    maker = session_factory.get_session_maker()
    with maker() as session:
        statement = select(DocumentSegment).where(DocumentSegment.id.in_(tuple(segment_ids)))
        if dataset_id is not None:
            statement = statement.where(DocumentSegment.dataset_id == dataset_id)
        if document_id is not None:
            statement = statement.where(DocumentSegment.document_id == document_id)
        rows = session.scalars(statement).all()

    grouped: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for segment in rows:
        actor_id = segment.disabled_by or segment.created_by
        grouped[(segment.tenant_id, segment.dataset_id, segment.document_id, actor_id)].append(segment.id)

    transitions = SQLAlchemySegmentStateTransitions(session_factory=maker)
    for (tenant_id, owned_dataset_id, owned_document_id, actor_id), owned_segment_ids in grouped.items():
        intents = transitions.request(
            document_ref=DatasetRef(tenant_id, owned_dataset_id).document(owned_document_id),
            segment_ids=owned_segment_ids,
            action=action,
            actor_id=actor_id,
            reconcile_settled=True,
        )
        publish_segment_state_intents([intent.id for intent in intents])

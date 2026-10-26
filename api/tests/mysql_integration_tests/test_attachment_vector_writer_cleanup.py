"""MySQL concurrency regression for durable attachment-vector writers."""

from __future__ import annotations

from threading import Event, Thread
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

import tasks.attachment_vector_cleanup_task as cleanup_task
from models.dataset import Dataset, SegmentAttachmentBinding
from models.model import AttachmentVectorCleanupOutbox
from repositories.knowledge.attachment_vector_repository import (
    AttachmentVectorCleanupClaimLostError,
    SQLAlchemyAttachmentVectorStore,
    delete_unreferenced_attachment_vectors,
)
from services.knowledge.entities.vector_route import VectorRoute


def test_detach_handoff_and_late_attachment_writer_require_second_cleanup(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out writer that completes after detach cannot resurrect its physical vector."""

    sessions = sessionmaker(bind=mysql_session.get_bind(), expire_on_commit=False)
    tenant_id = str(uuid4())
    dataset_id = str(uuid4())
    document_id = str(uuid4())
    segment_id = str(uuid4())
    attachment_id = str(uuid4())
    route = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name=f"attachment-route-{uuid4()}",
        collection_binding_id=str(uuid4()),
        indexing_technique="high_quality",
        doc_form="attachment",
    )
    binding = SegmentAttachmentBinding(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        segment_id=segment_id,
        attachment_id=attachment_id,
    )
    with sessions.begin() as seed:
        seed.add_all(
            [
                Dataset(
                    id=dataset_id,
                    tenant_id=tenant_id,
                    name="attachment-writer-cleanup",
                    created_by=str(uuid4()),
                    indexing_technique="high_quality",
                    index_struct=route.index_struct,
                    collection_binding_id=route.collection_binding_id,
                    is_multimodal=True,
                ),
                binding,
            ]
        )

    repository = SQLAlchemyAttachmentVectorStore(session_factory=sessions)
    (intent,) = repository.begin_writes(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        attachment_ids=(attachment_id,),
        route=route,
    )
    with sessions.begin() as detach:
        orphan_ids = delete_unreferenced_attachment_vectors(
            detach,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_ids=(document_id,),
            segment_ids=(segment_id,),
            route=route,
        )
        assert orphan_ids == (attachment_id,)
        detach.execute(delete(SegmentAttachmentBinding).where(SegmentAttachmentBinding.id == binding.id))

    with sessions() as after_detach:
        row = after_detach.get_one(AttachmentVectorCleanupOutbox, intent.id)
        assert row.cleanup_started_at is not None
        assert row.writer_finished_at is None
        assert row.claim_id is None
        assert after_detach.scalar(select(SegmentAttachmentBinding.id)) is None
        assert len(after_detach.scalars(select(AttachmentVectorCleanupOutbox.id)).all()) == 1

    vector_ids = {attachment_id}
    delete_calls: list[tuple[str, ...]] = []
    first_delete_completed = Event()
    allow_first_task_return = Event()
    first_task_errors: list[BaseException] = []

    class _ControlledVector:
        def __init__(self, dataset: Dataset, *, session: Session | None, vector_type: str) -> None:
            del dataset, session
            assert vector_type == route.vector_type

        def get_routing_target(self) -> tuple[str, str]:
            return route.vector_type, route.collection_name

        def delete_by_ids(self, node_ids: list[str]) -> None:
            delete_calls.append(tuple(node_ids))
            vector_ids.difference_update(node_ids)
            if len(delete_calls) == 1:
                first_delete_completed.set()
                assert allow_first_task_return.wait(timeout=10)

    monkeypatch.setattr(cleanup_task, "_repository", lambda: repository)
    monkeypatch.setattr(cleanup_task, "Vector", _ControlledVector)

    def run_first_cleanup() -> None:
        try:
            cleanup_task.process_attachment_vector_cleanup_task.run(intent.id)
        except BaseException as error:
            first_task_errors.append(error)

    first_cleanup = Thread(target=run_first_cleanup)
    try:
        first_cleanup.start()
        assert first_delete_completed.wait(timeout=10)

        # The first delete has completed remotely, but its transaction still
        # owns an unfinished-writer snapshot. The timed-out write now lands.
        vector_ids.add(attachment_id)
        with pytest.raises(AttachmentVectorCleanupClaimLostError):
            repository.finish_writes((intent,))

        allow_first_task_return.set()
        first_cleanup.join(timeout=10)
        assert not first_cleanup.is_alive()
        assert first_task_errors == []
        assert vector_ids == {attachment_id}

        with sessions() as after_first_cleanup:
            row = after_first_cleanup.get_one(AttachmentVectorCleanupOutbox, intent.id)
            assert row.writer_finished_at is not None
            assert row.claim_id is None

        cleanup_task.process_attachment_vector_cleanup_task.run(intent.id)

        assert delete_calls == [(attachment_id,), (attachment_id,)]
        assert vector_ids == set()
        with sessions() as verify:
            assert verify.get(AttachmentVectorCleanupOutbox, intent.id) is None
            assert verify.scalar(select(AttachmentVectorCleanupOutbox.id)) is None
            assert verify.scalar(select(SegmentAttachmentBinding.id)) is None
    finally:
        allow_first_task_return.set()
        if first_cleanup.ident is not None:
            first_cleanup.join(timeout=10)
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(AttachmentVectorCleanupOutbox).where(AttachmentVectorCleanupOutbox.dataset_id == dataset_id)
            )
            cleanup.execute(delete(SegmentAttachmentBinding).where(SegmentAttachmentBinding.dataset_id == dataset_id))
            cleanup.execute(delete(Dataset).where(Dataset.id == dataset_id))

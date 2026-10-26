"""Database-lock regression tests for shared attachment vectors."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

import tasks.attachment_vector_cleanup_task as cleanup_task
from models.dataset import Dataset, DocumentSegment, SegmentAttachmentBinding
from models.model import AttachmentVectorCleanupOutbox
from repositories.knowledge.attachment_vector_repository import (
    SQLAlchemyAttachmentVectorStore,
    delete_unreferenced_attachment_vectors,
    publish_bound_attachment_vectors,
)
from repositories.knowledge.segment_cleanup_repository import add_segment_attachment_bindings
from services.knowledge.entities.vector_route import VectorRoute
from tests.unit_tests.model_factories import make_document, make_upload_file


@pytest.mark.usefixtures("flask_app_with_containers")
def test_attachment_publish_waits_for_cleanup_and_restores_live_vector(
    db_session_with_containers: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A binding created behind the post-commit cleanup guard republishes after deletion."""
    tenant_id = str(uuid4())
    dataset_id = str(uuid4())
    route = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name=f"attachment-route-{uuid4()}",
        collection_binding_id=str(uuid4()),
        indexing_technique="high_quality",
        doc_form="attachment",
    )
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name="Attachment locking",
        created_by="account-1",
        indexing_technique="high_quality",
        index_struct=route.index_struct,
        collection_binding_id=route.collection_binding_id,
        is_multimodal=True,
        chunk_structure="text_model",
    )
    target_document = make_document(document_id=str(uuid4()), tenant_id=tenant_id, dataset_id=dataset.id, position=1)
    survivor_document = make_document(document_id=str(uuid4()), tenant_id=tenant_id, dataset_id=dataset.id, position=2)
    target_segment = DocumentSegment(
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        document_id=target_document.id,
        position=1,
        content="target",
        word_count=1,
        tokens=1,
        created_by="account-1",
    )
    survivor_segment = DocumentSegment(
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        document_id=survivor_document.id,
        position=1,
        content="survivor",
        word_count=1,
        tokens=1,
        created_by="account-1",
    )
    attachment = make_upload_file(
        file_id=str(uuid4()),
        tenant_id=tenant_id,
        key="uploads/shared-race.png",
        name="shared-race.png",
        extension="png",
        mime_type="image/png",
    )
    target_binding = SegmentAttachmentBinding(
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        document_id=target_document.id,
        segment_id=target_segment.id,
        attachment_id=attachment.id,
    )
    db_session_with_containers.add_all(
        [
            dataset,
            target_document,
            survivor_document,
            target_segment,
            survivor_segment,
            attachment,
            target_binding,
        ]
    )
    db_session_with_containers.flush()
    target_document_id = target_document.id
    target_segment_id = target_segment.id
    survivor_document_id = survivor_document.id
    survivor_segment_id = survivor_segment.id
    attachment_id = attachment.id
    target_binding_id = target_binding.id
    db_session_with_containers.commit()

    sessions = sessionmaker(bind=db_session_with_containers.get_bind(), expire_on_commit=False)
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sessions)
    with sessions.begin() as detach:
        assert delete_unreferenced_attachment_vectors(
            detach,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_ids=(target_document_id,),
            segment_ids=(target_segment_id,),
            route=route,
        ) == (attachment_id,)
        detach.execute(delete(SegmentAttachmentBinding).where(SegmentAttachmentBinding.id == target_binding_id))
    with sessions() as read_cleanup:
        cleanup_id = read_cleanup.scalars(select(AttachmentVectorCleanupOutbox.id)).one()

    cleanup_in_vector_io = Event()
    release_cleanup = Event()
    publisher_started = Event()
    vector_ids = {attachment_id}

    class _ControlledVector:
        def __init__(self, dataset: Dataset, *, session: Session | None, vector_type: str) -> None:
            del dataset, session
            assert vector_type == route.vector_type

        def get_routing_target(self) -> tuple[str, str]:
            return route.vector_type, route.collection_name

        def delete_by_ids(self, node_ids: list[str]) -> None:
            assert node_ids == [attachment_id]
            cleanup_in_vector_io.set()
            assert release_cleanup.wait(timeout=10)
            vector_ids.difference_update(node_ids)

    monkeypatch.setattr(cleanup_task, "_repository", lambda: repository)
    monkeypatch.setattr(cleanup_task, "Vector", _ControlledVector)

    def cleanup() -> None:
        cleanup_task.process_attachment_vector_cleanup_task.run(cleanup_id)

    def bind_and_publish() -> None:
        publisher_started.set()
        with sessions.begin() as session:
            add_segment_attachment_bindings(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=survivor_document_id,
                segment_id=survivor_segment_id,
                attachment_ids=(attachment_id,),
            )

            def publish(
                bound_refs: frozenset[tuple[str, str]],
                before_write: Callable[[Sequence[str]], None],
                after_write: Callable[[Sequence[str]], None],
            ) -> None:
                assert bound_refs == frozenset({(survivor_document_id, attachment_id)})
                before_write((attachment_id,))
                vector_ids.add(attachment_id)
                after_write((attachment_id,))

            publish_bound_attachment_vectors(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                attachment_refs=((survivor_document_id, attachment_id),),
                route=route,
                publish=publish,
                store=repository,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        cleanup_future = pool.submit(cleanup)
        assert cleanup_in_vector_io.wait(timeout=5)
        publisher_future = pool.submit(bind_and_publish)
        assert publisher_started.wait(timeout=5)
        try:
            with pytest.raises(TimeoutError):
                publisher_future.result(timeout=0.5)
        finally:
            release_cleanup.set()
        cleanup_future.result(timeout=5)
        publisher_future.result(timeout=5)

    assert vector_ids == {attachment_id}
    db_session_with_containers.expire_all()
    bindings = db_session_with_containers.scalars(
        select(SegmentAttachmentBinding).where(SegmentAttachmentBinding.attachment_id == attachment_id)
    ).all()
    assert [(binding.document_id, binding.segment_id) for binding in bindings] == [
        (survivor_document_id, survivor_segment_id)
    ]
    assert db_session_with_containers.scalar(select(AttachmentVectorCleanupOutbox.id)) is None

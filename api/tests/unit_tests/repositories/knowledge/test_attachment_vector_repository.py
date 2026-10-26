from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

import tasks.attachment_vector_cleanup_task as cleanup_task
from models.dataset import Dataset, SegmentAttachmentBinding
from models.model import AttachmentVectorCleanupOutbox
from repositories.knowledge import attachment_vector_repository
from repositories.knowledge.attachment_vector_repository import (
    AttachmentVectorCleanupClaimLostError,
    SQLAlchemyAttachmentVectorStore,
    delete_unreferenced_attachment_vectors,
    publish_bound_attachment_vectors,
)
from services.knowledge.entities.vector_route import VectorRoute
from tests.unit_tests.model_factories import make_dataset


def _route(collection_name: str = "attachments-a") -> VectorRoute:
    return VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name=collection_name,
        collection_binding_id=f"binding-{collection_name}",
        indexing_technique="high_quality",
        doc_form="attachment",
    )


def test_each_retry_registers_an_independent_writer_receipt(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)
    route = _route()

    first = repository.begin_writes(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        attachment_ids=("attachment-1",),
        route=route,
    )
    second = repository.begin_writes(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        attachment_ids=("attachment-1",),
        route=route,
    )

    assert first[0].id != second[0].id
    assert first[0].writer_id != second[0].writer_id
    with sqlite_session_factory() as session:
        assert len(session.scalars(select(AttachmentVectorCleanupOutbox)).all()) == 2


@pytest.mark.usefixtures("_unbound_session_factory")
def test_publication_uses_the_explicit_store_when_the_global_factory_is_unbound(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route = _route()
    dataset = make_dataset(
        dataset_id=route.dataset_id,
        tenant_id=route.tenant_id,
        indexing_technique="high_quality",
        index_struct=route.index_struct,
    )
    binding = SegmentAttachmentBinding(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        document_id="document-1",
        segment_id="segment-1",
        attachment_id="attachment-1",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([dataset, binding])
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)

    def publish(
        refs: frozenset[tuple[str, str]],
        before_write: Callable[[Sequence[str]], None],
        after_write: Callable[[Sequence[str]], None],
    ) -> None:
        assert refs == frozenset({("document-1", "attachment-1")})
        before_write(("attachment-1",))
        after_write(("attachment-1",))

    with sqlite_session_factory.begin() as session:
        published = publish_bound_attachment_vectors(
            session,
            tenant_id=route.tenant_id,
            dataset_id=route.dataset_id,
            attachment_refs=(("document-1", "attachment-1"),),
            route=route,
            publish=publish,
            store=repository,
        )

    assert published == frozenset({("document-1", "attachment-1")})
    with sqlite_session_factory() as session:
        assert session.scalar(select(AttachmentVectorCleanupOutbox.id)) is None


def test_multibatch_failure_retains_only_receipts_that_crossed_write_boundary(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route = _route()
    dataset = make_dataset(
        dataset_id=route.dataset_id,
        tenant_id=route.tenant_id,
        indexing_technique="high_quality",
        index_struct=route.index_struct,
    )
    bindings = [
        SegmentAttachmentBinding(
            tenant_id=route.tenant_id,
            dataset_id=route.dataset_id,
            document_id="document-1",
            segment_id=f"segment-{position}",
            attachment_id=f"attachment-{position}",
        )
        for position in (1, 2)
    ]
    with sqlite_session_factory.begin() as session:
        session.add_all([dataset, *bindings])
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)

    def publish(
        _refs: frozenset[tuple[str, str]],
        before_write: Callable[[Sequence[str]], None],
        after_write: Callable[[Sequence[str]], None],
    ) -> None:
        before_write(("attachment-1",))
        after_write(("attachment-1",))
        raise TimeoutError("second batch failed before remote submission")

    with pytest.raises(TimeoutError):
        with sqlite_session_factory() as session, session.begin():
            publish_bound_attachment_vectors(
                session,
                tenant_id=route.tenant_id,
                dataset_id=route.dataset_id,
                attachment_refs=(("document-1", "attachment-1"), ("document-1", "attachment-2")),
                route=route,
                publish=publish,
                store=repository,
            )

    with sqlite_session_factory() as session:
        rows = session.scalars(select(AttachmentVectorCleanupOutbox)).all()
        assert [row.attachment_id for row in rows] == ["attachment-1"]
        assert rows[0].writer_finished_at is not None


def test_writer_attempt_rejects_resubmitting_a_completed_attachment(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route = _route()
    dataset = make_dataset(
        dataset_id=route.dataset_id,
        tenant_id=route.tenant_id,
        indexing_technique="high_quality",
        index_struct=route.index_struct,
    )
    binding = SegmentAttachmentBinding(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        document_id="document-1",
        segment_id="segment-1",
        attachment_id="attachment-1",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([dataset, binding])
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)

    def publish(
        _refs: frozenset[tuple[str, str]],
        before_write: Callable[[Sequence[str]], None],
        after_write: Callable[[Sequence[str]], None],
    ) -> None:
        before_write(("attachment-1",))
        after_write(("attachment-1",))
        before_write(("attachment-1",))

    with pytest.raises(AttachmentVectorCleanupClaimLostError, match="submitted IDs more than once"):
        with sqlite_session_factory() as session, session.begin():
            publish_bound_attachment_vectors(
                session,
                tenant_id=route.tenant_id,
                dataset_id=route.dataset_id,
                attachment_refs=(("document-1", "attachment-1"),),
                route=route,
                publish=publish,
                store=repository,
            )

    with sqlite_session_factory() as session:
        row = session.scalar(select(AttachmentVectorCleanupOutbox))
        assert row is not None
        assert row.attachment_id == "attachment-1"
        assert row.writer_finished_at is not None


def test_writer_heartbeat_renews_completed_and_unfinished_batches(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    database_now = datetime(2026, 1, 1)
    monkeypatch.setattr(attachment_vector_repository, "_database_now", lambda _session: database_now)
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)
    first, second = repository.begin_writes(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        attachment_ids=("attachment-1", "attachment-2"),
        route=route,
    )
    repository.finish_writes((first,))
    short_expiry = database_now + timedelta(seconds=5)
    with sqlite_session_factory.begin() as session:
        session.get_one(AttachmentVectorCleanupOutbox, first.id).claim_expires_at = short_expiry
        session.get_one(AttachmentVectorCleanupOutbox, second.id).claim_expires_at = short_expiry

    repository.renew_writes((first, second))
    database_now += timedelta(seconds=6)
    assert repository.claim_pending_cleanup(first.id) is None
    assert repository.claim_pending_cleanup(second.id) is None

    with sqlite_session_factory() as session:
        first_row = session.get_one(AttachmentVectorCleanupOutbox, first.id)
        assert first_row.writer_finished_at is not None
        assert first_row.claim_expires_at is not None
        assert first_row.claim_expires_at > short_expiry
        renewed_expiry = session.get_one(AttachmentVectorCleanupOutbox, second.id).claim_expires_at
        assert renewed_expiry is not None
        assert renewed_expiry > short_expiry


def test_timeout_detach_late_write_requires_a_second_delete_before_ack(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route = _route()
    dataset = make_dataset(
        dataset_id=route.dataset_id,
        tenant_id=route.tenant_id,
        indexing_technique="high_quality",
        index_struct=route.index_struct,
    )
    dataset.is_multimodal = True
    with sqlite_session_factory.begin() as session:
        session.add(dataset)
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)
    (intent,) = repository.begin_writes(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        attachment_ids=("attachment-1",),
        route=route,
    )
    deleted: list[tuple[str, ...]] = []

    with sqlite_session_factory.begin() as session:
        result = delete_unreferenced_attachment_vectors(
            session,
            tenant_id=route.tenant_id,
            dataset_id=route.dataset_id,
            candidate_attachment_ids=(intent.attachment_id,),
            route=route,
        )

    assert result == (intent.attachment_id,)
    assert deleted == []
    with sqlite_session_factory() as session:
        handed_off = session.get_one(AttachmentVectorCleanupOutbox, intent.id)
        assert handed_off.cleanup_started_at is not None
        assert handed_off.writer_finished_at is None

    # The timed-out remote call returns after the first delete. Publication is
    # rejected, but its durable writer outcome makes a second delete provable.
    with pytest.raises(AttachmentVectorCleanupClaimLostError):
        repository.finish_writes((intent,))
    claim = repository.claim_pending_cleanup(intent.id)
    assert claim is not None
    assert claim.writer_finished
    with repository.guard_cleanup_delete(claim) as guard:
        assert not guard.has_live_binding
        deleted.append((claim.attachment_id,))

    assert deleted == [(intent.attachment_id,)]
    with sqlite_session_factory() as session:
        assert session.get(AttachmentVectorCleanupOutbox, intent.id) is None


def test_shared_live_binding_prevents_terminal_handoff_and_vector_delete(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route = _route()
    dataset = make_dataset(
        dataset_id=route.dataset_id,
        tenant_id=route.tenant_id,
        indexing_technique="high_quality",
        index_struct=route.index_struct,
    )
    dataset.is_multimodal = True
    binding = SegmentAttachmentBinding(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        document_id="document-other",
        segment_id="segment-other",
        attachment_id="attachment-1",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([dataset, binding])
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)
    (intent,) = repository.begin_writes(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        attachment_ids=(binding.attachment_id,),
        route=route,
    )
    with sqlite_session_factory.begin() as session:
        result = delete_unreferenced_attachment_vectors(
            session,
            tenant_id=route.tenant_id,
            dataset_id=route.dataset_id,
            candidate_attachment_ids=(binding.attachment_id,),
            route=route,
        )

    assert result == ()
    with sqlite_session_factory() as session:
        row = session.get_one(AttachmentVectorCleanupOutbox, intent.id)
        assert row.cleanup_started_at is None

    # Test cleanup must not leave an active writer receipt behind.
    repository.discard_unsubmitted_writes((intent,))
    with sqlite_session_factory.begin() as session:
        session.execute(delete(SegmentAttachmentBinding).where(SegmentAttachmentBinding.id == binding.id))


def test_detach_rollback_keeps_live_binding_and_creates_no_cleanup_receipt(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route = _route()
    dataset = make_dataset(
        dataset_id=route.dataset_id,
        tenant_id=route.tenant_id,
        indexing_technique="high_quality",
        index_struct=route.index_struct,
    )
    dataset.is_multimodal = True
    binding = SegmentAttachmentBinding(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        document_id="document-1",
        segment_id="segment-1",
        attachment_id="attachment-1",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([dataset, binding])

    def fail_detach_transaction() -> None:
        with sqlite_session_factory() as session, session.begin():
            result = delete_unreferenced_attachment_vectors(
                session,
                tenant_id=route.tenant_id,
                dataset_id=route.dataset_id,
                document_ids=(binding.document_id,),
                segment_ids=(binding.segment_id,),
                route=route,
            )
            assert result == (binding.attachment_id,)
            raise RuntimeError("relational detach failed")

    with pytest.raises(RuntimeError):
        fail_detach_transaction()

    with sqlite_session_factory() as session:
        assert session.get(SegmentAttachmentBinding, binding.id) is not None
        assert session.scalar(select(AttachmentVectorCleanupOutbox.id)) is None


def test_live_binding_on_new_route_does_not_ack_old_route_without_deleting_it(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_route = _route("attachments-a")
    new_route = _route("attachments-b")
    dataset = make_dataset(
        dataset_id=new_route.dataset_id,
        tenant_id=new_route.tenant_id,
        indexing_technique="high_quality",
        index_struct=new_route.index_struct,
    )
    dataset.is_multimodal = True
    binding = SegmentAttachmentBinding(
        tenant_id=new_route.tenant_id,
        dataset_id=new_route.dataset_id,
        document_id="document-1",
        segment_id="segment-1",
        attachment_id="attachment-1",
    )
    cleanup = AttachmentVectorCleanupOutbox(
        tenant_id=old_route.tenant_id,
        dataset_id=old_route.dataset_id,
        attachment_id=binding.attachment_id,
        writer_id="writer-a",
        vector_route=old_route.serialized,
        vector_route_hash=old_route.key,
        writer_finished_at=datetime(2026, 10, 25),
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([dataset, binding, cleanup])
        cleanup_id = cleanup.id
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)

    class CapturingVector:
        deleted: list[tuple[str, tuple[str, ...]]] = []

        @staticmethod
        def resolve_vector_type(dataset: Dataset, *, session: Session) -> str:
            del session
            index_struct = dataset.index_struct_dict
            assert index_struct is not None
            return index_struct["type"]

        def __init__(self, dataset: Dataset, *, session: Session, vector_type: str) -> None:
            del session
            index_struct = dataset.index_struct_dict
            assert index_struct is not None
            self._target = (vector_type, index_struct["vector_store"]["class_prefix"])

        def get_routing_target(self) -> tuple[str, str]:
            return self._target

        def delete_by_ids(self, node_ids: list[str]) -> None:
            self.deleted.append((self._target[1], tuple(node_ids)))

    monkeypatch.setattr(cleanup_task, "_repository", lambda: repository)
    monkeypatch.setattr(cleanup_task, "Vector", CapturingVector)

    cleanup_task.process_attachment_vector_cleanup_task.run(cleanup_id)

    assert CapturingVector.deleted == [(old_route.collection_name, (binding.attachment_id,))]
    with sqlite_session_factory() as session:
        assert session.get(AttachmentVectorCleanupOutbox, cleanup_id) is None
        assert session.get(SegmentAttachmentBinding, binding.id) is not None

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from threading import Event, Lock, Thread

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

import tasks.child_vector_cleanup_task as task_module
from models.dataset import ChildChunk, Dataset
from models.model import ChildVectorCleanupOutbox
from repositories.knowledge.child_vector_cleanup_repository import (
    ChildVectorCleanupClaimLostError,
    SQLAlchemyChildVectorCleanupRepository,
    _cleanup_dataset_lock_statement,
    _pending_cleanup_ids_statement,
)
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.segments import ChildVectorCleanupClaim, ChildVectorRoute


def _route(collection_name: str) -> ChildVectorRoute:
    return ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name=collection_name,
        collection_binding_id=f"binding-{collection_name}",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )


def _cleanup(position: int, *, last_attempted_at: datetime | None = None) -> ChildVectorCleanupOutbox:
    return ChildVectorCleanupOutbox(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        child_id=f"child-{position}",
        node_id=f"node-{position}",
        writer_id=f"writer-{position}",
        writer_finished_at=datetime(2026, 10, 24),
        last_attempted_at=last_attempted_at,
    )


def _routed_cleanup(route: ChildVectorRoute, *, node_id: str = "shared-node") -> ChildVectorCleanupOutbox:
    return ChildVectorCleanupOutbox(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        document_id="document-1",
        segment_id="segment-1",
        child_id="child-old",
        node_id=node_id,
        writer_id="writer-old",
        writer_finished_at=datetime(2026, 10, 24),
        vector_route=route.serialized,
        vector_route_hash=route.key,
    )


def test_pending_cleanup_order_compiles_null_first_for_postgresql() -> None:
    statement = _pending_cleanup_ids_statement(datetime(2026, 10, 25), limit=100)

    sql = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "CASE WHEN (child_vector_cleanup_outbox.last_attempted_at IS NULL) THEN 0 ELSE 1 END" in sql
    assert "child_vector_cleanup_outbox.last_attempted_at" in sql


def test_cleanup_guard_compiles_dataset_share_lock_for_mysql() -> None:
    route = _route("collection-a")
    claim = ChildVectorCleanupClaim(
        id="cleanup-1",
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        node_id="node-1",
        claim_id="claim-1",
        writer_finished=True,
        route=route,
    )

    sql = str(_cleanup_dataset_lock_statement(claim).compile(dialect=mysql.dialect()))

    assert "LOCK IN SHARE MODE" in sql


def test_new_cleanup_is_selected_ahead_of_more_than_one_batch_of_retries(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    attempted_at = datetime(2026, 10, 24)
    with sqlite_session_factory.begin() as session:
        old_rows = [_cleanup(position, last_attempted_at=attempted_at) for position in range(101)]
        new_row = _cleanup(101)
        session.add_all([*old_rows, new_row])
        new_id = new_row.id
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)

    pending = repository.list_pending_cleanup_ids(limit=100)

    assert len(pending) == 100
    assert pending[0] == new_id


@pytest.mark.parametrize("broker_fails", [False, True])
def test_drain_marks_every_broker_publish_attempt(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    broker_fails: bool,
) -> None:
    with sqlite_session_factory.begin() as session:
        cleanup = _cleanup(1)
        session.add(cleanup)
        cleanup_id = cleanup.id
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)
    published: list[str] = []

    def publish(attempted_id: str) -> None:
        published.append(attempted_id)
        if broker_fails:
            raise OSError("broker unavailable")

    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module.process_child_vector_cleanup_task, "delay", publish)

    task_module.drain_child_vector_cleanup_outbox_task.run()

    assert published == [cleanup_id]
    with sqlite_session_factory() as session:
        assert session.get_one(ChildVectorCleanupOutbox, cleanup_id).last_attempted_at is not None


def test_broker_failure_rotates_a_full_batch_to_new_work(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        rows = [_cleanup(position) for position in range(3)]
        session.add_all(rows)
        cleanup_ids = {row.id for row in rows}
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)
    first_batch = repository.list_pending_cleanup_ids(limit=2)
    (remaining_id,) = cleanup_ids.difference(first_batch)
    published: list[str] = []

    def fail_publish(cleanup_id: str) -> None:
        published.append(cleanup_id)
        raise OSError("broker unavailable")

    monkeypatch.setattr(task_module, "_DRAIN_BATCH_SIZE", 2)
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module.process_child_vector_cleanup_task, "delay", fail_publish)

    task_module.drain_child_vector_cleanup_outbox_task.run()
    second_batch = repository.list_pending_cleanup_ids(limit=2)
    task_module.drain_child_vector_cleanup_outbox_task.run()

    assert tuple(published[:2]) == first_batch
    assert second_batch[0] == remaining_id
    assert tuple(published[2:]) == second_batch


def test_pending_cleanup_excludes_a_live_worker_claim(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        active = _cleanup(1)
        available = _cleanup(2)
        session.add_all([active, available])
        active_id, available_id = active.id, available.id
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)
    claim = repository.claim_pending_cleanup(active_id)
    assert claim is not None

    assert repository.list_pending_cleanup_ids(limit=10) == (available_id,)

    with sqlite_session_factory() as session:
        active_row = session.get_one(ChildVectorCleanupOutbox, active_id)
        assert active_row.claim_id == claim.claim_id
        assert active_row.last_attempted_at is not None
        assert active_row.claim_expires_at is not None
        assert active_row.claim_expires_at > active_row.last_attempted_at


def test_cleanup_uses_persisted_route_after_dataset_switch_and_preserves_same_node_in_new_route(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_route = _route("collection-a")
    current_route = _route("collection-b")
    with sqlite_session_factory.begin() as session:
        session.add(
            Dataset(
                id=current_route.dataset_id,
                tenant_id=current_route.tenant_id,
                name="dataset",
                created_by="author-1",
                indexing_technique=current_route.indexing_technique,
                index_struct=current_route.index_struct,
                collection_binding_id=current_route.collection_binding_id,
            )
        )
        session.add(
            ChildChunk(
                tenant_id=current_route.tenant_id,
                dataset_id=current_route.dataset_id,
                document_id="document-2",
                segment_id="segment-2",
                position=1,
                content="current",
                word_count=1,
                created_by="author-1",
                index_node_id="shared-node",
                vector_route=current_route.serialized,
                vector_route_hash=current_route.key,
            )
        )
        cleanup = _routed_cleanup(old_route)
        session.add(cleanup)
        cleanup_id = cleanup.id
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)
    nodes = {"collection-a": {"shared-node"}, "collection-b": {"shared-node"}}

    class RoutedVector:
        def __init__(self, dataset: Dataset, *, session: Session | None, vector_type: str) -> None:
            assert session is None
            self.target = (vector_type, json.loads(dataset.index_struct)["vector_store"]["class_prefix"])

        def get_routing_target(self) -> tuple[str, str]:
            return self.target

        def delete_by_ids(self, node_ids: list[str]) -> None:
            nodes[self.target[1]].difference_update(node_ids)

    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "Vector", RoutedVector)

    task_module.process_child_vector_cleanup_task.run(cleanup_id)

    assert nodes == {"collection-a": set(), "collection-b": {"shared-node"}}
    with sqlite_session_factory() as session:
        assert session.get(ChildVectorCleanupOutbox, cleanup_id) is None


def test_reference_guard_treats_binding_changes_to_the_same_physical_collection_as_live(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    cleanup_route = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="shared-collection",
        collection_binding_id="binding-old",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    live_route = cleanup_route.model_copy(update={"collection_binding_id": "binding-current"})
    assert live_route.key == cleanup_route.key
    with sqlite_session_factory.begin() as session:
        session.add(
            Dataset(
                id=live_route.dataset_id,
                tenant_id=live_route.tenant_id,
                name="dataset",
                created_by="author-1",
                indexing_technique=live_route.indexing_technique,
                index_struct=live_route.index_struct,
                collection_binding_id=live_route.collection_binding_id,
            )
        )
        session.add(
            ChildChunk(
                tenant_id=live_route.tenant_id,
                dataset_id=live_route.dataset_id,
                document_id="document-2",
                segment_id="segment-2",
                position=1,
                content="current",
                word_count=1,
                created_by="author-1",
                index_node_id="shared-node",
                vector_route=live_route.serialized,
                vector_route_hash=live_route.key,
            )
        )
        cleanup = _routed_cleanup(cleanup_route)
        session.add(cleanup)
        cleanup_id = cleanup.id
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)
    claim = repository.claim_pending_cleanup(cleanup_id)
    assert claim is not None

    with repository.guard_cleanup_delete(claim) as referenced:
        assert referenced


def test_referenced_cleanup_and_detach_cannot_commit_without_a_receipt(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    """SQLite's snapshot conflict must preserve either the reference or a cleanup receipt."""
    route = _route("shared-collection")
    child_id = "child-current"
    with sqlite_session_factory.begin() as session:
        session.add(
            Dataset(
                id=route.dataset_id,
                tenant_id=route.tenant_id,
                name="dataset",
                created_by="author-1",
                indexing_technique=route.indexing_technique,
                index_struct=route.index_struct,
                collection_binding_id=route.collection_binding_id,
            )
        )
        child = ChildChunk(
            tenant_id=route.tenant_id,
            dataset_id=route.dataset_id,
            document_id="document-1",
            segment_id="segment-1",
            position=1,
            content="current",
            word_count=1,
            created_by="author-1",
            index_node_id="shared-node",
            vector_route=route.serialized,
            vector_route_hash=route.key,
        )
        child.id = child_id
        session.add(child)
        cleanup = _routed_cleanup(route)
        session.add(cleanup)
        cleanup_id = cleanup.id
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)
    claim = repository.claim_pending_cleanup(cleanup_id)
    assert claim is not None
    ref = DatasetRef(route.tenant_id, route.dataset_id).document("document-1").segment("segment-1")

    try:
        with repository.guard_cleanup_delete(claim) as referenced:
            assert referenced is True
            try:
                with sqlite_session_factory.begin() as detach:
                    child = detach.scalar(select(ChildChunk).where(ChildChunk.id == child_id).with_for_update())
                    assert child is not None
                    repository.add_terminal_cleanups(
                        detach,
                        ref=ref,
                        obsolete_nodes=((child.id, "shared-node", route),),
                    )
                    detach.delete(child)
            except OperationalError:
                # SQLite may reject the second writer before the guard upgrades
                # its snapshot. Either serialization winner is safe below.
                pass
    except (ChildVectorCleanupClaimLostError, OperationalError):
        pass

    with sqlite_session_factory() as observer:
        reference_exists = observer.get(ChildChunk, child_id) is not None
        receipt_exists = (
            observer.scalar(
                select(ChildVectorCleanupOutbox.id).where(
                    ChildVectorCleanupOutbox.tenant_id == route.tenant_id,
                    ChildVectorCleanupOutbox.node_id == "shared-node",
                    ChildVectorCleanupOutbox.vector_route_hash == route.key,
                )
            )
            is not None
        )
    assert reference_exists or receipt_exists


def test_cleanup_task_keeps_repository_guard_open_during_vector_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit control-flow check; InnoDB lock semantics belong to MySQL integration tests."""
    route = _route("collection-a")
    claim = ChildVectorCleanupClaim(
        id="cleanup-1",
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        node_id="shared-node",
        claim_id="claim-1",
        writer_finished=True,
        route=route,
    )
    dataset_lock = Lock()
    delete_started = Event()
    allow_delete = Event()
    rebuild_attempted = Event()
    rebuild_acquired = Event()
    order: list[str] = []

    class Repository:
        def claim_pending_cleanup(self, cleanup_id: str) -> ChildVectorCleanupClaim:
            assert cleanup_id == claim.id
            return claim

        @contextmanager
        def guard_cleanup_delete(self, guarded_claim: ChildVectorCleanupClaim) -> Generator[bool]:
            assert guarded_claim == claim
            with dataset_lock:
                order.append("reference-checked")
                yield False
                order.append("acknowledged")

        def renew_cleanup(self, guarded_claim: ChildVectorCleanupClaim) -> None:
            assert guarded_claim == claim

        def release_cleanup(self, guarded_claim: ChildVectorCleanupClaim) -> bool:
            assert guarded_claim == claim
            return True

    class CleanupVector:
        def delete_by_ids(self, node_ids: list[str]) -> None:
            assert node_ids == [claim.node_id]
            order.append("delete-started")
            delete_started.set()
            assert allow_delete.wait(timeout=1)
            order.append("delete-finished")

    def rebuild() -> None:
        assert delete_started.wait(timeout=1)
        rebuild_attempted.set()
        with dataset_lock:
            order.append("rebuild-acquired")
            rebuild_acquired.set()

    monkeypatch.setattr(task_module, "_repository", Repository)
    monkeypatch.setattr(task_module, "_vector_for_route", lambda _route: CleanupVector())
    cleanup_thread = Thread(target=task_module.process_child_vector_cleanup_task.run, args=(claim.id,))
    rebuild_thread = Thread(target=rebuild)

    cleanup_thread.start()
    assert delete_started.wait(timeout=1)
    rebuild_thread.start()
    assert rebuild_attempted.wait(timeout=1)
    assert not rebuild_acquired.wait(timeout=0.05)
    allow_delete.set()
    cleanup_thread.join(timeout=1)
    rebuild_thread.join(timeout=1)

    assert not cleanup_thread.is_alive()
    assert not rebuild_thread.is_alive()
    assert order == [
        "reference-checked",
        "delete-started",
        "delete-finished",
        "acknowledged",
        "rebuild-acquired",
    ]


def test_cleanup_uses_route_without_live_dataset(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route("deleted-dataset-collection")
    with sqlite_session_factory.begin() as session:
        cleanup = _routed_cleanup(route, node_id="orphan-node")
        session.add(cleanup)
        cleanup_id = cleanup.id
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)
    deleted: list[tuple[str, str]] = []

    class RoutedVector:
        def __init__(self, dataset: Dataset, *, session: Session | None, vector_type: str) -> None:
            assert session is None
            self.target = (vector_type, json.loads(dataset.index_struct)["vector_store"]["class_prefix"])

        def get_routing_target(self) -> tuple[str, str]:
            return self.target

        def delete_by_ids(self, node_ids: list[str]) -> None:
            deleted.extend((self.target[1], node_id) for node_id in node_ids)

    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "Vector", RoutedVector)

    task_module.process_child_vector_cleanup_task.run(cleanup_id)

    assert deleted == [("deleted-dataset-collection", "orphan-node")]
    with sqlite_session_factory() as session:
        assert session.get(ChildVectorCleanupOutbox, cleanup_id) is None


def test_cleanup_does_not_ack_when_persisted_route_resolves_elsewhere(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route("collection-a")
    with sqlite_session_factory.begin() as session:
        cleanup = _routed_cleanup(route)
        session.add(cleanup)
        cleanup_id = cleanup.id
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)

    class WrongVector:
        def __init__(self, dataset: Dataset, *, session: Session | None, vector_type: str) -> None:
            del dataset, session, vector_type

        def get_routing_target(self) -> tuple[str, str]:
            return "qdrant", "collection-b"

    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "Vector", WrongVector)

    with pytest.raises(RuntimeError, match="did not resolve"):
        task_module.process_child_vector_cleanup_task.run(cleanup_id)

    with sqlite_session_factory() as session:
        retained = session.get_one(ChildVectorCleanupOutbox, cleanup_id)
        assert retained.claim_id is None


def test_legacy_cleanup_without_route_is_retained_without_deleting(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        cleanup = _cleanup(1)
        session.add(cleanup)
        cleanup_id = cleanup.id
    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)

    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "Vector", pytest.fail)

    task_module.process_child_vector_cleanup_task.run(cleanup_id)

    with sqlite_session_factory() as session:
        retained = session.get_one(ChildVectorCleanupOutbox, cleanup_id)
        assert retained.claim_id is None

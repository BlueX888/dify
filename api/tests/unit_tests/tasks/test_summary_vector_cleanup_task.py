from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from threading import Event
from time import monotonic
from typing import override

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

import tasks.summary_vector_cleanup_task as task_module
from models.dataset import DocumentSegmentSummary
from models.model import SummaryVectorCleanupOutbox
from repositories.knowledge.summary_vector_cleanup_repository import (
    SQLAlchemySummaryVectorCleanupRepository,
    SummaryVectorCleanupClaimLostError,
    _pending_cleanup_ids_statement,
)
from services.knowledge.entities.summaries import SummaryVectorCleanupClaim
from services.knowledge.entities.vector_route import VectorRoute
from tests.unit_tests.model_factories import make_dataset


class RepositoryProbe(SQLAlchemySummaryVectorCleanupRepository):
    def __init__(
        self,
        claim: SummaryVectorCleanupClaim | None,
        *,
        referenced: bool = False,
        pending_ids: tuple[str, ...] = (),
    ) -> None:
        self.claim = claim
        self.referenced = referenced
        self.pending_ids = pending_ids
        self.attempted: list[str] = []
        self.finished: list[SummaryVectorCleanupClaim] = []
        self.released: list[SummaryVectorCleanupClaim] = []
        self.renewed: list[SummaryVectorCleanupClaim] = []
        self.renew_hook: Callable[[SummaryVectorCleanupClaim], None] | None = None

    @override
    def claim_pending_cleanup(self, cleanup_id: str) -> SummaryVectorCleanupClaim | None:
        if self.claim is not None:
            assert cleanup_id == self.claim.id
        return self.claim

    @contextmanager
    @override
    def guard_cleanup_delete(self, claim: SummaryVectorCleanupClaim) -> Generator[bool]:
        assert claim == self.claim
        yield self.referenced
        if claim.writer_finished:
            self.finished.append(claim)

    @override
    def release_cleanup(self, claim: SummaryVectorCleanupClaim) -> bool:
        self.released.append(claim)
        return True

    @override
    def renew_cleanup(self, claim: SummaryVectorCleanupClaim) -> None:
        self.renewed.append(claim)
        if self.renew_hook is not None:
            self.renew_hook(claim)

    @override
    def list_pending_cleanup_ids(self, *, limit: int) -> tuple[str, ...]:
        assert limit > 0
        return self.pending_ids

    @override
    def mark_attempted(self, cleanup_id: str) -> None:
        self.attempted.append(cleanup_id)


class VectorProbe:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.deleted: list[list[str]] = []

    def delete_by_ids(self, node_ids: list[str]) -> None:
        self.deleted.append(node_ids)
        if self.error is not None:
            raise self.error


class RouteResolverProbe:
    def __init__(self, vector: VectorProbe | None = None, *, error: Exception | None = None) -> None:
        self.vector = vector
        self.error = error
        self.routes: list[VectorRoute] = []

    def __call__(self, route: VectorRoute) -> VectorProbe:
        self.routes.append(route)
        if self.error is not None:
            raise self.error
        assert self.vector is not None
        return self.vector


def _route(collection_name: str = "summary-a") -> VectorRoute:
    return VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name=collection_name,
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="summary",
    )


def _claim(*, writer_finished: bool = True, route: VectorRoute | None = None) -> SummaryVectorCleanupClaim:
    return SummaryVectorCleanupClaim(
        id="cleanup-1",
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        node_id="node-1",
        claim_id="claim-1",
        writer_finished=writer_finished,
        route=_route() if route is None else route,
    )


def test_referenced_node_is_acknowledged_without_vector_deletion(monkeypatch: pytest.MonkeyPatch) -> None:
    claim = _claim()
    repository = RepositoryProbe(claim, referenced=True)
    vector_for_route = RouteResolverProbe()
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", vector_for_route)

    task_module.process_summary_vector_cleanup_task.run("cleanup-1")

    assert repository.finished == [claim]
    assert vector_for_route.routes == []


def test_unreferenced_node_is_deleted_before_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim()
    repository = RepositoryProbe(claim)
    vector = VectorProbe()
    vector_for_route = RouteResolverProbe(vector)
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", vector_for_route)

    task_module.process_summary_vector_cleanup_task.run("cleanup-1")

    assert vector_for_route.routes == [_route()]
    assert vector.deleted == [["node-1"]]
    assert repository.finished == [claim]


def test_delete_failure_releases_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim()
    repository = RepositoryProbe(claim)
    vector = VectorProbe(error=RuntimeError("vector unavailable"))
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", RouteResolverProbe(vector))

    with pytest.raises(RuntimeError, match="vector unavailable"):
        task_module.process_summary_vector_cleanup_task.run("cleanup-1")

    assert repository.released == [claim]
    assert repository.finished == []


def test_unfinished_writer_is_deleted_but_never_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim(writer_finished=False)
    repository = RepositoryProbe(claim)
    vector = VectorProbe()
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", RouteResolverProbe(vector))

    task_module.process_summary_vector_cleanup_task.run("cleanup-1")

    assert vector.deleted == [["node-1"]]
    assert repository.finished == []
    assert repository.released == [claim]


@pytest.mark.parametrize(
    ("writer_finished", "acknowledged"),
    [(False, False), (True, True)],
)
def test_missing_dataset_still_uses_persisted_route_and_acknowledges_only_terminal_writers(
    monkeypatch: pytest.MonkeyPatch,
    writer_finished: bool,
    acknowledged: bool,
) -> None:
    claim = _claim(writer_finished=writer_finished)
    repository = RepositoryProbe(claim)
    vector = VectorProbe()
    vector_for_route = RouteResolverProbe(vector)
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", vector_for_route)

    task_module.process_summary_vector_cleanup_task.run("cleanup-1")

    assert vector_for_route.routes == [_route()]
    assert vector.deleted == [["node-1"]]
    if acknowledged:
        assert repository.finished == [claim]
        assert repository.released == []
    else:
        assert repository.finished == []
        assert repository.released == [claim]


def test_missing_dataset_unfinished_writer_converges_after_writer_finishes(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    with sqlite_session_factory.begin() as session:
        cleanup = SummaryVectorCleanupOutbox(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            summary_id="summary-1",
            summary_version=1,
            node_id="node-1",
            writer_id="writer-1",
            vector_route=route.serialized,
            vector_route_hash=route.key,
        )
        session.add(cleanup)
        cleanup_id = cleanup.id
    repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
    vector = VectorProbe()
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", RouteResolverProbe(vector))

    task_module.process_summary_vector_cleanup_task.run(cleanup_id)

    with sqlite_session_factory.begin() as session:
        retained = session.get_one(SummaryVectorCleanupOutbox, cleanup_id)
        assert retained.claim_id is None
        retained.writer_finished_at = retained.created_at

    task_module.process_summary_vector_cleanup_task.run(cleanup_id)

    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, cleanup_id) is None
    assert vector.deleted == [["node-1"], ["node-1"]]


def test_cleanup_uses_persisted_route_after_dataset_switches_collection(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_a = _route("summary-a")
    route_b = _route("summary-b")
    with sqlite_session_factory.begin() as session:
        dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.index_struct = route_b.index_struct
        dataset.collection_binding_id = "binding-b"
        session.add(dataset)
        cleanup = SummaryVectorCleanupOutbox(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            summary_id="summary-1",
            summary_version=1,
            node_id="shared-node",
            writer_id="writer-1",
            vector_route=route_a.serialized,
            vector_route_hash=route_a.key,
            writer_finished_at=datetime(2026, 1, 1),
        )
        session.add(cleanup)
        cleanup_id = cleanup.id

    repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
    vector_a = VectorProbe()
    resolve_route = RouteResolverProbe(vector_a)
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", resolve_route)

    task_module.process_summary_vector_cleanup_task.run(cleanup_id)

    assert resolve_route.routes == [route_a]
    assert vector_a.deleted == [["shared-node"]]
    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, cleanup_id) is None


def test_same_physical_route_with_new_binding_protects_current_reference(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_route = _route("shared-summary")
    current_route = VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="shared-summary",
        collection_binding_id="binding-b",
        indexing_technique="high_quality",
        doc_form="summary-v2",
    )
    assert current_route.key == old_route.key
    with sqlite_session_factory.begin() as session:
        session.add(make_dataset(dataset_id="dataset-1", tenant_id="tenant-1"))
        summary = DocumentSegmentSummary(
            dataset_id="dataset-1",
            document_id="document-1",
            chunk_id="segment-1",
            summary_content="summary",
            summary_index_node_id="shared-node",
            vector_route=current_route.serialized,
            vector_route_hash=current_route.key,
            enabled=True,
        )
        session.add(summary)
        cleanup = SummaryVectorCleanupOutbox(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            summary_id=summary.id,
            summary_version=1,
            node_id="shared-node",
            writer_id="writer-1",
            vector_route=old_route.serialized,
            vector_route_hash=old_route.key,
            writer_finished_at=datetime(2026, 1, 1),
        )
        session.add(cleanup)
        cleanup_id = cleanup.id

    repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
    resolve_route = RouteResolverProbe()
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", resolve_route)

    task_module.process_summary_vector_cleanup_task.run(cleanup_id)

    assert resolve_route.routes == []
    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, cleanup_id) is None


def test_referenced_summary_cleanup_and_detach_cannot_commit_without_a_receipt(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    """A detach after the reference snapshot cannot consume the only receipt."""
    route = _route("shared-summary")
    with sqlite_session_factory.begin() as session:
        session.add(make_dataset(dataset_id=route.dataset_id, tenant_id=route.tenant_id))
        summary = DocumentSegmentSummary(
            dataset_id=route.dataset_id,
            document_id="document-1",
            chunk_id="segment-1",
            summary_content="summary",
            summary_index_node_id="shared-node",
            vector_route=route.serialized,
            vector_route_hash=route.key,
            enabled=True,
        )
        session.add(summary)
        cleanup = SummaryVectorCleanupOutbox(
            tenant_id=route.tenant_id,
            dataset_id=route.dataset_id,
            document_id="document-1",
            segment_id="segment-1",
            summary_id=summary.id,
            summary_version=summary.version,
            node_id="shared-node",
            writer_id="writer-1",
            vector_route=route.serialized,
            vector_route_hash=route.key,
            writer_finished_at=datetime(2026, 1, 1),
        )
        session.add(cleanup)
        summary_id, cleanup_id = summary.id, cleanup.id
    repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
    claim = repository.claim_pending_cleanup(cleanup_id)
    assert claim is not None

    try:
        with repository.guard_cleanup_delete(claim) as referenced:
            assert referenced is True
            try:
                with sqlite_session_factory.begin() as detach:
                    summary = detach.scalar(
                        select(DocumentSegmentSummary).where(DocumentSegmentSummary.id == summary_id).with_for_update()
                    )
                    assert summary is not None
                    repository.add_terminal_cleanups(detach, tenant_id=route.tenant_id, summaries=(summary,))
                    detach.delete(summary)
            except OperationalError:
                pass
    except (OperationalError, SummaryVectorCleanupClaimLostError):
        pass

    with sqlite_session_factory() as observer:
        reference_exists = observer.get(DocumentSegmentSummary, summary_id) is not None
        receipt_exists = (
            observer.scalar(
                select(SummaryVectorCleanupOutbox.id).where(
                    SummaryVectorCleanupOutbox.tenant_id == route.tenant_id,
                    SummaryVectorCleanupOutbox.node_id == "shared-node",
                    SummaryVectorCleanupOutbox.vector_route_hash == route.key,
                )
            )
            is not None
        )
    assert reference_exists or receipt_exists


def test_cleanup_without_trustworthy_route_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    claim = _claim(route=None)
    # ``_claim`` uses its default for None; construct the legacy evidence explicitly.
    claim = SummaryVectorCleanupClaim(
        id=claim.id,
        tenant_id=claim.tenant_id,
        dataset_id=claim.dataset_id,
        node_id=claim.node_id,
        claim_id=claim.claim_id,
        writer_finished=True,
        route=None,
    )
    repository = RepositoryProbe(claim)
    vector_for_route = RouteResolverProbe()
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", vector_for_route)

    task_module.process_summary_vector_cleanup_task.run(claim.id)

    assert vector_for_route.routes == []
    assert repository.released == [claim]
    assert repository.finished == []


def test_wrong_physical_route_resolution_does_not_acknowledge(monkeypatch: pytest.MonkeyPatch) -> None:
    claim = _claim()
    repository = RepositoryProbe(claim)
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(
        task_module,
        "vector_for_route",
        RouteResolverProbe(error=ValueError("route mismatch")),
    )

    with pytest.raises(ValueError, match="route mismatch"):
        task_module.process_summary_vector_cleanup_task.run(claim.id)

    assert repository.released == [claim]
    assert repository.finished == []


def test_drain_publishes_only_claimable_cleanup_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = RepositoryProbe(None, pending_ids=("cleanup-1", "cleanup-2"))
    published: list[str] = []
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "enqueue_summary_vector_cleanup", published.append)

    task_module.drain_summary_vector_cleanup_outbox_task.run()

    assert published == ["cleanup-1", "cleanup-2"]
    assert repository.attempted == published


def _retry_cleanup(position: int, *, attempted_at: datetime | None = None) -> SummaryVectorCleanupOutbox:
    return SummaryVectorCleanupOutbox(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=f"segment-{position}",
        summary_id=f"summary-{position}",
        summary_version=1,
        node_id=f"node-{position}",
        writer_id=f"writer-{position}",
        writer_finished_at=datetime(2026, 10, 24),
        last_attempted_at=attempted_at,
    )


def test_pending_cleanup_uses_null_first_order_on_postgresql() -> None:
    sql = str(
        _pending_cleanup_ids_statement(datetime(2026, 10, 25), limit=100).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )

    assert "CASE WHEN (summary_vector_cleanup_outbox.last_attempted_at IS NULL) THEN 0 ELSE 1 END" in sql


def test_new_summary_cleanup_passes_more_than_one_batch_of_unknown_route_retries(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    with sqlite_session_factory.begin() as session:
        retries = [_retry_cleanup(position, attempted_at=datetime(2026, 10, 24)) for position in range(101)]
        fresh = _retry_cleanup(101)
        fresh.vector_route = route.serialized
        fresh.vector_route_hash = route.key
        session.add_all([*retries, fresh])
        fresh_id = fresh.id
        retry_ids = {row.id for row in retries}
    repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
    vector = VectorProbe()
    published: list[str] = []

    def consume(cleanup_id: str) -> None:
        published.append(cleanup_id)
        task_module.process_summary_vector_cleanup_task.run(cleanup_id)

    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module, "vector_for_route", RouteResolverProbe(vector))
    monkeypatch.setattr(task_module.process_summary_vector_cleanup_task, "delay", consume)

    task_module.drain_summary_vector_cleanup_outbox_task.run()

    assert len(published) == 100
    assert published[0] == fresh_id
    assert vector.deleted == [["node-101"]]
    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, fresh_id) is None
        assert all(session.get(SummaryVectorCleanupOutbox, cleanup_id) is not None for cleanup_id in retry_ids)


@pytest.mark.parametrize("broker_fails", [False, True])
def test_summary_cleanup_drainer_records_every_publication_attempt(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    broker_fails: bool,
) -> None:
    with sqlite_session_factory.begin() as session:
        row = _retry_cleanup(1)
        session.add(row)
        cleanup_id = row.id
    repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
    published: list[str] = []

    def publish(attempted_id: str) -> None:
        published.append(attempted_id)
        if broker_fails:
            raise OSError("broker unavailable")

    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module.process_summary_vector_cleanup_task, "delay", publish)

    task_module.drain_summary_vector_cleanup_outbox_task.run()

    assert published == [cleanup_id]
    with sqlite_session_factory() as session:
        assert session.get_one(SummaryVectorCleanupOutbox, cleanup_id).last_attempted_at is not None


def test_rejected_summary_cleanup_batch_rotates_to_the_remaining_work(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        rows = [_retry_cleanup(position) for position in range(101)]
        session.add_all(rows)
        cleanup_ids = {row.id for row in rows}
    repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)
    first_batch = repository.list_pending_cleanup_ids(limit=100)
    (remaining_id,) = cleanup_ids.difference(first_batch)
    published: list[str] = []

    def reject(cleanup_id: str) -> None:
        published.append(cleanup_id)
        raise OSError("broker unavailable")

    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    monkeypatch.setattr(task_module.process_summary_vector_cleanup_task, "delay", reject)

    task_module.drain_summary_vector_cleanup_outbox_task.run()
    second_batch = repository.list_pending_cleanup_ids(limit=100)
    task_module.drain_summary_vector_cleanup_outbox_task.run()

    assert tuple(published[:100]) == first_batch
    assert second_batch[0] == remaining_id
    assert tuple(published[100:]) == second_batch


def test_cleanup_heartbeat_shutdown_does_not_wait_for_blocked_renewal() -> None:
    started = Event()
    unblock = Event()
    repository = RepositoryProbe(_claim())

    def blocked_renewal(_claim: SummaryVectorCleanupClaim) -> None:
        started.set()
        unblock.wait(timeout=5)

    repository.renew_hook = blocked_renewal
    before = monotonic()
    with task_module._maintain_cleanup_claim(repository, _claim(), interval_seconds=0.001):
        assert started.wait(timeout=1)
    elapsed = monotonic() - before
    unblock.set()

    assert elapsed < 1

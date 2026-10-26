from collections.abc import Generator
from contextlib import contextmanager

import pytest

import services.knowledge.summaries.application as application_module
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.summaries import (
    SummaryPublicationResult,
    SummaryVectorCleanupClaim,
    SummaryVectorInput,
    SummaryVectorResult,
    SummaryVectorWriteIntent,
)
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.summaries.application import SummaryIndexService, SummaryVectorGenerationLostError


class PreparedVectorProbe:
    def __init__(self, route: VectorRoute, events: list[str], *, error: Exception | None = None) -> None:
        self.route = route
        self.events = events
        self.error = error
        self.submit_count = 0

    def submit(self) -> None:
        self.submit_count += 1
        self.events.append("submit")
        if self.error is not None:
            raise self.error


class SummaryStoreProbe:
    def __init__(
        self,
        events: list[str],
        intent: SummaryVectorWriteIntent,
        claim: SummaryVectorCleanupClaim,
    ) -> None:
        self.events = events
        self.intent = intent
        self.claim = claim
        self.begin_count = 0
        self.failed_calls: list[str] = []
        self.discarded: list[SummaryVectorWriteIntent] = []

    def begin_write(
        self,
        command: SummaryVectorInput,
        result: SummaryVectorResult,
        route: VectorRoute,
    ) -> SummaryVectorWriteIntent:
        del command, result
        assert route == self.intent.route
        self.begin_count += 1
        self.events.append("begin")
        return self.intent

    def discard_unsubmitted_write(self, intent: SummaryVectorWriteIntent) -> None:
        assert intent == self.intent
        self.discarded.append(intent)
        self.events.append("discard")

    def renew_write(self, intent: SummaryVectorWriteIntent) -> None:
        assert intent == self.intent

    def finish_write(self, intent: SummaryVectorWriteIntent) -> SummaryVectorCleanupClaim:
        assert intent == self.intent
        self.events.append("finish")
        return self.claim

    def completed(
        self,
        command: SummaryVectorInput,
        result: SummaryVectorResult,
        claim: SummaryVectorCleanupClaim,
    ) -> SummaryPublicationResult:
        del command, result
        assert claim == self.claim
        self.events.append("complete")
        return SummaryPublicationResult(True)

    def is_published(
        self,
        command: SummaryVectorInput,
        result: SummaryVectorResult,
        claim: SummaryVectorCleanupClaim,
    ) -> bool:
        del command, result, claim
        return False

    @contextmanager
    def guard_cleanup_delete(self, claim: SummaryVectorCleanupClaim) -> Generator[bool]:
        assert claim == self.claim
        yield False

    def release_cleanup(self, claim: SummaryVectorCleanupClaim) -> bool:
        assert claim == self.claim
        return True

    def failed(self, command: SummaryVectorInput, error: str) -> bool:
        del command
        self.failed_calls.append(error)
        return True


class SummaryVectorBackendProbe:
    def __init__(self, prepared: PreparedVectorProbe) -> None:
        self.prepared = prepared
        self.prepare_count = 0

    def delete(self, claim: SummaryVectorCleanupClaim) -> None:
        raise AssertionError(f"Unexpected cleanup for {claim.node_id}")

    def count_tokens(self, content: str) -> int:
        assert content
        return 1

    def prepare(self, command: SummaryVectorInput, result: SummaryVectorResult) -> PreparedVectorProbe:
        del command, result
        self.prepare_count += 1
        self.prepared.events.append("prepare")
        return self.prepared


def _command() -> SummaryVectorInput:
    return SummaryVectorInput(
        DatasetRef("tenant-1", "dataset-1").document("document-1").segment("segment-1"),
        "summary-1",
        1,
        "summary",
        None,
    )


def _service(
    events: list[str],
) -> tuple[SummaryIndexService, SummaryStoreProbe, SummaryVectorBackendProbe]:
    route = VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="collection-1",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    intent = SummaryVectorWriteIntent("cleanup-1", "tenant-1", "dataset-1", "node-1", "writer-1", "claim-1", route)
    claim = SummaryVectorCleanupClaim("cleanup-1", "tenant-1", "dataset-1", "node-1", "claim-1", True, route)
    store = SummaryStoreProbe(events, intent, claim)
    prepared = PreparedVectorProbe(route, events)
    vectors = SummaryVectorBackendProbe(prepared)
    service = SummaryIndexService(
        store=store,
        vectors=vectors,
        new_id=lambda: "node-1",
        text_hash=lambda _content: "hash-1",
        sleep=lambda _seconds: None,
    )
    return service, store, vectors


def test_writer_is_registered_before_heartbeat_and_generation_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    service, _store, _vectors = _service(events)

    @contextmanager
    def heartbeat(*_args: object, **_kwargs: object) -> Generator[None]:
        events.append("heartbeat-enter")
        try:
            yield
        finally:
            events.append("heartbeat-exit")

    @contextmanager
    def generation_guard() -> Generator[None]:
        events.append("guard-enter")
        try:
            yield
        finally:
            events.append("guard-exit")

    monkeypatch.setattr(application_module, "_maintain_write_claim", heartbeat)

    result = service.vectorize(_command(), mutation_guard=generation_guard)

    assert result is not None
    assert events == [
        "prepare",
        "begin",
        "heartbeat-enter",
        "guard-enter",
        "submit",
        "finish",
        "complete",
        "guard-exit",
        "heartbeat-exit",
    ]


def test_generation_loss_discards_registered_unsubmitted_writer_without_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    service, store, vectors = _service(events)

    @contextmanager
    def heartbeat(*_args: object, **_kwargs: object) -> Generator[None]:
        events.append("heartbeat-enter")
        try:
            yield
        finally:
            events.append("heartbeat-exit")

    @contextmanager
    def lost_generation_guard() -> Generator[None]:
        events.append("guard-enter")
        raise SummaryVectorGenerationLostError()
        yield  # pragma: no cover

    monkeypatch.setattr(application_module, "_maintain_write_claim", heartbeat)

    result = service.vectorize(_command(), mutation_guard=lost_generation_guard)

    assert result is None
    assert events == ["prepare", "begin", "heartbeat-enter", "guard-enter", "heartbeat-exit", "discard"]
    assert vectors.prepare_count == 1
    assert store.failed_calls == []


def test_network_failure_cannot_publish_error_after_parent_generation_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    service, store, vectors = _service(events)
    prepared = PreparedVectorProbe(
        vectors.prepared.route,
        events,
        error=RuntimeError("network unavailable"),
    )
    vectors.prepared = prepared
    guard_entries = 0

    @contextmanager
    def heartbeat(*_args: object, **_kwargs: object) -> Generator[None]:
        yield

    @contextmanager
    def generation_guard() -> Generator[None]:
        nonlocal guard_entries
        guard_entries += 1
        if guard_entries == 4:
            raise SummaryVectorGenerationLostError()
        yield

    monkeypatch.setattr(application_module, "_maintain_write_claim", heartbeat)

    result = service.vectorize(_command(), mutation_guard=generation_guard)

    assert result is None
    assert guard_entries == 4
    assert prepared.submit_count == 3
    assert store.begin_count == 3
    assert store.failed_calls == []
    assert store.discarded == []

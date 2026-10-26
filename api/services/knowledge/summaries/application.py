"""Summary vectorization policy independent of ORM, model providers, and vector stores."""

import logging
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from threading import Event, Thread
from typing import Protocol

from services.knowledge.entities.summaries import (
    SummaryPublicationResult,
    SummaryVectorCleanupClaim,
    SummaryVectorGenerationLostError,
    SummaryVectorInput,
    SummaryVectorResult,
    SummaryVectorWriteIntent,
)
from services.knowledge.entities.vector_route import VectorRoute

logger = logging.getLogger(__name__)

SUMMARY_VECTOR_CLEANUP_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


class SummaryStore(Protocol):
    def begin_write(
        self,
        command: SummaryVectorInput,
        result: SummaryVectorResult,
        route: VectorRoute,
    ) -> SummaryVectorWriteIntent: ...
    def discard_unsubmitted_write(self, intent: SummaryVectorWriteIntent) -> None: ...
    def renew_write(self, intent: SummaryVectorWriteIntent) -> None: ...
    def finish_write(self, intent: SummaryVectorWriteIntent) -> SummaryVectorCleanupClaim: ...
    def completed(
        self,
        command: SummaryVectorInput,
        result: SummaryVectorResult,
        claim: SummaryVectorCleanupClaim,
    ) -> SummaryPublicationResult: ...
    def is_published(
        self,
        command: SummaryVectorInput,
        result: SummaryVectorResult,
        claim: SummaryVectorCleanupClaim,
    ) -> bool: ...
    def guard_cleanup_delete(self, claim: SummaryVectorCleanupClaim) -> AbstractContextManager[bool]: ...
    def release_cleanup(self, claim: SummaryVectorCleanupClaim) -> bool: ...
    def failed(self, command: SummaryVectorInput, error: str) -> bool: ...


class PreparedSummaryVectorWrite(Protocol):
    """A prepared write with an explicit external mutation boundary.

    ``submit`` may report ``SummaryVectorWriteNotSubmittedError`` only when it
    can prove the mutation boundary was not crossed. Every other exception is
    treated as if the external write may still complete.
    """

    @property
    def route(self) -> VectorRoute: ...

    def submit(self) -> None: ...


class SummaryVectorWriteNotSubmittedError(RuntimeError):
    """The vector backend proved that its mutation boundary was not crossed."""


class SummaryVectorBackend(Protocol):
    def delete(self, claim: SummaryVectorCleanupClaim) -> None: ...
    def count_tokens(self, content: str) -> int: ...
    def prepare(self, command: SummaryVectorInput, result: SummaryVectorResult) -> PreparedSummaryVectorWrite: ...


@contextmanager
def _maintain_write_claim(
    store: SummaryStore,
    intent: SummaryVectorWriteIntent,
    *,
    interval_seconds: float = SUMMARY_VECTOR_CLEANUP_HEARTBEAT_SECONDS,
) -> Generator[None]:
    if interval_seconds <= 0:
        raise ValueError("summary writer heartbeat interval must be positive")
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                store.renew_write(intent)
            except Exception:
                logger.exception("Summary vector writer lost ownership, cleanup_id=%s", intent.id)
                return

    thread = Thread(target=heartbeat, name=f"summary-vector-heartbeat-{intent.id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning("Summary vector writer heartbeat did not stop promptly, cleanup_id=%s", intent.id)


class SummaryIndexService:
    def __init__(
        self,
        *,
        store: SummaryStore,
        vectors: SummaryVectorBackend,
        new_id: Callable[[], str],
        text_hash: Callable[[str], str],
        sleep: Callable[[float], None],
    ) -> None:
        self._store, self._vectors = store, vectors
        self._new_id, self._hash, self._sleep = new_id, text_hash, sleep

    def vectorize(
        self,
        command: SummaryVectorInput,
        *,
        mutation_guard: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> SummaryVectorResult | None:
        if not command.content.strip():
            raise ValueError(f"Summary content is empty for segment {command.ref.segment_id}, cannot vectorize")
        tokens = 0
        try:
            tokens = self._vectors.count_tokens(command.content)
        except Exception:
            logger.warning("Failed to calculate summary embedding tokens", exc_info=True)
        content_hash = self._hash(command.content)
        for attempt in range(3):
            # Each submission attempt needs its own node and writer identity.
            # Once submit starts, an earlier attempt may still write after this
            # process has moved on.
            result = SummaryVectorResult(self._new_id(), content_hash, tokens)
            completion_outcome_unknown = False
            submission_outcome_unknown = False
            cleanup: SummaryVectorCleanupClaim | None = None
            publication: SummaryPublicationResult | None = None
            try:
                prepared = self._vectors.prepare(command, result)
                intent = self._store.begin_write(command, result, prepared.route)
                try:
                    # Register and commit the writer before taking the generation
                    # guard. On MySQL an UPDATE lock queued between two SHARE
                    # requests blocks the second SHARE; holding an outer SHARE
                    # while begin_write opens another session would self-deadlock.
                    with _maintain_write_claim(self._store, intent):
                        with mutation_guard() if mutation_guard is not None else nullcontext():
                            try:
                                prepared.submit()
                            except SummaryVectorWriteNotSubmittedError:
                                self._store.discard_unsubmitted_write(intent)
                                raise
                            except Exception:
                                # Entry into submit is the uncertainty boundary. The
                                # remote backend may complete after the caller sees an
                                # error, so only a later reconciliation pass may ACK.
                                submission_outcome_unknown = True
                                raise
                            # Cleanup may repeatedly delete an unfinished writer's node, but
                            # it cannot acknowledge the row. Only this stable writer identity
                            # can record the terminal marker after the external call returns.
                            cleanup = self._store.finish_write(intent)
                            try:
                                publication = self._store.completed(command, result, cleanup)
                            except Exception:
                                try:
                                    if self._store.is_published(command, result, cleanup):
                                        return result
                                except Exception:
                                    # The transaction may have committed even if the
                                    # caller observed an error. Retain the durable intent
                                    # until a worker can safely resolve the node.
                                    completion_outcome_unknown = True
                                    logger.exception(
                                        "Could not confirm summary completion outcome for %s",
                                        command.ref.segment_id,
                                    )
                                raise
                    # Cleanup takes its own Dataset SHARE lock from another
                    # session. Leave the publication guard first: on MySQL a
                    # queued Dataset UPDATE between two SHARE requests would
                    # otherwise make this worker wait on its own outer lock.
                    assert cleanup is not None
                    assert publication is not None
                    if not publication.published:
                        self._delete_claimed_vector(cleanup)
                        return None
                    if publication.previous_cleanup is not None:
                        self._delete_claimed_vector(publication.previous_cleanup)
                    return result
                except SummaryVectorGenerationLostError:
                    self._store.discard_unsubmitted_write(intent)
                    return None
            except SummaryVectorGenerationLostError:
                return None
            except Exception as error:
                if cleanup is not None and not completion_outcome_unknown and not submission_outcome_unknown:
                    self._delete_claimed_vector(cleanup)
                transient = any(
                    word in str(error).lower()
                    for word in ("connection", "disconnected", "timeout", "network", "weaviate")
                )
                if transient and attempt < 2 and not completion_outcome_unknown:
                    self._sleep(2.0 * 2**attempt)
                    continue
                if not completion_outcome_unknown:
                    try:
                        # The external failure may return after the Dataset
                        # generation guard is released. Reacquire that exact
                        # parent generation before publishing ERROR so an old
                        # worker cannot mark a newer configuration's summary.
                        with mutation_guard() if mutation_guard is not None else nullcontext():
                            self._store.failed(command, f"Vectorization failed: {error}")
                    except SummaryVectorGenerationLostError:
                        return None
                raise
        raise AssertionError("Unreachable vectorization retry state")

    def _delete_claimed_vector(self, claim: SummaryVectorCleanupClaim) -> None:
        try:
            if claim.route is None:
                self._store.release_cleanup(claim)
                return
            with self._store.guard_cleanup_delete(claim) as referenced:
                if not referenced:
                    self._vectors.delete(claim)
            if not claim.writer_finished:
                self._store.release_cleanup(claim)
        except Exception:
            try:
                self._store.release_cleanup(claim)
            except Exception:
                logger.exception("Failed to release summary cleanup claim %s", claim.id)
            logger.warning("Deferred cleanup of summary vector %s", claim.node_id, exc_info=True)

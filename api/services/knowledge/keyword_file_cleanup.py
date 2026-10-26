"""Retryable cleanup for immutable keyword-table files."""

import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, suppress

from repositories.knowledge.keyword_file_cleanup_repository import (
    KeywordFileCleanupArtifact,
    SQLAlchemyKeywordFileCleanupRepository,
)


def cleanup_keyword_file(
    artifact: KeywordFileCleanupArtifact,
    *,
    repository: SQLAlchemyKeywordFileCleanupRepository,
    delete_file: Callable[[str], None],
    publication_lock: Callable[[], AbstractContextManager[object]],
    logger: logging.Logger,
) -> bool:
    """Delete one obsolete key and acknowledge only a known-safe outcome."""
    try:
        with publication_lock():
            current = repository.get(artifact.id)
            if current is None:
                return True
            if repository.is_current_pointer(current):
                # A pre-write cleanup candidate may become the published key.
                # It is safe to forget now; a later replacement transaction
                # will create a fresh intent for the same key.
                repository.complete(current)
                return True
            current = repository.begin_cleanup(current)
            if current is None:
                return True
            with suppress(FileNotFoundError):
                delete_file(current.key)
            if current.writer_id is not None and current.writer_finished_at is None:
                # A timed-out PUT may still materialize after this DELETE. Keep
                # its tombstone so later drains reconcile that uncertain remote
                # outcome. A writer that confirmed PUT completion is safe to
                # acknowledge: UUID publication requires this marker, so its
                # absence permanently fences a late publisher.
                repository.mark_attempted(current)
                return True
            repository.complete(current)
            return True
    except Exception:
        logger.exception(
            "Keyword file cleanup failed; durable intent retained, cleanup_id=%s, dataset_id=%s, key=%s",
            artifact.id,
            artifact.dataset_id,
            artifact.key,
        )
        try:
            repository.mark_attempted(artifact)
        except Exception:
            logger.exception("Failed to update keyword cleanup retry order, cleanup_id=%s", artifact.id)
        return False


def drain_pending_keyword_file_cleanups(
    *,
    repository: SQLAlchemyKeywordFileCleanupRepository,
    delete_file: Callable[[str], None],
    publication_lock: Callable[[KeywordFileCleanupArtifact], AbstractContextManager[object]],
    logger: logging.Logger,
    limit: int,
) -> tuple[str, ...]:
    """Attempt a bounded durable cleanup batch and return retryable ids."""
    failed: list[str] = []
    pending: Sequence[KeywordFileCleanupArtifact] = repository.list_pending(limit=limit)
    for artifact in pending:
        if not cleanup_keyword_file(
            artifact,
            repository=repository,
            delete_file=delete_file,
            publication_lock=lambda artifact=artifact: publication_lock(artifact),
            logger=logger,
        ):
            failed.append(artifact.id)
    return tuple(failed)

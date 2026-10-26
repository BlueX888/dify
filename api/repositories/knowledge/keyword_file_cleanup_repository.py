"""Durable cleanup ownership for immutable keyword-table files."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import case, delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from models.dataset import Dataset, DatasetKeywordTable
from models.model import KeywordFileCleanupOutbox


@dataclass(frozen=True)
class KeywordFileCleanupArtifact:
    id: str
    tenant_id: str
    dataset_id: str
    key: str
    writer_id: str | None
    writer_finished_at: datetime | None
    cleanup_started_at: datetime | None


def _to_artifact(row: KeywordFileCleanupOutbox) -> KeywordFileCleanupArtifact:
    return KeywordFileCleanupArtifact(
        id=row.id,
        tenant_id=row.tenant_id,
        dataset_id=row.dataset_id,
        key=row.key,
        writer_id=row.writer_id,
        writer_finished_at=row.writer_finished_at,
        cleanup_started_at=row.cleanup_started_at,
    )


class KeywordFilePublicationLostError(RuntimeError):
    """A cleanup worker fenced a candidate before it could be published."""


def _is_managed_keyword_file_key(*, tenant_id: str, dataset_id: str, key: str) -> bool:
    """Return whether ``key`` belongs to the fenced immutable-file format."""
    prefix = f"keyword_files/{tenant_id}/{dataset_id}/"
    if not key.startswith(prefix) or not key.endswith(".txt"):
        return False
    identifier = key[len(prefix) : -len(".txt")]
    try:
        return str(UUID(identifier)) == identifier
    except ValueError:
        return False


def register_keyword_file_cleanup_in_session(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    key: str,
    writer_id: str | None = None,
) -> KeywordFileCleanupArtifact:
    """Idempotently retain one storage key in the caller's transaction."""
    row = session.scalar(
        select(KeywordFileCleanupOutbox)
        .where(
            KeywordFileCleanupOutbox.tenant_id == tenant_id,
            KeywordFileCleanupOutbox.dataset_id == dataset_id,
            KeywordFileCleanupOutbox.key == key,
        )
        .with_for_update()
    )
    if row is None:
        row = KeywordFileCleanupOutbox(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            key=key,
            writer_id=writer_id,
        )
        session.add(row)
        session.flush()
    return _to_artifact(row)


def begin_keyword_file_write_in_session(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    key: str,
) -> str:
    """Register a writer in a caller-owned transaction.

    This supports SQLite and legacy transaction owners that cannot open the
    independent transaction used by the production writer lifecycle. The row
    still enforces publication fencing, while its durability is bounded by the
    caller's transaction.
    """
    writer_id = str(uuid4())
    artifact = register_keyword_file_cleanup_in_session(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        key=key,
        writer_id=writer_id,
    )
    if artifact.writer_id != writer_id:
        raise RuntimeError("keyword file key already has cleanup ownership")
    return writer_id


def finish_keyword_file_write_in_session(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    key: str,
    writer_id: str,
) -> None:
    """Mark a caller-owned writer complete before pointer publication."""
    result = session.execute(
        update(KeywordFileCleanupOutbox)
        .where(
            KeywordFileCleanupOutbox.tenant_id == tenant_id,
            KeywordFileCleanupOutbox.dataset_id == dataset_id,
            KeywordFileCleanupOutbox.key == key,
            KeywordFileCleanupOutbox.writer_id == writer_id,
            KeywordFileCleanupOutbox.writer_finished_at.is_(None),
            KeywordFileCleanupOutbox.cleanup_started_at.is_(None),
        )
        .values(writer_finished_at=func.current_timestamp())
    )
    if cast(CursorResult[object], result).rowcount != 1:
        raise KeywordFilePublicationLostError("keyword file writer marker was lost")


def complete_keyword_file_cleanup_in_session(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    key: str,
) -> None:
    """Transfer a finished, unfenced candidate into the current pointer.

    Historical file rows used one fixed key and had no writer marker. UUID
    pointers are the immutable format introduced with durable cleanup, so they
    must never use that compatibility path: a missing marker means cleanup has
    already won and publication is permanently fenced.
    """
    row = session.scalar(
        select(KeywordFileCleanupOutbox)
        .where(
            KeywordFileCleanupOutbox.tenant_id == tenant_id,
            KeywordFileCleanupOutbox.dataset_id == dataset_id,
            KeywordFileCleanupOutbox.key == key,
        )
        .with_for_update()
    )
    managed_key = _is_managed_keyword_file_key(tenant_id=tenant_id, dataset_id=dataset_id, key=key)
    legacy_key = f"keyword_files/{tenant_id}/{dataset_id}.txt"
    if not managed_key and key != legacy_key:
        raise KeywordFilePublicationLostError("keyword file is not a recognized publication key")
    if row is None:
        if managed_key:
            raise KeywordFilePublicationLostError("keyword file writer marker was lost")
        return
    if managed_key and row.writer_id is None:
        raise KeywordFilePublicationLostError("keyword file has no writer marker")
    if row.writer_id is not None and (row.writer_finished_at is None or row.cleanup_started_at is not None):
        raise KeywordFilePublicationLostError("keyword file cleanup already owns the candidate")
    session.delete(row)


class SQLAlchemyKeywordFileCleanupRepository:
    def __init__(self, *, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def register(self, *, tenant_id: str, dataset_id: str, key: str) -> KeywordFileCleanupArtifact:
        """Persist cleanup ownership before the corresponding external delete/write."""
        with self._session_factory() as session, session.begin():
            return register_keyword_file_cleanup_in_session(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                key=key,
            )

    def begin_file_write(self, *, tenant_id: str, dataset_id: str, key: str) -> str:
        """Retain an unfinished marker before an external object write starts."""
        with self._session_factory() as session, session.begin():
            return begin_keyword_file_write_in_session(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                key=key,
            )

    def finish_file_write(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        key: str,
        writer_id: str,
    ) -> None:
        """Make a completed object write eligible for one-shot cleanup."""
        with self._session_factory() as session, session.begin():
            finish_keyword_file_write_in_session(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                key=key,
                writer_id=writer_id,
            )

    def get(self, cleanup_id: str) -> KeywordFileCleanupArtifact | None:
        with self._session_factory() as session:
            row = session.scalar(select(KeywordFileCleanupOutbox).where(KeywordFileCleanupOutbox.id == cleanup_id))
            return _to_artifact(row) if row is not None else None

    def get_by_key(self, *, tenant_id: str, dataset_id: str, key: str) -> KeywordFileCleanupArtifact | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(KeywordFileCleanupOutbox).where(
                    KeywordFileCleanupOutbox.tenant_id == tenant_id,
                    KeywordFileCleanupOutbox.dataset_id == dataset_id,
                    KeywordFileCleanupOutbox.key == key,
                )
            )
            return _to_artifact(row) if row is not None else None

    def list_pending(self, *, limit: int) -> tuple[KeywordFileCleanupArtifact, ...]:
        if limit <= 0:
            return ()
        with self._session_factory() as session:
            rows = session.scalars(
                select(KeywordFileCleanupOutbox)
                .order_by(
                    case((KeywordFileCleanupOutbox.last_attempted_at.is_(None), 0), else_=1),
                    KeywordFileCleanupOutbox.last_attempted_at,
                    KeywordFileCleanupOutbox.created_at,
                    KeywordFileCleanupOutbox.id,
                )
                .limit(limit)
            ).all()
            return tuple(_to_artifact(row) for row in rows)

    def is_current_pointer(self, artifact: KeywordFileCleanupArtifact) -> bool:
        """Re-read the tenant-owned pointer while the caller holds its publication mutex."""
        with self._session_factory() as session:
            row = session.execute(
                select(DatasetKeywordTable.data_source_type, DatasetKeywordTable.keyword_table)
                .join(Dataset, Dataset.id == DatasetKeywordTable.dataset_id)
                .where(
                    Dataset.id == artifact.dataset_id,
                    Dataset.tenant_id == artifact.tenant_id,
                )
            ).one_or_none()
            if row is None:
                return False
            storage_type, data = row
            if storage_type == "database":
                return False
            current_key = data or f"keyword_files/{artifact.tenant_id}/{artifact.dataset_id}.txt"
            return current_key == artifact.key

    def mark_attempted(self, artifact: KeywordFileCleanupArtifact) -> None:
        with self._session_factory() as session, session.begin():
            session.execute(
                update(KeywordFileCleanupOutbox)
                .where(
                    KeywordFileCleanupOutbox.id == artifact.id,
                    KeywordFileCleanupOutbox.tenant_id == artifact.tenant_id,
                    KeywordFileCleanupOutbox.dataset_id == artifact.dataset_id,
                    KeywordFileCleanupOutbox.key == artifact.key,
                )
                .values(last_attempted_at=func.current_timestamp())
            )

    def begin_cleanup(self, artifact: KeywordFileCleanupArtifact) -> KeywordFileCleanupArtifact | None:
        """Permanently fence publication before starting an external delete."""
        with self._session_factory() as session, session.begin():
            session.execute(
                update(KeywordFileCleanupOutbox)
                .where(
                    KeywordFileCleanupOutbox.id == artifact.id,
                    KeywordFileCleanupOutbox.tenant_id == artifact.tenant_id,
                    KeywordFileCleanupOutbox.dataset_id == artifact.dataset_id,
                    KeywordFileCleanupOutbox.key == artifact.key,
                    KeywordFileCleanupOutbox.cleanup_started_at.is_(None),
                )
                .values(cleanup_started_at=func.current_timestamp())
            )
            row = session.scalar(
                select(KeywordFileCleanupOutbox).where(
                    KeywordFileCleanupOutbox.id == artifact.id,
                    KeywordFileCleanupOutbox.tenant_id == artifact.tenant_id,
                    KeywordFileCleanupOutbox.dataset_id == artifact.dataset_id,
                    KeywordFileCleanupOutbox.key == artifact.key,
                )
            )
            return _to_artifact(row) if row is not None else None

    def complete(self, artifact: KeywordFileCleanupArtifact) -> None:
        with self._session_factory() as session, session.begin():
            session.execute(
                delete(KeywordFileCleanupOutbox).where(
                    KeywordFileCleanupOutbox.id == artifact.id,
                    KeywordFileCleanupOutbox.tenant_id == artifact.tenant_id,
                    KeywordFileCleanupOutbox.dataset_id == artifact.dataset_id,
                    KeywordFileCleanupOutbox.key == artifact.key,
                )
            )

"""Keyword table serialization, object storage, and tenant-scoped SQL persistence."""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Protocol, cast
from uuid import uuid4

import orjson
from sqlalchemy import event, exists, select
from sqlalchemy.orm import Session, SessionTransaction, sessionmaker
from sqlalchemy.sql import Select

from models.dataset import Dataset, DatasetKeywordTable, DocumentSegment
from repositories.knowledge.keyword_file_cleanup_repository import (
    SQLAlchemyKeywordFileCleanupRepository,
    complete_keyword_file_cleanup_in_session,
    register_keyword_file_cleanup_in_session,
)

logger = logging.getLogger(__name__)


class _KeywordStorage(Protocol):
    def load_once(self, key: str, /) -> bytes: ...

    def save(self, key: str, data: bytes, /) -> None: ...

    def delete(self, key: str, /) -> None: ...


@dataclass(frozen=True)
class _PendingFileCleanup:
    storage: _KeywordStorage
    tenant_id: str
    dataset_id: str
    previous_key: str | None
    new_key: str | None
    published: bool
    publication_lock: Callable[[], AbstractContextManager[object]] | None
    repository: SQLAlchemyKeywordFileCleanupRepository


@dataclass(frozen=True)
class _HeldPublicationLock:
    context: AbstractContextManager[object]
    value: object


@dataclass(frozen=True)
class _DeletedKeywordTableSnapshot:
    id: str
    storage_type: str
    data: str


_PENDING_FILE_CLEANUPS = "keyword_table_pending_file_cleanups"
_FILE_CLEANUP_OUTCOME = "keyword_table_file_cleanup_outcome"
_HELD_PUBLICATION_LOCKS = "keyword_table_held_publication_locks"
_KEYWORD_TRANSACTION_LISTENERS = "keyword_table_transaction_listeners"


def _cleanup_repository_for_session(session: Session) -> SQLAlchemyKeywordFileCleanupRepository:
    return SQLAlchemyKeywordFileCleanupRepository(
        session_factory=sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    )


def _cleanup_registered_keyword_file(
    cleanup: _PendingFileCleanup,
    key: str,
    *,
    writer_can_acknowledge: bool = False,
) -> None:
    artifact = cleanup.repository.get_by_key(
        tenant_id=cleanup.tenant_id,
        dataset_id=cleanup.dataset_id,
        key=key,
    )
    if artifact is None:
        return
    assert cleanup.publication_lock is not None
    with cleanup.publication_lock():
        artifact = cleanup.repository.get(artifact.id)
        if artifact is None:
            return
        if cleanup.repository.is_current_pointer(artifact):
            cleanup.repository.complete(artifact)
            return
        artifact = cleanup.repository.begin_cleanup(artifact)
        if artifact is None:
            return
        with suppress(FileNotFoundError):
            cleanup.storage.delete(artifact.key)
        if artifact.writer_id is not None and artifact.writer_finished_at is None and not writer_can_acknowledge:
            cleanup.repository.mark_attempted(artifact)
            return
        cleanup.repository.complete(artifact)


def _cleanup_keyword_file_now(
    *,
    storage: _KeywordStorage,
    repository: SQLAlchemyKeywordFileCleanupRepository,
    tenant_id: str,
    dataset_id: str,
    key: str,
    publication_lock: Callable[[], AbstractContextManager[object]],
    writer_can_acknowledge: bool = False,
) -> None:
    cleanup = _PendingFileCleanup(
        storage=storage,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        previous_key=key,
        new_key=None,
        published=True,
        publication_lock=publication_lock,
        repository=repository,
    )
    _cleanup_registered_keyword_file(
        cleanup,
        key,
        writer_can_acknowledge=writer_can_acknowledge,
    )


def _mark_keyword_file_commit(session: Session) -> None:
    session.info[_FILE_CLEANUP_OUTCOME] = "commit"


def _mark_keyword_file_rollback(session: Session) -> None:
    session.info[_FILE_CLEANUP_OUTCOME] = "rollback"


def _finish_keyword_transaction(session: Session, transaction: SessionTransaction) -> None:
    if transaction.parent is not None:
        return
    held_locks = cast(dict[str, _HeldPublicationLock], session.info.pop(_HELD_PUBLICATION_LOCKS, {}))
    for held in reversed(held_locks.values()):
        try:
            held.context.__exit__(None, None, None)
        except Exception:
            logger.exception("Failed to release keyword table publication lock")

    cleanups = cast(list[_PendingFileCleanup], session.info.pop(_PENDING_FILE_CLEANUPS, []))
    outcome = session.info.pop(_FILE_CLEANUP_OUTCOME, "rollback")
    for cleanup in cleanups:
        if not cleanup.published or outcome != "commit":
            if cleanup.new_key is not None:
                try:
                    cleanup.repository.register(
                        tenant_id=cleanup.tenant_id,
                        dataset_id=cleanup.dataset_id,
                        key=cleanup.new_key,
                    )
                    _cleanup_registered_keyword_file(
                        cleanup,
                        cleanup.new_key,
                        writer_can_acknowledge=True,
                    )
                except Exception:
                    logger.exception("Failed to retain unpublished keyword file cleanup intent: %s", cleanup.new_key)
        elif cleanup.previous_key is not None:
            try:
                _cleanup_registered_keyword_file(cleanup, cleanup.previous_key)
            except Exception:
                logger.exception("Failed to reclaim previous keyword table file: %s", cleanup.previous_key)


def _ensure_keyword_transaction_listeners(session: Session) -> None:
    if session.info.get(_KEYWORD_TRANSACTION_LISTENERS):
        return
    session.info[_KEYWORD_TRANSACTION_LISTENERS] = True
    event.listen(session, "after_commit", _mark_keyword_file_commit)
    event.listen(session, "after_rollback", _mark_keyword_file_rollback)
    event.listen(session, "after_transaction_end", _finish_keyword_transaction)


@contextmanager
def keyword_transaction_lock(
    session: Session,
    *,
    lock_name: str,
    lock_factory: Callable[[], AbstractContextManager[object]],
):
    """Hold one dataset's publication mutex through the caller-owned transaction.

    Legacy keyword writes prepare an immutable file, flush its pointer and segment
    keywords, and retain this mutex until the caller commits or rolls back. Reads
    from an already-active caller transaction use the same boundary so their
    current database read and this mutex have one lock order. This helper never
    commits unrelated caller state. File reclamation runs only after the
    transaction has ended.
    """
    held_locks = cast(dict[str, _HeldPublicationLock], session.info.setdefault(_HELD_PUBLICATION_LOCKS, {}))
    held = held_locks.get(lock_name)
    if held is not None:
        yield held.value
        return

    context = lock_factory()
    value = context.__enter__()
    held_locks[lock_name] = _HeldPublicationLock(context=context, value=value)
    _ensure_keyword_transaction_listeners(session)
    try:
        yield value
    finally:
        registered = session.info.get(_HELD_PUBLICATION_LOCKS)
        if not session.in_transaction() and registered is held_locks and held_locks.pop(lock_name, None) is not None:
            context.__exit__(None, None, None)


@contextmanager
def keyword_publication_lock(
    session: Session,
    *,
    lock_name: str,
    lock_factory: Callable[[], AbstractContextManager[object]],
):
    """Reuse the caller's write lock, or briefly lock a standalone read."""
    held_locks = cast(dict[str, _HeldPublicationLock], session.info.get(_HELD_PUBLICATION_LOCKS, {}))
    held = held_locks.get(lock_name)
    if held is not None:
        yield held.value
        return
    with lock_factory() as value:
        yield value


def _defer_keyword_file_cleanup(session: Session, cleanup: _PendingFileCleanup) -> None:
    pending = cast(list[_PendingFileCleanup], session.info.setdefault(_PENDING_FILE_CLEANUPS, []))
    pending.append(cleanup)
    _ensure_keyword_transaction_listeners(session)


def complete_keyword_file_write(
    *,
    storage: _KeywordStorage,
    tenant_id: str,
    dataset_id: str,
    previous_key: str | None,
    new_key: str,
    transaction_session: Session | None,
    publication_lock: Callable[[], AbstractContextManager[object]],
    cleanup_repository: SQLAlchemyKeywordFileCleanupRepository | None = None,
) -> None:
    """Reclaim the replaced file after its new pointer is durably published."""
    if transaction_session is None:
        if previous_key is None:
            return
        if cleanup_repository is None:
            try:
                storage.delete(previous_key)
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception("Failed to clean up keyword table file: %s", previous_key)
            return
        artifact = cleanup_repository.get_by_key(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            key=previous_key,
        )
        if artifact is not None:
            try:
                _cleanup_keyword_file_now(
                    storage=storage,
                    repository=cleanup_repository,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    key=artifact.key,
                    publication_lock=publication_lock,
                )
            except Exception:
                logger.exception("Failed to reclaim previous keyword table file: %s", previous_key)
        return
    cleanup_repository = cleanup_repository or _cleanup_repository_for_session(transaction_session)
    _defer_keyword_file_cleanup(
        transaction_session,
        _PendingFileCleanup(
            storage=storage,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            previous_key=previous_key,
            new_key=new_key,
            published=True,
            publication_lock=publication_lock,
            repository=cleanup_repository,
        ),
    )


def discard_keyword_file_write(
    *,
    storage: _KeywordStorage,
    tenant_id: str,
    dataset_id: str,
    new_key: str,
    transaction_session: Session | None,
    publication_lock: Callable[[], AbstractContextManager[object]],
    cleanup_repository: SQLAlchemyKeywordFileCleanupRepository | None = None,
) -> None:
    """Reclaim an unpublished file without replacing the write failure."""
    if transaction_session is None:
        if cleanup_repository is None:
            try:
                storage.delete(new_key)
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception("Failed to clean up keyword table file: %s", new_key)
            return
        artifact = cleanup_repository.get_by_key(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            key=new_key,
        ) or cleanup_repository.register(tenant_id=tenant_id, dataset_id=dataset_id, key=new_key)
        try:
            _cleanup_keyword_file_now(
                storage=storage,
                repository=cleanup_repository,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                key=artifact.key,
                publication_lock=publication_lock,
                writer_can_acknowledge=True,
            )
        except Exception:
            logger.exception("Failed to reclaim unpublished keyword table file: %s", new_key)
        return
    cleanup_repository = cleanup_repository or _cleanup_repository_for_session(transaction_session)
    _defer_keyword_file_cleanup(
        transaction_session,
        _PendingFileCleanup(
            storage=storage,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            previous_key=None,
            new_key=new_key,
            published=False,
            publication_lock=publication_lock,
            repository=cleanup_repository,
        ),
    )


def complete_keyword_file_delete(
    *,
    storage: _KeywordStorage,
    tenant_id: str,
    dataset_id: str,
    previous_key: str,
    transaction_session: Session,
    publication_lock: Callable[[], AbstractContextManager[object]],
    cleanup_repository: SQLAlchemyKeywordFileCleanupRepository | None = None,
) -> None:
    """Delete a removed table's file only after its database row commits."""
    register_keyword_file_cleanup_in_session(
        transaction_session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        key=previous_key,
    )
    cleanup_repository = cleanup_repository or _cleanup_repository_for_session(transaction_session)
    _defer_keyword_file_cleanup(
        transaction_session,
        _PendingFileCleanup(
            storage=storage,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            previous_key=previous_key,
            new_key=None,
            published=True,
            publication_lock=publication_lock,
            repository=cleanup_repository,
        ),
    )


def keyword_file_key(tenant_id: str, dataset_id: str) -> str:
    return f"keyword_files/{tenant_id}/{dataset_id}.txt"


def _new_keyword_file_key(tenant_id: str, dataset_id: str) -> str:
    return f"keyword_files/{tenant_id}/{dataset_id}/{uuid4()}.txt"


def _decode_sets(values: dict[str, Any]) -> dict[str, Any]:
    return {key: set(value) if isinstance(value, list) else value for key, value in values.items()}


def _encode_sets(value: object) -> list[object]:
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_keyword_table(
    *, storage: _KeywordStorage, tenant_id: str, dataset_id: str, storage_type: str, data: str | None
) -> dict[str, Any] | None:
    if storage_type == "database":
        return json.loads(data, object_hook=_decode_sets) if data else None
    # Rows written before file pointers were introduced stored an empty value
    # and used the dataset's fixed object key.
    file_key = data or keyword_file_key(tenant_id, dataset_id)
    try:
        payload = storage.load_once(file_key)
        return json.loads(payload.decode("utf-8"), object_hook=_decode_sets) if payload else None
    except FileNotFoundError:
        return None


def _locked_keyword_table_query(*, tenant_id: str, dataset_id: str, read: bool) -> Select[tuple[DatasetKeywordTable]]:
    return (
        select(DatasetKeywordTable)
        .where(
            DatasetKeywordTable.dataset_id == dataset_id,
            exists().where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id),
        )
        .with_for_update(read=read)
        .execution_options(populate_existing=True)
    )


def lock_dataset_keyword_table(
    session: Session, *, tenant_id: str, dataset_id: str, read: bool = False
) -> DatasetKeywordTable | None:
    """Current-read the keyword row without locking the Dataset again.

    The publishing owner already holds its Dataset generation guard. In
    particular, MySQL's locking JOIN would also lock Dataset from this separate
    keyword session and can wait behind a queued configuration UPDATE while the
    publishing session is still holding SHARE. The ownership subquery remains a
    non-locking read; only the keyword row participates in this lock request.
    """
    return session.scalar(_locked_keyword_table_query(tenant_id=tenant_id, dataset_id=dataset_id, read=read))


def delete_deleted_dataset_keyword_table(
    *,
    session_factory: sessionmaker[Session],
    tenant_id: str,
    dataset_id: str,
    storage: _KeywordStorage,
    lock_factory: Callable[[], AbstractContextManager[object]],
) -> bool:
    """Delete a keyword table after its trusted dataset deletion has committed.

    Dataset keyword rows do not retain a tenant column. The durable dataset
    cleanup intent supplies the original owner, while current reads prevent
    this recovery path from touching a live dataset.

    The publication lock spans two short SQL transactions. For file-backed
    tables, blob deletion runs between them so a storage failure leaves the row
    and its retry key intact. If the second transaction fails after the blob was
    deleted, the row remains and the idempotent storage contract makes a retry
    safe. This narrow ordering is specific to cleanup after Dataset deletion.
    """
    with lock_factory():
        with session_factory.begin() as session:
            live_dataset_id = session.scalar(
                select(Dataset.id)
                .where(Dataset.id == dataset_id)
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if live_dataset_id is not None:
                raise LookupError("Dataset still exists")

            keyword_table = session.scalar(
                select(DatasetKeywordTable)
                .where(DatasetKeywordTable.dataset_id == dataset_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if keyword_table is None:
                return False
            snapshot = _DeletedKeywordTableSnapshot(
                id=keyword_table.id,
                storage_type=keyword_table.data_source_type,
                data=keyword_table.keyword_table,
            )

        if snapshot.storage_type != "database":
            storage.delete(snapshot.data or keyword_file_key(tenant_id, dataset_id))

        with session_factory.begin() as session:
            live_dataset_id = session.scalar(
                select(Dataset.id)
                .where(Dataset.id == dataset_id)
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if live_dataset_id is not None:
                raise LookupError("Dataset still exists")
            keyword_table = session.scalar(
                select(DatasetKeywordTable)
                .where(
                    DatasetKeywordTable.id == snapshot.id,
                    DatasetKeywordTable.dataset_id == dataset_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if keyword_table is None:
                return True
            if keyword_table.data_source_type != snapshot.storage_type or keyword_table.keyword_table != snapshot.data:
                raise RuntimeError("Dataset keyword table changed during cleanup")
            session.delete(keyword_table)
        return True


def save_keyword_table(
    *,
    storage: _KeywordStorage,
    tenant_id: str,
    dataset_id: str,
    storage_type: str,
    table: dict[str, set[str]],
    begin_file_write: Callable[[str], str] | None = None,
    finish_file_write: Callable[[str, str], None] | None = None,
) -> str:
    data = orjson.dumps(
        {"__type__": "keyword_table", "__data__": {"index_id": dataset_id, "summary": None, "table": table}},
        default=_encode_sets,
    )
    if storage_type != "database":
        if begin_file_write is None or finish_file_write is None:
            raise ValueError("keyword file writes require a durable publication marker")
        file_key = _new_keyword_file_key(tenant_id, dataset_id)
        writer_id = begin_file_write(file_key)
        storage.save(file_key, data)
        finish_file_write(file_key, writer_id)
        return file_key
    return data.decode("utf-8")


def persist_keyword_table(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    storage_type: str,
    data: str,
    keywords: Mapping[str, Sequence[str]],
) -> None:
    """Flush a tenant-owned table and segment keywords without ending the caller transaction.

    A caller that spans immutable file publication and this SQL write must retain
    the dataset publication mutex until commit or rollback. Cleanup of replaced
    files is deferred until that transaction has ended.
    """
    owned_dataset_id = session.scalar(
        select(Dataset.id).where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
    )
    if owned_dataset_id is None:
        raise LookupError("Dataset no longer exists")

    keyword_table = session.scalar(
        select(DatasetKeywordTable)
        .where(DatasetKeywordTable.dataset_id == dataset_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    new_key = data if storage_type != "database" else None
    if keyword_table is not None and keyword_table.data_source_type != "database":
        previous_key = keyword_table.keyword_table or keyword_file_key(tenant_id, dataset_id)
        if previous_key != new_key:
            register_keyword_file_cleanup_in_session(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                key=previous_key,
            )
    if new_key is not None:
        # File writers may pre-register the candidate before storage I/O. Its
        # cleanup ownership transfers atomically to the replaced key here.
        complete_keyword_file_cleanup_in_session(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            key=new_key,
        )
    if keyword_table is None:
        keyword_table = DatasetKeywordTable(
            dataset_id=dataset_id,
            data_source_type=storage_type,
            keyword_table=data,
        )
        session.add(keyword_table)
    else:
        keyword_table.data_source_type = storage_type
        keyword_table.keyword_table = data
    if keywords:
        segments = session.scalars(
            select(DocumentSegment).where(
                DocumentSegment.tenant_id == tenant_id,
                DocumentSegment.dataset_id == dataset_id,
                DocumentSegment.index_node_id.in_(keywords),
            )
        )
        for segment in segments:
            assert segment.index_node_id is not None
            segment.keywords = list(keywords[segment.index_node_id])
    session.flush()

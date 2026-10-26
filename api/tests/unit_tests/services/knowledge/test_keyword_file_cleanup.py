import logging
from contextlib import nullcontext
from dataclasses import dataclass, field

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from models.dataset import Dataset, DatasetKeywordTable
from models.model import KeywordFileCleanupOutbox
from repositories.knowledge.keyword_file_cleanup_repository import (
    KeywordFilePublicationLostError,
    SQLAlchemyKeywordFileCleanupRepository,
)
from repositories.knowledge.keyword_table_repository import (
    discard_keyword_file_write,
    persist_keyword_table,
    save_keyword_table,
)
from services.knowledge.keyword_file_cleanup import drain_pending_keyword_file_cleanups


@dataclass
class KeywordStorage:
    files: set[str] = field(default_factory=set)
    failures: int = 0
    save_failures: int = 0
    deletes: list[str] = field(default_factory=list)

    def load_once(self, key: str) -> bytes:
        if key not in self.files:
            raise FileNotFoundError(key)
        return b""

    def save(self, key: str, _data: bytes) -> None:
        self.files.add(key)
        if self.save_failures:
            self.save_failures -= 1
            raise TimeoutError("write outcome unknown")

    def delete(self, key: str) -> None:
        self.deletes.append(key)
        if self.failures:
            self.failures -= 1
            raise OSError("storage unavailable")
        if key not in self.files:
            raise FileNotFoundError(key)
        self.files.remove(key)


def _repository(factory: sessionmaker[Session]) -> SQLAlchemyKeywordFileCleanupRepository:
    return SQLAlchemyKeywordFileCleanupRepository(session_factory=factory)


def _drain(factory: sessionmaker[Session], storage: KeywordStorage) -> tuple[str, ...]:
    return drain_pending_keyword_file_cleanups(
        repository=_repository(factory),
        delete_file=storage.delete,
        publication_lock=lambda _artifact: nullcontext(),
        logger=logging.getLogger(__name__),
        limit=100,
    )


def _seed_dataset(factory: sessionmaker[Session], *, pointer: str | None = None) -> None:
    with factory.begin() as session:
        session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
        if pointer is not None:
            session.add(
                DatasetKeywordTable(
                    dataset_id="dataset-1",
                    data_source_type="file",
                    keyword_table=pointer,
                )
            )


def test_replaced_key_delete_failure_survives_a_new_session_and_drains(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    old_key = "keyword_files/tenant-1/dataset-1/00000000-0000-0000-0000-000000000001.txt"
    new_key = "keyword_files/tenant-1/dataset-1/00000000-0000-0000-0000-000000000002.txt"
    _seed_dataset(sqlite_session_factory, pointer=old_key)
    storage = KeywordStorage(files={old_key, new_key}, failures=1)
    repository = _repository(sqlite_session_factory)
    writer_id = repository.begin_file_write(tenant_id="tenant-1", dataset_id="dataset-1", key=new_key)
    repository.finish_file_write(tenant_id="tenant-1", dataset_id="dataset-1", key=new_key, writer_id=writer_id)

    with sqlite_session_factory.begin() as session:
        persist_keyword_table(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            storage_type="file",
            data=new_key,
            keywords={},
        )

    failed = _drain(sqlite_session_factory, storage)
    assert len(failed) == 1
    with sqlite_session_factory() as fresh_session:
        cleanup = fresh_session.scalar(select(KeywordFileCleanupOutbox))
        assert cleanup is not None
        assert cleanup.key == old_key
        assert cleanup.cleanup_started_at is not None

    assert _drain(sqlite_session_factory, storage) == ()
    with sqlite_session_factory() as fresh_session:
        assert fresh_session.scalar(select(func.count()).select_from(KeywordFileCleanupOutbox)) == 0
        current = fresh_session.scalar(select(DatasetKeywordTable).where(DatasetKeywordTable.dataset_id == "dataset-1"))
        assert current is not None
        assert current.keyword_table == new_key
    assert storage.files == {new_key}


def test_normal_publication_consumes_writer_marker_and_persists_replaced_file_cleanup(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    old_key = "keyword_files/tenant-1/dataset-1/00000000-0000-0000-0000-000000000008.txt"
    new_key = "keyword_files/tenant-1/dataset-1/00000000-0000-0000-0000-000000000009.txt"
    _seed_dataset(sqlite_session_factory, pointer=old_key)
    repository = _repository(sqlite_session_factory)
    writer_id = repository.begin_file_write(tenant_id="tenant-1", dataset_id="dataset-1", key=new_key)
    repository.finish_file_write(tenant_id="tenant-1", dataset_id="dataset-1", key=new_key, writer_id=writer_id)

    with sqlite_session_factory.begin() as session:
        persist_keyword_table(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            storage_type="file",
            data=new_key,
            keywords={},
        )

    assert repository.get_by_key(tenant_id="tenant-1", dataset_id="dataset-1", key=new_key) is None
    old_cleanup = repository.get_by_key(tenant_id="tenant-1", dataset_id="dataset-1", key=old_key)
    assert old_cleanup is not None
    assert old_cleanup.writer_id is None
    with sqlite_session_factory() as session:
        current = session.scalar(select(DatasetKeywordTable))
        assert current is not None
        assert current.keyword_table == new_key


def test_file_to_database_transition_retains_old_file_cleanup_in_same_transaction(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    old_key = "keyword_files/tenant-1/dataset-1/old.txt"
    _seed_dataset(sqlite_session_factory, pointer=old_key)
    storage = KeywordStorage(files={old_key})

    with sqlite_session_factory.begin() as session:
        persist_keyword_table(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            storage_type="database",
            data='{"__type__":"keyword_table","__data__":{"table":{}}}',
            keywords={},
        )

    [artifact] = _repository(sqlite_session_factory).list_pending(limit=10)
    assert artifact.key == old_key
    assert _drain(sqlite_session_factory, storage) == ()
    assert storage.files == set()


def test_rollback_delete_failure_retains_unpublished_key_for_retry(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    _seed_dataset(sqlite_session_factory)
    unpublished_key = "keyword_files/tenant-1/dataset-1/unpublished.txt"
    storage = KeywordStorage(files={unpublished_key}, failures=1)

    with sqlite_session_factory() as session:
        session.execute(select(Dataset.id).where(Dataset.id == "dataset-1"))
        discard_keyword_file_write(
            storage=storage,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            new_key=unpublished_key,
            transaction_session=session,
            publication_lock=nullcontext,
        )
        session.rollback()

    with sqlite_session_factory() as fresh_session:
        cleanup = fresh_session.scalar(select(KeywordFileCleanupOutbox))
        assert cleanup is not None
        assert cleanup.key == unpublished_key
    assert unpublished_key in storage.files

    assert _drain(sqlite_session_factory, storage) == ()
    assert unpublished_key not in storage.files


def test_current_pointer_is_never_deleted(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    current_key = "keyword_files/tenant-1/dataset-1/current.txt"
    _seed_dataset(sqlite_session_factory, pointer=current_key)
    storage = KeywordStorage(files={current_key})
    _repository(sqlite_session_factory).register(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        key=current_key,
    )

    assert _drain(sqlite_session_factory, storage) == ()
    assert storage.files == {current_key}
    assert storage.deletes == []
    with sqlite_session_factory() as session:
        assert session.scalar(select(func.count()).select_from(KeywordFileCleanupOutbox)) == 0


def test_timed_out_put_keeps_an_unfinished_tombstone_after_reconciliation(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    _seed_dataset(sqlite_session_factory)
    storage = KeywordStorage(save_failures=1)
    repository = _repository(sqlite_session_factory)

    with pytest.raises(TimeoutError, match="outcome unknown"):
        save_keyword_table(
            storage=storage,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            storage_type="file",
            table={"keyword": {"node-1"}},
            begin_file_write=lambda key: repository.begin_file_write(
                tenant_id="tenant-1",
                dataset_id="dataset-1",
                key=key,
            ),
            finish_file_write=lambda key, writer_id: repository.finish_file_write(
                tenant_id="tenant-1",
                dataset_id="dataset-1",
                key=key,
                writer_id=writer_id,
            ),
        )

    [artifact] = repository.list_pending(limit=10)
    assert artifact.writer_id is not None
    assert artifact.writer_finished_at is None
    assert artifact.key in storage.files
    assert _drain(sqlite_session_factory, storage) == ()
    assert artifact.key not in storage.files
    assert repository.get(artifact.id) is not None


def test_unfinished_writer_is_reconciled_without_ack_and_fences_publication(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    old_key = "keyword_files/tenant-1/dataset-1/00000000-0000-0000-0000-000000000003.txt"
    candidate_key = "keyword_files/tenant-1/dataset-1/00000000-0000-0000-0000-000000000004.txt"
    _seed_dataset(sqlite_session_factory, pointer=old_key)
    storage = KeywordStorage(files={candidate_key})
    repository = _repository(sqlite_session_factory)
    writer_id = repository.begin_file_write(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        key=candidate_key,
    )

    assert _drain(sqlite_session_factory, storage) == ()
    artifact = repository.get_by_key(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        key=candidate_key,
    )
    assert artifact is not None
    assert artifact.cleanup_started_at is not None

    # Simulate a remote PUT that completed after the first DELETE.
    storage.files.add(candidate_key)
    assert _drain(sqlite_session_factory, storage) == ()
    assert candidate_key not in storage.files
    assert repository.get(artifact.id) is not None

    with pytest.raises(KeywordFilePublicationLostError):
        repository.finish_file_write(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            key=candidate_key,
            writer_id=writer_id,
        )
    with pytest.raises(KeywordFilePublicationLostError):
        with sqlite_session_factory.begin() as session:
            persist_keyword_table(
                session,
                tenant_id="tenant-1",
                dataset_id="dataset-1",
                storage_type="file",
                data=candidate_key,
                keywords={},
            )
    with sqlite_session_factory() as session:
        current = session.scalar(select(DatasetKeywordTable))
        assert current is not None
        assert current.keyword_table == old_key


def test_finished_writer_is_acked_after_cleanup_and_late_publication_is_rejected(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    old_key = "keyword_files/tenant-1/dataset-1/00000000-0000-0000-0000-000000000005.txt"
    candidate_key = "keyword_files/tenant-1/dataset-1/00000000-0000-0000-0000-000000000006.txt"
    _seed_dataset(sqlite_session_factory, pointer=old_key)
    storage = KeywordStorage(files={candidate_key})
    repository = _repository(sqlite_session_factory)
    writer_id = repository.begin_file_write(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        key=candidate_key,
    )
    repository.finish_file_write(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        key=candidate_key,
        writer_id=writer_id,
    )

    assert _drain(sqlite_session_factory, storage) == ()
    artifact = repository.get_by_key(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        key=candidate_key,
    )
    assert artifact is None
    assert candidate_key not in storage.files

    with pytest.raises(KeywordFilePublicationLostError):
        with sqlite_session_factory.begin() as session:
            persist_keyword_table(
                session,
                tenant_id="tenant-1",
                dataset_id="dataset-1",
                storage_type="file",
                data=candidate_key,
                keywords={},
            )
    with sqlite_session_factory() as session:
        current = session.scalar(select(DatasetKeywordTable))
        assert current is not None
        assert current.keyword_table == old_key
        assert session.scalar(select(func.count()).select_from(KeywordFileCleanupOutbox)) == 0

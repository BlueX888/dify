from contextlib import nullcontext

import pytest
from sqlalchemy import event, select
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Session, sessionmaker

from models.dataset import Dataset, DatasetKeywordTable, DocumentSegment
from repositories.knowledge.keyword_file_cleanup_repository import (
    begin_keyword_file_write_in_session,
    finish_keyword_file_write_in_session,
)
from repositories.knowledge.keyword_table_repository import (
    _locked_keyword_table_query,
    delete_deleted_dataset_keyword_table,
    load_keyword_table,
    persist_keyword_table,
)


class KeywordStorage:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def load_once(self, key: str) -> bytes:
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]

    def save(self, key: str, data: bytes) -> None:
        self.files[key] = data

    def delete(self, key: str) -> None:
        self.files.pop(key, None)


@pytest.mark.parametrize(
    ("read", "lock_clause"),
    [
        (False, "FOR UPDATE"),
        (True, "LOCK IN SHARE MODE"),
    ],
)
def test_mysql_keyword_lock_query_uses_a_current_read(read: bool, lock_clause: str) -> None:
    statement = _locked_keyword_table_query(tenant_id="tenant-1", dataset_id="dataset-1", read=read)

    sql = str(statement.compile(dialect=mysql.dialect()))

    assert lock_clause in sql


def test_load_file_keyword_table_supports_legacy_empty_pointer() -> None:
    legacy_key = "keyword_files/tenant-1/dataset-1.txt"
    storage = KeywordStorage({legacy_key: b'{"__data__":{"table":{"legacy":["node-1"]}}}'})

    payload = load_keyword_table(
        storage=storage,
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        storage_type="file",
        data="",
    )

    assert payload == {"__data__": {"table": {"legacy": {"node-1"}}}}


@pytest.mark.parametrize("existing_table", [False, True])
def test_keyword_writes_roll_back_with_the_callers_transaction(sqlite_session: Session, existing_table: bool) -> None:
    sqlite_session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
    segment = DocumentSegment(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        position=1,
        content="original",
        word_count=8,
        tokens=1,
        created_by="author",
        index_node_id="node-1",
        keywords=["original"],
    )
    sqlite_session.add(segment)
    if existing_table:
        sqlite_session.add(DatasetKeywordTable(dataset_id="dataset-1", keyword_table="original"))
    sqlite_session.commit()

    persist_keyword_table(
        sqlite_session,
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        storage_type="database",
        data="replacement",
        keywords={"node-1": ["replacement"]},
    )
    assert segment.keywords == ["replacement"]
    sqlite_session.rollback()

    assert segment.keywords == ["original"]
    row = sqlite_session.scalar(select(DatasetKeywordTable))
    if existing_table:
        assert row is not None
        assert row.keyword_table == "original"
    else:
        assert row is None


@pytest.mark.parametrize(("tenant_id", "dataset_id"), [("tenant-2", "dataset-1"), ("tenant-1", "missing")])
def test_keyword_writes_require_a_tenant_owned_dataset(
    sqlite_session: Session, tenant_id: str, dataset_id: str
) -> None:
    sqlite_session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
    sqlite_session.commit()

    with pytest.raises(LookupError, match="Dataset no longer exists"):
        persist_keyword_table(
            sqlite_session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            storage_type="database",
            data="replacement",
            keywords={},
        )

    assert sqlite_session.scalar(select(DatasetKeywordTable)) is None


def test_keyword_write_updates_storage_type_and_file_pointer(sqlite_session: Session) -> None:
    sqlite_session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
    sqlite_session.add(
        DatasetKeywordTable(dataset_id="dataset-1", data_source_type="database", keyword_table="database payload")
    )
    sqlite_session.commit()

    file_key = "keyword_files/tenant-1/dataset-1/00000000-0000-0000-0000-000000000007.txt"
    writer_id = begin_keyword_file_write_in_session(
        sqlite_session,
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        key=file_key,
    )
    finish_keyword_file_write_in_session(
        sqlite_session,
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        key=file_key,
        writer_id=writer_id,
    )
    persist_keyword_table(
        sqlite_session,
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        storage_type="file",
        data=file_key,
        keywords={},
    )
    sqlite_session.commit()

    row = sqlite_session.scalar(select(DatasetKeywordTable))
    assert row is not None
    assert row.data_source_type == "file"
    assert row.keyword_table == file_key


@pytest.mark.parametrize("storage_type", ["database", "file"])
def test_deleted_dataset_keyword_cleanup_commits_database_and_file_state(
    sqlite_session_factory: sessionmaker[Session], storage_type: str
) -> None:
    file_key = "keyword_files/tenant-1/dataset-1/version.txt"
    storage = KeywordStorage({file_key: b"payload"} if storage_type == "file" else {})
    with sqlite_session_factory.begin() as session:
        session.add(
            DatasetKeywordTable(
                dataset_id="dataset-1",
                data_source_type=storage_type,
                keyword_table=file_key if storage_type == "file" else "database payload",
            )
        )

    deleted = delete_deleted_dataset_keyword_table(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        storage=storage,
        lock_factory=nullcontext,
    )

    assert deleted
    with sqlite_session_factory() as session:
        assert session.scalar(select(DatasetKeywordTable)) is None
    assert storage.files == {}


def test_deleted_dataset_keyword_cleanup_storage_failure_preserves_the_retry_key(
    sqlite_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    file_key = "keyword_files/tenant-1/dataset-1/version.txt"
    storage = KeywordStorage({file_key: b"payload"})
    with sqlite_session_factory.begin() as session:
        session.add(
            DatasetKeywordTable(
                dataset_id="dataset-1",
                data_source_type="file",
                keyword_table=file_key,
            )
        )
    delete_attempts = 0
    delete = storage.delete

    def fail_first_delete(key: str) -> None:
        nonlocal delete_attempts
        delete_attempts += 1
        if delete_attempts == 1:
            raise RuntimeError("storage unavailable")
        delete(key)

    monkeypatch.setattr(storage, "delete", fail_first_delete)

    with pytest.raises(RuntimeError, match="storage unavailable"):
        delete_deleted_dataset_keyword_table(
            session_factory=sqlite_session_factory,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            storage=storage,
            lock_factory=nullcontext,
        )

    with sqlite_session_factory() as session:
        row = session.scalar(select(DatasetKeywordTable))
        assert row is not None
        assert row.keyword_table == file_key
    assert storage.files == {file_key: b"payload"}

    assert delete_deleted_dataset_keyword_table(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        storage=storage,
        lock_factory=nullcontext,
    )
    with sqlite_session_factory() as session:
        assert session.scalar(select(DatasetKeywordTable)) is None
    assert storage.files == {}


def test_deleted_dataset_keyword_cleanup_retries_after_the_final_commit_fails(
    sqlite_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    file_key = "keyword_files/tenant-1/dataset-1/version.txt"
    storage = KeywordStorage({file_key: b"payload"})
    with sqlite_session_factory.begin() as session:
        session.add(DatasetKeywordTable(dataset_id="dataset-1", data_source_type="file", keyword_table=file_key))
    fail_commit = False

    def fail_one_commit(_session: Session) -> None:
        nonlocal fail_commit
        if fail_commit:
            fail_commit = False
            raise RuntimeError("commit failed")

    event.listen(sqlite_session_factory, "before_commit", fail_one_commit)
    delete = storage.delete

    def delete_then_fail_commit(key: str) -> None:
        nonlocal fail_commit
        delete(key)
        fail_commit = True

    monkeypatch.setattr(storage, "delete", delete_then_fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            delete_deleted_dataset_keyword_table(
                session_factory=sqlite_session_factory,
                tenant_id="tenant-1",
                dataset_id="dataset-1",
                storage=storage,
                lock_factory=nullcontext,
            )
    finally:
        event.remove(sqlite_session_factory, "before_commit", fail_one_commit)

    with sqlite_session_factory() as session:
        row = session.scalar(select(DatasetKeywordTable))
        assert row is not None
        assert row.keyword_table == file_key
    assert storage.files == {}

    assert delete_deleted_dataset_keyword_table(
        session_factory=sqlite_session_factory,
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        storage=storage,
        lock_factory=nullcontext,
    )
    with sqlite_session_factory() as session:
        assert session.scalar(select(DatasetKeywordTable)) is None


def test_deleted_dataset_keyword_cleanup_rejects_a_live_dataset(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    storage = KeywordStorage({})
    with sqlite_session_factory.begin() as session:
        session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
        session.add(DatasetKeywordTable(dataset_id="dataset-1", data_source_type="database", keyword_table="payload"))

    with pytest.raises(LookupError, match="Dataset still exists"):
        delete_deleted_dataset_keyword_table(
            session_factory=sqlite_session_factory,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            storage=storage,
            lock_factory=nullcontext,
        )

    with sqlite_session_factory() as session:
        assert session.scalar(select(DatasetKeywordTable)) is not None

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from core.rag.datasource.keyword.jieba import jieba as jieba_module
from core.rag.datasource.keyword.jieba.jieba import Jieba
from core.rag.models.document import Document
from models.dataset import Dataset, DatasetKeywordTable, DocumentSegment
from repositories.knowledge.dataset_read_repository import get_dataset_keyword_table
from repositories.knowledge.keyword_file_cleanup_repository import (
    begin_keyword_file_write_in_session,
    finish_keyword_file_write_in_session,
)
from repositories.knowledge.keyword_table_repository import (
    load_keyword_table,
    persist_keyword_table,
    save_keyword_table,
)
from tests.unit_tests.config_override import apply_config_overrides
from tests.unit_tests.model_factories import make_dataset


@dataclass
class KeywordStorage:
    files: dict[str, bytes] = field(default_factory=dict)
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)

    def load_once(self, key: str) -> bytes:
        self.reads.append(key)
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]

    def exists(self, key: str) -> bool:
        return key in self.files

    def save(self, key: str, data: bytes) -> None:
        self.files[key] = data
        self.writes.append(key)

    def delete(self, key: str) -> None:
        del self.files[key]


@dataclass
class KeywordLocks:
    acquired: list[str] = field(default_factory=list)
    held: bool = False

    @contextmanager
    def lock(self, name: str, *, timeout: int) -> Generator[None]:
        assert timeout == 600
        assert not self.held
        self.acquired.append(name)
        self.held = True
        try:
            yield
        finally:
            self.held = False


@dataclass
class BlockingKeywordLocks:
    locks: dict[str, Lock] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    guard: Lock = field(default_factory=Lock)
    second_attempt: Event = field(default_factory=Event)

    @contextmanager
    def lock(self, name: str, *, timeout: int) -> Generator[None]:
        assert timeout == 600
        with self.guard:
            lock = self.locks.setdefault(name, Lock())
            self.attempts[name] = self.attempts.get(name, 0) + 1
            if self.attempts[name] == 2:
                self.second_attempt.set()
        lock.acquire()
        try:
            yield
        finally:
            lock.release()


@dataclass
class KeywordRuntime:
    session: Session
    dataset: Dataset
    storage: KeywordStorage
    locks: KeywordLocks
    extracted: list[tuple[str, int]]

    def table(self) -> dict[str, set[str]]:
        row = get_dataset_keyword_table(self.dataset, session=self.session)
        assert row is not None
        payload = load_keyword_table(
            storage=self.storage,
            tenant_id=self.dataset.tenant_id,
            dataset_id=self.dataset.id,
            storage_type=row.data_source_type,
            data=row.keyword_table,
        )
        assert payload is not None
        return payload["__data__"]["table"]


@pytest.fixture(params=["database", "file"])
def runtime(request: pytest.FixtureRequest, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch) -> KeywordRuntime:
    storage = KeywordStorage()
    locks = KeywordLocks()
    extracted: list[tuple[str, int]] = []

    class KeywordExtractor:
        def extract_keywords(self, text: str, keyword_number: int = 10) -> set[str]:
            extracted.append((text, keyword_number))
            return set(text.split()[:keyword_number])

    monkeypatch.setattr(jieba_module, "JiebaKeywordTableHandler", KeywordExtractor)
    monkeypatch.setattr(jieba_module, "redis_client", locks)
    monkeypatch.setattr(jieba_module, "storage", storage)
    apply_config_overrides(monkeypatch, KEYWORD_DATA_SOURCE_TYPE=request.param)
    dataset = make_dataset(name="Test", created_by="author", keyword_number=2)
    sqlite_session.add(dataset)
    sqlite_session.commit()
    return KeywordRuntime(sqlite_session, dataset, storage, locks, extracted)


def segment(node_id: str, *, tenant_id: str = "tenant-1") -> DocumentSegment:
    return DocumentSegment(
        tenant_id=tenant_id,
        dataset_id="dataset-1",
        document_id="document-1",
        position=1,
        content=node_id,
        word_count=len(node_id),
        tokens=0,
        created_by="author",
        index_node_id=node_id,
    )


def single_connection_sessions(sqlite_engine: Engine) -> tuple[Engine, sessionmaker[Session]]:
    engine = create_engine(
        sqlite_engine.url,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
    )
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


@pytest.mark.parametrize("storage_type", ["database", "file"])
def test_write_uses_the_callers_checked_out_connection(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    storage_type: str,
) -> None:
    keyword_storage = KeywordStorage()
    locks = KeywordLocks()
    monkeypatch.setattr(jieba_module, "redis_client", locks)
    monkeypatch.setattr(jieba_module, "storage", keyword_storage)
    apply_config_overrides(monkeypatch, KEYWORD_DATA_SOURCE_TYPE=storage_type)
    engine, sessions = single_connection_sessions(sqlite_engine)
    try:
        with sessions.begin() as session:
            session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
            session.add(segment("node-1"))

        with sessions() as caller:
            dataset = caller.get(Dataset, "dataset-1")
            assert dataset is not None
            assert caller.in_transaction()
            Jieba(dataset).add_texts(
                [Document(page_content="alpha", metadata={"doc_id": "node-1"})],
                caller,
                keywords_list=[["alpha"]],
            )
            assert locks.held is True
            caller.commit()
            assert locks.held is False

        with sessions() as verifier:
            dataset = verifier.get(Dataset, "dataset-1")
            assert dataset is not None
            row = get_dataset_keyword_table(dataset, session=verifier)
            assert row is not None
            payload = load_keyword_table(
                storage=keyword_storage,
                tenant_id=dataset.tenant_id,
                dataset_id=dataset.id,
                storage_type=row.data_source_type,
                data=row.keyword_table,
            )
            assert payload is not None
            assert payload["__data__"]["table"] == {"alpha": {"node-1"}}
    finally:
        engine.dispose()


@pytest.mark.parametrize("storage_type", ["database", "file"])
@pytest.mark.parametrize("consumer", ["search", "text_exists"])
def test_read_uses_the_callers_checked_out_connection(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    storage_type: str,
    consumer: str,
) -> None:
    keyword_storage = KeywordStorage()
    locks = KeywordLocks()
    monkeypatch.setattr(jieba_module, "redis_client", locks)
    monkeypatch.setattr(jieba_module, "storage", keyword_storage)
    apply_config_overrides(monkeypatch, KEYWORD_DATA_SOURCE_TYPE=storage_type)
    engine, sessions = single_connection_sessions(sqlite_engine)
    try:
        with sessions.begin() as session:
            session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
            session.add(segment("node-1"))
        with sessions() as seed:
            dataset = seed.get(Dataset, "dataset-1")
            assert dataset is not None
            Jieba(dataset).add_texts(
                [Document(page_content="alpha", metadata={"doc_id": "node-1"})],
                seed,
                keywords_list=[["alpha"]],
            )
            seed.commit()

        with sessions() as caller:
            dataset = caller.get(Dataset, "dataset-1")
            assert dataset is not None
            assert caller.in_transaction()
            keyword = Jieba(dataset)
            if consumer == "search":
                assert [document.metadata["doc_id"] for document in keyword.search("alpha", session=caller)] == [
                    "node-1"
                ]
            else:
                assert keyword.text_exists("node-1", session=caller)
            assert locks.held is True
            caller.commit()
            assert locks.held is False
    finally:
        engine.dispose()


@pytest.mark.parametrize("storage_type", ["database", "file"])
def test_two_committed_sessions_preserve_both_keyword_updates(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    storage_type: str,
) -> None:
    keyword_storage = KeywordStorage()
    locks = BlockingKeywordLocks()
    monkeypatch.setattr(jieba_module, "redis_client", locks)
    monkeypatch.setattr(jieba_module, "storage", keyword_storage)
    apply_config_overrides(monkeypatch, KEYWORD_DATA_SOURCE_TYPE=storage_type)
    with sqlite_session_factory.begin() as session:
        session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
        session.add_all([segment("node-1"), segment("node-2")])

    first_flushed = Event()
    allow_first_commit = Event()
    errors: list[Exception] = []

    def mutate(node_id: str, keyword_value: str, *, wait_before_commit: bool) -> None:
        try:
            with sqlite_session_factory() as session:
                dataset = session.get(Dataset, "dataset-1")
                assert dataset is not None
                Jieba(dataset).add_texts(
                    [Document(page_content=keyword_value, metadata={"doc_id": node_id})],
                    session,
                    keywords_list=[[keyword_value]],
                )
                if wait_before_commit:
                    first_flushed.set()
                    assert allow_first_commit.wait(timeout=5)
                session.commit()
        except Exception as error:
            errors.append(error)
            first_flushed.set()
            allow_first_commit.set()

    first = Thread(target=mutate, args=("node-1", "alpha"), kwargs={"wait_before_commit": True})
    second = Thread(target=mutate, args=("node-2", "beta"), kwargs={"wait_before_commit": False})
    first.start()
    assert first_flushed.wait(timeout=5)
    second.start()
    assert locks.second_attempt.wait(timeout=5)
    assert second.is_alive()
    allow_first_commit.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    with sqlite_session_factory() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        row = get_dataset_keyword_table(dataset, session=session)
        assert row is not None
        payload = load_keyword_table(
            storage=keyword_storage,
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            storage_type=row.data_source_type,
            data=row.keyword_table,
        )
        assert payload is not None
        assert payload["__data__"]["table"] == {"alpha": {"node-1"}, "beta": {"node-2"}}
        segments = session.scalars(
            select(DocumentSegment)
            .where(DocumentSegment.dataset_id == "dataset-1")
            .order_by(DocumentSegment.index_node_id)
        ).all()
        assert [segment.keywords for segment in segments] == [["alpha"], ["beta"]]
    if storage_type == "file":
        assert len(keyword_storage.files) == 1


@pytest.mark.parametrize("storage_type", ["database", "file"])
def test_stale_caller_identity_map_reads_latest_table_after_acquiring_lock(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    storage_type: str,
) -> None:
    keyword_storage = KeywordStorage()
    locks = KeywordLocks()
    monkeypatch.setattr(jieba_module, "redis_client", locks)
    monkeypatch.setattr(jieba_module, "storage", keyword_storage)
    apply_config_overrides(monkeypatch, KEYWORD_DATA_SOURCE_TYPE=storage_type)
    with sqlite_session_factory.begin() as session:
        session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
        session.add_all([segment("node-1"), segment("node-2"), segment("node-3")])

    with sqlite_session_factory() as seed:
        dataset = seed.get(Dataset, "dataset-1")
        assert dataset is not None
        Jieba(dataset).add_texts(
            [Document(page_content="alpha", metadata={"doc_id": "node-1"})],
            seed,
            keywords_list=[["alpha"]],
        )
        seed.commit()

    with sqlite_session_factory() as caller:
        dataset = caller.get(Dataset, "dataset-1")
        assert dataset is not None
        stale_row = get_dataset_keyword_table(dataset, session=caller)
        assert stale_row is not None
        stale_data = stale_row.keyword_table
        stale_file = keyword_storage.files.get(stale_data)
        caller.commit()

        with sqlite_session_factory() as concurrent_writer:
            concurrent_dataset = concurrent_writer.get(Dataset, "dataset-1")
            assert concurrent_dataset is not None
            Jieba(concurrent_dataset).add_texts(
                [Document(page_content="beta", metadata={"doc_id": "node-2"})],
                concurrent_writer,
                keywords_list=[["beta"]],
            )
            concurrent_writer.commit()

        assert stale_row.keyword_table == stale_data
        if stale_file is not None:
            keyword_storage.files[stale_data] = stale_file
        keyword = Jieba(dataset)
        assert keyword.text_exists("node-2", session=caller)
        assert [document.metadata["doc_id"] for document in keyword.search("beta", session=caller)] == ["node-2"]
        keyword.add_texts(
            [Document(page_content="gamma", metadata={"doc_id": "node-3"})],
            caller,
            keywords_list=[["gamma"]],
        )
        caller.commit()

    with sqlite_session_factory() as verifier:
        dataset = verifier.get(Dataset, "dataset-1")
        assert dataset is not None
        row = get_dataset_keyword_table(dataset, session=verifier)
        assert row is not None
        payload = load_keyword_table(
            storage=keyword_storage,
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            storage_type=row.data_source_type,
            data=row.keyword_table,
        )
        assert payload is not None
        assert payload["__data__"]["table"] == {
            "alpha": {"node-1"},
            "beta": {"node-2"},
            "gamma": {"node-3"},
        }


def test_repeated_writes_in_one_transaction_reuse_uncommitted_table(runtime: KeywordRuntime) -> None:
    runtime.session.add_all([segment("node-1"), segment("node-2")])
    keyword = Jieba(runtime.dataset)

    keyword.add_texts(
        [Document(page_content="alpha", metadata={"doc_id": "node-1"})],
        runtime.session,
        keywords_list=[["alpha"]],
    )
    keyword.add_texts(
        [Document(page_content="beta", metadata={"doc_id": "node-2"})],
        runtime.session,
        keywords_list=[["beta"]],
    )

    assert runtime.locks.held is True
    assert len(runtime.locks.acquired) == 1
    runtime.session.commit()
    assert runtime.table() == {"alpha": {"node-1"}, "beta": {"node-2"}}


def test_create_extracts_keywords_and_persists_the_shared_format(runtime: KeywordRuntime) -> None:
    chunk = segment("node-1")
    foreign = segment("node-1", tenant_id="tenant-2")
    runtime.session.add_all([chunk, foreign])
    keyword = Jieba(runtime.dataset)
    result = keyword.create(
        [Document(page_content="中文 alpha ignored", metadata={"doc_id": "node-1"})], runtime.session
    )
    assert runtime.locks.held is True
    assert keyword.text_exists("node-1", session=runtime.session)
    runtime.session.commit()

    assert result is keyword
    assert set(chunk.keywords) == {"中文", "alpha"}
    assert foreign.keywords is None
    assert runtime.table() == {"中文": {"node-1"}, "alpha": {"node-1"}}
    assert runtime.extracted == [("中文 alpha ignored", 2)]
    assert runtime.locks.acquired == ["keyword_indexing_lock_dataset-1"]
    assert runtime.locks.held is False
    row = get_dataset_keyword_table(runtime.dataset, session=runtime.session)
    assert row is not None
    payload = load_keyword_table(
        storage=runtime.storage,
        tenant_id=runtime.dataset.tenant_id,
        dataset_id=runtime.dataset.id,
        storage_type=row.data_source_type,
        data=row.keyword_table,
    )
    assert payload is not None
    assert payload["__type__"] == "keyword_table"
    assert payload["__data__"]["index_id"] == "dataset-1"
    if row.data_source_type == "file":
        assert row.keyword_table == runtime.storage.writes[0]
        assert row.keyword_table.startswith("keyword_files/tenant-1/dataset-1/")
        assert row.keyword_table.endswith(".txt")
    else:
        assert runtime.storage.reads == []
        assert runtime.storage.writes == []


def test_add_uses_manual_keywords_and_extracts_empty_selections(runtime: KeywordRuntime) -> None:
    chunks = [segment("node-1"), segment("node-2")]
    runtime.session.add_all(chunks)
    keyword = Jieba(runtime.dataset)
    keyword.add_texts(
        [
            Document(page_content="automatic", metadata={"doc_id": "node-1"}),
            Document(page_content="ignored", metadata={"doc_id": "node-2"}),
        ],
        runtime.session,
        keywords_list=[[], ["manual"]],
    )
    runtime.session.commit()
    assert runtime.table() == {"automatic": {"node-1"}, "manual": {"node-2"}}
    assert runtime.extracted == [("automatic", 2)]
    assert chunks[0].keywords == ["automatic"]
    assert chunks[1].keywords == ["manual"]


def test_delete_ids_preserves_unrelated_entries(runtime: KeywordRuntime) -> None:
    keyword = Jieba(runtime.dataset)
    keyword.add_texts(
        [
            Document(page_content="shared first", metadata={"doc_id": "node-1"}),
            Document(page_content="shared second", metadata={"doc_id": "node-2"}),
        ],
        runtime.session,
    )
    runtime.session.commit()
    keyword.delete_by_ids(["node-1"], runtime.session)
    runtime.session.commit()
    assert runtime.table() == {"shared": {"node-2"}, "second": {"node-2"}}
    assert keyword.text_exists("node-1", session=runtime.session) is False
    assert keyword.text_exists("node-2", session=runtime.session) is True
    keyword.delete_by_ids(["node-2"], runtime.session)
    runtime.session.commit()
    assert runtime.table() == {}
    assert keyword.text_exists("node-2", session=runtime.session) is False


@pytest.mark.parametrize("runtime", ["file"], indirect=True)
def test_lost_writer_cleans_only_its_unpublished_file(runtime: KeywordRuntime) -> None:
    keyword = Jieba(runtime.dataset)
    keyword.add_texts(
        [Document(page_content="base", metadata={"doc_id": "base-node"})],
        runtime.session,
    )
    runtime.session.commit()
    row = get_dataset_keyword_table(runtime.dataset, session=runtime.session)
    assert row is not None
    initial_pointer = row.keyword_table
    stale_pointer: str | None = None
    active_pointer: str | None = None

    runtime.session.rollback()
    sessions = sessionmaker(bind=runtime.session.get_bind(), expire_on_commit=False)

    def read() -> tuple[str, str | None]:
        with sessions() as session:
            current = get_dataset_keyword_table(runtime.dataset, session=session)
            assert current is not None
            return current.data_source_type, current.keyword_table

    def persist(storage_type: str, data: str, keywords: Mapping[str, Sequence[str]]) -> None:
        with sessions.begin() as session:
            persist_keyword_table(
                session,
                tenant_id=runtime.dataset.tenant_id,
                dataset_id=runtime.dataset.id,
                storage_type=storage_type,
                data=data,
                keywords=keywords,
            )

    def reject_stale_write(storage_type: str, data: str, _keywords: Mapping[str, Sequence[str]]) -> None:
        nonlocal active_pointer, stale_pointer
        stale_pointer = data
        keyword.update_texts(
            [Document(page_content="active", metadata={"doc_id": "active-node"})],
            read=read,
            write=persist,
            lock=nullcontext(),
            cleanup_session_factory=sessions,
            keywords_list=[["active"]],
        )
        _, active_pointer = read()
        raise RuntimeError("claim lost")

    with pytest.raises(RuntimeError, match="claim lost"):
        keyword.update_texts(
            [Document(page_content="stale", metadata={"doc_id": "stale-node"})],
            read=read,
            write=reject_stale_write,
            lock=nullcontext(),
            cleanup_session_factory=sessions,
            keywords_list=[["stale"]],
        )

    assert stale_pointer is not None
    assert active_pointer is not None
    assert active_pointer != initial_pointer != stale_pointer
    assert stale_pointer not in runtime.storage.files
    assert initial_pointer not in runtime.storage.files
    assert active_pointer in runtime.storage.files
    assert runtime.table() == {"base": {"base-node"}, "active": {"active-node"}}


@pytest.mark.parametrize("runtime", ["file"], indirect=True)
def test_file_update_reclaims_previous_blob_after_commit(runtime: KeywordRuntime) -> None:
    keyword = Jieba(runtime.dataset)
    keyword.add_texts(
        [Document(page_content="first", metadata={"doc_id": "node-1"})],
        runtime.session,
    )
    runtime.session.commit()
    first_pointer = next(iter(runtime.storage.files))
    delete = runtime.storage.delete

    def delete_outside_transaction(key: str) -> None:
        assert not runtime.session.in_transaction()
        assert runtime.locks.held
        delete(key)

    runtime.storage.delete = delete_outside_transaction

    keyword.add_texts(
        [Document(page_content="second", metadata={"doc_id": "node-2"})],
        runtime.session,
    )
    row = get_dataset_keyword_table(runtime.dataset, session=runtime.session)
    assert row is not None
    second_pointer = row.keyword_table
    assert set(runtime.storage.files) == {first_pointer, second_pointer}

    runtime.session.commit()

    assert set(runtime.storage.files) == {second_pointer}
    assert runtime.table() == {"first": {"node-1"}, "second": {"node-2"}}


@pytest.mark.parametrize("runtime", ["file"], indirect=True)
@pytest.mark.parametrize("consumer", ["search", "text_exists"])
def test_reader_refreshes_a_reclaimed_pointer_under_the_publication_lock(
    runtime: KeywordRuntime, consumer: str
) -> None:
    keyword = Jieba(runtime.dataset)
    keyword.add_texts([Document(page_content="old", metadata={"doc_id": "node-1"})], runtime.session)
    runtime.session.add(segment("node-2"))
    runtime.session.commit()
    stale_row = get_dataset_keyword_table(runtime.dataset, session=runtime.session)
    assert stale_row is not None
    old_pointer = stale_row.keyword_table

    # Publish and reclaim before this reader acquires the mutex, leaving its
    # caller-owned Session with a previously fetched, obsolete pointer.
    with Session(bind=runtime.session.get_bind()) as writer:
        new_pointer = save_keyword_table(
            storage=runtime.storage,
            tenant_id=runtime.dataset.tenant_id,
            dataset_id=runtime.dataset.id,
            storage_type="file",
            table={"active": {"node-2"}},
            begin_file_write=lambda key: begin_keyword_file_write_in_session(
                writer,
                tenant_id=runtime.dataset.tenant_id,
                dataset_id=runtime.dataset.id,
                key=key,
            ),
            finish_file_write=lambda key, writer_id: finish_keyword_file_write_in_session(
                writer,
                tenant_id=runtime.dataset.tenant_id,
                dataset_id=runtime.dataset.id,
                key=key,
                writer_id=writer_id,
            ),
        )
        persist_keyword_table(
            writer,
            tenant_id=runtime.dataset.tenant_id,
            dataset_id=runtime.dataset.id,
            storage_type="file",
            data=new_pointer,
            keywords={},
        )
        writer.commit()
    runtime.storage.delete(old_pointer)
    assert stale_row.keyword_table == old_pointer

    load = runtime.storage.load_once

    def load_while_locked(key: str) -> bytes:
        assert runtime.locks.held
        return load(key)

    runtime.storage.load_once = load_while_locked
    if consumer == "search":
        results = keyword.search("active", session=runtime.session)
        assert [result.metadata["doc_id"] for result in results] == ["node-2"]
    else:
        assert keyword.text_exists("node-2", session=runtime.session)
    assert runtime.storage.reads[-1:] == [new_pointer]


@pytest.mark.parametrize("runtime", ["file"], indirect=True)
def test_file_update_rollback_reclaims_unpublished_blob(runtime: KeywordRuntime) -> None:
    keyword = Jieba(runtime.dataset)
    keyword.add_texts(
        [Document(page_content="first", metadata={"doc_id": "node-1"})],
        runtime.session,
    )
    runtime.session.commit()
    first_pointer = next(iter(runtime.storage.files))
    delete = runtime.storage.delete

    def delete_outside_transaction(key: str) -> None:
        assert not runtime.session.in_transaction()
        delete(key)

    runtime.storage.delete = delete_outside_transaction

    keyword.add_texts(
        [Document(page_content="second", metadata={"doc_id": "node-2"})],
        runtime.session,
    )
    row = get_dataset_keyword_table(runtime.dataset, session=runtime.session)
    assert row is not None
    unpublished_pointer = row.keyword_table
    runtime.session.rollback()

    assert unpublished_pointer not in runtime.storage.files
    assert set(runtime.storage.files) == {first_pointer}
    assert runtime.table() == {"first": {"node-1"}}


@pytest.mark.parametrize("runtime", ["file"], indirect=True)
def test_file_cleanup_failure_does_not_replace_write_error(
    runtime: KeywordRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_delete(_key: str) -> None:
        raise OSError("storage unavailable")

    monkeypatch.setattr(runtime.storage, "delete", fail_delete)

    def fail_write(_storage_type: str, _data: str, _keywords: Mapping[str, Sequence[str]]) -> None:
        raise RuntimeError("claim lost")

    cleanup_sessions = sessionmaker(bind=runtime.session.get_bind(), expire_on_commit=False)

    with pytest.raises(RuntimeError, match="claim lost"):
        Jieba(runtime.dataset).update_texts(
            [Document(page_content="stale", metadata={"doc_id": "stale-node"})],
            read=lambda: ("file", None),
            write=fail_write,
            lock=nullcontext(),
            cleanup_session_factory=cleanup_sessions,
            keywords_list=[["stale"]],
        )


def test_delete_without_an_existing_table_is_idempotent(runtime: KeywordRuntime) -> None:
    keyword = Jieba(runtime.dataset)
    assert keyword.text_exists("missing", session=runtime.session) is False
    keyword.delete_by_ids(["missing"], runtime.session)
    runtime.session.commit()
    keyword.delete_by_ids(["missing"], runtime.session)
    runtime.session.commit()
    assert runtime.table() == {}


def test_search_preserves_ranking_and_document_filters(runtime: KeywordRuntime) -> None:
    first, second = segment("node-1"), segment("node-2")
    second.document_id = "document-2"
    runtime.session.add_all([first, second])
    keyword = Jieba(runtime.dataset)
    keyword.add_texts(
        [
            Document(page_content="alpha", metadata={"doc_id": "node-1"}),
            Document(page_content="alpha beta", metadata={"doc_id": "node-2"}),
        ],
        runtime.session,
    )
    runtime.session.commit()
    documents = keyword.search("alpha beta", session=runtime.session, top_k=2)
    assert [doc.metadata["doc_id"] for doc in documents] == ["node-2", "node-1"]
    documents = keyword.search("alpha beta", session=runtime.session, top_k=1, document_ids_filter=["document-1"])
    assert [doc.metadata["doc_id"] for doc in documents] == ["node-1"]


def test_delete_removes_the_database_record_and_file(runtime: KeywordRuntime) -> None:
    keyword = Jieba(runtime.dataset)
    keyword.create([Document(page_content="alpha", metadata={"doc_id": "node-1"})], runtime.session)
    runtime.session.commit()
    keyword.delete(session=runtime.session)
    assert runtime.locks.held is True
    runtime.session.commit()
    assert runtime.session.scalar(select(DatasetKeywordTable)) is None
    assert runtime.storage.files == {}


@pytest.mark.parametrize("runtime", ["file"], indirect=True)
def test_delete_rollback_keeps_the_published_file(runtime: KeywordRuntime) -> None:
    keyword = Jieba(runtime.dataset)
    keyword.create([Document(page_content="alpha", metadata={"doc_id": "node-1"})], runtime.session)
    runtime.session.commit()
    row = get_dataset_keyword_table(runtime.dataset, session=runtime.session)
    assert row is not None
    active_pointer = row.keyword_table

    keyword.delete(session=runtime.session)
    assert runtime.locks.held is True
    assert active_pointer in runtime.storage.files
    runtime.session.rollback()

    restored = get_dataset_keyword_table(runtime.dataset, session=runtime.session)
    assert restored is not None
    assert restored.keyword_table == active_pointer
    assert active_pointer in runtime.storage.files
    assert runtime.locks.held is False


@pytest.mark.parametrize("storage_type", ["database", "file"])
def test_delete_refreshes_stale_row_and_reclaims_the_latest_file(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    storage_type: str,
) -> None:
    keyword_storage = KeywordStorage()
    locks = KeywordLocks()
    monkeypatch.setattr(jieba_module, "redis_client", locks)
    monkeypatch.setattr(jieba_module, "storage", keyword_storage)
    apply_config_overrides(monkeypatch, KEYWORD_DATA_SOURCE_TYPE=storage_type)
    with sqlite_session_factory.begin() as session:
        session.add(Dataset(id="dataset-1", tenant_id="tenant-1", name="Test", created_by="author"))
        session.add_all([segment("node-1"), segment("node-2")])

    with sqlite_session_factory() as seed:
        dataset = seed.get(Dataset, "dataset-1")
        assert dataset is not None
        Jieba(dataset).add_texts(
            [Document(page_content="alpha", metadata={"doc_id": "node-1"})],
            seed,
            keywords_list=[["alpha"]],
        )
        seed.commit()

    with sqlite_session_factory() as caller:
        dataset = caller.get(Dataset, "dataset-1")
        assert dataset is not None
        stale_row = get_dataset_keyword_table(dataset, session=caller)
        assert stale_row is not None
        stale_data = stale_row.keyword_table
        caller.commit()

        with sqlite_session_factory() as concurrent_writer:
            concurrent_dataset = concurrent_writer.get(Dataset, "dataset-1")
            assert concurrent_dataset is not None
            Jieba(concurrent_dataset).add_texts(
                [Document(page_content="beta", metadata={"doc_id": "node-2"})],
                concurrent_writer,
                keywords_list=[["beta"]],
            )
            concurrent_writer.commit()

        assert stale_row.keyword_table == stale_data
        Jieba(dataset).delete(session=caller)
        caller.commit()

    with sqlite_session_factory() as verifier:
        assert verifier.scalar(select(DatasetKeywordTable).where(DatasetKeywordTable.dataset_id == "dataset-1")) is None
    assert keyword_storage.files == {}


def test_mutation_does_not_replace_an_unreadable_table(
    runtime: KeywordRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyword = Jieba(runtime.dataset)
    keyword.create([Document(page_content="original", metadata={"doc_id": "node-1"})], runtime.session)
    runtime.session.commit()
    row = get_dataset_keyword_table(runtime.dataset, session=runtime.session)
    assert row is not None
    stored_files = dict(runtime.storage.files)
    writes = list(runtime.storage.writes)
    if row.data_source_type == "database":
        row.keyword_table = "invalid json"
        runtime.session.commit()
        expected_error = ValueError
    else:

        def fail_read(_key: str) -> bytes:
            raise OSError("storage unavailable")

        monkeypatch.setattr(runtime.storage, "load_once", fail_read)
        expected_error = OSError
    with pytest.raises(expected_error):
        keyword.add_texts([Document(page_content="replacement", metadata={"doc_id": "node-1"})], runtime.session)
    runtime.session.rollback()
    assert runtime.storage.files == stored_files
    assert runtime.storage.writes == writes
    assert runtime.locks.held is False
    if row.data_source_type == "database":
        assert row.keyword_table == "invalid json"


def test_deletion_does_not_initialize_the_keyword_extractor(
    runtime: KeywordRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyword = Jieba(runtime.dataset)
    keyword.create([Document(page_content="original", metadata={"doc_id": "node-1"})], runtime.session)
    runtime.session.commit()

    def unexpected_extractor() -> None:
        pytest.fail("Index cleanup must not initialize the text extractor")

    monkeypatch.setattr(jieba_module, "JiebaKeywordTableHandler", unexpected_extractor)
    keyword.delete_by_ids(["node-1"], runtime.session)
    runtime.session.commit()
    assert runtime.table() == {}

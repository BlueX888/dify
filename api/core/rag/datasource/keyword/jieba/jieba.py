from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, override

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker

from configs import dify_config
from core.rag.datasource.keyword.jieba.jieba_keyword_table_handler import JiebaKeywordTableHandler
from core.rag.datasource.keyword.keyword_base import BaseKeyword
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from models.dataset import Dataset, DocumentSegment
from repositories.knowledge.dataset_read_repository import get_dataset_keyword_table
from repositories.knowledge.keyword_file_cleanup_repository import (
    SQLAlchemyKeywordFileCleanupRepository,
    begin_keyword_file_write_in_session,
    finish_keyword_file_write_in_session,
)
from repositories.knowledge.keyword_table_repository import (
    complete_keyword_file_delete,
    complete_keyword_file_write,
    discard_keyword_file_write,
    keyword_file_key,
    keyword_publication_lock,
    keyword_transaction_lock,
    load_keyword_table,
    lock_dataset_keyword_table,
    persist_keyword_table,
    save_keyword_table,
)


class KeywordTableConfig(BaseModel):
    max_keywords_per_chunk: int = 10


class Jieba(BaseKeyword):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self._config = KeywordTableConfig()

    @override
    def create(self, texts: list[Document], session: Session, **kwargs: Any) -> BaseKeyword:
        self.add_texts(texts, session)
        return self

    @override
    def add_texts(self, texts: list[Document], session: Session, **kwargs: Any):
        lock_name = f"keyword_indexing_lock_{self.dataset.id}"
        self.update_texts(
            texts,
            read=lambda: self._read_keyword_record(session, current=True),
            write=lambda storage_type, data, keywords: persist_keyword_table(
                session,
                tenant_id=self.dataset.tenant_id,
                dataset_id=self.dataset.id,
                storage_type=storage_type,
                data=data,
                keywords=keywords,
            ),
            lock=keyword_transaction_lock(
                session,
                lock_name=lock_name,
                lock_factory=lambda: redis_client.lock(lock_name, timeout=600),
            ),
            transaction_session=session,
            keywords_list=kwargs.get("keywords_list"),
        )

    def update_texts(
        self,
        texts: Sequence[Document],
        *,
        read: Callable[[], tuple[str, str | None]],
        write: Callable[[str, str, Mapping[str, Sequence[str]]], None],
        lock: AbstractContextManager[object],
        before_write: Callable[[], None] | None = None,
        transaction_session: Session | None = None,
        cleanup_session_factory: Callable[[], Session] | None = None,
        keywords_list: Sequence[Sequence[str] | None] | None = None,
        replace_existing: bool = False,
        deleted_ids: Sequence[str] = (),
    ) -> None:
        """Mutate the shared index under one dataset publication lock.

        Repository-owned callbacks may close short transactions before storage I/O.
        Legacy callers own a wider transaction: they prepare an immutable file and
        flush its pointer while retaining the mutex through commit or rollback.
        This method does not end that caller transaction; replaced-file cleanup is
        deferred until it has ended.
        """
        with lock:
            tenant_id = self.dataset.tenant_id
            dataset_id = self.dataset.id
            if cleanup_session_factory is None and transaction_session is not None:
                bind = transaction_session.get_bind()
                # Production databases can durably pre-register through a
                # second short transaction. SQLite cannot safely upgrade the
                # caller's open read transaction after that independent commit;
                # its rollback listener remains the test/local fallback.
                if bind.dialect.name != "sqlite":
                    cleanup_session_factory = sessionmaker(bind=bind, expire_on_commit=False)
            cleanup_repository = (
                SQLAlchemyKeywordFileCleanupRepository(session_factory=cleanup_session_factory)
                if cleanup_session_factory is not None
                else None
            )
            begin_file_write: Callable[[str], str] | None = None
            finish_file_write: Callable[[str, str], None] | None = None
            if cleanup_repository is not None:

                def begin_file_write(key: str) -> str:
                    return cleanup_repository.begin_file_write(
                        tenant_id=tenant_id,
                        dataset_id=dataset_id,
                        key=key,
                    )

                def finish_file_write(key: str, writer_id: str) -> None:
                    cleanup_repository.finish_file_write(
                        tenant_id=tenant_id,
                        dataset_id=dataset_id,
                        key=key,
                        writer_id=writer_id,
                    )

            elif transaction_session is not None:
                # SQLite cannot durably pre-register through a second writer
                # transaction while this caller owns its connection. Keep the
                # same publication fence in the caller transaction; production
                # databases use the independent durable repository above.
                def begin_file_write(key: str) -> str:
                    return begin_keyword_file_write_in_session(
                        transaction_session,
                        tenant_id=tenant_id,
                        dataset_id=dataset_id,
                        key=key,
                    )

                def finish_file_write(key: str, writer_id: str) -> None:
                    finish_keyword_file_write_in_session(
                        transaction_session,
                        tenant_id=tenant_id,
                        dataset_id=dataset_id,
                        key=key,
                        writer_id=writer_id,
                    )

            storage_type, data = read()
            payload = load_keyword_table(
                storage=storage,
                tenant_id=self.dataset.tenant_id,
                dataset_id=self.dataset.id,
                storage_type=storage_type,
                data=data,
            )
            table = dict(payload["__data__"]["table"] or {}) if payload else {}
            removed = list(deleted_ids)
            if replace_existing:
                removed.extend(text.metadata["doc_id"] for text in texts)
            if removed:
                table = self._delete_ids_from_keyword_table(table, removed)
            handler: JiebaKeywordTableHandler | None = None
            keyword_number = self.dataset.keyword_number or self._config.max_keywords_per_chunk
            selected: dict[str, Sequence[str]] = {}
            for position, text in enumerate(texts):
                keywords = keywords_list[position] if keywords_list else None
                if not keywords:
                    if handler is None:
                        handler = JiebaKeywordTableHandler()
                    keywords = list(handler.extract_keywords(text.page_content, keyword_number))
                node_id = text.metadata["doc_id"]
                selected[node_id] = list(keywords)
                table = self._add_text_to_keyword_table(table, node_id, list(keywords))
            if before_write is not None:
                before_write()
            stored_data = save_keyword_table(
                storage=storage,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                storage_type=storage_type,
                table=table,
                begin_file_write=begin_file_write,
                finish_file_write=finish_file_write,
            )
            try:
                write(storage_type, stored_data, selected)
            except Exception:
                if storage_type != "database":
                    discard_keyword_file_write(
                        storage=storage,
                        tenant_id=tenant_id,
                        dataset_id=dataset_id,
                        new_key=stored_data,
                        transaction_session=transaction_session,
                        publication_lock=(
                            (lambda: redis_client.lock(f"keyword_indexing_lock_{dataset_id}", timeout=600))
                            if transaction_session is not None
                            else nullcontext
                        ),
                        cleanup_repository=cleanup_repository,
                    )
                raise
            if storage_type != "database":
                previous_key = None
                if data is not None:
                    previous_key = data or keyword_file_key(tenant_id, dataset_id)
                complete_keyword_file_write(
                    storage=storage,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    previous_key=previous_key,
                    new_key=stored_data,
                    transaction_session=transaction_session,
                    publication_lock=(
                        (lambda: redis_client.lock(f"keyword_indexing_lock_{dataset_id}", timeout=600))
                        if transaction_session is not None
                        else nullcontext
                    ),
                    cleanup_repository=cleanup_repository,
                )

    def _read_keyword_record(self, session: Session, *, current: bool, shared: bool = False) -> tuple[str, str | None]:
        """Read with the caller's current transaction, or a short standalone transaction."""
        if current:
            row = lock_dataset_keyword_table(
                session,
                tenant_id=self.dataset.tenant_id,
                dataset_id=self.dataset.id,
                read=shared,
            )
            return (row.data_source_type, row.keyword_table) if row else (dify_config.KEYWORD_DATA_SOURCE_TYPE, None)
        bind = session.get_bind()
        reader_bind = bind.engine if isinstance(bind, Connection) else bind
        with Session(bind=reader_bind) as reader:
            row = get_dataset_keyword_table(self.dataset, session=reader)
            return (row.data_source_type, row.keyword_table) if row else (dify_config.KEYWORD_DATA_SOURCE_TYPE, None)

    @override
    def text_exists(self, id: str, *, session: Session) -> bool:
        keyword_table = None
        lock_name = f"keyword_indexing_lock_{self.dataset.id}"
        current = session.in_transaction() or isinstance(session.get_bind(), Connection)
        lock = (
            keyword_transaction_lock(
                session,
                lock_name=lock_name,
                lock_factory=lambda: redis_client.lock(lock_name, timeout=600),
            )
            if current
            else keyword_publication_lock(
                session,
                lock_name=lock_name,
                lock_factory=lambda: redis_client.lock(lock_name, timeout=600),
            )
        )
        with lock:
            storage_type, data = self._read_keyword_record(session, current=current, shared=True)
            keyword_table_dict = load_keyword_table(
                storage=storage,
                tenant_id=self.dataset.tenant_id,
                dataset_id=self.dataset.id,
                storage_type=storage_type,
                data=data,
            )
        if keyword_table_dict:
            data: Any = keyword_table_dict["__data__"]
            keyword_table = dict(data["table"])
        if keyword_table is None:
            return False
        return any(id in node_ids for node_ids in keyword_table.values())

    @override
    def delete_by_ids(self, ids: list[str], session: Session, **kwargs: Any):
        lock_name = f"keyword_indexing_lock_{self.dataset.id}"
        self.update_texts(
            [],
            read=lambda: self._read_keyword_record(session, current=True),
            write=lambda storage_type, data, keywords: persist_keyword_table(
                session,
                tenant_id=self.dataset.tenant_id,
                dataset_id=self.dataset.id,
                storage_type=storage_type,
                data=data,
                keywords=keywords,
            ),
            lock=keyword_transaction_lock(
                session,
                lock_name=lock_name,
                lock_factory=lambda: redis_client.lock(lock_name, timeout=600),
            ),
            transaction_session=session,
            deleted_ids=ids,
        )

    @override
    def search(self, query: str, *, session: Session, **kwargs: Any) -> list[Document]:
        keyword_table = None
        lock_name = f"keyword_indexing_lock_{self.dataset.id}"
        current = session.in_transaction() or isinstance(session.get_bind(), Connection)
        lock = (
            keyword_transaction_lock(
                session,
                lock_name=lock_name,
                lock_factory=lambda: redis_client.lock(lock_name, timeout=600),
            )
            if current
            else keyword_publication_lock(
                session,
                lock_name=lock_name,
                lock_factory=lambda: redis_client.lock(lock_name, timeout=600),
            )
        )
        with lock:
            storage_type, data = self._read_keyword_record(session, current=current, shared=True)
            keyword_table_dict = load_keyword_table(
                storage=storage,
                tenant_id=self.dataset.tenant_id,
                dataset_id=self.dataset.id,
                storage_type=storage_type,
                data=data,
            )
        if keyword_table_dict:
            data: Any = keyword_table_dict["__data__"]
            keyword_table = dict(data["table"])

        k = kwargs.get("top_k", 4)
        document_ids_filter = kwargs.get("document_ids_filter")
        segment_map: dict[str, DocumentSegment] | None = None
        candidate_ids: set[str] | None = None
        if document_ids_filter is not None:
            segments = session.scalars(
                select(DocumentSegment).where(
                    DocumentSegment.tenant_id == self.dataset.tenant_id,
                    DocumentSegment.dataset_id == self.dataset.id,
                    DocumentSegment.document_id.in_(document_ids_filter),
                )
            ).all()
            segment_map = {segment.index_node_id: segment for segment in segments if segment.index_node_id is not None}
            candidate_ids = set(segment_map)
        sorted_chunk_indices = self._retrieve_ids_by_query(keyword_table or {}, query, k, candidate_ids=candidate_ids)

        documents = []
        if segment_map is None:
            segments = session.scalars(
                select(DocumentSegment).where(
                    DocumentSegment.tenant_id == self.dataset.tenant_id,
                    DocumentSegment.dataset_id == self.dataset.id,
                    DocumentSegment.index_node_id.in_(sorted_chunk_indices),
                )
            ).all()
            segment_map = {segment.index_node_id: segment for segment in segments if segment.index_node_id is not None}
        for chunk_index in sorted_chunk_indices:
            segment = segment_map.get(chunk_index)

            if segment:
                documents.append(
                    Document(
                        page_content=segment.content,
                        metadata={
                            "doc_id": chunk_index,
                            "doc_hash": segment.index_node_hash,
                            "document_id": segment.document_id,
                            "dataset_id": segment.dataset_id,
                        },
                    )
                )

        return documents

    @override
    def delete(self, *, session: Session):
        lock_name = f"keyword_indexing_lock_{self.dataset.id}"
        with keyword_transaction_lock(
            session,
            lock_name=lock_name,
            lock_factory=lambda: redis_client.lock(lock_name, timeout=600),
        ):
            dataset_keyword_table = lock_dataset_keyword_table(
                session,
                tenant_id=self.dataset.tenant_id,
                dataset_id=self.dataset.id,
            )
            if dataset_keyword_table:
                file_key = None
                if dataset_keyword_table.data_source_type != "database":
                    file_key = dataset_keyword_table.keyword_table or keyword_file_key(
                        self.dataset.tenant_id, self.dataset.id
                    )
                session.delete(dataset_keyword_table)
                session.flush()
                if file_key is not None:
                    complete_keyword_file_delete(
                        storage=storage,
                        tenant_id=self.dataset.tenant_id,
                        dataset_id=self.dataset.id,
                        previous_key=file_key,
                        transaction_session=session,
                        publication_lock=lambda: redis_client.lock(lock_name, timeout=600),
                    )

    def _add_text_to_keyword_table(
        self, keyword_table: dict[str, set[str]], id: str, keywords: list[str]
    ) -> dict[str, set[str]]:
        for keyword in keywords:
            if keyword not in keyword_table:
                keyword_table[keyword] = set()
            keyword_table[keyword].add(id)
        return keyword_table

    def _delete_ids_from_keyword_table(self, keyword_table: dict[str, set[str]], ids: list[str]) -> dict[str, set[str]]:
        # get set of ids that correspond to node
        node_idxs_to_delete = set(ids)

        # delete node_idxs from keyword to node idxs mapping
        keywords_to_delete = set()
        for keyword, node_idxs in keyword_table.items():
            if node_idxs_to_delete.intersection(node_idxs):
                keyword_table[keyword] = node_idxs.difference(node_idxs_to_delete)
                if not keyword_table[keyword]:
                    keywords_to_delete.add(keyword)

        for keyword in keywords_to_delete:
            del keyword_table[keyword]

        return keyword_table

    def _retrieve_ids_by_query(
        self,
        keyword_table: dict[str, set[str]],
        query: str,
        k: int = 4,
        *,
        candidate_ids: set[str] | None = None,
    ) -> list[str]:
        keyword_table_handler = JiebaKeywordTableHandler()
        keywords = keyword_table_handler.extract_keywords(query)

        # go through text chunks in order of most matching keywords
        chunk_indices_count: dict[str, int] = defaultdict(int)
        keywords_list = [keyword for keyword in keywords if keyword in set(keyword_table.keys())]
        for keyword in keywords_list:
            for node_id in keyword_table[keyword]:
                if candidate_ids is None or node_id in candidate_ids:
                    chunk_indices_count[node_id] += 1

        sorted_chunk_indices = sorted(
            chunk_indices_count.keys(),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )

        return sorted_chunk_indices[:k]

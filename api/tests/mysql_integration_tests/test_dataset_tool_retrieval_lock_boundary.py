"""MySQL regressions for retrieval authorization and hit-count lock boundaries."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import delete, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from core.app.app_config.entities import DatasetRetrieveConfigEntity
from core.app.apps.base_app_queue_manager import AppQueueManager
from core.app.entities.app_invoke_entities import InvokeFrom
from core.callback_handler.index_tool_callback_handler import DatasetIndexToolCallbackHandler
from core.rag.entities.retrieval_settings import ExternalDatasetRetriever
from core.rag.models.document import Document as RagDocument
from core.tools.utils.dataset_retriever import dataset_multi_retriever_tool as multi_retriever_module
from core.tools.utils.dataset_retriever import dataset_retriever_tool as single_retriever_module
from core.tools.utils.dataset_retriever.dataset_multi_retriever_tool import DatasetMultiRetrieverTool
from core.tools.utils.dataset_retriever.dataset_retriever_tool import DatasetRetrieverTool
from models.dataset import Dataset, DatasetQuery, Document, DocumentSegment
from models.enums import DataSourceType, DocumentCreatedFrom, SegmentStatus
from services.knowledge.retrieval.finalization import KnowledgeRetrievalFinalizer


class _UnexpectedQueueBoundary:
    def publish(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("return_resource=False must not publish queue events")


class _UnexpectedExternalRetrieval:
    def retrieve(self, **_kwargs: object) -> list[dict[str, Any]]:
        raise AssertionError("vendor datasets must not use external retrieval")


class _IdentityRerankRunner:
    def __init__(self, _model_instance: object, *, session: Session) -> None:
        self._session = session

    def run(
        self,
        _query: str,
        documents: list[RagDocument],
        _score_threshold: float | None,
        _top_n: int,
    ) -> list[RagDocument]:
        return documents


class _RerankModelBoundary:
    def get_model_instance(self, **_kwargs: object) -> object:
        return object()


_PREVIOUS_LOCK_TIMEOUT_KEY = "dataset_tool_test_previous_lock_wait_timeout"


def _set_mysql_lock_wait_timeout(dbapi_connection, connection_record, _connection_proxy) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SELECT @@SESSION.innodb_lock_wait_timeout")
        connection_record.info[_PREVIOUS_LOCK_TIMEOUT_KEY] = int(cursor.fetchone()[0])
        cursor.execute("SET SESSION innodb_lock_wait_timeout = 2")
    finally:
        cursor.close()


def _restore_mysql_lock_wait_timeout(dbapi_connection, connection_record) -> None:
    previous = connection_record.info.pop(_PREVIOUS_LOCK_TIMEOUT_KEY, None)
    if previous is None or dbapi_connection is None:
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(f"SET SESSION innodb_lock_wait_timeout = {int(previous)}")
    finally:
        cursor.close()


def _candidate(document: Document, segment: DocumentSegment) -> RagDocument:
    return RagDocument(
        page_content=segment.content,
        provider="dify",
        metadata={
            "doc_id": segment.index_node_id,
            "document_id": document.id,
            "dataset_id": document.dataset_id,
            "score": 0.9,
        },
    )


@pytest.mark.parametrize("tool_kind", ["single", "multi"])
def test_dataset_tool_releases_authorization_share_locks_before_real_hit_callback(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    tool_kind: str,
) -> None:
    """The real callback's independent UPDATE must not wait on the tool's authorization read."""

    engine = cast(Engine, mysql_session.get_bind())
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id = str(uuid4())
    dataset_id = str(uuid4())
    document_id = str(uuid4())
    segment_id = str(uuid4())
    actor_id = str(uuid4())
    with sessions.begin() as seed:
        dataset = Dataset(
            id=dataset_id,
            tenant_id=tenant_id,
            name="retrieval-lock-boundary",
            provider="vendor",
            data_source_type=DataSourceType.UPLOAD_FILE,
            indexing_technique="high_quality",
            retrieval_model={
                "search_method": "semantic_search",
                "score_threshold_enabled": False,
                "reranking_enable": False,
                "reranking_mode": "reranking_model",
                "top_k": 1,
            },
            created_by=actor_id,
        )
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch="batch-1",
            name="document",
            created_from=DocumentCreatedFrom.WEB,
            created_by=actor_id,
            indexing_status="completed",
            enabled=True,
            archived=False,
        )
        segment = DocumentSegment(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            index_node_id=f"node-{uuid4()}",
            content="authorized content",
            position=1,
            word_count=2,
            tokens=2,
            hit_count=7,
            created_by=actor_id,
            status=SegmentStatus.COMPLETED,
            enabled=True,
        )
        segment.id = segment_id
        seed.add_all([dataset, document, segment])

    callback = DatasetIndexToolCallbackHandler(
        queue_manager=cast(AppQueueManager, _UnexpectedQueueBoundary()),
        app_id=str(uuid4()),
        message_id=str(uuid4()),
        user_id=actor_id,
        invoke_from=InvokeFrom.EXPLORE,
    )
    candidate = _candidate(document, segment)
    finalizer = KnowledgeRetrievalFinalizer()

    event.listen(engine, "checkout", _set_mysql_lock_wait_timeout)
    event.listen(engine, "checkin", _restore_mysql_lock_wait_timeout)
    try:
        with sessions() as caller:
            if tool_kind == "single":

                def retrieve_candidate(**_kwargs: object) -> list[RagDocument]:
                    return [candidate]

                monkeypatch.setattr(single_retriever_module.RetrievalService, "retrieve", retrieve_candidate)
                tool = DatasetRetrieverTool(
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    external_retrieval=cast(ExternalDatasetRetriever, _UnexpectedExternalRetrieval()),
                    retrieve_config=DatasetRetrieveConfigEntity(
                        retrieve_strategy=DatasetRetrieveConfigEntity.RetrieveStrategy.SINGLE,
                        metadata_filtering_mode="disabled",
                    ),
                    return_resource=False,
                    retriever_from="prod",
                    hit_callbacks=[callback],
                    inputs={},
                    top_k=1,
                    retrieval_finalizer=finalizer,
                )
            else:

                def append_candidate(
                    self: DatasetMultiRetrieverTool,
                    flask_app: object,
                    dataset_id: str,
                    query: str,
                    all_documents: list[RagDocument],
                    hit_callbacks: list[DatasetIndexToolCallbackHandler],
                ) -> None:
                    del self, flask_app
                    hit_callbacks[0].on_query(query, dataset_id, caller)
                    all_documents.append(candidate)

                rerank_boundary = _RerankModelBoundary()
                monkeypatch.setattr(DatasetMultiRetrieverTool, "_retriever", append_candidate)
                monkeypatch.setattr(
                    multi_retriever_module.ModelManager,
                    "for_tenant",
                    lambda **_kwargs: rerank_boundary,
                )
                monkeypatch.setattr(multi_retriever_module, "RerankModelRunner", _IdentityRerankRunner)
                tool = DatasetMultiRetrieverTool(
                    tenant_id=tenant_id,
                    dataset_ids=[dataset_id],
                    reranking_provider_name="provider",
                    reranking_model_name="model",
                    return_resource=False,
                    retriever_from="prod",
                    hit_callbacks=[callback],
                    top_k=1,
                    retrieval_finalizer=finalizer,
                )

            assert tool.run(session=caller, query="question") == "authorized content"

        with sessions() as verify:
            assert verify.get_one(DocumentSegment, segment_id).hit_count == 8
    finally:
        event.remove(engine, "checkout", _set_mysql_lock_wait_timeout)
        event.remove(engine, "checkin", _restore_mysql_lock_wait_timeout)
        with sessions.begin() as cleanup:
            cleanup.execute(delete(DatasetQuery).where(DatasetQuery.dataset_id == dataset_id))
            cleanup.execute(delete(DocumentSegment).where(DocumentSegment.id == segment_id))
            cleanup.execute(delete(Document).where(Document.id == document_id))
            cleanup.execute(delete(Dataset).where(Dataset.id == dataset_id))

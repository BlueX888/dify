import json
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import Engine, event
from sqlalchemy.orm import Session, sessionmaker

from core.rag.models.document import Document
from extensions.application_services.knowledge_index import build_retrieval_finalizer
from models.dataset import Dataset, ExternalKnowledgeApis, ExternalKnowledgeBindings
from repositories.knowledge.external_repository import SQLAlchemyExternalKnowledgeRepository
from services.knowledge.external.adapters import ExternalDatasetRetrievalAdapter, ExternalKnowledgeGateway


@contextmanager
def _track_checked_out_connections(engine: Engine) -> Generator[set[object]]:
    checked_out: set[object] = set()

    def checkout(_connection: object, record: object, _proxy: object) -> None:
        checked_out.add(record)

    def checkin(_connection: object, record: object) -> None:
        checked_out.discard(record)

    event.listen(engine, "checkout", checkout)
    event.listen(engine, "checkin", checkin)

    try:
        yield checked_out
    finally:
        event.remove(engine, "checkout", checkout)
        event.remove(engine, "checkin", checkin)


@pytest.mark.parametrize(
    ("stored_settings", "override_settings", "expected_settings"),
    [
        (None, None, {"top_k": 2, "score_threshold": 0.0}),
        (None, {}, {"top_k": 2, "score_threshold": 0.0}),
        ({}, None, {"top_k": 2, "score_threshold": 0.0}),
        ({"top_k": 5}, None, {"top_k": 5, "score_threshold": 0.0}),
        ({"top_k": 5}, {}, {"top_k": 5, "score_threshold": 0.0}),
        ({"top_k": 5}, {"top_k": 3}, {"top_k": 3, "score_threshold": 0.0}),
        (
            None,
            {"top_k": 3, "score_threshold_enabled": True, "score_threshold": 0.7},
            {"top_k": 3, "score_threshold": 0.7},
        ),
    ],
)
def test_runtime_retrieval_applies_settings_and_preserves_query_outside_transaction(
    sqlite_session_factory: sessionmaker[Session],
    sqlite_engine: Engine,
    stored_settings: dict[str, object] | None,
    override_settings: dict[str, object] | None,
    expected_settings: dict[str, object],
) -> None:
    with sqlite_session_factory.begin() as session:
        template = ExternalKnowledgeApis(
            tenant_id="tenant",
            created_by="actor",
            updated_by="actor",
            name="API",
            description="",
            settings=json.dumps({"endpoint": "https://example", "api_key": "secret"}),
        )
        template.id = "template"
        session.add_all(
            [
                template,
                Dataset(
                    id="dataset",
                    tenant_id="tenant",
                    created_by="actor",
                    name="Dataset",
                    provider="external",
                    retrieval_model=stored_settings,
                ),
                ExternalKnowledgeBindings(
                    tenant_id="tenant",
                    dataset_id="dataset",
                    external_knowledge_api_id="template",
                    external_knowledge_id="knowledge",
                    created_by="actor",
                ),
            ]
        )

    with _track_checked_out_connections(sqlite_engine) as checked_out:

        def send(_url: str, **_kwargs: object) -> httpx.Response:
            assert not checked_out
            return httpx.Response(200, json={"records": [{"content": "answer", "score": 0.9}]})

        adapter = ExternalDatasetRetrievalAdapter(
            repository=SQLAlchemyExternalKnowledgeRepository(session_factory=sqlite_session_factory),
            gateway=ExternalKnowledgeGateway(),
        )
        with patch("services.knowledge.external.adapters.ssrf_proxy.post", autospec=True, side_effect=send) as request:
            result = adapter.retrieve(
                tenant_id="tenant",
                dataset_id="dataset",
                query=r'查找 "hello" C:\docs\hello',
                retrieval_model=override_settings,
                metadata_filters=None,
            )

    assert result == [{"content": "answer", "score": 0.9, "metadata": {}}]
    request.assert_called_once()
    payload = json.loads(request.call_args.kwargs["data"])
    assert payload["query"] == r'查找 "hello" C:\docs\hello'
    assert payload["retrieval_setting"] == expected_settings


def test_runtime_retrieval_missing_fields_map_and_rank_safely(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        template = ExternalKnowledgeApis(
            tenant_id="tenant",
            created_by="actor",
            updated_by="actor",
            name="API",
            description="",
            settings=json.dumps({"endpoint": "https://example", "api_key": "secret"}),
        )
        template.id = "template"
        session.add_all(
            [
                template,
                Dataset(
                    id="dataset",
                    tenant_id="tenant",
                    created_by="actor",
                    name="Dataset",
                    provider="external",
                ),
                ExternalKnowledgeBindings(
                    tenant_id="tenant",
                    dataset_id="dataset",
                    external_knowledge_api_id="template",
                    external_knowledge_id="knowledge",
                    created_by="actor",
                ),
            ]
        )

    adapter = ExternalDatasetRetrievalAdapter(
        repository=SQLAlchemyExternalKnowledgeRepository(session_factory=sqlite_session_factory),
        gateway=ExternalKnowledgeGateway(),
    )
    with patch(
        "services.knowledge.external.adapters.ssrf_proxy.post",
        autospec=True,
        return_value=httpx.Response(
            200,
            json={"records": [{}, {"content": "ranked", "score": 0.6, "title": "Ranked"}]},
        ),
    ):
        records = adapter.retrieve(
            tenant_id="tenant",
            dataset_id="dataset",
            query="question",
            retrieval_model=None,
            metadata_filters=None,
        )

    documents: list[Document] = []
    for record in records:
        document = Document.model_validate(
            {
                "page_content": record.get("content"),
                "metadata": record.get("metadata"),
                "provider": "external",
            }
        )
        document.metadata["score"] = record.get("score")
        documents.append(document)

    with sqlite_session_factory.begin() as session:
        selected = build_retrieval_finalizer().finalize(
            format_session=session,
            authorization_session=session,
            candidates=documents,
            tenant_id="tenant",
            dataset_ids=["dataset"],
            top_k=2,
        )
    assert [item.page_content for item in selected.items if isinstance(item, Document)] == ["ranked", ""]
    assert documents[0].metadata["score"] == 0.0

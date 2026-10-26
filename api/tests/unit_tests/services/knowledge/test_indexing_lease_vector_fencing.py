from __future__ import annotations

from typing import Self

import pytest
from flask import Flask

from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.models.document import Document as IndexDocument
from models.dataset import Dataset, Document
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.dataset_indexing_outbox_repository import SQLAlchemyDatasetIndexingOutboxRepository
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef
from services.knowledge.entities.segments import ChildVectorRoute
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.adapters.execution import IndexingExecutionAdapter
from services.knowledge.indexing.errors import (
    DocumentIndexingClaimLostError,
    DocumentIsDeletedPausedError,
    DocumentIsPausedError,
)
from services.knowledge.indexing.estimate import StoredSource
from services.knowledge.indexing.execution import IndexingDocument


@pytest.mark.parametrize(
    ("termination", "expected_error"),
    [
        ("claim_lost", DocumentIndexingClaimLostError),
        ("deleted", DocumentIsDeletedPausedError),
        ("paused", DocumentIsPausedError),
    ],
)
def test_ended_indexing_owner_cleans_only_vector_nodes_created_by_that_execution(
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
    expected_error: type[Exception],
) -> None:
    dataset = Dataset(
        id="dataset-1",
        tenant_id="tenant-1",
        name="Dataset",
        description="",
        provider="vendor",
        created_by="account-1",
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        indexing_request_id="generation-1",
    )
    document = Document(
        id="document-1",
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type="notion_import",
        batch="batch-1",
        name="Document",
        created_from="api",
        created_by="account-1",
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    ref = DatasetRef(dataset.tenant_id, dataset.id).document(document.id)
    indexing_document = IndexingDocument(
        source=StoredSource(ref, "notion_import", {}, IndexStructureType.PARAGRAPH_INDEX),
        processing_rule={},
        doc_language="English",
        need_summary=False,
        claim_id="claim-1",
        dataset_generation_id="generation-1",
    )
    chunk = IndexDocument(page_content="content", metadata={"doc_id": "owned-node", "doc_hash": "hash"})

    class Documents:
        def get_indexing_models(self, requested_ref: DocumentRef, *, claim_id: str | None) -> tuple[Dataset, Document]:
            assert requested_ref == ref
            assert claim_id == indexing_document.claim_id
            return dataset, document

    class Segments:
        def complete_indexing_segments(
            self,
            requested_ref: DocumentRef,
            node_ids: list[str],
            *,
            claim_id: str | None,
            dataset_generation_id: str | None,
            child_vector_route: ChildVectorRoute | None = None,
        ) -> None:
            assert requested_ref == ref
            assert node_ids == ["owned-node"]
            assert claim_id == indexing_document.claim_id
            assert dataset_generation_id == indexing_document.dataset_generation_id
            assert child_vector_route is None
            if termination == "claim_lost":
                raise DocumentIndexingClaimLostError()
            if termination == "deleted":
                raise DocumentIsDeletedPausedError()
            raise AssertionError("completion must not run after pause")

    class Uploads:
        def get_files(self, *, workspace_id: str, file_ids: list[str]) -> list[object]:
            assert workspace_id == dataset.tenant_id
            assert file_ids == []
            return []

    class Redis:
        calls = 0

        def get(self, _key: str) -> bytes | None:
            self.calls += 1
            return b"paused" if termination == "paused" and self.calls == 2 else None

    class FencedVector:
        created: list[list[str]] = []
        deleted: list[list[str]] = []

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def resolve_vector_type(_dataset: Dataset, *, session: object) -> str:
            del session
            return "test"

        def create(self, documents: list[IndexDocument], **_kwargs: object) -> None:
            self.created.append([str(item.metadata["doc_id"]) for item in documents])

        def create_multimodal(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("no attachments expected")

        @staticmethod
        def get_routing_target() -> tuple[str, str]:
            return "test", "collection-1"

        def delete_by_ids(self, node_ids: list[str]) -> None:
            self.deleted.append(node_ids)

    monkeypatch.setattr("services.knowledge.indexing.adapters.execution.Vector", FencedVector)
    monkeypatch.setattr("services.knowledge.indexing.adapters.execution.redis_client", Redis())
    monkeypatch.setattr(
        "services.knowledge.indexing.adapters.execution.refresh_locked_indexing_claim",
        lambda *_args, **_kwargs: None,
    )
    recorded_routes: list[VectorRoute] = []

    def record_route(
        session: object,
        *,
        dataset: Dataset,
        generation_id: str | None,
        route: VectorRoute,
    ) -> bool:
        assert isinstance(session, _UnusedSession)
        assert dataset.id == "dataset-1"
        assert generation_id == "generation-1"
        recorded_routes.append(route)
        return True

    monkeypatch.setattr(
        SQLAlchemyDatasetIndexingOutboxRepository,
        "record_generation_writer_route_locked",
        record_route,
    )
    adapter = IndexingExecutionAdapter(
        session_factory=_UnusedSessionFactory(),  # type: ignore[arg-type]
        documents=Documents(),  # type: ignore[arg-type]
        segments=Segments(),  # type: ignore[arg-type]
        uploads=Uploads(),  # type: ignore[arg-type]
        attachment_vectors=SQLAlchemyAttachmentVectorStore(
            session_factory=_UnusedSessionFactory(),  # type: ignore[arg-type]
        ),
        sources=None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        adapter,
        "_require_generation",
        lambda _session, _indexing_document, **_kwargs: (dataset, document),
    )

    with pytest.raises(expected_error):
        adapter._process_chunk(Flask(__name__), indexing_document, [chunk], False)

    assert FencedVector.created == [["owned-node"]]
    assert FencedVector.deleted == [["owned-node"]]
    assert [(route.vector_type, route.collection_name) for route in recorded_routes] == [("test", "collection-1")]


class _UnusedSession:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _UnusedSessionFactory:
    @staticmethod
    def begin() -> _UnusedSession:
        return _UnusedSession()

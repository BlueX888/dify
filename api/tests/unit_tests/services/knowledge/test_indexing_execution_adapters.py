import base64
import json
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call, create_autospec, patch

import pytest
from docx import Document as DocxDocument
from flask import Flask
from openpyxl import Workbook
from openpyxl.drawing.image import Image as SpreadsheetImage
from sqlalchemy import Connection, Engine, event, inspect, select
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.exceptions import NotFound

from core.errors.error import LLMBadRequestError, ProviderTokenNotInitError, QuotaExceededError
from core.plugin.impl.exc import PluginDaemonClientSideError
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.extractor.entity.extract_setting import CreatedUploadArtifact, ExtractionResult, WebsiteJobRef
from core.rag.extractor.excel_extractor import ExcelExtractor
from core.rag.extractor.word_extractor import WordExtractor
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.models.document import AttachmentDocument, ChildDocument
from core.rag.models.document import Document as IndexDocument
from extensions.storage.storage_type import StorageType
from models import Account
from models.dataset import (
    Dataset,
    DatasetKeywordTable,
    DatasetProcessRule,
    Document,
    DocumentSegment,
    SegmentAttachmentBinding,
)
from models.enums import CreatorUserRole, DataSourceType, IndexingStatus, ProcessRuleMode, SegmentStatus
from models.model import AttachmentVectorCleanupOutbox, DatasetIndexingOutbox, UploadFile, UploadFileCleanupOutbox
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from repositories.knowledge.segment_repository import SQLAlchemySegmentRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.data_source.website_crawl_cache import website_crawl_provider_instance_id
from services.knowledge.entities.datasets import DatasetIndexingRepresentation
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.adapters.execution import IndexingExecutionAdapter
from services.knowledge.indexing.adapters.sources import CompositeStoredSourceResolver, WebsiteSourceAdapter
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError, DocumentIsPausedError
from services.knowledge.indexing.execution import DocumentIndexingService, IndexingDocument
from tests.unit_tests.config_override import apply_config_overrides
from tests.unit_tests.repositories.knowledge.test_document_repository import _dataset, _document

MODULE = "services.knowledge.indexing.adapters.execution"


BackendFixture = tuple[
    IndexingExecutionAdapter,
    SQLAlchemyDocumentRepository,
    SQLAlchemySegmentRepository,
    MagicMock,
    IndexingDocument,
    MagicMock,
]


@pytest.fixture
def backend(sqlite_session_factory: sessionmaker[Session]) -> Iterator[BackendFixture]:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [_dataset("dataset-1", "workspace-1"), _document("document-1", status=IndexingStatus.SPLITTING)]
        )
    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    segments = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    sources = MagicMock()
    adapter = IndexingExecutionAdapter(
        session_factory=sqlite_session_factory,
        documents=documents,
        segments=segments,
        uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory),
        attachment_vectors=SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory),
        sources=sources,
    )
    job = replace(
        documents.get_indexing_document(DatasetRef("workspace-1", "dataset-1").document("document-1")),
        processing_rule={"mode": "automatic", "rules": {}},
    )
    with patch(f"{MODULE}.redis_client") as redis:
        redis.get.return_value = None
        yield adapter, documents, segments, sources, job, redis


@pytest.fixture
def active_transactions(sqlite_engine: Engine) -> Iterator[set[Connection]]:
    active: set[Connection] = set()
    begin, finish = active.add, active.discard
    event.listen(sqlite_engine, "begin", begin)
    event.listen(sqlite_engine, "commit", finish)
    event.listen(sqlite_engine, "rollback", finish)
    try:
        yield active
    finally:
        event.remove(sqlite_engine, "begin", begin)
        event.remove(sqlite_engine, "commit", finish)
        event.remove(sqlite_engine, "rollback", finish)


@pytest.mark.parametrize("doc_form", list(IndexStructureType))
def test_vector_io_releases_transactions_and_preserves_chunk_structure(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
    active_transactions: set[Connection],
    monkeypatch: pytest.MonkeyPatch,
    doc_form: IndexStructureType,
) -> None:
    adapter, _, segments, _, job, _ = backend
    apply_config_overrides(monkeypatch, VECTOR_STORE="qdrant", VECTOR_STORE_WHITELIST_ENABLE=True)
    files = [
        UploadFile(
            tenant_id=tenant_id,
            storage_type=StorageType.LOCAL,
            key=f"image-{tenant_id}",
            name="image.png",
            size=1,
            extension="png",
            mime_type="image/png",
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by="account-1",
            created_at=datetime(2026, 1, 1),
            used=True,
        )
        for tenant_id in ("workspace-1", "workspace-2")
    ]
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        document = session.get(Document, "document-1")
        assert dataset is not None
        assert document is not None
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.is_multimodal = True
        document.doc_form = doc_form
        session.add_all(files)
    child = ChildDocument(page_content="child", metadata={"doc_id": "child-node", "doc_hash": "child-hash"})
    chunk = IndexDocument(
        page_content="parent",
        metadata={"doc_id": "node", "doc_hash": "hash"},
        children=[child] if doc_form == IndexStructureType.PARENT_CHILD_INDEX else None,
        attachments=[
            AttachmentDocument(page_content="image", metadata={"doc_id": file.id, "doc_type": "image"})
            for file in files[:1]
        ],
    )
    segments.save_for_indexing(job, [chunk], [5])
    vector_backend = MagicMock()
    vector_factory = MagicMock()
    vector_factory.return_value.init_vector.return_value = vector_backend

    def text_external_call(*_args, **_kwargs) -> list[list[float]]:
        assert active_transactions, "Index I/O must hold the Dataset generation fence"
        return [[1.0]]

    def attachment_external_call(*_args, **_kwargs) -> list[list[float]]:
        assert active_transactions, "Attachment I/O must hold the Dataset ownership lock"
        return [[1.0]]

    def get_vector_factory(vector_type: str) -> MagicMock:
        assert active_transactions, "Vector initialization must remain inside the generation fence"
        assert vector_type == "qdrant"
        return vector_factory

    def load_image(key: str) -> bytes:
        assert active_transactions, "Attachment storage reads share the vector publication fence"
        assert key == files[0].key
        return b"image"

    vector_backend.create.side_effect = [None, None]
    with (
        patch.object(Vector, "get_vector_factory", side_effect=get_vector_factory),
        patch(
            "core.rag.datasource.vdb.vector_factory._LazyEmbeddings.embed_documents", side_effect=text_external_call
        ) as embed,
        patch(
            "core.rag.datasource.vdb.vector_factory._LazyEmbeddings.embed_multimodal_documents",
            side_effect=attachment_external_call,
        ) as embed_images,
        patch("core.rag.datasource.vdb.vector_factory.storage") as storage,
    ):
        storage.load_once.side_effect = load_image
        adapter._process_chunk(Flask(__name__), job, [chunk], False)
    embed.assert_called_once_with(["child" if doc_form == IndexStructureType.PARENT_CHILD_INDEX else "parent"])
    assert embed_images.call_args.args[0] == [{"content": "aW1hZ2U=", "content_type": "image", "file_id": files[0].id}]
    storage.load_once.assert_called_once_with(files[0].key)
    assert vector_backend.create.call_count == 2
    indexed_text = vector_backend.create.call_args_list[0].kwargs["texts"][0]
    assert indexed_text.metadata["doc_id"] == (
        "child-node" if doc_form == IndexStructureType.PARENT_CHILD_INDEX else "node"
    )
    assert segments.resume_indexing(job) == ([], 5)


def test_attachment_publication_refreshes_claim_after_vector_io_outlives_lease(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    adapter, documents, segments, _, _, _ = backend
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    request_id = "request-1"
    claim_id = "claim-1"
    operation_started_at = datetime(2099, 1, 1)
    generation_id = "dataset-generation-1"
    upload = UploadFile(
        tenant_id=ref.dataset.tenant_id,
        storage_type=StorageType.LOCAL,
        key="image-1",
        name="image.png",
        size=1,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="account-1",
        created_at=datetime(2026, 1, 1),
        used=True,
    )
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, ref.dataset.dataset_id)
        document = session.get_one(Document, ref.document_id)
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.is_multimodal = True
        route = VectorRoute.capture(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            vector_type="qdrant",
            collection_name="collection-b",
            collection_binding_id=None,
            indexing_technique="high_quality",
            doc_form=str(document.doc_form),
        )
        representation = DatasetIndexingRepresentation(
            indexing_technique="high_quality",
            index_struct=route.index_struct,
            collection_binding_id=None,
            embedding_model=dataset.embedding_model,
            embedding_model_provider=dataset.embedding_model_provider,
            keyword_number=dataset.keyword_number,
            is_multimodal=True,
            doc_form=str(document.doc_form),
            summary_index_setting=dataset.summary_index_setting,
        )
        dataset.indexing_request_id = generation_id
        document.indexing_status = IndexingStatus.INDEXING
        document.indexing_request_id = request_id
        document.indexing_claim_id = claim_id
        document.indexing_claim_expires_at = operation_started_at + timedelta(minutes=5)
        document.indexing_dataset_generation_id = generation_id
        session.add_all(
            [
                upload,
                DatasetIndexingOutbox(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    generation_id=generation_id,
                    operation="rebuild",
                    action="update",
                    source_representation=representation.model_dump_json(),
                    target_representation=representation.model_dump_json(),
                ),
            ]
        )

    job = replace(
        documents.get_indexing_document(ref, claim_id=claim_id),
        processing_rule={"mode": "automatic", "rules": {}},
    )
    assert job.indexing_request_id == request_id
    chunk = IndexDocument(
        page_content="content",
        metadata={"doc_id": "node-1", "doc_hash": "hash-1"},
        attachments=[
            AttachmentDocument(
                page_content="image",
                metadata={"doc_id": upload.id, "doc_type": "image"},
            )
        ],
    )
    segments.save_for_indexing(job, [chunk], [1])

    database_now = operation_started_at

    def publish_slow_multimodal(*_args: object, **_kwargs: object) -> None:
        nonlocal database_now
        database_now += timedelta(minutes=5, seconds=1)

    with (
        patch("repositories.knowledge.document_repository._database_now", side_effect=lambda _session: database_now),
        patch(f"{MODULE}.Vector") as vector,
    ):
        vector.resolve_vector_type.return_value = "qdrant"
        vector.return_value.get_routing_target.return_value = ("qdrant", "collection-b")
        vector.return_value.create_multimodal.side_effect = publish_slow_multimodal

        adapter._process_chunk(Flask(__name__), job, [chunk], False)

        with sqlite_session_factory() as session:
            persisted = session.get_one(Document, ref.document_id)
            segment = session.scalar(select(DocumentSegment).where(DocumentSegment.document_id == ref.document_id))
            assert persisted.indexing_claim_expires_at == database_now + timedelta(minutes=5)
            assert segment is not None
            assert segment.status == SegmentStatus.COMPLETED
            receipt = session.scalar(select(DatasetIndexingOutbox))
            assert receipt is not None
            target = DatasetIndexingRepresentation.model_validate_json(receipt.target_representation)
            assert target.vector_route is None
            assert target.cleanup_routes == (route,)

        documents.complete_indexing(
            ref,
            claim_id=claim_id,
            tokens=1,
            latency=301.0,
            dataset_generation_id=generation_id,
        )

    vector.return_value.create_multimodal.assert_called_once()
    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, ref.document_id)
        assert persisted.indexing_status == IndexingStatus.COMPLETED
        assert persisted.indexing_request_id is None
        assert persisted.indexing_claim_id is None


@pytest.mark.parametrize("storage_type", ["database", "file"])
def test_keyword_io_releases_transactions_and_commits_before_unlocking(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
    active_transactions: set[Connection],
    storage_type: str,
) -> None:
    adapter, _, segments, _, job, redis = backend
    original = json.dumps({"__type__": "keyword_table", "__data__": {"table": {"existing": ["old-node"]}}})
    with sqlite_session_factory.begin() as session:
        session.add(DatasetKeywordTable(dataset_id="dataset-1", data_source_type=storage_type, keyword_table=original))
    chunk = IndexDocument(page_content="hello", metadata={"doc_id": "node", "doc_hash": "hash"})
    segments.save_for_indexing(job, [chunk], [5])
    locked = False

    @contextmanager
    def dataset_lock() -> Generator[None]:
        nonlocal locked
        locked = True
        yield
        assert active_transactions
        # A separate session must see the write before the next lock holder can read.
        with sqlite_session_factory() as session:
            segment = session.scalar(select(DocumentSegment))
            assert segment is not None
            assert segment.keywords == ["hello"]
        locked = False

    redis.lock.side_effect = lambda *_args, **_kwargs: dataset_lock()

    def external_call(*_args, **_kwargs) -> None:
        assert locked
        assert active_transactions, "Keyword I/O must hold the Dataset generation fence"

    def load_keywords(*_args) -> bytes:
        external_call()
        return original.encode()

    def file_exists(*_args) -> bool:
        external_call()
        return True

    def extract_keywords(*_args) -> set[str]:
        external_call()
        return {"hello"}

    with (
        patch("core.rag.datasource.keyword.jieba.jieba.redis_client", redis),
        patch("core.rag.datasource.keyword.jieba.jieba.storage") as storage,
        patch("core.rag.datasource.keyword.jieba.jieba.JiebaKeywordTableHandler") as handler,
    ):
        storage.load_once.side_effect = load_keywords
        storage.exists.side_effect = file_exists
        storage.delete.side_effect = external_call
        storage.save.side_effect = external_call
        handler.return_value.extract_keywords.side_effect = extract_keywords
        adapter._process_chunk(Flask(__name__), job, [chunk], True)
    assert not locked
    assert redis.lock.call_args.args == ("keyword_indexing_lock_dataset-1",)
    _, encoded = segments.get_keyword_table(job.ref.dataset)
    if storage_type == "file":
        storage.save.assert_called_once()
        encoded = storage.save.call_args.args[1]
    else:
        storage.save.assert_not_called()
    assert encoded is not None
    assert json.loads(encoded)["__data__"]["table"] == {"existing": ["old-node"], "hello": ["node"]}
    assert segments.resume_indexing(job) == ([], 5)


def test_lost_indexing_lease_cannot_overwrite_file_keyword_table(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    adapter, _, segments, _, job, _ = backend
    original = json.dumps({"__type__": "keyword_table", "__data__": {"table": {"existing": ["old-node"]}}})
    with sqlite_session_factory.begin() as session:
        session.add(DatasetKeywordTable(dataset_id="dataset-1", data_source_type="file", keyword_table=""))
    chunk = IndexDocument(page_content="hello", metadata={"doc_id": "node", "doc_hash": "hash"})
    segments.save_for_indexing(job, [chunk], [5])
    with sqlite_session_factory.begin() as session:
        document = session.get(Document, "document-1")
        assert document is not None
        document.indexing_claim_id = "old-owner"
        document.indexing_claim_expires_at = datetime(2099, 1, 1)
    claimed_job = replace(job, claim_id="old-owner")

    def replace_owner(*_args) -> set[str]:
        with sqlite_session_factory.begin() as session:
            document = session.get(Document, "document-1")
            assert document is not None
            document.indexing_claim_id = "new-owner"
            document.indexing_claim_expires_at = datetime(2099, 1, 1)
        return {"hello"}

    with (
        patch("core.rag.datasource.keyword.jieba.jieba.storage") as storage,
        patch("core.rag.datasource.keyword.jieba.jieba.JiebaKeywordTableHandler") as handler,
    ):
        storage.load_once.return_value = original.encode()
        handler.return_value.extract_keywords.side_effect = replace_owner
        with pytest.raises(DocumentIndexingClaimLostError):
            adapter._process_chunk(Flask(__name__), claimed_job, [chunk], True)
    storage.exists.assert_not_called()
    storage.delete.assert_not_called()
    storage.save.assert_not_called()


def test_extract_attaches_document_metadata_and_owns_local_session(backend: BackendFixture) -> None:
    adapter, _, _, sources, job, _ = backend
    text = IndexDocument(page_content="source")
    with patch(f"{MODULE}.IndexProcessorFactory") as factory:
        processor = factory.return_value.init_index_processor.return_value

        def extract(
            setting: object,
            *,
            session: Session,
            process_rule_mode: str,
            on_upload_artifact_created: Callable[[CreatedUploadArtifact], None],
        ) -> ExtractionResult:
            assert setting is sources.resolve_for_indexing.return_value
            assert not session.in_transaction()
            assert process_rule_mode == "automatic"
            artifact = CreatedUploadArtifact("created-upload-1", "created-key")
            on_upload_artifact_created(artifact)
            return ExtractionResult(documents=[text], created_upload_artifacts=(artifact,))

        processor.extract.side_effect = extract
        extraction = adapter.extract(job)
        assert extraction.documents == [text]
        assert extraction.created_upload_file_ids == ("created-upload-1",)
    sources.resolve_for_indexing.assert_called_once_with(job.source, attach_document_for_updates=True)
    assert text.metadata == {"dataset_id": "dataset-1", "document_id": "document-1"}

    sources.resolve_for_indexing.reset_mock()
    with patch(f"{MODULE}.IndexProcessorFactory") as factory:
        factory.return_value.init_index_processor.return_value.extract.return_value = ExtractionResult(
            documents=[text],
            source_version="2026-10-24T00:00:00Z",
        )
        claimed_extraction = adapter.extract(replace(job, claim_id="claim-1"))
    sources.resolve_for_indexing.assert_called_once_with(job.source, attach_document_for_updates=False)
    assert claimed_extraction.source_version == "2026-10-24T00:00:00Z"


def test_extract_commit_failure_persists_artifact_key_before_blob_retry(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _, _, job, _ = backend
    created_id: list[str] = []

    def extract(
        _setting: object,
        *,
        session: Session,
        process_rule_mode: str,
        on_upload_artifact_created: Callable[[CreatedUploadArtifact], None],
    ) -> ExtractionResult:
        assert process_rule_mode == "automatic"
        upload = UploadFile(
            tenant_id="workspace-1",
            storage_type=StorageType.LOCAL,
            key="workspace-1/indexing-commit-unknown.png",
            name="temp.png",
            size=1,
            extension="png",
            mime_type="image/png",
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by="account-1",
            created_at=datetime(2026, 1, 1),
            used=True,
        )
        session.add(upload)
        artifact = CreatedUploadArtifact(upload.id, upload.key)
        on_upload_artifact_created(artifact)
        created_id.append(upload.id)
        return ExtractionResult(documents=[IndexDocument(page_content="source")], created_upload_artifacts=(artifact,))

    commit_attempts = 0

    def fail_first_commit(_session: Session) -> None:
        nonlocal commit_attempts
        commit_attempts += 1
        # The writer marker commits first; fail the UploadFile publication.
        if commit_attempts == 2:
            raise RuntimeError("commit result unknown")

    event.listen(sqlite_session_factory.class_, "before_commit", fail_first_commit)
    monkeypatch.setattr(
        f"{MODULE}.storage.delete",
        lambda _key: (_ for _ in ()).throw(OSError("storage unavailable")),
    )
    try:
        with patch(f"{MODULE}.IndexProcessorFactory") as factory:
            factory.return_value.init_index_processor.return_value.extract.side_effect = extract
            with pytest.raises(RuntimeError, match="commit result unknown"):
                adapter.extract(job)
    finally:
        event.remove(sqlite_session_factory.class_, "before_commit", fail_first_commit)

    with sqlite_session_factory() as session:
        assert session.get(UploadFile, created_id[0]) is None
        pending = session.scalar(
            select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == created_id[0])
        )
        assert pending is not None
        assert pending.key == "workspace-1/indexing-commit-unknown.png"


def test_failed_indexing_cleanup_deletes_only_unreferenced_created_files(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
    active_transactions: set[Connection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, segments, _, job, _ = backend
    uploads = {
        name: UploadFile(
            tenant_id="workspace-1",
            storage_type=StorageType.LOCAL,
            key=f"image-{name}",
            name=f"{name}.png",
            size=1,
            extension="png",
            mime_type="image/png",
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by="account-1",
            created_at=datetime(2026, 1, 1),
            used=True,
        )
        for name in ("orphan", "bound", "body", "child", "existing")
    }
    with sqlite_session_factory.begin() as session:
        session.add_all(uploads.values())
        document = session.get(Document, job.ref.document_id)
        assert document is not None
        document.doc_form = IndexStructureType.PARENT_CHILD_INDEX
    hierarchical_job = replace(job, source=job.source._replace(document_model=IndexStructureType.PARENT_CHILD_INDEX))
    chunk = IndexDocument(
        page_content=f"body ![image](https://files.local/files/{uploads['body'].id}/file-preview)",
        metadata={"doc_id": "node", "doc_hash": "hash"},
        attachments=[
            AttachmentDocument(
                page_content="bound",
                metadata={"doc_id": uploads["bound"].id, "doc_type": "image"},
            )
        ],
        children=[
            ChildDocument(
                page_content=f"child ![image](https://files.local/files/{uploads['child'].id}/file-preview)",
                metadata={"doc_id": "child-node", "doc_hash": "child-hash"},
            )
        ],
    )
    segments.save_for_indexing(hierarchical_job, [chunk], [5])
    deleted_keys: list[str] = []

    def delete_storage(key: str) -> None:
        assert not active_transactions
        deleted_keys.append(key)

    monkeypatch.setattr(f"{MODULE}.storage.delete", delete_storage)

    adapter.cleanup_created_upload_files(
        hierarchical_job,
        [uploads[name].id for name in ("orphan", "bound", "body", "child")],
    )

    with sqlite_session_factory() as session:
        remaining_ids = set(session.scalars(select(UploadFile.id)))
    assert uploads["orphan"].id not in remaining_ids
    assert remaining_ids == {uploads[name].id for name in ("bound", "body", "child", "existing")}
    assert deleted_keys == [uploads["orphan"].key]

    dangling_chunk = IndexDocument(
        page_content="retry",
        metadata={"doc_id": "retry-node", "doc_hash": "retry-hash"},
        attachments=[
            AttachmentDocument(
                page_content="deleted",
                metadata={"doc_id": uploads["orphan"].id, "doc_type": "image"},
            )
        ],
    )
    with pytest.raises(LookupError, match="Attachment files no longer exist"):
        segments.save_for_indexing(hierarchical_job, [dangling_chunk], [1])


@pytest.mark.parametrize("file_type", ["xlsx", "docx"])
def test_real_office_extraction_files_are_claimed_by_failed_indexing_cleanup(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    file_type: str,
) -> None:
    adapter, _, _, _, job, _ = backend
    apply_config_overrides(monkeypatch, FILES_URL="https://files.local", STORAGE_TYPE="local")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    )
    file_path = tmp_path / f"source.{file_type}"
    if file_type == "xlsx":
        workbook = Workbook()
        sheet = workbook.active
        assert sheet is not None
        sheet.append(["content", "image"])
        sheet.append(["row", ""])
        sheet.add_image(SpreadsheetImage(image_path), "B2")
        workbook.save(file_path)
    else:
        document = DocxDocument()
        document.add_paragraph("content")
        document.add_picture(str(image_path))
        document.save(str(file_path))

    saved_keys: list[str] = []
    deleted_keys: list[str] = []
    monkeypatch.setattr("core.rag.extractor.excel_extractor.storage.save", lambda key, _data: saved_keys.append(key))
    monkeypatch.setattr("core.rag.extractor.word_extractor.storage.save", lambda key, _data: saved_keys.append(key))
    monkeypatch.setattr(f"{MODULE}.storage.delete", deleted_keys.append)

    if file_type == "xlsx":
        extractor = ExcelExtractor(
            str(file_path),
            tenant_id="workspace-1",
            user_id="account-1",
            source_file_id="source-file-1",
        )
        extractor.extract()
    else:
        with sqlite_session_factory() as session:
            extractor = WordExtractor(
                str(file_path),
                tenant_id="workspace-1",
                user_id="account-1",
                session=session,
            )
            extractor.extract()
            session.commit()

    created_ids = extractor.created_upload_file_ids
    assert len(created_ids) == 1
    adapter.cleanup_created_upload_files(job, created_ids)

    with sqlite_session_factory() as session:
        assert session.scalars(select(UploadFile).where(UploadFile.id.in_(created_ids))).all() == []
    assert deleted_keys == saved_keys


def test_lost_claim_discards_persisted_attempt_segments_before_claiming_created_files(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, segments, _, job, _ = backend
    created_upload = UploadFile(
        tenant_id="workspace-1",
        storage_type=StorageType.LOCAL,
        key="lost-attempt-image",
        name="image.png",
        size=1,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="account-1",
        created_at=datetime(2026, 1, 1),
        used=True,
    )
    with sqlite_session_factory.begin() as session:
        session.add(created_upload)
        dataset = session.get(Dataset, job.ref.dataset.dataset_id)
        assert dataset is not None
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
    chunk = IndexDocument(
        page_content=f"![image](https://files.local/files/{created_upload.id}/file-preview)",
        metadata={"doc_id": "lost-node", "doc_hash": "hash"},
        attachments=[
            AttachmentDocument(
                page_content="image",
                metadata={"doc_id": created_upload.id, "doc_type": "image"},
            )
        ],
    )
    retained_chunk = IndexDocument(
        page_content="new attempt",
        metadata={"doc_id": "new-node", "doc_hash": "new-hash"},
    )
    segments.save_for_indexing(job, [chunk, retained_chunk], [1, 2])
    deleted_keys: list[str] = []
    monkeypatch.setattr(f"{MODULE}.storage.delete", deleted_keys.append)

    with patch(f"{MODULE}.Vector") as vector:
        vector.resolve_vector_type.return_value = "qdrant"
        vector.return_value.get_routing_target.return_value = ("qdrant", "dataset-1-attachments")
        adapter.cleanup_created_upload_files(
            job,
            (created_upload.id,),
            discard_persisted_attempt=True,
        )

    assert vector.return_value.delete_by_ids.call_args_list == [call(["lost-node"])]

    with sqlite_session_factory() as session:
        assert session.get(UploadFile, created_upload.id) is None
        attachment_cleanup = session.scalar(
            select(AttachmentVectorCleanupOutbox).where(
                AttachmentVectorCleanupOutbox.attachment_id == created_upload.id
            )
        )
        assert attachment_cleanup is not None
        assert attachment_cleanup.writer_finished_at is not None
        assert (
            VectorRoute.model_validate_json(attachment_cleanup.vector_route).collection_name == "dataset-1-attachments"
        )
        retained_segments = session.scalars(
            select(DocumentSegment).where(DocumentSegment.document_id == job.ref.document_id)
        ).all()
        assert [segment.index_node_id for segment in retained_segments] == ["new-node"]
        assert (
            session.scalars(
                select(SegmentAttachmentBinding).where(SegmentAttachmentBinding.document_id == job.ref.document_id)
            ).all()
            == []
        )
    assert deleted_keys == [created_upload.key]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ProviderTokenNotInitError(), "Provider Token Not Init"),
        (ProviderTokenNotInitError("configure an embedding provider"), "configure an embedding provider"),
        (LLMBadRequestError("invalid model input"), "invalid model input"),
        (QuotaExceededError(), "Quota Exceeded"),
        (PluginDaemonClientSideError("plugin unavailable"), "plugin unavailable"),
        (NotFound("source file missing"), "source file missing"),
        (RuntimeError("index unavailable"), "index unavailable"),
    ],
)
def test_indexing_preserves_provider_and_source_error_descriptions(
    backend: BackendFixture, sqlite_session_factory: sessionmaker[Session], error: Exception, message: str
) -> None:
    adapter, documents, segments, sources, job, _ = backend
    sources.resolve_for_indexing.side_effect = error
    with sqlite_session_factory.begin() as session:
        rule = DatasetProcessRule(
            dataset_id=job.ref.dataset.dataset_id, mode=ProcessRuleMode.AUTOMATIC, rules="{}", created_by="account-1"
        )
        session.add(rule)
        session.flush()
        document = session.get(Document, job.ref.document_id)
        assert document is not None
        document.dataset_process_rule_id = rule.id

    DocumentIndexingService(documents=documents, segments=segments, backend=adapter).run([job.ref])

    sources.resolve_for_indexing.assert_called_once()
    with sqlite_session_factory() as session:
        document = session.get(Document, job.ref.document_id)
        assert document is not None
        assert document.indexing_status == IndexingStatus.ERROR
        assert document.error == message
        assert document.completed_at is None


@pytest.mark.parametrize("provider", [None, "embedding-provider"])
def test_transform_selects_embedding_model_and_preserves_processor_options(
    backend: BackendFixture, sqlite_session_factory: sessionmaker[Session], provider: str | None
) -> None:
    adapter, documents, _, _, job, _ = backend
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.embedding_model_provider = provider
        dataset.embedding_model = "embedding-model"
    with (
        patch.object(documents, "get_indexing_user", return_value=MagicMock()) as user,
        patch(f"{MODULE}.ModelManager.for_tenant") as manager,
        patch(f"{MODULE}.IndexProcessorFactory") as factory,
    ):
        processor = factory.return_value.init_index_processor.return_value
        processor.transform.side_effect = lambda *_args, **kwargs: (
            kwargs["on_upload_file_created"]("transform-upload") or list[IndexDocument]()
        )
        transformation = adapter.transform(job, [])
        selected = (
            manager.return_value.get_model_instance if provider else manager.return_value.get_default_model_instance
        )
        selected.assert_called_once()
        assert processor.transform.call_args.kwargs["embedding_model_instance"] is selected.return_value
        assert processor.transform.call_args.kwargs["process_rule"] == job.processing_rule
        assert processor.transform.call_args.args == ([], user.return_value)
        assert transformation.chunks == []
        assert transformation.created_upload_file_ids == ("transform-upload",)


def test_transform_failure_cleans_uploads_reported_by_the_processor(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, documents, _, _, job, _ = backend
    created_upload = UploadFile(
        tenant_id="workspace-1",
        storage_type=StorageType.LOCAL,
        key="transform-created-image",
        name="image.png",
        size=1,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="account-1",
        created_at=datetime(2026, 1, 1),
        used=True,
    )
    deleted_keys: list[str] = []
    monkeypatch.setattr(f"{MODULE}.storage.delete", deleted_keys.append)

    with (
        patch.object(
            documents,
            "get_indexing_user",
            return_value=create_autospec(Account, instance=True, spec_set=True),
        ),
        patch(f"{MODULE}.IndexProcessorFactory") as factory,
    ):
        processor = factory.return_value.init_index_processor.return_value

        def transform(
            *_args: object,
            on_upload_file_created: Callable[[str], None],
            **_kwargs: object,
        ) -> list[IndexDocument]:
            with sqlite_session_factory.begin() as session:
                session.add(created_upload)
            on_upload_file_created(created_upload.id)
            raise RuntimeError("transform failed")

        processor.transform.side_effect = transform
        with pytest.raises(RuntimeError, match="transform failed"):
            adapter.transform(job, [])

    with sqlite_session_factory() as session:
        assert session.get(UploadFile, created_upload.id) is None
    assert deleted_keys == [created_upload.key]


def test_hash_groups_load_worker_owned_inputs_before_completion(
    backend: BackendFixture, sqlite_session_factory: sessionmaker[Session]
) -> None:
    adapter, _, segments, _, job, _ = backend
    with sqlite_session_factory.begin() as session:
        persisted_dataset = session.get(Dataset, "dataset-1")
        assert persisted_dataset is not None
        persisted_dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
    chunks = [
        IndexDocument(page_content=content, metadata={"doc_id": f"node-{i}", "doc_hash": f"hash-{i}"})
        for i, content in enumerate(["same", "other", "same"])
    ]
    segments.save_for_indexing(job, chunks, [3, 5, 7])
    calls: list[tuple[Dataset, list[str]]] = []
    main_thread = threading.get_ident()

    def vector(dataset: Dataset, *, session: Session | None, vector_type: str) -> MagicMock:
        assert threading.get_ident() != main_thread
        assert inspect(dataset).persistent
        assert session is None
        assert vector_type == "qdrant"

        def create(group: list[IndexDocument]) -> None:
            with sqlite_session_factory() as reader:
                persisted = reader.scalars(select(DocumentSegment)).all()
                assert len(persisted) == 3
                assert all(
                    row.status == SegmentStatus.INDEXING
                    for row in persisted
                    if row.index_node_id in {chunk.metadata["doc_id"] for chunk in group}
                )
            calls.append((dataset, [chunk.page_content for chunk in group]))

        backend = MagicMock()
        backend.create.side_effect = create
        return backend

    with Flask(__name__).app_context(), patch(f"{MODULE}.Vector", side_effect=vector) as factory:
        factory.resolve_vector_type.return_value = "qdrant"
        adapter.load(job, chunks)
    assert sum(group.count("same") for _, group in calls) == 2
    assert sum("same" in group for _, group in calls) == 1
    assert len({id(dataset) for dataset, _ in calls}) == len(calls)
    assert segments.resume_indexing(job) == ([], 15)


@pytest.mark.parametrize("technique", ["economy", "high_quality"])
def test_worker_failure_marks_document_error_and_keeps_retryable_segments(
    backend: BackendFixture, sqlite_session_factory: sessionmaker[Session], technique: str
) -> None:
    adapter, documents, segments, _, job, _ = backend
    with sqlite_session_factory.begin() as session:
        persisted_dataset = session.get(Dataset, "dataset-1")
        assert persisted_dataset is not None
        persisted_dataset.indexing_technique = IndexTechniqueType(technique)
    chunks = [IndexDocument(page_content="text", metadata={"doc_id": "node", "doc_hash": "hash"})]
    segments.save_for_indexing(job, chunks, [9])
    with (
        Flask(__name__).app_context(),
        patch(f"{MODULE}.Jieba") as keyword,
        patch(f"{MODULE}.Vector") as vector,
    ):
        keyword.return_value.update_texts.side_effect = RuntimeError("worker failed")
        vector.return_value.create.side_effect = RuntimeError("worker failed")
        DocumentIndexingService(documents=documents, segments=segments, backend=adapter).run_in_indexing_status(job.ref)
    with sqlite_session_factory() as session:
        row = session.get(Document, job.ref.document_id)
        assert row is not None
        assert row.indexing_status == IndexingStatus.ERROR
        assert row.error == "worker failed"
    assert len(segments.resume_indexing(job)[0]) == 1


def test_worker_pause_does_not_mark_segments_complete(backend: BackendFixture) -> None:
    adapter, _, segments, _, job, redis = backend
    chunks = [IndexDocument(page_content="text", metadata={"doc_id": "node", "doc_hash": "hash"})]
    segments.save_for_indexing(job, chunks, [9])
    redis.get.return_value = b"paused"
    with Flask(__name__).app_context(), pytest.raises(DocumentIsPausedError):
        adapter.load(job, chunks)
    assert len(segments.resume_indexing(job)[0]) == 1


@pytest.mark.parametrize("technique", ["economy", "high_quality"])
@pytest.mark.parametrize("doc_form", ["text_model", "qa_model", "hierarchical_model"])
def test_full_indexing_commits_phases_through_real_repositories(
    backend: BackendFixture, sqlite_session_factory: sessionmaker[Session], technique: str, doc_form: str
) -> None:
    adapter, documents, segments, _, job, _ = backend
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.indexing_technique = IndexTechniqueType(technique)
        row = session.get(Document, "document-1")
        assert row is not None
        row.doc_form = IndexStructureType(doc_form)
        rule = DatasetProcessRule(
            dataset_id="dataset-1", mode=ProcessRuleMode.AUTOMATIC, rules="{}", created_by="account-1"
        )
        rule.id = "rule-1"
        row.dataset_process_rule_id = rule.id
        session.add(rule)
    chunk = IndexDocument(page_content="text", metadata={"doc_id": "node", "doc_hash": "hash"})
    with (
        Flask(__name__).app_context(),
        patch.object(documents, "get_indexing_user", return_value=MagicMock()),
        patch(f"{MODULE}.IndexProcessorFactory") as factory,
        patch(f"{MODULE}.ModelManager.for_tenant"),
        patch(f"{MODULE}.calculate_segment_token_counts", return_value=[19]),
        patch(f"{MODULE}.Jieba"),
        patch(f"{MODULE}.Vector") as vector_cls,
    ):
        processor = factory.return_value.init_index_processor.return_value
        processor.extract.return_value = ExtractionResult(documents=[chunk])
        processor.transform.return_value = [chunk]
        vector_cls.return_value.get_routing_target.return_value = ("qdrant", "collection-a")
        DocumentIndexingService(documents=documents, segments=segments, backend=adapter).run([job.ref])
    with sqlite_session_factory() as session:
        row = session.get(Document, "document-1")
        assert row is not None
        assert row.indexing_status == IndexingStatus.COMPLETED
        assert row.tokens == 19
        assert row.parsing_completed_at is not None
        assert row.splitting_completed_at is not None
        assert row.completed_at is not None


def test_claimed_jina_sync_refetches_empty_job_source_and_completes_real_pipeline(
    backend: BackendFixture,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _adapter, documents, segments, _sources, job, _redis = backend
    request_id = "website-sync-request"
    claim_id = "website-sync-claim"
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.data_source_type = DataSourceType.WEBSITE_CRAWL
        document = session.get_one(Document, "document-1")
        document.data_source_type = DataSourceType.WEBSITE_CRAWL
        document.data_source_info = json.dumps(
            {
                "provider": "jinareader",
                "provider_instance_id": website_crawl_provider_instance_id(provider="jinareader"),
                "job_id": "",
                "url": "https://example.com/current",
                "mode": "scrape",
                "only_main_content": True,
            }
        )
        document.indexing_request_id = request_id
        document.indexing_claim_id = claim_id
        document.indexing_claim_expires_at = datetime.now() + timedelta(minutes=5)
        rule = DatasetProcessRule(
            dataset_id=dataset.id,
            mode=ProcessRuleMode.AUTOMATIC,
            rules="{}",
            created_by="account-1",
        )
        rule.id = "jina-sync-rule"
        document.dataset_process_rule_id = rule.id
        session.add(rule)

    sources = CompositeStoredSourceResolver(adapters={"website_crawl": WebsiteSourceAdapter()})
    adapter = IndexingExecutionAdapter(
        session_factory=sqlite_session_factory,
        documents=documents,
        segments=segments,
        uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory),
        attachment_vectors=SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory),
        sources=sources,
    )
    crawl_calls: list[tuple[WebsiteJobRef, str, bool]] = []

    class WebsiteServiceStub:
        @staticmethod
        def get_crawl_url_data_by_ref(job_ref: WebsiteJobRef, url: str, *, use_job: bool) -> dict[str, str]:
            crawl_calls.append((job_ref, url, use_job))
            return {
                "content": "fresh Jina content",
                "url": "https://example.com/current",
                "title": "Current page",
            }

    indexing_user = Account(name="Indexer", email="indexer@example.com")
    indexing_user.id = "account-1"
    monkeypatch.setattr(documents, "get_indexing_user", lambda *_args, **_kwargs: indexing_user)
    monkeypatch.setattr(
        "extensions.application_services.data_sources.build_website_service",
        lambda **_kwargs: WebsiteServiceStub(),
    )
    service = DocumentIndexingService(documents=documents, segments=segments, backend=adapter)
    with (
        Flask(__name__).app_context(),
        patch(f"{MODULE}.calculate_segment_token_counts", return_value=[18]),
        patch(f"{MODULE}.Jieba"),
    ):
        service.restart_claimed(job.ref, claim_id=claim_id)

    assert crawl_calls == [
        (
            WebsiteJobRef(
                tenant_id="workspace-1",
                provider="jinareader",
                provider_instance_id=website_crawl_provider_instance_id(provider="jinareader"),
                job_id="",
            ),
            "https://example.com/current",
            False,
        )
    ]
    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.indexing_status == IndexingStatus.COMPLETED
        assert document.tokens == 18
        persisted_segments = session.scalars(
            select(DocumentSegment).where(DocumentSegment.document_id == document.id)
        ).all()
        assert [segment.content for segment in persisted_segments] == ["fresh Jina content"]


@pytest.mark.parametrize("change", ["paused", "deleted"])
def test_document_changed_during_worker_io_cannot_be_completed(
    backend: BackendFixture, sqlite_session_factory: sessionmaker[Session], change: str
) -> None:
    adapter, documents, segments, _, job, _ = backend
    chunks = [IndexDocument(page_content="text", metadata={"doc_id": "node", "doc_hash": "hash"})]
    segments.save_for_indexing(job, chunks, [9])

    def change_document(*_args, **_kwargs) -> None:
        with sqlite_session_factory.begin() as writer:
            row = writer.get(Document, "document-1")
            assert row is not None
            if change == "paused":
                row.is_paused = True
            else:
                writer.delete(row)

    service = DocumentIndexingService(documents=documents, segments=segments, backend=adapter)
    with Flask(__name__).app_context(), patch(f"{MODULE}.Jieba") as keyword:
        keyword.return_value.update_texts.side_effect = change_document
        if change == "paused":
            with pytest.raises(DocumentIsPausedError):
                service.run_in_indexing_status(job.ref)
        else:
            service.run_in_indexing_status(job.ref)
    with sqlite_session_factory() as session:
        segment = session.scalar(select(DocumentSegment))
        assert segment is not None
        assert segment.status == SegmentStatus.INDEXING


def test_partial_worker_failure_retries_only_incomplete_segments_and_keeps_total(
    backend: BackendFixture, sqlite_session_factory: sessionmaker[Session]
) -> None:
    adapter, documents, segments, _, job, _ = backend
    with sqlite_session_factory.begin() as session:
        persisted_dataset = session.get(Dataset, "dataset-1")
        assert persisted_dataset is not None
        persisted_dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
    chunks = [
        IndexDocument(page_content=content, metadata={"doc_id": content, "doc_hash": content})
        for content in ["good", "bad"]
    ]
    segments.save_for_indexing(job, chunks, [13, 17])
    service = DocumentIndexingService(documents=documents, segments=segments, backend=adapter)

    def load(group: list[IndexDocument]) -> None:
        if group[0].page_content == "bad":
            raise RuntimeError("one shard failed")

    with (
        Flask(__name__).app_context(),
        patch(f"{MODULE}.Vector") as factory,
        patch(f"{MODULE}.helper.generate_text_hash", side_effect=lambda content: "0" if content == "good" else "1"),
    ):
        vector = factory.return_value
        vector.create.side_effect = load
        service.run_in_indexing_status(job.ref)
        pending, tokens = segments.resume_indexing(job)
        assert [chunk.page_content for chunk in pending] == ["bad"]
        assert tokens == 30
        vector.create.reset_mock()
        vector.create.side_effect = None
        service.run_in_indexing_status(job.ref)
        vector.create.assert_called_once()
        assert [chunk.page_content for chunk in vector.create.call_args.args[0]] == ["bad"]
    with sqlite_session_factory() as session:
        row = session.get(Document, "document-1")
        assert row is not None
        assert row.indexing_status == IndexingStatus.COMPLETED
        assert row.tokens == 30
        assert row.error is None

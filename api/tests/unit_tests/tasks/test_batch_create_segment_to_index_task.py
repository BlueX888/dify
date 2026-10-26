from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from sqlalchemy import event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from extensions.storage.storage_type import StorageType
from graphon.model_runtime.entities.model_entities import ModelType
from models.dataset import Dataset, Document, DocumentSegment
from models.enums import CreatorUserRole, DataSourceType, DocumentCreatedFrom, IndexingStatus, SegmentStatus
from models.model import UploadFile
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef
from tasks import batch_create_segment_to_index_task as task_module
from tasks.batch_create_segment_to_index_task import batch_create_segment_to_index_task


@dataclass
class RedisRecorder:
    writes: list[tuple[str, int, str]] = field(default_factory=list)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.writes.append((key, ttl, value))


@dataclass
class SegmentMutationsStub:
    error: Exception | None = None
    calls: list[tuple[DocumentRef, list[str]]] = field(default_factory=list)

    def index_segments(self, document_ref: DocumentRef, *, segment_ids: list[str]) -> None:
        self.calls.append((document_ref, segment_ids))
        if self.error is not None:
            raise self.error


@dataclass
class SegmentsStub:
    mutations: SegmentMutationsStub


@dataclass
class KnowledgeStub:
    segments: SegmentsStub


@dataclass
class ApplicationServicesStub:
    knowledge: KnowledgeStub


def _services(mutations: SegmentMutationsStub) -> ApplicationServicesStub:
    return ApplicationServicesStub(knowledge=KnowledgeStub(segments=SegmentsStub(mutations=mutations)))


def _records(
    sqlite_session: Session,
    *,
    indexing_technique: IndexTechniqueType = IndexTechniqueType.ECONOMY,
) -> tuple[str, str, Dataset, Document, UploadFile]:
    tenant_id = str(uuid4())
    user_id = str(uuid4())
    dataset = Dataset(
        id=str(uuid4()),
        tenant_id=tenant_id,
        name="Batch import dataset",
        data_source_type=DataSourceType.UPLOAD_FILE,
        created_by=user_id,
        indexing_technique=indexing_technique,
        embedding_model_provider="provider" if indexing_technique == IndexTechniqueType.HIGH_QUALITY else None,
        embedding_model="model" if indexing_technique == IndexTechniqueType.HIGH_QUALITY else None,
    )
    document = Document(
        id=str(uuid4()),
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name="document.txt",
        created_from=DocumentCreatedFrom.WEB,
        created_by=user_id,
        indexing_status=IndexingStatus.COMPLETED,
        enabled=True,
        archived=False,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        word_count=0,
    )
    upload_file = UploadFile(
        tenant_id=tenant_id,
        storage_type=StorageType.LOCAL,
        key="batch-import.csv",
        name="batch-import.csv",
        size=32,
        extension=".csv",
        mime_type="text/csv",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=user_id,
        created_at=datetime.now(UTC),
        used=False,
    )
    sqlite_session.add_all([dataset, document, upload_file])
    sqlite_session.commit()
    return tenant_id, user_id, dataset, document, upload_file


@pytest.mark.parametrize("foreign_owner", [None, "dataset", "document_tenant", "document_dataset", "upload"])
def test_batch_create_segment_to_index_task_validates_scope_and_indexes_committed_batch(
    sqlite_session: Session,
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    foreign_owner: str | None,
) -> None:
    tenant_id, user_id, dataset, document, upload_file = _records(sqlite_session)
    existing_segment = DocumentSegment(
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=7,
        content="existing",
        word_count=1,
        tokens=1,
        created_by=user_id,
        index_node_id="existing-node",
        index_node_hash="existing-hash",
        status=SegmentStatus.COMPLETED,
    )
    if foreign_owner == "dataset":
        dataset.tenant_id = str(uuid4())
    elif foreign_owner == "document_tenant":
        document.tenant_id = str(uuid4())
    elif foreign_owner == "document_dataset":
        document.dataset_id = str(uuid4())
    elif foreign_owner == "upload":
        upload_file.tenant_id = str(uuid4())
    sqlite_session.add(existing_segment)
    sqlite_session.commit()

    download_calls: list[str] = []

    def mock_download(_key: str, file_path: str) -> None:
        download_calls.append(file_path)
        Path(file_path).write_text("content\nfirst\nsecond\nthird\n", encoding="utf-8")

    max_position_queries: list[str] = []

    def count_max_position_query(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        normalized_statement = statement.lower()
        if "max(" in normalized_statement and "document_segments.position" in normalized_statement:
            max_position_queries.append(statement)

    event.listen(sqlite_engine, "before_cursor_execute", count_max_position_query)
    mutations = SegmentMutationsStub()
    redis = RedisRecorder()
    monkeypatch.setattr(task_module.storage, "download", mock_download)
    monkeypatch.setattr(task_module, "redis_client", redis)
    from extensions import ext_application_services

    monkeypatch.setattr(ext_application_services, "application_services", lambda: _services(mutations))
    job_id = str(uuid4())
    try:
        batch_create_segment_to_index_task(
            job_id=job_id,
            upload_file_id=upload_file.id,
            dataset_id=dataset.id,
            document_id=document.id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", count_max_position_query)

    if foreign_owner is not None:
        assert not download_calls
        assert not mutations.calls
        assert not max_position_queries
        assert sqlite_session.scalars(select(DocumentSegment.id)).all() == [existing_segment.id]
        assert redis.writes == [(f"segment_batch_import_{job_id}", 600, "error")]
        return

    positions = sqlite_session.scalars(
        select(DocumentSegment.position)
        .where(DocumentSegment.document_id == document.id)
        .order_by(DocumentSegment.position)
    ).all()
    assert positions == [7, 8, 9, 10]
    assert len(max_position_queries) == 1
    assert len(mutations.calls) == 1
    document_ref, segment_ids = mutations.calls[0]
    assert document_ref == DocumentRef(DatasetRef(tenant_id, dataset.id), document.id)
    assert len(segment_ids) == 3
    assert redis.writes == [(f"segment_batch_import_{job_id}", 600, "completed")]


class FailingModelManager:
    def get_model_instance(
        self,
        tenant_id: str,
        provider: str,
        model_type: ModelType,
        model: str,
        *,
        request_metadata: Mapping[str, object] | None = None,
    ) -> None:
        _ = tenant_id, provider, model_type, model, request_metadata
        raise RuntimeError("provider failed")


@pytest.mark.parametrize("failure_stage", ["download", "csv", "provider", "database", "indexing"])
def test_batch_create_segment_to_index_task_records_execution_failure(
    sqlite_session: Session,
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    technique = IndexTechniqueType.HIGH_QUALITY if failure_stage == "provider" else IndexTechniqueType.ECONOMY
    tenant_id, user_id, dataset, document, upload_file = _records(
        sqlite_session,
        indexing_technique=technique,
    )
    redis = RedisRecorder()
    mutations = SegmentMutationsStub(error=RuntimeError("indexing failed") if failure_stage == "indexing" else None)
    monkeypatch.setattr(task_module, "redis_client", redis)

    def download(_key: str, file_path: str) -> None:
        if failure_stage == "download":
            raise RuntimeError("download failed")
        csv_content = 'content\n"unterminated\n' if failure_stage == "csv" else "content\nsegment\n"
        Path(file_path).write_text(csv_content, encoding="utf-8")

    monkeypatch.setattr(task_module.storage, "download", download)
    if failure_stage == "provider":

        def failing_for_tenant(*, tenant_id: str) -> FailingModelManager:
            assert tenant_id == dataset.tenant_id
            return FailingModelManager()

        monkeypatch.setattr(
            task_module.ModelManager,
            "for_tenant",
            staticmethod(failing_for_tenant),
        )

    from extensions import ext_application_services

    monkeypatch.setattr(ext_application_services, "application_services", lambda: _services(mutations))

    def fail_segment_insert(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if failure_stage == "database" and statement.lstrip().upper().startswith("INSERT INTO DOCUMENT_SEGMENTS"):
            raise RuntimeError("database failed")

    event.listen(sqlite_engine, "before_cursor_execute", fail_segment_insert)
    job_id = str(uuid4())
    try:
        with pytest.raises((RuntimeError, pd.errors.ParserError)):
            batch_create_segment_to_index_task(
                job_id=job_id,
                upload_file_id=upload_file.id,
                dataset_id=dataset.id,
                document_id=document.id,
                tenant_id=tenant_id,
                user_id=user_id,
            )
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", fail_segment_insert)

    assert redis.writes == [(f"segment_batch_import_{job_id}", 600, "error")]
    sqlite_session.expire_all()
    persisted_segments = sqlite_session.scalars(
        select(DocumentSegment).where(DocumentSegment.document_id == document.id)
    ).all()
    if failure_stage == "indexing":
        assert len(persisted_segments) == 1
    else:
        assert persisted_segments == []
        persisted_document = sqlite_session.get(Document, document.id)
        assert persisted_document is not None
        assert persisted_document.word_count == 0

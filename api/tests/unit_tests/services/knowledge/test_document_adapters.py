import json
from datetime import datetime
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from controllers.console.datasets.datasets_document import DocumentDetailResponse, DocumentWithSegmentsListResponse
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from models.dataset import (
    Dataset,
    DatasetMetadata,
    DatasetMetadataBinding,
    DatasetProcessRule,
    Document,
    DocumentPipelineExecutionLog,
    DocumentSegment,
    Pipeline,
)
from models.enums import DatasetRuntimeMode, IndexingStatus, ProcessRuleMode, SegmentStatus
from models.model import DocumentIndexingOutbox, ResourceCleanupOutbox
from models.workflow import Workflow
from repositories.knowledge.metadata_repository import (
    SQLAlchemyMetadataRepository,
    lock_metadata_dataset,
    lock_metadata_document,
)
from services.knowledge.dataset_service import DatasetService
from services.knowledge.documents.adapters import SQLAlchemyDocumentOperations
from services.knowledge.documents.application import (
    DocumentIndexingStateError,
    DocumentListFilter,
    DocumentNotFoundError,
)
from services.knowledge.entities.resource_scope import DatasetRef
from tests.unit_tests.model_factories import make_tenant

REF = DatasetRef("tenant-1", "dataset-1")


class _UnexpectedActorCredentials:
    def resolve(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        credential_id: str,
        provider: str,
        plugin_id: str,
    ) -> dict[str, object]:
        del workspace_id, actor_id, credential_id, provider, plugin_id
        raise AssertionError("document adapter test did not select a datasource credential")


class _RetryLock:
    def acquire(self, *, blocking: bool) -> bool:
        del blocking
        return True

    def release(self) -> None:
        pass


class _RetryRedis:
    def lock(self, _key: str, *, timeout: int, thread_local: bool) -> _RetryLock:
        del timeout, thread_local
        return _RetryLock()


def _dataset(**values: object) -> Dataset:
    row = Dataset(
        id="dataset-1",
        tenant_id="tenant-1",
        name="Dataset",
        created_by="actor-1",
        maintainer="actor-1",
        indexing_technique="economy",
        chunk_structure="text_model",
    )
    for key, value in values.items():
        setattr(row, key, value)
    return row


def _document(**values: object) -> Document:
    return Document(
        **{
            "id": "document-1",
            "tenant_id": "tenant-1",
            "dataset_id": "dataset-1",
            "position": 1,
            "data_source_type": "local_file",
            "data_source_info": json.dumps({"file_path": "/path/to/file"}),
            "batch": "batch-1",
            "name": "Document",
            "created_from": "web",
            "created_by": "actor-1",
            "indexing_status": IndexingStatus.COMPLETED,
            "doc_form": "text_model",
            "enabled": True,
            "archived": False,
            "word_count": 10,
            "created_at": datetime(2024, 1, 1),
            **values,
        }
    )


def _segment(
    segment_id: str,
    *,
    tenant_id: str = "tenant-1",
    dataset_id: str = "dataset-1",
    document_id: str = "document-1",
    status: SegmentStatus = SegmentStatus.COMPLETED,
    completed_at: datetime | None = datetime(2024, 1, 1),
    hit_count: int = 0,
) -> DocumentSegment:
    segment = DocumentSegment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        position=1,
        content="content",
        word_count=7,
        tokens=2,
        created_by="actor-1",
        status=status,
        completed_at=completed_at,
        hit_count=hit_count,
    )
    segment.id = segment_id
    return segment


@pytest.fixture
def operations(sqlite_session_factory: sessionmaker[Session]) -> SQLAlchemyDocumentOperations:
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id="tenant-1"), _dataset(), _document()])
    return SQLAlchemyDocumentOperations(
        session_factory=sqlite_session_factory,
        actor_credentials=_UnexpectedActorCredentials(),
    )


def test_retry_documents_adapter_persists_exact_pipeline_source_ownership(
    operations: SQLAlchemyDocumentOperations,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, REF.dataset_id)
        dataset.runtime_mode = DatasetRuntimeMode.RAG_PIPELINE
        dataset.pipeline_id = "pipeline-1"
        pipeline = Pipeline(
            tenant_id=REF.tenant_id,
            name="Pipeline",
            description="",
            workflow_id="workflow-v2",
            is_published=True,
        )
        pipeline.id = "pipeline-1"
        workflow = Workflow(
            id="workflow-v2",
            tenant_id=REF.tenant_id,
            app_id=pipeline.id,
            type="rag-pipeline",
            version="v2",
            marked_name="",
            marked_comment="",
            graph="{}",
            features="{}",
            created_by="actor-1",
            created_at=datetime(2024, 1, 1),
            updated_by=None,
            updated_at=datetime(2024, 1, 1),
            environment_variables=[],
            conversation_variables=[],
            rag_pipeline_variables=[],
        )
        document = session.get_one(Document, "document-1")
        document.indexing_status = IndexingStatus.ERROR
        source_log = DocumentPipelineExecutionLog(
            pipeline_id="pipeline-1",
            document_id=document.id,
            datasource_type="local_file",
            datasource_info="{}",
            datasource_node_id="source-v1",
            input_data={"content": "v1"},
            created_by="original-actor",
            workflow_id="workflow-v1",
            original_document_id=document.id,
        )
        session.add_all([pipeline, workflow, source_log])
        session.flush()
        source_log_id = source_log.id
    published: list[str] = []
    monkeypatch.setattr("services.knowledge.dataset_service.redis_client", _RetryRedis())
    monkeypatch.setattr("services.knowledge.dataset_service.enqueue_document_indexing_intent", published.append)

    operations.retry_documents(REF, ["document-1"], actor_id="retry-actor")

    with sqlite_session_factory() as session:
        intent = session.scalar(select(DocumentIndexingOutbox))
        assert intent is not None
        assert intent.actor_id == "retry-actor"
        assert intent.pipeline_execution_log_id != source_log_id
        retry_log = session.get_one(DocumentPipelineExecutionLog, intent.pipeline_execution_log_id)
        assert retry_log.workflow_id == workflow.id
        assert retry_log.created_by == "retry-actor"
        assert retry_log.datasource_info == source_log.datasource_info
        assert retry_log.input_data == source_log.input_data
        assert retry_log.original_document_id == document.id
        assert json.loads(intent.document_ids) == ["document-1"]
        document = session.get_one(Document, "document-1")
        assert document.indexing_status == IndexingStatus.WAITING
        assert document.indexing_request_id == intent.id
        assert published == [intent.id]


@pytest.mark.parametrize("foreign", [{"tenant_id": "other"}, {"dataset_id": "other"}])
def test_document_reads_and_writes_reject_incorrect_owner(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session], foreign: dict[str, str]
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_document(id="foreign", **foreign))
    ref = REF.document("foreign")
    assert operations.get_state(ref) is None
    with pytest.raises(DocumentNotFoundError):
        operations.get_detail(ref, metadata_only=False)
    with pytest.raises(DocumentNotFoundError):
        operations.update_metadata(ref, doc_type="book", doc_metadata={"title": "changed"})
    with pytest.raises(DocumentNotFoundError):
        operations.rename_document(ref, "changed")
    with pytest.raises(DocumentNotFoundError):
        operations.delete_document(ref)
    with sqlite_session_factory() as session:
        document = session.get(Document, "foreign")
        assert document is not None
        assert document.name == "Document"
        assert session.scalar(select(ResourceCleanupOutbox)) is None


def test_rename_locks_dataset_before_document(
    operations: SQLAlchemyDocumentOperations,
) -> None:
    lock_order: list[str] = []

    def lock_dataset(session: Session, ref: DatasetRef) -> Dataset | None:
        lock_order.append("dataset")
        return lock_metadata_dataset(session, ref)

    def lock_document(session: Session, ref: DatasetRef, document_id: str) -> Document | None:
        lock_order.append("document")
        return lock_metadata_document(session, ref, document_id)

    with (
        patch("services.knowledge.documents.adapters.lock_metadata_dataset", side_effect=lock_dataset),
        patch("services.knowledge.documents.adapters.lock_metadata_document", side_effect=lock_document),
    ):
        operations.rename_document(REF.document("document-1"), "Renamed")

    assert lock_order == ["dataset", "document"]


def test_delete_ignores_broken_model_credentials_and_persists_cleanup_intent(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session]
) -> None:
    with (
        patch.object(
            DatasetService,
            "check_dataset_model_setting",
            side_effect=AssertionError("embedding credentials are invalid"),
        ) as check_model,
        patch("services.knowledge.dataset_service.enqueue_resource_cleanup_intent") as enqueue,
    ):
        operations.delete_document(REF.document("document-1"))

    check_model.assert_not_called()
    with sqlite_session_factory() as session:
        assert session.get(Document, "document-1") is None
        cleanup = session.scalar(select(ResourceCleanupOutbox))
        assert cleanup is not None
        assert cleanup.tenant_id == REF.tenant_id
        assert cleanup.dataset_id == REF.dataset_id
        assert cleanup.resource_id == "document-1"
        assert cleanup.resource_type == "document"
    enqueue.assert_called_once_with(cleanup.id)


def test_delete_rejects_document_with_active_indexing_claim(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session]
) -> None:
    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, "document-1")
        document.indexing_claim_id = "active-claim"
        document.indexing_claim_expires_at = datetime(2099, 1, 1)

    with pytest.raises(DocumentIndexingStateError):
        operations.delete_document(REF.document("document-1"))

    with sqlite_session_factory() as session:
        assert session.get(Document, "document-1") is not None
        assert session.scalar(select(ResourceCleanupOutbox)) is None


def test_document_lookup_uses_active_workspace(operations: SQLAlchemyDocumentOperations) -> None:
    assert operations.find_document(workspace_id="other", document_id="document-1") is None
    assert operations.find_document(workspace_id="tenant-1", document_id="document-1") == REF.document("document-1")


def test_listing_filters_scope_and_materializes_values(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session]
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _document(id="matching", name="Needle", position=2),
                _document(id="disabled", name="Needle", enabled=False),
                _document(id="foreign", name="Needle", tenant_id="other"),
                _document(id="other-dataset", name="Needle", dataset_id="other"),
            ]
        )
    result = operations.list_documents(REF, DocumentListFilter(search="Needle", status="available", fetch=True))
    response = DocumentWithSegmentsListResponse.model_validate(result).model_dump(mode="json")
    assert response["total"] == 1
    assert response["data"][0]["id"] == "matching"
    assert response["data"][0]["data_source_info"] == {"file_path": "/path/to/file"}
    assert response["data"][0]["completed_segments"] == 0
    assert response["data"][0]["total_segments"] == 0


def test_listing_hit_count_sort_and_exact_last_page(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session]
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _document(id="popular", position=2),
                _segment("segment-1", hit_count=1),
                _segment("segment-2", document_id="popular", hit_count=10),
                _segment("foreign", tenant_id="other", hit_count=1000),
            ]
        )
    first = operations.list_documents(REF, DocumentListFilter(sort="-hit_count", limit=1))
    last = operations.list_documents(REF, DocumentListFilter(sort="-hit_count", limit=1, page=2))
    assert [row["id"] for row in first["data"]] == ["popular"]
    assert first["has_more"] is True
    assert last["has_more"] is False
    assert last["total"] == 2


def test_status_counts_use_one_aggregate_and_complete_owner_chain(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session], sqlite_engine: Engine
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _segment("complete"),
                _segment("waiting", completed_at=None),
                _segment("excluded", status=SegmentStatus.RE_SEGMENT),
                _segment("foreign", tenant_id="other"),
                _segment("wrong-dataset", dataset_id="other"),
                _document(id="second", is_paused=True, indexing_status=IndexingStatus.INDEXING),
            ]
        )
    statements: list[str] = []

    def record(
        _conn: object, _cursor: object, statement: str, _parameters: object, _context: object, _many: bool
    ) -> None:
        statements.append(statement)

    event.listen(sqlite_engine, "before_cursor_execute", record)
    try:
        result = operations.get_batch_indexing_status(REF, "batch-1")
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", record)
    rows = {row["id"]: row for row in result["data"]}
    assert rows["document-1"]["completed_segments"] == 1
    assert rows["document-1"]["total_segments"] == 2
    assert rows["second"]["indexing_status"] == "paused"
    assert sum("FROM document_segments" in statement for statement in statements) == 1


def test_unknown_batch_is_not_found(operations: SQLAlchemyDocumentOperations) -> None:
    with pytest.raises(DocumentNotFoundError):
        operations.get_batch_indexing_status(REF, "missing")


@pytest.mark.parametrize("rules", [None, {"segmentation": {"delimiter": "\\n", "max_tokens": 500}}])
def test_process_rule_preserves_null_and_legacy_delimiter(
    operations: SQLAlchemyDocumentOperations,
    sqlite_session_factory: sessionmaker[Session],
    rules: dict[str, object] | None,
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(
            DatasetProcessRule(
                dataset_id="dataset-1",
                mode=ProcessRuleMode.CUSTOM,
                rules=json.dumps(rules) if rules else None,
                created_by="actor-1",
            )
        )
    result = operations.get_process_rule(REF)
    assert result["mode"] == "custom"
    assert result["rules"] == rules


def test_detail_is_serializable_after_session_closes(operations: SQLAlchemyDocumentOperations) -> None:
    result = operations.get_detail(REF.document("document-1"), metadata_only=False)
    response = DocumentDetailResponse.model_validate(result).model_dump(mode="json")
    assert response["id"] == "document-1"
    assert response["data_source_info"] == {"file_path": "/path/to/file"}
    assert response["data_source_detail_dict"] == {}
    assert response["created_at"] == int(datetime(2024, 1, 1).timestamp())


def test_metadata_only_avoids_loading_full_details(operations: SQLAlchemyDocumentOperations) -> None:
    with patch("services.knowledge.documents.adapters.get_document_source_detail") as source:
        result = operations.get_detail(REF.document("document-1"), metadata_only=True)
    assert set(result) == {"id", "doc_type", "doc_metadata"}
    source.assert_not_called()


def test_metadata_update_is_committed_before_returning(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session]
) -> None:
    operations.update_metadata(
        REF.document("document-1"),
        doc_type="book",
        doc_metadata={"title": "Committed title"},
    )
    with sqlite_session_factory() as session:
        document = session.get(Document, "document-1")
        assert document is not None
        assert document.doc_type == "book"
        assert document.doc_metadata == {"title": "Committed title"}


def test_metadata_update_preserves_bound_custom_value_for_later_rename(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session]
) -> None:
    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, "document-1")
        document.doc_metadata = {"custom_author": "Ada", "old_doc_type_field": "discard me"}
        field = DatasetMetadata(
            tenant_id=REF.tenant_id,
            dataset_id=REF.dataset_id,
            type="string",
            name="custom_author",
            created_by="schema-author",
        )
        session.add(field)
        session.flush()
        session.add(
            DatasetMetadataBinding(
                tenant_id=REF.tenant_id,
                dataset_id=REF.dataset_id,
                metadata_id=field.id,
                document_id="document-1",
                created_by="binding-author",
            )
        )
        field_id = field.id

    operations.update_metadata(
        REF.document("document-1"),
        doc_type="book",
        doc_metadata={"title": "New title", "custom_author": "must not overwrite"},
    )

    store = SQLAlchemyMetadataRepository(session_factory=sqlite_session_factory)
    renamed = store.rename(REF, field_id, name="renamed_author", actor_id="schema-editor")
    assert renamed.updated_by == "schema-editor"
    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        bindings = session.scalars(
            select(DatasetMetadataBinding).where(
                DatasetMetadataBinding.tenant_id == REF.tenant_id,
                DatasetMetadataBinding.dataset_id == REF.dataset_id,
                DatasetMetadataBinding.document_id == "document-1",
            )
        ).all()
        assert document.doc_metadata == {"title": "New title", "renamed_author": "Ada"}
        assert len(bindings) == 1
        assert bindings[0].created_by == "binding-author"

    store.delete(REF, field_id)
    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.doc_metadata == {"title": "New title"}
        assert (
            session.scalar(
                select(DatasetMetadataBinding).where(
                    DatasetMetadataBinding.tenant_id == REF.tenant_id,
                    DatasetMetadataBinding.dataset_id == REF.dataset_id,
                    DatasetMetadataBinding.document_id == "document-1",
                )
            )
            is None
        )


def test_metadata_rename_does_not_invent_null_for_missing_bound_value(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id="tenant-1"), _dataset(), _document(doc_metadata={"title": "Book"})])
        field = DatasetMetadata(
            tenant_id=REF.tenant_id,
            dataset_id=REF.dataset_id,
            type="string",
            name="missing_author",
            created_by="schema-author",
        )
        session.add(field)
        session.flush()
        session.add(
            DatasetMetadataBinding(
                tenant_id=REF.tenant_id,
                dataset_id=REF.dataset_id,
                metadata_id=field.id,
                document_id="document-1",
                created_by="binding-author",
            )
        )
        field_id = field.id

    store = SQLAlchemyMetadataRepository(session_factory=sqlite_session_factory)
    store.rename(REF, field_id, name="writer", actor_id="schema-editor")

    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.doc_metadata == {"title": "Book"}
        assert "writer" not in document.doc_metadata


def test_metadata_update_cannot_revive_disabled_built_in_or_foreign_binding(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session]
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.built_in_field_enabled = True
        document = session.get_one(Document, "document-1")
        document.doc_metadata = {
            "document_name": "Document",
            "foreign_custom": "must be removed",
        }
        foreign_field = DatasetMetadata(
            tenant_id="other-tenant",
            dataset_id=REF.dataset_id,
            type="string",
            name="foreign_custom",
            created_by="foreign-author",
        )
        session.add(foreign_field)
        session.flush()
        session.add(
            DatasetMetadataBinding(
                tenant_id="other-tenant",
                dataset_id=REF.dataset_id,
                metadata_id=foreign_field.id,
                document_id="document-1",
                created_by="foreign-author",
            )
        )

    store = SQLAlchemyMetadataRepository(session_factory=sqlite_session_factory)
    store.set_built_in(REF, enabled=False)
    operations.update_metadata(
        REF.document("document-1"),
        doc_type="others",
        doc_metadata={"document_name": "stale name", "free_form": "current"},
    )

    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.doc_metadata == {"free_form": "current"}
        assert (
            session.scalar(
                select(DatasetMetadataBinding).where(
                    DatasetMetadataBinding.tenant_id == REF.tenant_id,
                    DatasetMetadataBinding.dataset_id == REF.dataset_id,
                    DatasetMetadataBinding.document_id == "document-1",
                )
            )
            is None
        )


def test_latest_pipeline_log_and_empty_result(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session]
) -> None:
    assert operations.get_execution_log(REF.document("document-1")) == {}
    with sqlite_session_factory.begin() as session:
        for year in [2024, 2025]:
            row = DocumentPipelineExecutionLog(
                pipeline_id="pipeline-1",
                document_id="document-1",
                datasource_type="local_file",
                datasource_info=json.dumps({"year": year}),
                datasource_node_id="node-1",
                input_data={"year": year},
                created_by="actor-1",
            )
            row.created_at = datetime(year, 1, 1)
            session.add(row)
    result = operations.get_execution_log(REF.document("document-1"))
    assert result["datasource_info"] == {"year": 2025}
    assert result["input_data"] == {"year": 2025}


@pytest.mark.parametrize("settings", [None, {"enable": False}])
def test_summary_guard_rejects_disabled_summary(
    operations: SQLAlchemyDocumentOperations,
    sqlite_session_factory: sessionmaker[Session],
    settings: dict[str, object] | None,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.summary_index_setting = settings
    with pytest.raises(ValueError, match="not enabled"):
        operations.require_summary_enabled(REF)


def test_enable_summary_only_updates_requested_owner(
    operations: SQLAlchemyDocumentOperations, sqlite_session_factory: sessionmaker[Session]
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_document(id="foreign", tenant_id="other", need_summary=False))
    operations.enable_summary(REF, ["document-1", "foreign"])
    with sqlite_session_factory() as session:
        document = session.get(Document, "document-1")
        foreign = session.get(Document, "foreign")
        assert document is not None
        assert foreign is not None
        assert document.need_summary is True
        assert foreign.need_summary is False

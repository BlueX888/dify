"""Unit tests for DocumentService behaviors in dataset_service."""

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import event, func, select
from sqlalchemy.orm import Session, sessionmaker

from core.model_manager import ModelInstance
from core.rag.extractor.entity.extract_setting import WebsiteJobRef
from core.rag.index_processor.index_processor import IndexProcessorFactory
from core.rag.models.document import Document as IndexDocument
from core.workflow.file_reference import build_file_reference
from enums import DeploymentEdition
from models import TenantAccountJoin, TenantAccountRole
from models.dataset import (
    Dataset,
    DatasetCollectionBinding,
    DocumentPipelineExecutionLog,
    DocumentSegment,
    DocumentSegmentSummary,
    Pipeline,
)
from models.enums import (
    DatasetRuntimeMode,
    DataSourceType,
    DocumentCreatedFrom,
    IndexingStatus,
    SegmentStatus,
    SummaryStatus,
)
from models.model import (
    DocumentIndexingOutbox,
    ResourceCleanupOutbox,
    SegmentStateOutbox,
    UploadFile,
    UploadFileCleanupOutbox,
)
from models.workflow import Workflow
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    SQLAlchemyDocumentIndexingOutboxRepository,
    create_document_indexing_intents,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from repositories.knowledge.segment_repository import SQLAlchemySegmentRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.data_source.website_crawl_cache import website_crawl_provider_instance_id
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.indexing_estimate import EstimateValidationError, normalize_process_rule
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.execution import IndexingExecutionAdapter
from services.knowledge.indexing.adapters.sources import CompositeStoredSourceResolver
from tasks.segment_state_outbox_task import drain_segment_state_outbox
from tests.unit_tests.config_override import config_overrides_context
from tests.unit_tests.model_factories import make_account, make_tenant, make_upload_file

from .dataset_service_test_helpers import (
    Account,
    BuiltInField,
    CloudPlan,
    DatasetProcessRule,
    DatasetService,
    DataSource,
    Document,
    DocumentIndexingError,
    DocumentService,
    FileInfo,
    FileNotExistsError,
    IndexStructureType,
    InfoList,
    KnowledgeConfig,
    NotionIcon,
    NotionInfo,
    NotionPage,
    PreProcessingRule,
    ProcessRule,
    RerankingModel,
    RetrievalModel,
    Rule,
    Segmentation,
    WebsiteInfo,
    _make_features,
    _make_lock_context,
    _make_upload_knowledge_config,
    json,
    patch,
    pytest,
)


def _account(*, account_id: str = "user-1", tenant_id: str = "tenant-1") -> Account:
    return make_account(
        account_id=account_id,
        name="User",
        email=f"{account_id}@example.com",
        tenant=make_tenant(tenant_id=tenant_id, name="Tenant"),
    )


def _indexing_outbox_jobs(session: Session) -> list[tuple[DocumentIndexingAction, tuple[str, ...]]]:
    rows = session.scalars(select(DocumentIndexingOutbox).order_by(DocumentIndexingOutbox.created_at)).all()
    return [(DocumentIndexingAction(row.action), tuple(json.loads(row.document_ids))) for row in rows]


def _dataset_row(
    *,
    dataset_id: str = "dataset-1",
    tenant_id: str = "tenant-1",
    built_in_field_enabled: bool = False,
    data_source_type: str | None = None,
    indexing_technique: str | None = "economy",
) -> Dataset:
    return Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name="Dataset",
        description="",
        provider="vendor",
        created_by="user-1",
        maintainer="user-1",
        built_in_field_enabled=built_in_field_enabled,
        chunk_structure=IndexStructureType.PARAGRAPH_INDEX,
        data_source_type=data_source_type,
        indexing_technique=indexing_technique,
    )


def _document_row(
    *,
    document_id: str = "document-1",
    dataset_id: str = "dataset-1",
    tenant_id: str = "tenant-1",
    name: str = "Document",
    indexing_status: str = IndexingStatus.COMPLETED,
    data_source_type: str = DataSourceType.UPLOAD_FILE,
    data_source_info: str = "{}",
    enabled: bool = True,
    archived: bool = False,
    is_paused: bool = False,
) -> Document:
    return Document(
        id=document_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        position=1,
        data_source_type=data_source_type,
        data_source_info=data_source_info,
        batch="batch-1",
        name=name,
        created_from=DocumentCreatedFrom.API,
        created_by="user-1",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 2),
        indexing_status=indexing_status,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        word_count=10,
        enabled=enabled,
        archived=archived,
        is_paused=is_paused,
        completed_at=datetime(2026, 1, 2) if indexing_status == IndexingStatus.COMPLETED else None,
    )


def _persist_owner_chain(session: Session, dataset: Dataset, *documents: Document) -> None:
    tenant = make_tenant(tenant_id=dataset.tenant_id, name="Tenant")
    session.add_all([tenant, dataset, *documents])
    session.commit()


def _published_pipeline_workflow(*, workflow_id: str) -> tuple[Pipeline, Workflow]:
    pipeline = Pipeline(
        tenant_id="tenant-1",
        name="Pipeline",
        description="",
        workflow_id=workflow_id,
        is_published=True,
    )
    pipeline.id = "pipeline-1"
    workflow = Workflow(
        id=workflow_id,
        tenant_id="tenant-1",
        app_id=pipeline.id,
        type="rag-pipeline",
        version=f"version-{workflow_id}",
        marked_name="",
        marked_comment="",
        graph="{}",
        features="{}",
        created_by="user-1",
        created_at=datetime(2026, 1, 1),
        updated_by=None,
        updated_at=datetime(2026, 1, 1),
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )
    return pipeline, workflow


def _upload_file(*, file_id: str, tenant_id: str = "tenant-1", name: str = "upload.txt") -> UploadFile:
    return make_upload_file(
        file_id=file_id,
        tenant_id=tenant_id,
        storage_type="opendal",
        key=f"key-{file_id}",
        name=name,
        size=1,
        created_by="user-1",
        created_at=datetime(2026, 1, 1),
    )


def _process_rule(
    *, dataset_id: str = "dataset-1", rule_id: str = "rule-1", mode: str = "automatic"
) -> DatasetProcessRule:
    rules = (
        json.dumps(DatasetProcessRule.AUTOMATIC_RULES)
        if mode == "automatic"
        else Rule(
            pre_processing_rules=[PreProcessingRule(id="remove_extra_spaces", enabled=True)],
            segmentation=Segmentation(separator="\n", max_tokens=100),
        ).model_dump_json()
    )
    process_rule = DatasetProcessRule(dataset_id=dataset_id, mode=mode, rules=rules, created_by="user-1")
    process_rule.id = rule_id
    return process_rule


class _DocumentStatusRedisBoundary:
    def __init__(self) -> None:
        self.cached: list[tuple[str, int, int]] = []

    def get(self, _key: str) -> None:
        return None

    def setex(self, key: str, ttl: int, value: int) -> None:
        self.cached.append((key, ttl, value))


class _DocumentEnableTaskBoundary:
    def __init__(self) -> None:
        self.document_ids: list[str] = []

    def delay(self, document_id: str) -> None:
        self.document_ids.append(document_id)


class _RejectingDocumentEnableTaskBoundary:
    def __init__(self) -> None:
        self.document_ids: list[str] = []

    def delay(self, document_id: str) -> None:
        self.document_ids.append(document_id)
        raise RuntimeError("broker unavailable")


class _RetryFlagLock:
    def __init__(self, store: "_RetryFlagStore", key: str):
        self.store = store
        self.key = key
        self.token = f"owner-{store.next_token}"
        store.next_token += 1

    def acquire(self, *, blocking: bool):
        assert blocking is False
        if self.key in self.store.values:
            if self.store.replacement_on_conflict:
                replacement_key, replacement_value = self.store.replacement_on_conflict
                self.store.values[replacement_key] = replacement_value
            return False
        self.store.values[self.key] = self.token
        return True

    def release(self):
        if self.store.values.get(self.key) == self.token:
            self.store.values.pop(self.key)


class _RetryFlagStore:
    def __init__(
        self,
        values: dict[str, str] | None = None,
        replacement_on_conflict: tuple[str, str] | None = None,
    ):
        self.values = values or {}
        self.replacement_on_conflict = replacement_on_conflict
        self.next_token = 1

    def lock(self, key: str, *, timeout: int, thread_local: bool):
        assert timeout == 600
        assert thread_local is False
        return _RetryFlagLock(self, key)


class TestDocumentServiceDisplayStatus:
    """Unit tests for DocumentService display-status helpers."""

    @pytest.mark.parametrize(
        ("raw_status", "expected"),
        [
            ("enabled", "available"),
            ("AVAILABLE", "available"),
            ("paused", "paused"),
            ("unknown", None),
            (None, None),
        ],
    )
    def test_normalize_display_status(self, raw_status, expected):
        assert DocumentService.normalize_display_status(raw_status) == expected

    def test_build_display_status_filters_returns_empty_tuple_for_unknown_status(self):
        assert DocumentService.build_display_status_filters("missing") == ()

    def test_apply_display_status_filter_returns_original_query_for_unknown_status(self):
        query = select(Document)

        result = DocumentService.apply_display_status_filter(query, "missing")

        assert result is query

    def test_apply_display_status_filter_applies_where_for_known_status(self):
        query = select(Document)

        result = DocumentService.apply_display_status_filter(query, "enabled")

        assert result is not query
        assert "documents.enabled" in str(result)


class TestDocumentServiceRetrieval:
    def test_get_document_by_id_uses_provided_session(self, sqlite_session: Session):
        document = _document_row()
        sqlite_session.add(document)
        sqlite_session.commit()

        assert DocumentService.get_document_by_id(document.id, session=sqlite_session) is document
        assert DocumentService.get_document_by_id("missing", session=sqlite_session) is None

    def test_get_document_with_tenant_checks_both_dataset_and_document(self, sqlite_session: Session) -> None:
        dataset = _dataset_row()
        document = _document_row()
        malformed_owner = _document_row(document_id="foreign", tenant_id="tenant-2")
        sqlite_session.add_all((dataset, document, malformed_owner))
        sqlite_session.commit()

        assert (
            DocumentService.get_document(dataset.id, document.id, session=sqlite_session, tenant_id=dataset.tenant_id)
            is document
        )
        assert (
            DocumentService.get_document(dataset.id, document.id, session=sqlite_session, tenant_id="tenant-2") is None
        )
        assert (
            DocumentService.get_document(dataset.id, malformed_owner.id, session=sqlite_session, tenant_id="tenant-2")
            is None
        )

    def test_get_document_by_ids_enforces_dataset_owner_and_state(self, sqlite_session: Session):
        dataset = _dataset_row()
        expected = _document_row(document_id="expected")
        sqlite_session.add_all(
            [
                dataset,
                expected,
                _document_row(document_id="disabled", enabled=False),
                _document_row(document_id="archived", archived=True),
                _document_row(document_id="waiting", indexing_status=IndexingStatus.WAITING),
                _document_row(document_id="other-dataset", dataset_id="dataset-2"),
                _document_row(document_id="other-tenant", tenant_id="tenant-2"),
            ]
        )
        sqlite_session.commit()

        documents = DocumentService.get_document_by_ids(
            DatasetRef(tenant_id=dataset.tenant_id, dataset_id=dataset.id),
            ["expected", "disabled", "archived", "waiting", "other-dataset", "other-tenant"],
            sqlite_session,
        )

        assert [document.id for document in documents] == [expected.id]


class TestDocumentServiceMutations:
    """Unit tests for DocumentService mutation and orchestration helpers."""

    @pytest.mark.parametrize(("archived", "expected"), [(True, True), (False, False)])
    def test_check_archived_returns_boolean_status(self, archived, expected):
        document = _document_row(archived=archived)

        assert DocumentService.check_archived(document) is expected

    def test_delete_documents_limits_query_and_cleanup_to_dataset_ref(
        self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch
    ):
        dataset = _dataset_row()
        document = _document_row(
            document_id="doc-1",
            data_source_info=json.dumps({"upload_file_id": "file-1"}),
        )
        local_file_document = _document_row(
            document_id="doc-local-file",
            data_source_type=DataSourceType.LOCAL_FILE,
            data_source_info=json.dumps({"reference": build_file_reference(record_id="file-2")}),
        )
        other_dataset = _document_row(document_id="other-dataset", dataset_id="dataset-2")
        other_tenant = _document_row(document_id="other-tenant", tenant_id="tenant-2")
        _persist_owner_chain(sqlite_session, dataset, document, local_file_document, other_dataset, other_tenant)

        enqueued: list[str] = []
        monkeypatch.setattr("services.knowledge.dataset_service.enqueue_resource_cleanup_intent", enqueued.append)
        dataset_ref = DatasetRef(tenant_id=dataset.tenant_id, dataset_id=dataset.id)
        DocumentService.delete_documents(
            dataset_ref,
            [document.id, local_file_document.id, other_dataset.id, other_tenant.id],
            IndexStructureType.PARAGRAPH_INDEX,
            sqlite_session,
        )

        assert sqlite_session.get(Document, document.id) is None
        assert sqlite_session.get(Document, local_file_document.id) is None
        assert sqlite_session.get(Document, other_dataset.id) is other_dataset
        assert sqlite_session.get(Document, other_tenant.id) is other_tenant
        intents = sqlite_session.scalars(
            select(ResourceCleanupOutbox).order_by(ResourceCleanupOutbox.resource_id)
        ).all()
        assert [(intent.resource_id, intent.source_file_id) for intent in intents] == [
            (document.id, "file-1"),
            (local_file_document.id, "file-2"),
        ]
        assert enqueued == [intent.id for intent in intents]

    def test_delete_local_file_document_persists_real_upload_file_id(
        self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch
    ):
        dataset = _dataset_row()
        document = _document_row(
            data_source_type=DataSourceType.LOCAL_FILE,
            data_source_info=json.dumps({"related_id": "file-1"}),
        )
        _persist_owner_chain(sqlite_session, dataset, document)

        enqueued: list[str] = []
        monkeypatch.setattr("services.knowledge.dataset_service.enqueue_resource_cleanup_intent", enqueued.append)
        DocumentService.delete_document(document, sqlite_session)

        intent = sqlite_session.scalar(select(ResourceCleanupOutbox))
        assert intent is not None
        assert (intent.tenant_id, intent.dataset_id, intent.resource_id) == (
            dataset.tenant_id,
            dataset.id,
            document.id,
        )
        assert intent.source_file_id == "file-1"
        assert enqueued == [intent.id]

    def test_delete_document_keeps_cleanup_intent_when_broker_publish_fails(
        self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = _dataset_row()
        document = _document_row(
            data_source_type=DataSourceType.LOCAL_FILE,
            data_source_info=json.dumps({"related_id": "file-1"}),
        )
        _persist_owner_chain(sqlite_session, dataset, document)

        def fail_publish(_intent_id: str) -> None:
            raise OSError("broker unavailable")

        monkeypatch.setattr(
            "tasks.resource_cleanup_task.process_resource_cleanup_intent_task.delay",
            fail_publish,
        )

        DocumentService.delete_document(document, sqlite_session)

        assert sqlite_session.get(Document, document.id) is None
        intent = sqlite_session.scalar(select(ResourceCleanupOutbox))
        assert intent is not None
        assert (intent.resource_id, intent.source_file_id) == (document.id, "file-1")

    def test_delete_document_commit_failure_rolls_back_cleanup_intent(
        self, sqlite_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = _dataset_row()
        document = _document_row()
        _persist_owner_chain(sqlite_session, dataset, document)
        enqueued: list[str] = []
        monkeypatch.setattr("services.knowledge.dataset_service.enqueue_resource_cleanup_intent", enqueued.append)

        def fail_commit(_session: Session) -> None:
            raise RuntimeError("commit failed")

        event.listen(sqlite_session, "before_commit", fail_commit)
        try:
            with pytest.raises(RuntimeError, match="commit failed"):
                DocumentService.delete_document(document, sqlite_session)
        finally:
            event.remove(sqlite_session, "before_commit", fail_commit)
            sqlite_session.rollback()

        assert sqlite_session.get(Document, document.id) is not None
        assert sqlite_session.scalar(select(ResourceCleanupOutbox)) is None
        assert enqueued == []

    def test_delete_documents_with_empty_ids_does_not_commit(self, sqlite_session: Session):
        commits = 0

        def count_commit(_session):
            nonlocal commits
            commits += 1

        event.listen(sqlite_session, "after_commit", count_commit)
        dataset = _dataset_row()
        DocumentService.delete_documents(
            DatasetRef(tenant_id=dataset.tenant_id, dataset_id=dataset.id),
            [],
            IndexStructureType.PARAGRAPH_INDEX,
            sqlite_session,
        )
        event.remove(sqlite_session, "after_commit", count_commit)
        assert commits == 0

    def test_rename_document_raises_when_dataset_is_missing(self, sqlite_session: Session):
        with patch("services.knowledge.dataset_service.current_user", _account()):
            with pytest.raises(ValueError, match="Dataset not found"):
                DocumentService.rename_document("dataset-1", "doc-1", "New Name", sqlite_session)

    def test_rename_document_raises_when_document_is_missing(self, sqlite_session: Session):
        dataset = _dataset_row()
        sqlite_session.add(dataset)
        sqlite_session.commit()
        with patch("services.knowledge.dataset_service.current_user", _account()):
            with pytest.raises(ValueError, match="Document not found"):
                DocumentService.rename_document(dataset.id, "doc-1", "New Name", sqlite_session)

    def test_rename_document_rejects_cross_tenant_access(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(tenant_id="tenant-other")
        sqlite_session.add_all([dataset, document])
        sqlite_session.commit()
        with patch("services.knowledge.dataset_service.current_user", _account()):
            with pytest.raises(ValueError, match="Document not found"):
                DocumentService.rename_document(dataset.id, document.id, "New Name", sqlite_session)

    def test_rename_document_updates_document_metadata_and_upload_file_name(self, sqlite_session: Session):
        dataset = _dataset_row(built_in_field_enabled=True)
        document = _document_row(
            data_source_info=json.dumps({"upload_file_id": "file-1"}),
        )
        document.doc_metadata = {BuiltInField.document_name: "Old"}
        upload_file = UploadFile(
            tenant_id=dataset.tenant_id,
            storage_type="opendal",
            key="key",
            name="old.txt",
            size=1,
            extension="txt",
            mime_type="text/plain",
            created_by_role="account",
            created_by="user-1",
            created_at=datetime(2026, 1, 1),
            used=False,
        )
        upload_file.id = "file-1"
        sqlite_session.add_all([dataset, document, upload_file])
        sqlite_session.commit()
        commits = 0

        def count_commit(_session):
            nonlocal commits
            commits += 1

        event.listen(sqlite_session, "after_commit", count_commit)
        with patch("services.knowledge.dataset_service.current_user", _account()):
            result = DocumentService.rename_document(dataset.id, document.id, "New Name", sqlite_session)
        event.remove(sqlite_session, "after_commit", count_commit)

        assert result is document
        assert document.name == "New Name"
        assert document.doc_metadata[BuiltInField.document_name] == "New Name"
        assert sqlite_session.get(UploadFile, upload_file.id).name == "New Name"
        assert commits == 0

    def test_rename_document_refreshes_disabled_built_ins_before_merging_metadata(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        dataset = _dataset_row(built_in_field_enabled=True)
        document = _document_row()
        document.doc_metadata = {BuiltInField.document_name: "Old", "custom": "old"}
        _persist_owner_chain(sqlite_session, dataset, document)

        # Keep the old ORM snapshots loaded while another metadata transaction
        # disables built-ins and publishes newer custom metadata.
        assert sqlite_session.get(Dataset, dataset.id) is dataset
        assert sqlite_session.get(Document, document.id) is document
        with sqlite_session_factory.begin() as concurrent:
            current_dataset = concurrent.get(Dataset, dataset.id)
            current_document = concurrent.get(Document, document.id)
            assert current_dataset is not None
            assert current_document is not None
            current_dataset.built_in_field_enabled = False
            current_document.doc_metadata = {"custom": "fresh"}

        with patch("services.knowledge.dataset_service.current_user", _account()):
            DocumentService.rename_document(dataset.id, document.id, "New Name", sqlite_session)
        sqlite_session.commit()

        with sqlite_session_factory() as verification:
            renamed = verification.get(Document, document.id)
            assert renamed is not None
            assert renamed.name == "New Name"
            assert renamed.doc_metadata == {"custom": "fresh"}

    def test_rename_document_merges_latest_custom_metadata(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        dataset = _dataset_row(built_in_field_enabled=True)
        document = _document_row()
        document.doc_metadata = {BuiltInField.document_name: "Old", "custom": "old"}
        _persist_owner_chain(sqlite_session, dataset, document)

        assert sqlite_session.get(Document, document.id) is document
        with sqlite_session_factory.begin() as concurrent:
            current_document = concurrent.get(Document, document.id)
            assert current_document is not None
            current_document.doc_metadata = {BuiltInField.document_name: "Old", "custom": "fresh"}

        with patch("services.knowledge.dataset_service.current_user", _account()):
            DocumentService.rename_document(dataset.id, document.id, "New Name", sqlite_session)
        sqlite_session.commit()

        with sqlite_session_factory() as verification:
            renamed = verification.get(Document, document.id)
            assert renamed is not None
            assert renamed.doc_metadata == {BuiltInField.document_name: "New Name", "custom": "fresh"}

    def test_recover_document_raises_when_document_is_not_paused(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(is_paused=False)
        _persist_owner_chain(sqlite_session, dataset, document)
        with pytest.raises(DocumentIndexingError):
            DocumentService.recover_document(document, sqlite_session)

    def test_recover_document_persists_and_dispatches(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(is_paused=True)
        _persist_owner_chain(sqlite_session, dataset, document)
        with (
            patch("services.knowledge.dataset_service.redis_client") as redis,
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent") as enqueue,
        ):
            DocumentService.recover_document(document, sqlite_session)

        sqlite_session.expire_all()
        recovered = sqlite_session.get(Document, document.id)
        assert recovered is not None
        assert recovered.is_paused is False
        redis.delete.assert_called_once_with(f"document_{document.id}_is_paused")
        enqueue.assert_called_once_with(recovered.indexing_request_id)
        assert _indexing_outbox_jobs(sqlite_session) == [(DocumentIndexingAction.RECOVER, (document.id,))]

    def test_recover_document_requeues_paused_retry_with_original_actor(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(is_paused=True, indexing_status=IndexingStatus.CLEANING)
        _persist_owner_chain(sqlite_session, dataset, document)
        (retry_intent_id,) = create_document_indexing_intents(
            sqlite_session,
            DocumentIndexingJobs(
                DatasetRef(dataset.tenant_id, dataset.id),
                retried=(document.id,),
                actor_id="retry-actor",
            ),
        )
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.redis_client"),
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent") as enqueue,
        ):
            DocumentService.recover_document(document, sqlite_session)

        sqlite_session.expire_all()
        recovered = sqlite_session.get_one(Document, document.id)
        assert recovered.is_paused is False
        assert recovered.indexing_status == IndexingStatus.WAITING
        assert recovered.indexing_request_id == retry_intent_id
        intent = sqlite_session.get_one(DocumentIndexingOutbox, retry_intent_id)
        assert intent.action == DocumentIndexingAction.RETRY.value
        assert intent.actor_id == "retry-actor"
        enqueue.assert_called_once_with(retry_intent_id)

    def test_retry_document_raises_when_retry_flag_is_already_set(self, sqlite_session: Session):
        document = _document_row(indexing_status=IndexingStatus.ERROR)
        sqlite_session.add(document)
        sqlite_session.commit()
        retry_flags = _RetryFlagStore({f"document_{document.id}_is_retried": "other-request"})
        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            patch("services.knowledge.dataset_service.redis_client", retry_flags),
        ):
            with pytest.raises(ValueError, match="being retried"):
                DocumentService.retry_document("dataset-1", [document], sqlite_session, actor_id="user-1")

    def test_retry_document_leaves_batch_unchanged_when_later_document_is_already_being_retried(
        self, sqlite_session: Session
    ):
        first_document = _document_row(document_id="doc-1", indexing_status=IndexingStatus.ERROR)
        second_document = _document_row(document_id="doc-2", indexing_status=IndexingStatus.ERROR)
        sqlite_session.add_all([first_document, second_document])
        sqlite_session.commit()
        first_retry_key = "document_doc-1_is_retried"
        second_retry_key = "document_doc-2_is_retried"
        retry_flags = _RetryFlagStore({second_retry_key: "other-request"})
        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            patch("services.knowledge.dataset_service.redis_client", retry_flags),
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent") as enqueue,
        ):
            with pytest.raises(ValueError, match="being retried"):
                DocumentService.retry_document(
                    "dataset-1", [first_document, second_document], sqlite_session, actor_id="user-1"
                )

        assert first_document.indexing_status == IndexingStatus.ERROR
        assert second_document.indexing_status == IndexingStatus.ERROR
        assert first_retry_key not in retry_flags.values
        assert retry_flags.values[second_retry_key] == "other-request"
        enqueue.assert_not_called()

    def test_retry_document_does_not_release_a_retry_flag_reacquired_by_another_request(self, sqlite_session: Session):
        first_retry_key = "document_doc-1_is_retried"
        second_retry_key = "document_doc-2_is_retried"
        retry_flags = _RetryFlagStore(
            {second_retry_key: "other-request"},
            replacement_on_conflict=(first_retry_key, "new-owner"),
        )
        documents = [
            _document_row(document_id="doc-1", indexing_status=IndexingStatus.ERROR),
            _document_row(document_id="doc-2", indexing_status=IndexingStatus.ERROR),
        ]
        sqlite_session.add_all(documents)
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            patch("services.knowledge.dataset_service.redis_client", retry_flags),
        ):
            with pytest.raises(ValueError, match="being retried"):
                DocumentService.retry_document("dataset-1", documents, sqlite_session, actor_id="user-1")

        assert retry_flags.values[first_retry_key] == "new-owner"
        assert retry_flags.values[second_retry_key] == "other-request"

    def test_retry_document_releases_flags_when_status_commit_fails(self, sqlite_session: Session):
        retry_flags = _RetryFlagStore()
        dataset = _dataset_row()
        document = _document_row(indexing_status=IndexingStatus.ERROR)
        _persist_owner_chain(sqlite_session, dataset, document)

        def fail_commit(_session):
            raise RuntimeError("database unavailable")

        event.listen(sqlite_session, "before_commit", fail_commit)
        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            patch("services.knowledge.dataset_service.redis_client", retry_flags),
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent") as enqueue,
        ):
            with pytest.raises(RuntimeError, match="database unavailable"):
                DocumentService.retry_document("dataset-1", [document], sqlite_session, actor_id="user-1")
        event.remove(sqlite_session, "before_commit", fail_commit)

        assert retry_flags.values == {}
        enqueue.assert_not_called()

    def test_retry_document_persists_status_and_dispatches(self, sqlite_session: Session):
        dataset = _dataset_row()
        documents = [
            _document_row(document_id="doc-1", indexing_status=IndexingStatus.ERROR),
            _document_row(document_id="doc-2", indexing_status=IndexingStatus.PAUSED),
        ]
        _persist_owner_chain(sqlite_session, dataset, *documents)
        retry_flags = _RetryFlagStore()
        with (
            patch("services.knowledge.dataset_service.current_user", _account()),
            patch("services.knowledge.dataset_service.redis_client", retry_flags),
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent") as enqueue,
        ):
            DocumentService.retry_document("dataset-1", documents, sqlite_session, actor_id="user-1")

        sqlite_session.expire_all()
        statuses = sqlite_session.scalars(select(Document.indexing_status).order_by(Document.id)).all()
        assert statuses == [IndexingStatus.WAITING, IndexingStatus.WAITING]
        intent = sqlite_session.scalar(select(DocumentIndexingOutbox))
        assert intent is not None
        assert intent.actor_id == "user-1"
        assert _indexing_outbox_jobs(sqlite_session) == [(DocumentIndexingAction.RETRY, ("doc-1", "doc-2"))]
        enqueue.assert_called_once_with(intent.id)

    def test_pipeline_retry_binds_each_document_to_its_exact_source_for_a_different_actor(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _dataset_row()
        dataset.runtime_mode = DatasetRuntimeMode.RAG_PIPELINE
        dataset.pipeline_id = "pipeline-1"
        pipeline, current_workflow = _published_pipeline_workflow(workflow_id="workflow-v2")
        documents = [
            _document_row(document_id="doc-1", indexing_status=IndexingStatus.ERROR),
            _document_row(document_id="doc-2", indexing_status=IndexingStatus.PAUSED),
        ]
        logs = [
            DocumentPipelineExecutionLog(
                pipeline_id="pipeline-1",
                document_id=document.id,
                datasource_type="local_file",
                datasource_info=json.dumps({"name": document.id}),
                datasource_node_id="source-v1",
                input_data={"document": document.id, "source": "v1"},
                created_by="original-actor",
                workflow_id="workflow-v1",
                original_document_id=document.id,
            )
            for document in documents
        ]
        sqlite_session.add_all(
            [
                make_tenant(tenant_id=dataset.tenant_id, name="Tenant"),
                pipeline,
                current_workflow,
                dataset,
                *documents,
                *logs,
            ]
        )
        sqlite_session.commit()
        retry_flags = _RetryFlagStore()
        published: list[str] = []
        monkeypatch.setattr("services.knowledge.dataset_service.redis_client", retry_flags)
        monkeypatch.setattr("services.knowledge.dataset_service.enqueue_document_indexing_intent", published.append)

        DocumentService.retry_document(dataset.id, documents, sqlite_session, actor_id="retry-actor")

        sqlite_session.expire_all()
        intents = sqlite_session.scalars(select(DocumentIndexingOutbox).order_by(DocumentIndexingOutbox.id)).all()
        assert len(intents) == 2
        assert {intent.actor_id for intent in intents} == {"retry-actor"}
        retry_logs = {
            log.document_id: log
            for log in sqlite_session.scalars(
                select(DocumentPipelineExecutionLog).where(
                    DocumentPipelineExecutionLog.id.in_(intent.pipeline_execution_log_id for intent in intents)
                )
            )
        }
        assert set(retry_logs) == {document.id for document in documents}
        assert {log.workflow_id for log in retry_logs.values()} == {current_workflow.id}
        assert {log.created_by for log in retry_logs.values()} == {"retry-actor"}
        assert {log.original_document_id for log in retry_logs.values()} == {document.id for document in documents}
        assert all(log.id not in {source.id for source in logs} for log in retry_logs.values())
        for source in logs:
            retry_log = retry_logs[source.document_id]
            assert retry_log.datasource_info == source.datasource_info
            assert retry_log.datasource_node_id == source.datasource_node_id
            assert retry_log.input_data == source.input_data
        assert set(published) == {intent.id for intent in intents}
        assert sqlite_session.scalars(select(Document.indexing_status).order_by(Document.id)).all() == [
            IndexingStatus.WAITING,
            IndexingStatus.WAITING,
        ]

        successor_workflow = _published_pipeline_workflow(workflow_id="workflow-v3")[1]
        pipeline.workflow_id = successor_workflow.id
        sqlite_session.add(successor_workflow)
        sqlite_session.commit()
        replay_store = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
        first_intent = intents[0]
        first_document_id = json.loads(first_intent.document_ids)[0]
        retry_log = retry_logs[first_document_id]
        replay = replay_store.get_pipeline_intent(
            first_intent.id,
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            pipeline_id=pipeline.id,
            document_id=first_document_id,
            actor_id="retry-actor",
            workflow_id=current_workflow.id,
            datasource_type=retry_log.datasource_type,
            datasource_info=json.loads(retry_log.datasource_info),
            datasource_node_id=retry_log.datasource_node_id,
            batch=documents[0].batch,
            original_document_id=first_document_id,
        )
        assert replay is not None
        assert replay.pipeline_execution_log_id == retry_log.id

    def test_pipeline_retry_replaces_an_unpinned_legacy_source_with_the_current_workflow(
        self,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _dataset_row()
        dataset.runtime_mode = DatasetRuntimeMode.RAG_PIPELINE
        dataset.pipeline_id = "pipeline-1"
        pipeline, current_workflow = _published_pipeline_workflow(workflow_id="workflow-v2")
        document = _document_row(indexing_status=IndexingStatus.ERROR)
        source_log = DocumentPipelineExecutionLog(
            pipeline_id="pipeline-1",
            document_id=document.id,
            datasource_type="local_file",
            datasource_info="{}",
            datasource_node_id="source-v1",
            input_data={},
            created_by="original-actor",
            workflow_id=None,
            original_document_id=None,
        )
        sqlite_session.add_all(
            [
                make_tenant(tenant_id=dataset.tenant_id, name="Tenant"),
                pipeline,
                current_workflow,
                dataset,
                document,
                source_log,
            ]
        )
        sqlite_session.commit()
        retry_flags = _RetryFlagStore()
        published: list[str] = []
        monkeypatch.setattr("services.knowledge.dataset_service.redis_client", retry_flags)
        monkeypatch.setattr("services.knowledge.dataset_service.enqueue_document_indexing_intent", published.append)

        DocumentService.retry_document(dataset.id, [document], sqlite_session, actor_id="retry-actor")

        sqlite_session.expire_all()
        persisted = sqlite_session.get_one(Document, document.id)
        assert persisted.indexing_status == IndexingStatus.WAITING
        intent = sqlite_session.scalar(select(DocumentIndexingOutbox))
        assert intent is not None
        retry_log = sqlite_session.get_one(DocumentPipelineExecutionLog, intent.pipeline_execution_log_id)
        assert retry_log.id != source_log.id
        assert retry_log.workflow_id == current_workflow.id
        assert retry_log.original_document_id is None
        assert retry_log.created_by == "retry-actor"
        assert source_log.workflow_id is None
        assert published == [intent.id]

    def test_retry_document_refreshes_a_stale_loaded_document_before_checking_claim(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        dataset = _dataset_row(data_source_type=DataSourceType.NOTION_IMPORT)
        document = _document_row(
            indexing_status=IndexingStatus.ERROR,
            data_source_type=DataSourceType.NOTION_IMPORT,
            data_source_info=json.dumps(
                {
                    "notion_workspace_id": "notion-workspace",
                    "notion_page_id": "notion-page",
                    "type": "page",
                    "last_edited_time": "2026-01-01T00:00:00Z",
                }
            ),
        )
        _persist_owner_chain(sqlite_session, dataset, document)
        assert document.indexing_claim_id is None

        repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
        claim = repository.claim_notion_sync(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id),
            claim_id="11111111-1111-1111-1111-111111111111",
        )
        assert claim is not None

        retry_flags = _RetryFlagStore()
        monkeypatch.setattr("services.knowledge.dataset_service.redis_client", retry_flags)

        with pytest.raises(DocumentIndexingError):
            DocumentService.retry_document(dataset.id, [document], sqlite_session, actor_id="user-1")

        assert retry_flags.values == {}
        sqlite_session.expire_all()
        persisted = sqlite_session.get_one(Document, document.id)
        assert persisted.indexing_status == IndexingStatus.ERROR
        assert persisted.indexing_claim_id == claim.claim_id

    def test_delete_document_rejects_an_active_notion_sync_claim(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        dataset = _dataset_row(data_source_type=DataSourceType.NOTION_IMPORT)
        document = _document_row(
            data_source_type=DataSourceType.NOTION_IMPORT,
            data_source_info=json.dumps(
                {
                    "notion_workspace_id": "notion-workspace",
                    "notion_page_id": "notion-page",
                    "type": "page",
                    "last_edited_time": "2026-01-01T00:00:00Z",
                }
            ),
        )
        _persist_owner_chain(sqlite_session, dataset, document)
        repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
        claim = repository.claim_notion_sync(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id),
            claim_id="22222222-2222-2222-2222-222222222222",
        )
        assert claim is not None

        with pytest.raises(DocumentIndexingError):
            DocumentService.delete_document(document, sqlite_session)

        sqlite_session.expire_all()
        assert sqlite_session.get(Document, document.id) is not None

    def test_retry_document_reclaims_an_expired_notion_sync_lease(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        dataset = _dataset_row(data_source_type=DataSourceType.NOTION_IMPORT)
        document = _document_row(
            indexing_status=IndexingStatus.ERROR,
            data_source_type=DataSourceType.NOTION_IMPORT,
            data_source_info=json.dumps(
                {
                    "notion_workspace_id": "notion-workspace",
                    "notion_page_id": "notion-page",
                    "type": "page",
                    "last_edited_time": "2026-01-01T00:00:00Z",
                }
            ),
        )
        _persist_owner_chain(sqlite_session, dataset, document)
        repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
        claim = repository.claim_notion_sync(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id),
            claim_id="44444444-4444-4444-4444-444444444444",
        )
        assert claim is not None
        with sqlite_session_factory.begin() as session:
            database_now = session.scalar(select(func.current_timestamp()))
            assert isinstance(database_now, datetime)
            current = session.get_one(Document, document.id)
            current.indexing_claim_expires_at = database_now - timedelta(seconds=1)

        retry_flags = _RetryFlagStore()
        published: list[str] = []
        monkeypatch.setattr("services.knowledge.dataset_service.redis_client", retry_flags)
        monkeypatch.setattr("services.knowledge.dataset_service.enqueue_document_indexing_intent", published.append)

        DocumentService.retry_document(dataset.id, [document], sqlite_session, actor_id="user-1")

        sqlite_session.expire_all()
        persisted = sqlite_session.get_one(Document, document.id)
        assert persisted.indexing_status == IndexingStatus.WAITING
        assert persisted.indexing_claim_id is None
        assert persisted.indexing_claim_expires_at is None
        assert published == [persisted.indexing_request_id]

    def test_delete_document_reclaims_an_expired_notion_sync_lease(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        dataset = _dataset_row(data_source_type=DataSourceType.NOTION_IMPORT)
        document = _document_row(
            data_source_type=DataSourceType.NOTION_IMPORT,
            data_source_info=json.dumps(
                {
                    "notion_workspace_id": "notion-workspace",
                    "notion_page_id": "notion-page",
                    "type": "page",
                    "last_edited_time": "2026-01-01T00:00:00Z",
                }
            ),
        )
        _persist_owner_chain(sqlite_session, dataset, document)
        repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
        claim = repository.claim_notion_sync(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id),
            claim_id="66666666-6666-6666-6666-666666666666",
        )
        assert claim is not None
        with sqlite_session_factory.begin() as session:
            database_now = session.scalar(select(func.current_timestamp()))
            assert isinstance(database_now, datetime)
            current = session.get_one(Document, document.id)
            current.indexing_claim_expires_at = database_now - timedelta(seconds=1)

        enqueued: list[str] = []
        monkeypatch.setattr("services.knowledge.dataset_service.enqueue_resource_cleanup_intent", enqueued.append)

        DocumentService.delete_document(document, sqlite_session)

        assert sqlite_session.get(Document, document.id) is None
        intent = sqlite_session.scalar(select(ResourceCleanupOutbox))
        assert intent is not None
        assert intent.resource_id == document.id
        assert enqueued == [intent.id]

    def test_sync_website_document_rejects_current_durable_sync_request(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(data_source_type=DataSourceType.WEBSITE_CRAWL)
        _persist_owner_chain(sqlite_session, dataset, document)
        (intent_id,) = create_document_indexing_intents(
            sqlite_session,
            DocumentIndexingJobs(DatasetRef(dataset.tenant_id, dataset.id), website_sync=document.id),
        )
        sqlite_session.commit()
        with patch("services.knowledge.dataset_service.redis_client") as mock_redis:
            mock_redis.get.return_value = intent_id

            with pytest.raises(ValueError, match="being synced"):
                DocumentService.sync_website_document(dataset, document, sqlite_session)

        mock_redis.get.assert_not_called()

    @pytest.mark.parametrize(
        ("document_dataset_id", "document_tenant_id"),
        [("dataset-2", "tenant-1"), ("dataset-1", "tenant-2")],
    )
    def test_sync_website_document_rejects_document_outside_dataset(
        self, document_dataset_id: str, document_tenant_id: str, sqlite_session: Session
    ):
        dataset = _dataset_row()
        document = _document_row(dataset_id=document_dataset_id, tenant_id=document_tenant_id)

        with (
            pytest.raises(ValueError, match="Document not found"),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
        ):
            DocumentService.sync_website_document(dataset, document, sqlite_session)

        mock_redis.get.assert_not_called()

    def test_sync_website_document_persists_generation_and_enqueues_intent(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(
            data_source_type=DataSourceType.WEBSITE_CRAWL,
            data_source_info=json.dumps({"mode": "crawl"}),
        )
        _persist_owner_chain(sqlite_session, dataset, document)

        with (
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent") as enqueue,
        ):
            mock_redis.get.return_value = None

            DocumentService.sync_website_document(dataset, document, sqlite_session)

        sqlite_session.expire_all()
        synced = sqlite_session.get(Document, document.id)
        assert synced is not None
        assert synced.indexing_status == IndexingStatus.WAITING
        assert synced.data_source_info_dict["mode"] == "scrape"
        [(action, document_ids)] = _indexing_outbox_jobs(sqlite_session)
        assert action == DocumentIndexingAction.WEBSITE_SYNC
        assert document_ids == (document.id,)
        intent = sqlite_session.scalar(select(DocumentIndexingOutbox))
        assert intent is not None
        assert synced.indexing_request_id == intent.id
        mock_redis.setex.assert_called_once_with(f"document_{document.id}_is_sync", 600, intent.id)
        enqueue.assert_called_once_with(intent.id)

    def test_sync_website_document_ignores_legacy_hint_without_durable_owner(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(data_source_type=DataSourceType.WEBSITE_CRAWL)
        _persist_owner_chain(sqlite_session, dataset, document)

        with (
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent"),
        ):
            mock_redis.get.return_value = "1"
            DocumentService.sync_website_document(dataset, document, sqlite_session)

        intent = sqlite_session.scalar(select(DocumentIndexingOutbox))
        assert intent is not None
        mock_redis.get.assert_not_called()
        mock_redis.setex.assert_called_once_with(f"document_{document.id}_is_sync", 600, intent.id)

    def test_sync_website_document_removes_hint_when_task_finishes_before_setex_reconciliation(
        self, sqlite_session: Session
    ):
        dataset = _dataset_row()
        document = _document_row(data_source_type=DataSourceType.WEBSITE_CRAWL)
        _persist_owner_chain(sqlite_session, dataset, document)
        cache: dict[str, str] = {}

        def finish_request_after_setex(key: str, _ttl: int, owner_id: str) -> None:
            cache[key] = owner_id
            stored = sqlite_session.get_one(Document, document.id)
            stored.indexing_request_id = None
            stored.indexing_claim_id = None
            stored.indexing_claim_expires_at = None
            outbox = sqlite_session.get_one(DocumentIndexingOutbox, owner_id)
            sqlite_session.delete(outbox)
            sqlite_session.commit()

        def register_script(_source: str) -> Callable[..., int]:
            def compare_and_delete(*, keys: list[str], args: list[str]) -> int:
                if cache.get(keys[0]) != args[0]:
                    return 0
                del cache[keys[0]]
                return 1

            return compare_and_delete

        class RedisStub:
            @staticmethod
            def get(_key: str) -> None:
                return None

            @staticmethod
            def setex(key: str, ttl: int, owner_id: str) -> None:
                finish_request_after_setex(key, ttl, owner_id)

            @staticmethod
            def register_script(source: str) -> Callable[..., int]:
                return register_script(source)

        with (
            patch("services.knowledge.dataset_service.redis_client", RedisStub()),
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent") as enqueue,
        ):
            DocumentService.sync_website_document(dataset, document, sqlite_session)

        assert cache == {}
        enqueue.assert_called_once()

    def test_sync_website_document_survives_initial_broker_failure(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(
            data_source_type=DataSourceType.WEBSITE_CRAWL,
            data_source_info=json.dumps({"mode": "crawl"}),
        )
        _persist_owner_chain(sqlite_session, dataset, document)

        with (
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch(
                "tasks.document_indexing_outbox_task.process_document_indexing_intent_task.delay",
                side_effect=OSError("broker unavailable"),
            ),
        ):
            mock_redis.get.return_value = None
            DocumentService.sync_website_document(dataset, document, sqlite_session)

        sqlite_session.expire_all()
        stored = sqlite_session.get(Document, document.id)
        intent = sqlite_session.scalar(select(DocumentIndexingOutbox))
        assert stored is not None
        assert intent is not None
        assert intent.action == DocumentIndexingAction.WEBSITE_SYNC.value
        assert stored.indexing_request_id == intent.id


class TestDocumentServiceSaveDocumentWithDatasetIdUpdate:
    """Integration of the production save owner with document updates."""

    def test_explicit_doc_language_reaches_the_index_processor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        tenant = make_tenant(tenant_id="tenant-1", name="Tenant")
        account = make_account(
            account_id="user-1",
            name="User",
            email="user-1@example.com",
            tenant=tenant,
            role=TenantAccountRole.OWNER,
        )
        join = TenantAccountJoin(
            tenant_id=tenant.id,
            account_id=account.id,
            role=TenantAccountRole.OWNER,
            current=True,
        )
        dataset = _dataset_row(data_source_type=DataSourceType.UPLOAD_FILE)
        document = _document_row(
            document_id="doc-1",
            data_source_info='{"upload_file_id": "file-1"}',
        )
        document.doc_language = "English"
        process_rule = _process_rule()
        document.dataset_process_rule_id = process_rule.id
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=1,
            content="content",
            word_count=1,
            tokens=1,
            created_by=account.id,
        )
        with sqlite_session_factory.begin() as session:
            session.add_all([tenant, account, join, dataset, process_rule, document, segment])

        document_data = KnowledgeConfig.model_validate(
            {
                "original_document_id": document.id,
                "indexing_technique": "economy",
                "process_rule": {"mode": "automatic"},
                "doc_form": IndexStructureType.PARAGRAPH_INDEX,
                "doc_language": "Japanese",
            }
        )
        dispatched: list[object] = []
        monkeypatch.setattr(DatasetService, "check_dataset_model_setting", lambda _dataset: None)
        monkeypatch.setattr(DocumentService, "dispatch_document_indexing", dispatched.append)
        with sqlite_session_factory() as session:
            with DocumentService.save_document_with_dataset_id(
                dataset,
                document_data,
                account,
                session=session,
            ) as (documents, _batch):
                updated = documents[0]
                assert updated.doc_language == "Japanese"

        documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
        ref = DatasetRef(dataset.tenant_id, dataset.id).document(document.id)
        with sqlite_session_factory() as session:
            persisted = session.get_one(Document, document.id)
            persisted_segment = session.get_one(DocumentSegment, segment.id)
            assert persisted.doc_language == "Japanese"
            assert persisted.indexing_status == IndexingStatus.WAITING
            assert persisted_segment.status == "re_segment"
            indexing_request_id = persisted.indexing_request_id
            assert indexing_request_id is not None
        assert len(dispatched) == 1

        (claim,) = documents.claim_indexing_request(
            indexing_request_id,
            expected_action=DocumentIndexingAction.UPDATE.value,
            processing_started_at=datetime(2026, 10, 24),
            admission_error=lambda _count: None,
        )
        indexing_document = documents.get_indexing_document(ref, claim_id=claim.claim_id)
        assert indexing_document.doc_language == "Japanese"

        processor_languages: list[str] = []

        class RecordingProcessor:
            def transform(self, _texts: list[IndexDocument], _user: Account, **kwargs: object) -> list[IndexDocument]:
                language = kwargs["doc_language"]
                assert isinstance(language, str)
                processor_languages.append(language)
                return []

        class RecordingProcessorFactory:
            def __init__(self, _doc_form: str) -> None:
                pass

            def init_index_processor(self) -> RecordingProcessor:
                return RecordingProcessor()

        monkeypatch.setattr(
            "services.knowledge.indexing.adapters.execution.IndexProcessorFactory",
            RecordingProcessorFactory,
        )
        adapter = IndexingExecutionAdapter(
            session_factory=sqlite_session_factory,
            documents=documents,
            segments=SQLAlchemySegmentRepository(session_factory=sqlite_session_factory),
            uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory),
            attachment_vectors=SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory),
            sources=CompositeStoredSourceResolver(adapters={}),
        )

        transform_result = adapter.transform(indexing_document, [])
        assert transform_result.chunks == []
        assert transform_result.created_upload_file_ids == ()
        assert processor_languages == ["Japanese"]


class TestDocumentServiceUploadSourceSwapCleanup:
    """The file domain owns cleanup after a document source replacement."""

    @staticmethod
    def _persist_source_rows(
        session_factory: sessionmaker[Session],
        *,
        shared_old_file: bool = False,
        foreign_new_file: bool = False,
    ) -> None:
        tenant = make_tenant(tenant_id="tenant-1", name="Tenant")
        dataset = _dataset_row(data_source_type=DataSourceType.UPLOAD_FILE)
        document = _document_row(
            document_id="doc-1",
            data_source_info=json.dumps({"upload_file_id": "old-file"}),
        )
        new_file_tenant_id = "tenant-2" if foreign_new_file else tenant.id
        rows: list[object] = [
            tenant,
            dataset,
            document,
            _upload_file(file_id="old-file"),
            _upload_file(file_id="new-file", tenant_id=new_file_tenant_id),
        ]
        if foreign_new_file:
            rows.append(make_tenant(tenant_id="tenant-2", name="Foreign Tenant"))
        if shared_old_file:
            shared_dataset = _dataset_row(dataset_id="dataset-2", data_source_type=DataSourceType.UPLOAD_FILE)
            shared_document = _document_row(
                document_id="doc-2",
                dataset_id=shared_dataset.id,
                data_source_info=json.dumps({"upload_file_id": "old-file"}),
            )
            rows.extend((shared_dataset, shared_document))
        with session_factory.begin() as session:
            session.add_all(rows)

    @staticmethod
    def _replace_source(session: Session, *, file_id: str = "new-file") -> Document:
        dataset = session.get_one(Dataset, "dataset-1")
        return DocumentService.update_prepared_document(
            dataset,
            _make_upload_knowledge_config(original_document_id="doc-1", file_ids=[file_id]),
            _account(),
            session=session,
        )

    def test_replacing_source_persists_candidate_then_drainer_reclaims_unreferenced_old_row(
        self, sqlite_session_factory: sessionmaker[Session]
    ) -> None:
        self._persist_source_rows(sqlite_session_factory)

        with sqlite_session_factory.begin() as session:
            updated = self._replace_source(session)
            assert json.loads(updated.data_source_info or "{}")["upload_file_id"] == "new-file"

        with sqlite_session_factory() as session:
            assert session.get(UploadFile, "old-file") is not None
            assert session.get(UploadFile, "new-file") is not None
            cleanup = session.scalar(
                select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == "old-file")
            )
            assert cleanup is not None
            assert cleanup.tenant_id == "tenant-1"
            assert cleanup.key == "key-old-file"

        artifacts = SQLAlchemyKnowledgeUploadRepository(
            session_factory=sqlite_session_factory
        ).list_pending_cleanup_artifacts(limit=100)

        assert [artifact.id for artifact in artifacts] == ["old-file"]
        with sqlite_session_factory() as session:
            assert session.get(UploadFile, "old-file") is None

    def test_replacing_source_with_same_file_is_a_noop_for_cleanup(
        self, sqlite_session_factory: sessionmaker[Session]
    ) -> None:
        self._persist_source_rows(sqlite_session_factory)

        with sqlite_session_factory.begin() as session:
            self._replace_source(session, file_id="old-file")

        with sqlite_session_factory() as session:
            assert session.get(UploadFile, "old-file") is not None
            assert session.scalar(select(UploadFileCleanupOutbox)) is None

    def test_source_swap_rollback_restores_old_source_without_cleanup_intent(
        self, sqlite_session_factory: sessionmaker[Session]
    ) -> None:
        self._persist_source_rows(sqlite_session_factory)

        with sqlite_session_factory() as session:
            self._replace_source(session)
            session.rollback()

        with sqlite_session_factory() as session:
            document = session.get_one(Document, "doc-1")
            assert json.loads(document.data_source_info or "{}")["upload_file_id"] == "old-file"
            assert session.get(UploadFile, "old-file") is not None
            assert session.scalar(select(UploadFileCleanupOutbox)) is None

    def test_source_swap_rejects_cross_tenant_new_file_without_cleanup(
        self, sqlite_session_factory: sessionmaker[Session]
    ) -> None:
        self._persist_source_rows(sqlite_session_factory, foreign_new_file=True)

        with sqlite_session_factory() as session:
            with pytest.raises(FileNotExistsError):
                self._replace_source(session)
            session.rollback()

        with sqlite_session_factory() as session:
            document = session.get_one(Document, "doc-1")
            assert json.loads(document.data_source_info or "{}")["upload_file_id"] == "old-file"
            assert session.get(UploadFile, "old-file") is not None
            assert session.scalar(select(UploadFileCleanupOutbox)) is None

    def test_source_swap_preserves_old_file_still_used_by_another_dataset(
        self, sqlite_session_factory: sessionmaker[Session]
    ) -> None:
        self._persist_source_rows(sqlite_session_factory, shared_old_file=True)

        with sqlite_session_factory.begin() as session:
            self._replace_source(session)

        with sqlite_session_factory() as session:
            assert session.get(UploadFile, "old-file") is not None
            assert (
                session.scalar(
                    select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == "old-file")
                )
                is not None
            )
            shared_document = session.get_one(Document, "doc-2")
            assert json.loads(shared_document.data_source_info or "{}")["upload_file_id"] == "old-file"

        assert (
            SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory).list_pending_cleanup_artifacts(
                limit=100
            )
            == ()
        )
        with sqlite_session_factory() as session:
            assert session.get(UploadFile, "old-file") is not None
            assert (
                session.scalar(
                    select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == "old-file")
                )
                is not None
            )


class TestDocumentServiceCreateValidation:
    """Unit tests for document creation validation helpers."""

    def test_document_create_args_validate_requires_data_source_or_process_rule(self):
        knowledge_config = KnowledgeConfig(indexing_technique="economy")

        with pytest.raises(ValueError, match="Data source or Process rule is required"):
            DocumentService.document_create_args_validate(knowledge_config)

    def test_document_create_args_validate_delegates_to_sub_validators(self):
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(
                info_list=InfoList(
                    data_source_type="upload_file",
                    file_info_list=FileInfo(file_ids=["file-1"]),
                )
            ),
            process_rule=ProcessRule(mode="automatic"),
        )

        with (
            patch.object(DocumentService, "data_source_args_validate") as validate_data_source,
            patch.object(DocumentService, "process_rule_args_validate") as validate_process_rule,
        ):
            DocumentService.document_create_args_validate(knowledge_config)

        validate_data_source.assert_called_once_with(knowledge_config)
        validate_process_rule.assert_called_once_with(knowledge_config)

    @pytest.mark.parametrize(
        ("data_source_type", "message"),
        [
            ("upload_file", "File source info is required"),
            ("notion_import", "Notion source info is required"),
            ("website_crawl", "Website source info is required"),
        ],
    )
    def test_data_source_args_validate_requires_source_specific_info(self, data_source_type, message):
        info_list = InfoList.model_validate({"data_source_type": data_source_type})
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(info_list=info_list),
        )

        with pytest.raises(ValueError, match=message):
            DocumentService.data_source_args_validate(knowledge_config)

    def test_process_rule_args_validate_clears_rules_for_automatic_mode(self):
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(
                info_list=InfoList(
                    data_source_type="upload_file",
                    file_info_list=FileInfo(file_ids=["file-1"]),
                )
            ),
            process_rule=ProcessRule(
                mode="automatic",
                rules=Rule(
                    pre_processing_rules=[PreProcessingRule(id="remove_stopwords", enabled=True)],
                    segmentation=Segmentation(separator="\n", max_tokens=128),
                ),
            ),
        )

        DocumentService.process_rule_args_validate(knowledge_config)

        assert knowledge_config.process_rule is not None
        assert knowledge_config.process_rule.rules is None

    @pytest.mark.parametrize(
        "automatic_rules",
        [
            pytest.param("missing", id="missing"),
            pytest.param(None, id="none"),
            pytest.param({"pre_processing_rules": "invalid"}, id="invalid-mapping"),
            pytest.param([], id="list"),
        ],
    )
    def test_automatic_process_rule_ignores_any_rules_in_estimate_and_create(self, automatic_rules: object) -> None:
        raw_process_rule: dict[str, object] = {"mode": "automatic"}
        if automatic_rules != "missing":
            raw_process_rule["rules"] = automatic_rules

        preview_rule = normalize_process_rule(raw_process_rule)
        knowledge_config = KnowledgeConfig.model_validate(
            {"indexing_technique": "economy", "process_rule": raw_process_rule}
        )
        assert knowledge_config.process_rule is not None
        assert ("rules" in knowledge_config.process_rule.model_fields_set) is (automatic_rules != "missing")
        DocumentService.process_rule_args_validate(knowledge_config)

        assert preview_rule == {"mode": "automatic", "rules": {}}
        assert knowledge_config.process_rule.rules is None

    def test_process_rule_args_validate_deduplicates_rules_and_skips_max_tokens_for_full_doc_hierarchical(self):
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(
                info_list=InfoList(
                    data_source_type="upload_file",
                    file_info_list=FileInfo(file_ids=["file-1"]),
                )
            ),
            process_rule=ProcessRule(
                mode="hierarchical",
                rules=Rule(
                    pre_processing_rules=[
                        PreProcessingRule(id="remove_stopwords", enabled=True),
                        PreProcessingRule(id="remove_stopwords", enabled=False),
                    ],
                    segmentation=Segmentation(separator="\n", max_tokens=0),
                    parent_mode="full-doc",
                    subchunk_segmentation=Segmentation(separator="\n", max_tokens=128),
                ),
            ),
        )

        DocumentService.process_rule_args_validate(knowledge_config)

        assert knowledge_config.process_rule is not None
        assert knowledge_config.process_rule.rules is not None
        assert len(knowledge_config.process_rule.rules.pre_processing_rules) == 1
        assert knowledge_config.process_rule.rules.pre_processing_rules[0].enabled is False

    def test_process_rule_args_validate_hierarchical_defaults_parent_mode_to_paragraph(self):
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(
                info_list=InfoList(
                    data_source_type="upload_file",
                    file_info_list=FileInfo(file_ids=["file-1"]),
                )
            ),
            process_rule=ProcessRule(
                mode="hierarchical",
                rules=Rule(
                    pre_processing_rules=[
                        PreProcessingRule(id="remove_extra_spaces", enabled=True),
                    ],
                    segmentation=Segmentation(separator="\n", max_tokens=1024),
                    subchunk_segmentation=Segmentation(separator="\n", max_tokens=512),
                ),
            ),
        )

        DocumentService.process_rule_args_validate(knowledge_config)

        assert knowledge_config.process_rule is not None
        assert knowledge_config.process_rule.rules is not None
        assert knowledge_config.process_rule.rules.parent_mode == "paragraph"

    @pytest.mark.parametrize("pre_processing_rules", [pytest.param("missing", id="missing"), None])
    @pytest.mark.parametrize("mode", ["custom", "hierarchical"])
    def test_process_rule_validation_matches_estimate_for_optional_preprocessing_and_splitter_fields(
        self,
        mode: str,
        pre_processing_rules: object,
    ) -> None:
        rules: dict[str, object] = {
            "segmentation": {"delimiter": "||", "max_tokens": 128, "chunk_overlap": 37},
        }
        if mode == "hierarchical":
            rules["subchunk_segmentation"] = {
                "delimiter": "||",
                "max_tokens": 64,
                "chunk_overlap": 7,
            }
        if pre_processing_rules != "missing":
            rules["pre_processing_rules"] = pre_processing_rules
        raw_process_rule = {"mode": mode, "rules": rules}
        preview_rule = normalize_process_rule(raw_process_rule)
        process_rule = ProcessRule.model_validate(raw_process_rule)
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(
                info_list=InfoList(
                    data_source_type="upload_file",
                    file_info_list=FileInfo(file_ids=["file-1"]),
                )
            ),
            process_rule=process_rule,
        )

        DocumentService.process_rule_args_validate(knowledge_config)

        assert process_rule.rules is not None
        assert process_rule.rules.pre_processing_rules == []
        assert process_rule.rules.segmentation is not None
        assert process_rule.rules.segmentation.separator == "||"
        assert process_rule.rules.segmentation.chunk_overlap == 37
        created_rule = process_rule.model_dump(mode="json", exclude_none=True, exclude_unset=True)
        assert created_rule == preview_rule

    def test_missing_preprocessing_survives_create_json_and_matches_preview_transform(
        self, sqlite_session: Session
    ) -> None:
        raw_process_rule = {
            "mode": "custom",
            "rules": {
                "segmentation": {"delimiter": "||", "max_tokens": 50, "chunk_overlap": 37},
            },
        }
        preview_rule = normalize_process_rule(raw_process_rule)
        knowledge_config = KnowledgeConfig.model_validate(
            {"indexing_technique": "economy", "process_rule": raw_process_rule}
        )
        DocumentService.process_rule_args_validate(knowledge_config)
        assert knowledge_config.process_rule is not None
        assert knowledge_config.process_rule.rules is not None
        persisted_rules = json.loads(knowledge_config.process_rule.rules.model_dump_json())
        assert persisted_rules["pre_processing_rules"] == []
        assert persisted_rules["segmentation"] == {
            "separator": "||",
            "max_tokens": 50,
            "chunk_overlap": 37,
        }

        processor = IndexProcessorFactory("text_model").init_index_processor()
        content = ("alpha beta gamma delta " * 30) + "||" + ("epsilon zeta eta theta " * 30)

        def transform(process_rule: dict[str, object]) -> list[IndexDocument]:
            return processor.transform(
                [IndexDocument(page_content=content, metadata={})],
                None,
                process_rule=process_rule,
                tenant_id="tenant-1",
                doc_language="English",
                session=sqlite_session,
            )

        preview_chunks = transform(preview_rule)
        persisted_chunks = transform({"mode": "custom", "rules": persisted_rules})
        assert len(persisted_chunks) > 1
        assert [chunk.page_content for chunk in persisted_chunks] == [chunk.page_content for chunk in preview_chunks]

    @pytest.mark.parametrize(
        "raw_process_rule",
        [
            {"mode": "custom", "rules": None},
            {"mode": "custom", "rules": {"segmentation": None}},
            {"mode": "custom", "rules": {}},
            {
                "mode": "hierarchical",
                "rules": {"segmentation": {"separator": "\n", "max_tokens": 128}},
            },
            {
                "mode": "custom",
                "rules": {
                    "pre_processing_rules": "invalid",
                    "segmentation": {"separator": "\n", "max_tokens": 128},
                },
            },
        ],
    )
    def test_create_and_estimate_reject_the_same_invalid_process_rule(
        self, raw_process_rule: dict[str, object]
    ) -> None:
        with pytest.raises(EstimateValidationError):
            normalize_process_rule(raw_process_rule)
        try:
            process_rule = ProcessRule.model_validate(raw_process_rule)
        except ValueError:
            return
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(
                info_list=InfoList(
                    data_source_type="upload_file",
                    file_info_list=FileInfo(file_ids=["file-1"]),
                )
            ),
            process_rule=process_rule,
        )
        with pytest.raises(EstimateValidationError):
            DocumentService.process_rule_args_validate(knowledge_config)


class TestDocumentServiceSaveDocumentWithDatasetId:
    """Unit tests for non-SQL validation branches in save_document_with_dataset_id."""

    @pytest.fixture
    def account_context(self, sqlite_session: Session):
        account = _account()
        sqlite_session.add(make_tenant(tenant_id="tenant-1", name="Tenant"))
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.current_user", account),
            patch.object(DatasetService, "check_doc_form"),
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent"),
        ):
            yield account

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_response_materialization_failure_rolls_back_without_dispatch(
        self, account_context, sqlite_session: Session
    ):
        from controllers.service_api.dataset.document import _document_and_batch_response

        dataset = _dataset_row()
        sqlite_session.add(dataset)
        sqlite_session.commit()
        knowledge_config = _make_upload_knowledge_config(file_ids=["file-1"])
        document = _document_row(document_id="response-failure-document")
        jobs = DocumentIndexingJobs(DatasetRef(dataset.tenant_id, dataset.id), created=(document.id,))

        def save_prepared_documents(*_args, session: Session, **_kwargs):
            session.add(document)
            session.flush()
            return [document], "batch-1", jobs

        with (
            patch.object(DocumentService, "prepare_document_creation", return_value=knowledge_config),
            patch.object(DocumentService, "save_prepared_documents", side_effect=save_prepared_documents),
            patch.object(DocumentService, "dispatch_document_indexing") as dispatch,
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch(
                "controllers.service_api.dataset.document.load_document_detail",
                side_effect=RuntimeError("response failed"),
            ),
        ):
            mock_redis.lock.return_value = _make_lock_context()
            with pytest.raises(RuntimeError, match="response failed"):
                with DocumentService.save_document_with_dataset_id(
                    dataset,
                    knowledge_config,
                    account_context,
                    session=sqlite_session,
                ) as (documents, batch):
                    _document_and_batch_response(documents[0], batch, session=sqlite_session)

        sqlite_session.expire_all()
        assert sqlite_session.get(Document, document.id) is None
        dispatch.assert_not_called()

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
    def test_save_document_with_dataset_id_requires_file_info_for_upload_source(
        self, account_context, unbound_session: Session
    ):
        dataset = _dataset_row()
        knowledge_config = _make_upload_knowledge_config(file_ids=None)

        with patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()):
            with pytest.raises(ValueError, match="File source info is required"):
                with DocumentService.save_document_with_dataset_id(
                    dataset,
                    knowledge_config,
                    account_context,
                    session=unbound_session,
                ):
                    pass

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
    def test_save_document_with_dataset_id_blocks_batch_upload_for_sandbox_plan(
        self, account_context, unbound_session: Session
    ):
        dataset = _dataset_row()
        knowledge_config = _make_upload_knowledge_config(file_ids=["file-1", "file-2"])

        with (
            patch(
                "services.knowledge.dataset_service.FeatureService.get_features",
                return_value=_make_features(plan=CloudPlan.SANDBOX),
            ),
            patch.object(DocumentService, "check_documents_upload_quota") as check_quota,
        ):
            with pytest.raises(ValueError, match="does not support batch upload"):
                with DocumentService.save_document_with_dataset_id(
                    dataset,
                    knowledge_config,
                    account_context,
                    session=unbound_session,
                ):
                    pass

        check_quota.assert_not_called()

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
    def test_save_document_with_dataset_id_enforces_batch_upload_limit(
        self,
        account_context,
        unbound_session: Session,
        config_overrides: Callable[..., None],
    ):
        config_overrides(BATCH_UPLOAD_LIMIT=1)
        dataset = _dataset_row()
        knowledge_config = _make_upload_knowledge_config(file_ids=["file-1", "file-2"])

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch.object(DocumentService, "check_documents_upload_quota") as check_quota,
        ):
            with pytest.raises(ValueError, match="batch upload limit of 1"):
                with DocumentService.save_document_with_dataset_id(
                    dataset,
                    knowledge_config,
                    account_context,
                    session=unbound_session,
                ):
                    pass

        check_quota.assert_not_called()

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_updates_existing_document_and_data_source_type(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=None)
        sqlite_session.add(dataset)
        sqlite_session.commit()
        knowledge_config = _make_upload_knowledge_config(original_document_id="doc-1", file_ids=["file-1"])
        updated_document = _document_row(document_id="doc-1")
        updated_document.batch = "batch-existing"

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch.object(DocumentService, "update_prepared_document", return_value=updated_document) as update_document,
        ):
            with DocumentService.save_document_with_dataset_id(
                dataset,
                knowledge_config,
                account_context,
                session=sqlite_session,
            ) as (documents, batch):
                pass

        assert dataset.data_source_type == "upload_file"
        assert documents == [updated_document]
        assert batch == "batch-existing"
        update_document.assert_called_once_with(
            dataset,
            knowledge_config,
            account_context,
            website_job_ref=None,
            session=sqlite_session,
        )

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_requires_data_source_for_new_documents(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row()
        sqlite_session.add(dataset)
        sqlite_session.commit()
        knowledge_config = _make_upload_knowledge_config(data_source=None)

        with patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()):
            with pytest.raises(ValueError, match="Data source is required when creating new documents"):
                with DocumentService.save_document_with_dataset_id(
                    dataset,
                    knowledge_config,
                    account_context,
                    session=sqlite_session,
                ):
                    pass

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_requires_existing_process_rule_for_custom_mode(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row()
        sqlite_session.add_all([dataset, _upload_file(file_id="file-1")])
        sqlite_session.commit()
        knowledge_config = _make_upload_knowledge_config(
            file_ids=["file-1"],
            process_rule=ProcessRule(mode="custom"),
        )

        with patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()):
            with pytest.raises(ValueError, match="No process rule found"):
                with DocumentService.save_document_with_dataset_id(
                    dataset,
                    knowledge_config,
                    account_context,
                    session=sqlite_session,
                ):
                    pass

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_upload_file_creates_and_reindexes_documents(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=DataSourceType.UPLOAD_FILE)
        dataset_process_rule = _process_rule()
        knowledge_config = _make_upload_knowledge_config(file_ids=["file-1", "file-2"])
        duplicate_document = _document_row(document_id="doc-duplicate", name="existing.txt")
        upload_file_a = _upload_file(file_id="file-1", name="existing.txt")
        upload_file_b = _upload_file(file_id="file-2", name="new.txt")
        sqlite_session.add_all([dataset, dataset_process_rule, duplicate_document, upload_file_a, upload_file_b])
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch("services.knowledge.dataset_service.naive_utc_now", return_value=datetime(2026, 2, 1)),
            patch("services.knowledge.dataset_service.time.strftime", return_value="20260101010101"),
            patch("services.knowledge.dataset_service.secrets.randbelow", return_value=23),
        ):
            mock_redis.lock.return_value = _make_lock_context()
            with DocumentService.save_document_with_dataset_id(
                dataset,
                knowledge_config,
                account_context,
                dataset_process_rule=dataset_process_rule,
                session=sqlite_session,
            ) as (documents, batch):
                pass

        assert [document.name for document in documents] == ["existing.txt", "new.txt"]
        assert batch == "20260101010101100023"
        assert duplicate_document.dataset_process_rule_id == "rule-1"
        assert duplicate_document.updated_at == datetime(2026, 2, 1)
        assert duplicate_document.batch == batch
        assert duplicate_document.indexing_status == IndexingStatus.WAITING
        created_document = next(document for document in documents if document.name == "new.txt")
        sqlite_session.expire_all()
        assert sqlite_session.get(Document, created_document.id) is not None
        assert _indexing_outbox_jobs(sqlite_session) == [
            (DocumentIndexingAction.CREATE, (created_document.id,)),
            (DocumentIndexingAction.DUPLICATE, ("doc-duplicate",)),
        ]

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_notion_import_truncates_names_and_cleans_removed_pages(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=DataSourceType.NOTION_IMPORT)
        dataset_process_rule = _process_rule()
        notion_page_name = "a" * 300
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(
                info_list=InfoList(
                    data_source_type="notion_import",
                    notion_info_list=[
                        NotionInfo(
                            credential_id="credential-1",
                            workspace_id="workspace-1",
                            pages=[
                                NotionPage(page_id="page-keep", page_name="Keep page", type="page"),
                                NotionPage(
                                    page_id="page-new",
                                    page_name=notion_page_name,
                                    page_icon=NotionIcon(type="emoji", emoji="page"),
                                    type="page",
                                ),
                            ],
                        )
                    ],
                )
            ),
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            doc_language="English",
        )
        existing_keep = _document_row(
            document_id="doc-keep",
            data_source_type=DataSourceType.NOTION_IMPORT,
        )
        existing_keep.data_source_info = json.dumps({"notion_page_id": "page-keep"})
        existing_remove = _document_row(
            document_id="doc-remove",
            data_source_type=DataSourceType.NOTION_IMPORT,
        )
        existing_remove.data_source_info = json.dumps({"notion_page_id": "page-remove"})
        old_removal = DocumentIndexingOutbox(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            action=DocumentIndexingAction.REMOVE_NOTION.value,
            document_ids=json.dumps([existing_keep.id]),
        )
        sqlite_session.add_all([dataset, dataset_process_rule, existing_keep, existing_remove, old_removal])
        sqlite_session.flush()
        existing_keep.indexing_request_id = old_removal.id
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch("services.knowledge.dataset_service.uuid.uuid4", return_value="doc-new"),
        ):
            mock_redis.lock.return_value = _make_lock_context()
            with DocumentService.save_document_with_dataset_id(
                dataset,
                knowledge_config,
                account_context,
                dataset_process_rule=dataset_process_rule,
                session=sqlite_session,
            ) as (documents, _):
                pass

        created_document = next(document for document in documents if document.id == "doc-new")
        assert created_document in documents
        assert len(created_document.name) == 255
        assert sqlite_session.get(Document, created_document.id) is created_document
        assert _indexing_outbox_jobs(sqlite_session) == [
            (DocumentIndexingAction.REMOVE_NOTION, ("doc-keep",)),
            (DocumentIndexingAction.REMOVE_NOTION, ("doc-remove",)),
            (DocumentIndexingAction.CREATE, ("doc-new",)),
            (DocumentIndexingAction.DUPLICATE, ("doc-keep",)),
        ]
        duplicate_intent = sqlite_session.scalar(
            select(DocumentIndexingOutbox).where(
                DocumentIndexingOutbox.action == DocumentIndexingAction.DUPLICATE.value
            )
        )
        assert duplicate_intent is not None
        sqlite_session.refresh(existing_keep)
        assert existing_keep.indexing_status == IndexingStatus.WAITING
        assert existing_keep.indexing_request_id == duplicate_intent.id

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_website_crawl_truncates_long_urls(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=DataSourceType.WEBSITE_CRAWL)
        dataset_process_rule = _process_rule()
        sqlite_session.add_all([dataset, dataset_process_rule])
        sqlite_session.commit()
        long_url = "https://example.com/" + ("a" * 260)
        short_url = "https://example.com/short"
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(
                info_list=InfoList(
                    data_source_type="website_crawl",
                    website_info_list=WebsiteInfo(
                        provider="firecrawl",
                        job_id="job-1",
                        urls=[long_url, short_url],
                        only_main_content=True,
                    ),
                )
            ),
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            doc_language="English",
        )
        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
        ):
            mock_redis.lock.return_value = _make_lock_context()

            with DocumentService.save_document_with_dataset_id(
                dataset,
                knowledge_config,
                account_context,
                dataset_process_rule=dataset_process_rule,
                session=sqlite_session,
                website_job_ref=WebsiteJobRef(
                    tenant_id=dataset.tenant_id,
                    provider="firecrawl",
                    provider_instance_id=website_crawl_provider_instance_id(provider="firecrawl"),
                    job_id="job-1",
                ),
            ) as (documents, _):
                pass

        assert [document.name for document in documents] == [long_url[:200] + "...", short_url]
        assert all(
            document.data_source_info_dict["provider_instance_id"]
            == website_crawl_provider_instance_id(provider="firecrawl")
            for document in documents
        )
        assert sqlite_session.scalars(select(Document).order_by(Document.position)).all() == documents
        assert _indexing_outbox_jobs(sqlite_session) == [
            (DocumentIndexingAction.CREATE, tuple(document.id for document in documents))
        ]


class TestDocumentServiceBatchUpdateStatus:
    """Unit tests for batch_update_document_status orchestration and helper branches."""

    def test_prepare_disable_update_requires_completed_document(self):
        document = _document_row(indexing_status=IndexingStatus.WAITING)
        document.completed_at = None

        with pytest.raises(DocumentIndexingError, match="is not completed"):
            DocumentService._prepare_disable_update(document, user=_account(), now=datetime(2026, 2, 1))

    def test_prepare_archive_update_sets_durable_segment_reconciliation_for_enabled_document(self):
        document = _document_row(enabled=True, archived=False)

        result = DocumentService._prepare_archive_update(document, user=_account(), now=datetime(2026, 2, 1))

        assert result is not None
        assert result["updates"]["archived"] is True
        assert result["set_cache"] is False
        assert result["async_task"] is None
        assert result["segment_action"] == "disable"

    def test_prepare_unarchive_update_sets_async_task_for_enabled_document(self):
        document = _document_row(enabled=True, archived=True)

        result = DocumentService._prepare_unarchive_update(document, now=datetime(2026, 2, 1))

        assert result is not None
        assert result["updates"]["archived"] is False
        assert result["set_cache"] is False
        assert result["async_task"]["args"] == [document.id]

    @pytest.mark.parametrize(
        ("action", "archived"),
        [("enable", False), ("un_archive", True)],
        ids=["document-enable", "document-unarchive"],
    )
    def test_enable_commits_segment_intent_and_summary_revision_with_document_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        action: Literal["enable", "un_archive"],
        archived: bool,
    ) -> None:
        dataset = _dataset_row(indexing_technique="high_quality")
        document = _document_row(enabled=action == "un_archive", archived=archived)
        _persist_owner_chain(sqlite_session, dataset, document)
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=1,
            content="content",
            word_count=1,
            tokens=1,
            created_by="user-1",
            enabled=False,
            status=SegmentStatus.COMPLETED,
        )
        segment.id = "segment-1"
        summary = DocumentSegmentSummary(
            dataset_id=dataset.id,
            document_id=document.id,
            chunk_id=segment.id,
            summary_content="retained summary",
            summary_index_node_id="summary-node-1",
            status=SummaryStatus.COMPLETED,
            enabled=False,
            version=2,
        )
        summary.id = "summary-1"
        superseded_enable = SegmentStateOutbox(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            segment_id=segment.id,
            action="enable",
            actor_id="user-1",
            summary_id=summary.id,
            summary_version=1,
        )
        segment.enabled = True
        segment.indexing_request_id = superseded_enable.id
        sqlite_session.add_all([segment, summary, superseded_enable])
        sqlite_session.commit()
        redis = _DocumentStatusRedisBoundary()
        task = _DocumentEnableTaskBoundary()
        checked_datasets: list[str] = []

        def check_model(current: Dataset) -> None:
            checked_datasets.append(current.id)

        monkeypatch.setattr("services.knowledge.dataset_service.redis_client", redis)
        monkeypatch.setattr("services.knowledge.dataset_service.add_document_to_index_task", task)
        monkeypatch.setattr(DatasetService, "check_dataset_model_setting", check_model)

        DocumentService.batch_update_document_status(
            dataset,
            [document.id],
            action,
            _account(),
            sqlite_session,
        )

        assert task.document_ids == [document.id]
        assert checked_datasets == [dataset.id]
        assert redis.cached == []
        with sqlite_session_factory() as observer:
            current_document = observer.get_one(Document, document.id)
            current_segment = observer.get_one(DocumentSegment, segment.id)
            intent = observer.get_one(SegmentStateOutbox, current_segment.indexing_request_id)
            assert current_document.enabled is True
            assert current_document.archived is False
            assert current_segment.enabled is True
            assert intent.id != superseded_enable.id
            assert (intent.action, intent.summary_id, intent.summary_version) == (
                "enable",
                summary.id,
                summary.version,
            )
            assert observer.get(SegmentStateOutbox, superseded_enable.id) is None
            assert observer.get_one(DocumentSegmentSummary, summary.id).enabled is False

    def test_batch_update_document_status_rejects_indexing_documents(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(name="Busy document")
        _persist_owner_chain(sqlite_session, dataset, document)

        with patch("services.knowledge.dataset_service.redis_client") as mock_redis:
            mock_redis.get.return_value = "1"

            with pytest.raises(DocumentIndexingError, match="Busy document is being indexed"):
                DocumentService.batch_update_document_status(
                    dataset, [document.id], "archive", _account(), sqlite_session
                )

        sqlite_session.refresh(document)
        assert document.archived is False

    def test_batch_update_document_status_rolls_back_when_commit_fails(self, sqlite_session: Session):
        dataset = _dataset_row()
        document = _document_row(enabled=False)
        _persist_owner_chain(sqlite_session, dataset, document)

        def fail_commit(_session):
            raise RuntimeError("commit failed")

        event.listen(sqlite_session, "before_commit", fail_commit)
        with (
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
        ):
            mock_redis.get.return_value = None

            with pytest.raises(RuntimeError, match="commit failed"):
                DocumentService.batch_update_document_status(
                    dataset, [document.id], "enable", _account(), sqlite_session
                )
        event.remove(sqlite_session, "before_commit", fail_commit)

        sqlite_session.refresh(document)
        assert document.enabled is False

    @pytest.mark.parametrize(
        ("action", "enabled", "archived"),
        [("enable", False, False), ("un_archive", True, True)],
        ids=["enable", "unarchive"],
    )
    def test_batch_update_document_status_accepts_committed_intent_when_bridge_publish_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
        action: Literal["enable", "un_archive"],
        enabled: bool,
        archived: bool,
    ) -> None:
        dataset = _dataset_row()
        document = _document_row(enabled=enabled, archived=archived)
        _persist_owner_chain(sqlite_session, dataset, document)
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=1,
            content="content",
            word_count=1,
            tokens=1,
            created_by="user-1",
            enabled=False,
            status=SegmentStatus.COMPLETED,
        )
        segment.id = "segment-1"
        sqlite_session.add(segment)
        sqlite_session.commit()
        redis = _DocumentStatusRedisBoundary()
        bridge = _RejectingDocumentEnableTaskBoundary()
        monkeypatch.setattr("services.knowledge.dataset_service.redis_client", redis)
        monkeypatch.setattr("services.knowledge.dataset_service.add_document_to_index_task", bridge)

        DocumentService.batch_update_document_status(
            dataset,
            [document.id],
            action,
            _account(),
            sqlite_session,
        )

        assert bridge.document_ids == [document.id]
        assert redis.cached == []
        with sqlite_session_factory() as observer:
            current_document = observer.get_one(Document, document.id)
            current_segment = observer.get_one(DocumentSegment, segment.id)
            intent = observer.get_one(SegmentStateOutbox, current_segment.indexing_request_id)
            assert current_document.enabled is True
            assert current_document.archived is False
            assert current_segment.enabled is True
            assert intent.action == "enable"

        recovered: list[tuple[str, ...]] = []

        def publish(intent_ids: Sequence[str]) -> None:
            recovered.append(tuple(intent_ids))

        drain_segment_state_outbox(new_session=sqlite_session_factory, publish=publish)
        assert recovered == [(intent.id,)]

    def test_batch_update_document_status_rejects_a_current_claim_from_a_stale_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        dataset = _dataset_row(data_source_type=DataSourceType.NOTION_IMPORT)
        document = _document_row(
            data_source_type=DataSourceType.NOTION_IMPORT,
            data_source_info=json.dumps(
                {
                    "notion_workspace_id": "notion-workspace",
                    "notion_page_id": "notion-page",
                    "type": "page",
                    "last_edited_time": "2026-01-01T00:00:00Z",
                }
            ),
        )
        _persist_owner_chain(sqlite_session, dataset, document)
        repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
        claim = repository.claim_notion_sync(
            DatasetRef(dataset.tenant_id, dataset.id).document(document.id),
            claim_id="33333333-3333-3333-3333-333333333333",
        )
        assert claim is not None

        class EmptyIndexingCache:
            def get(self, _key: str) -> None:
                return None

        monkeypatch.setattr("services.knowledge.dataset_service.redis_client", EmptyIndexingCache())

        with pytest.raises(DocumentIndexingError):
            DocumentService.batch_update_document_status(
                dataset,
                [document.id],
                "archive",
                _account(),
                sqlite_session,
            )

        sqlite_session.expire_all()
        persisted = sqlite_session.get_one(Document, document.id)
        assert persisted.archived is False
        assert persisted.indexing_claim_id == claim.claim_id


class TestDocumentServiceTenantAndUpdateEdges:
    """Unit tests for tenant-count and update edge cases."""

    @pytest.fixture
    def account_context(self):
        account = _account()

        with patch("services.knowledge.dataset_service.current_user", account):
            yield account

    def test_get_tenant_documents_count_scopes_state_and_tenant(self, account_context, sqlite_session: Session):
        sqlite_session.add_all(
            [
                _document_row(document_id="one"),
                _document_row(document_id="two"),
                _document_row(document_id="disabled", enabled=False),
                _document_row(document_id="archived", archived=True),
                _document_row(document_id="unfinished", indexing_status=IndexingStatus.WAITING),
                _document_row(document_id="foreign", tenant_id="tenant-2"),
            ]
        )
        sqlite_session.commit()

        result = DocumentService.get_tenant_documents_count(sqlite_session)

        assert result == 2


class TestDocumentServiceSaveDocumentAdditionalBranches:
    """Additional unit tests for dataset bootstrap and process-rule branches."""

    @pytest.fixture
    def account_context(self, sqlite_session: Session):
        account = _account()
        sqlite_session.add(make_tenant(tenant_id="tenant-1", name="Tenant"))
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.current_user", account),
            patch.object(DatasetService, "check_doc_form"),
            patch("services.knowledge.dataset_service.enqueue_document_indexing_intent"),
        ):
            yield account

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_initializes_high_quality_dataset_from_default_embedding_model(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=None, indexing_technique=None)
        sqlite_session.add(dataset)
        sqlite_session.commit()
        knowledge_config = _make_upload_knowledge_config(original_document_id="doc-1", file_ids=["file-1"])
        knowledge_config.indexing_technique = "high_quality"
        knowledge_config.embedding_model = None
        knowledge_config.embedding_model_provider = None
        updated_document = _document_row(document_id="doc-1")
        updated_document.batch = "batch-existing"
        binding = DatasetCollectionBinding(
            provider_name="default-provider",
            model_name="default-embedding",
            type="dataset",
            collection_name="collection",
        )
        binding.id = "binding-1"

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls,
            patch(
                "services.knowledge.dataset_service.DatasetCollectionBindingService.get_dataset_collection_binding",
                return_value=binding,
            ) as get_binding,
            patch.object(DocumentService, "update_prepared_document", return_value=updated_document),
        ):
            default_model = object.__new__(ModelInstance)
            default_model.model_name = "default-embedding"
            default_model.provider = "default-provider"
            model_manager_cls.for_tenant.return_value.get_default_model_instance.return_value = default_model

            with DocumentService.save_document_with_dataset_id(
                dataset,
                knowledge_config,
                account_context,
                session=sqlite_session,
            ) as (documents, batch):
                pass

        assert documents == [updated_document]
        assert batch == "batch-existing"
        assert dataset.data_source_type == "upload_file"
        assert dataset.indexing_technique == "high_quality"
        assert dataset.embedding_model == "default-embedding"
        assert dataset.embedding_model_provider == "default-provider"
        assert dataset.collection_binding_id == "binding-1"
        assert dataset.retrieval_model == {
            "search_method": "semantic_search",
            "reranking_enable": False,
            "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
            "top_k": 4,
            "score_threshold_enabled": False,
        }
        get_binding.assert_called_once_with("default-provider", "default-embedding", sqlite_session)

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_uses_explicit_embedding_and_retrieval_model(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(indexing_technique=None)
        sqlite_session.add(dataset)
        sqlite_session.commit()
        knowledge_config = _make_upload_knowledge_config(original_document_id="doc-1", file_ids=["file-1"])
        knowledge_config.indexing_technique = "high_quality"
        knowledge_config.embedding_model = "explicit-model"
        knowledge_config.embedding_model_provider = "explicit-provider"
        knowledge_config.retrieval_model = RetrievalModel(
            search_method="semantic_search",
            reranking_enable=True,
            reranking_model=RerankingModel(
                reranking_provider_name="rerank-provider",
                reranking_model_name="rerank-model",
            ),
            top_k=7,
            score_threshold_enabled=True,
            score_threshold=0.3,
        )
        binding = DatasetCollectionBinding(
            provider_name="explicit-provider",
            model_name="explicit-model",
            type="dataset",
            collection_name="collection",
        )
        binding.id = "binding-2"
        updated_document = _document_row(document_id="doc-1")

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.ModelManager") as model_manager_cls,
            patch(
                "services.knowledge.dataset_service.DatasetCollectionBindingService.get_dataset_collection_binding",
                return_value=binding,
            ) as get_binding,
            patch.object(DocumentService, "update_prepared_document", return_value=updated_document),
        ):
            with DocumentService.save_document_with_dataset_id(
                dataset, knowledge_config, account_context, session=sqlite_session
            ):
                pass

        model_manager_cls.for_tenant.return_value.get_default_model_instance.assert_not_called()
        get_binding.assert_called_once_with("explicit-provider", "explicit-model", sqlite_session)
        assert dataset.embedding_model == "explicit-model"
        assert dataset.embedding_model_provider == "explicit-provider"
        assert dataset.retrieval_model == knowledge_config.retrieval_model.model_dump()

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_creates_custom_process_rule_for_new_upload_document(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=DataSourceType.UPLOAD_FILE)
        knowledge_config = _make_upload_knowledge_config(
            file_ids=["file-1"],
            process_rule=ProcessRule(
                mode="custom",
                rules=Rule(
                    pre_processing_rules=[PreProcessingRule(id="remove_stopwords", enabled=True)],
                    segmentation=Segmentation(separator="\n", max_tokens=128),
                ),
            ),
        )
        upload_file = _upload_file(file_id="file-1", name="file.txt")
        sqlite_session.add_all([dataset, upload_file])
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch("services.knowledge.dataset_service.time.strftime", return_value="20260101010101"),
            patch("services.knowledge.dataset_service.secrets.randbelow", return_value=23),
        ):
            mock_redis.lock.return_value = _make_lock_context()
            with DocumentService.save_document_with_dataset_id(
                dataset,
                knowledge_config,
                account_context,
                session=sqlite_session,
            ) as (documents, batch):
                pass

        assert len(documents) == 1
        created_document = documents[0]
        assert created_document.name == "file.txt"
        assert batch == "20260101010101100023"
        created_rule = sqlite_session.get(DatasetProcessRule, created_document.dataset_process_rule_id)
        assert created_rule is not None
        assert created_rule.mode == "custom"
        assert created_rule.rules == knowledge_config.process_rule.rules.model_dump_json()
        assert _indexing_outbox_jobs(sqlite_session) == [(DocumentIndexingAction.CREATE, (created_document.id,))]

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_creates_automatic_process_rule_for_new_upload_document(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=DataSourceType.UPLOAD_FILE)
        knowledge_config = _make_upload_knowledge_config(
            file_ids=["file-1"],
            process_rule=ProcessRule(mode="automatic"),
        )
        sqlite_session.add_all([dataset, _upload_file(file_id="file-1", name="file.txt")])
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch("services.knowledge.dataset_service.time.strftime", return_value="20260101010101"),
            patch("services.knowledge.dataset_service.secrets.randbelow", return_value=23),
        ):
            mock_redis.lock.return_value = _make_lock_context()
            with DocumentService.save_document_with_dataset_id(
                dataset,
                knowledge_config,
                account_context,
                session=sqlite_session,
            ) as (documents, _):
                pass

        created_rule = sqlite_session.get(DatasetProcessRule, documents[0].dataset_process_rule_id)
        assert created_rule is not None
        assert created_rule.mode == "automatic"
        assert created_rule.rules == json.dumps(DatasetProcessRule.AUTOMATIC_RULES)
        assert sqlite_session.get(Document, documents[0].id) is documents[0]

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_creates_fallback_automatic_process_rule_when_latest_is_missing(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=DataSourceType.UPLOAD_FILE)
        knowledge_config = _make_upload_knowledge_config(file_ids=["file-1"], process_rule=None)
        sqlite_session.add_all([dataset, _upload_file(file_id="file-1", name="file.txt")])
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch("services.knowledge.dataset_service.time.strftime", return_value="20260101010101"),
            patch("services.knowledge.dataset_service.secrets.randbelow", return_value=23),
        ):
            mock_redis.lock.return_value = _make_lock_context()
            with DocumentService.save_document_with_dataset_id(
                dataset,
                knowledge_config,
                account_context,
                session=sqlite_session,
            ) as (documents, _):
                pass

        created_rule = sqlite_session.get(DatasetProcessRule, documents[0].dataset_process_rule_id)
        assert created_rule is not None
        assert created_rule.mode == "automatic"
        assert created_rule.rules == json.dumps(DatasetProcessRule.AUTOMATIC_RULES)

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_raises_when_upload_file_lookup_is_incomplete(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=DataSourceType.UPLOAD_FILE)
        knowledge_config = _make_upload_knowledge_config(file_ids=["file-1", "file-2"])
        sqlite_session.add_all([dataset, _upload_file(file_id="file-1", name="file.txt")])
        sqlite_session.commit()

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
            patch("services.knowledge.dataset_service.time.strftime", return_value="20260101010101"),
            patch("services.knowledge.dataset_service.secrets.randbelow", return_value=23),
        ):
            mock_redis.lock.return_value = _make_lock_context()
            with pytest.raises(FileNotExistsError, match="One or more files not found"):
                with DocumentService.save_document_with_dataset_id(
                    dataset,
                    knowledge_config,
                    account_context,
                    session=sqlite_session,
                ):
                    pass

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_requires_notion_info_list_for_notion_import(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=DataSourceType.NOTION_IMPORT)
        process_rule = _process_rule()
        sqlite_session.add_all([dataset, process_rule])
        sqlite_session.commit()
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(info_list=InfoList(data_source_type="notion_import")),
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            doc_language="English",
        )

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
        ):
            mock_redis.lock.return_value = _make_lock_context()
            with pytest.raises(ValueError, match="No notion info list found"):
                with DocumentService.save_document_with_dataset_id(
                    dataset,
                    knowledge_config,
                    account_context,
                    dataset_process_rule=process_rule,
                    session=sqlite_session,
                ):
                    pass

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    def test_save_document_with_dataset_id_requires_website_info_list_for_website_crawl(
        self, account_context, sqlite_session: Session
    ):
        dataset = _dataset_row(data_source_type=DataSourceType.WEBSITE_CRAWL)
        process_rule = _process_rule()
        sqlite_session.add_all([dataset, process_rule])
        sqlite_session.commit()
        knowledge_config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(info_list=InfoList(data_source_type="website_crawl")),
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            doc_language="English",
        )

        with (
            patch("services.knowledge.dataset_service.FeatureService.get_features", return_value=_make_features()),
            patch("services.knowledge.dataset_service.redis_client") as mock_redis,
        ):
            mock_redis.lock.return_value = _make_lock_context()
            with pytest.raises(ValueError, match="No website info list found"):
                with DocumentService.save_document_with_dataset_id(
                    dataset,
                    knowledge_config,
                    account_context,
                    dataset_process_rule=process_rule,
                    session=sqlite_session,
                ):
                    pass

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.index_processor.constant.index_type import IndexStructureType
from models.account import Tenant
from models.dataset import Dataset, DatasetProcessRule, Document, DocumentSegment
from models.enums import (
    DataSourceType,
    DocumentCreatedFrom,
    IndexingStatus,
    ProcessRuleMode,
    SegmentStatus,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.errors import (
    DocumentIndexingClaimLostError,
    DocumentIsDeletedPausedError,
    DocumentIsPausedError,
)


def _dataset(dataset_id: str, workspace_id: str) -> Dataset:
    return Dataset(
        id=dataset_id,
        tenant_id=workspace_id,
        name=f"Dataset {dataset_id}",
        description="",
        provider="vendor",
        created_by="account-1",
        maintainer="account-1",
        data_source_type="notion_import",
        indexing_technique="economy",
    )


def _document(
    document_id: str,
    *,
    workspace_id: str = "workspace-1",
    dataset_id: str = "dataset-1",
    source_type: str = DataSourceType.NOTION_IMPORT,
    enabled: bool = True,
    archived: bool = False,
    status: str = IndexingStatus.COMPLETED,
    batch: str = "batch-1",
    doc_metadata: dict[str, object] | None = None,
) -> Document:
    source_info = (
        {"notion_page_id": f"page-{document_id}"}
        if source_type == DataSourceType.NOTION_IMPORT
        else {"upload_file_id": f"file-{document_id}"}
    )
    return Document(
        id=document_id,
        tenant_id=workspace_id,
        dataset_id=dataset_id,
        position=1,
        data_source_type=source_type,
        data_source_info=json.dumps(source_info),
        batch=batch,
        name=f"Document {document_id}",
        created_from=DocumentCreatedFrom.API,
        created_by="account-1",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        indexing_status=status,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        word_count=1,
        enabled=enabled,
        archived=archived,
        is_paused=False,
        doc_metadata=doc_metadata,
    )


def test_document_queries_enforce_the_complete_owner_chain(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _dataset("dataset-2", "workspace-2"),
                _document("document-1"),
                _document("document-2", workspace_id="workspace-2", dataset_id="dataset-2"),
            ]
        )

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")

    record = repository.get_estimate_document(document_ref)

    assert record is not None
    assert record.workspace_id == "workspace-1"
    assert record.data_source_info == {"notion_page_id": "page-document-1"}
    assert repository.get_estimate_document(DatasetRef("workspace-2", "dataset-1").document("document-1")) is None


def test_retrieval_document_batches_and_metadata_filters_validate_owners_and_state(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    document_ids = [
        "owned",
        "foreign",
        "broken-chain",
        "wrong-document-tenant",
        "disabled",
        "archived",
        "pending",
        "missing",
    ]
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _dataset("dataset-2", "workspace-2"),
                _document("owned"),
                _document("foreign", workspace_id="workspace-2", dataset_id="dataset-2"),
                _document("broken-chain", dataset_id="dataset-2"),
                _document("wrong-document-tenant", workspace_id="workspace-2"),
                _document("disabled", enabled=False),
                _document("archived", archived=True),
                _document("pending", status=IndexingStatus.INDEXING),
            ]
        )
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef("workspace-1", "dataset-1")
    documents = repository.get_retrieval_documents(ref, document_ids)
    assert set(documents) == {"owned"}
    assert documents["owned"].values["name"] == "Document owned"
    common_filter = repository.get_metadata_filter_document_ids(
        tenant_id=ref.tenant_id,
        dataset_ids=(ref.dataset_id,),
        conditions=None,
    )
    assert common_filter == {"dataset-1": ["owned"]}
    assert repository.get_hit_testing_document_filter(ref, {"conditions": None}) == common_filter["dataset-1"]
    for invalid_ref in [DatasetRef("workspace-2", "dataset-1"), DatasetRef("workspace-1", "dataset-2")]:
        assert repository.get_retrieval_documents(invalid_ref, document_ids) == {}
        assert repository.get_hit_testing_document_filter(invalid_ref, {"conditions": None}) == []
    assert repository.get_retrieval_documents(ref, []) == {}


def test_metadata_empty_filters_use_extracted_scalar_null_semantics(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    metadata_by_document: dict[str, dict[str, object]] = {
        "missing": {},
        "json-null": {"topic": None},
        "text": {"topic": "value"},
        "zero": {"topic": 0},
        "false": {"topic": False},
        "empty-string": {"topic": ""},
    }
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add_all(
            [_document(document_id, doc_metadata=metadata) for document_id, metadata in metadata_by_document.items()]
        )

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef("workspace-1", "dataset-1")

    def matching(operator: str) -> set[str]:
        return set(
            repository.get_hit_testing_document_filter(
                ref,
                {
                    "logical_operator": "and",
                    "conditions": [{"name": "topic", "comparison_operator": operator, "value": None}],
                },
            )
        )

    assert matching("empty") == {"missing", "json-null"}
    assert matching("not empty") == {"text", "zero", "false", "empty-string"}


def test_pipeline_document_store_owns_sessions_and_scopes_updates(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add(_document("document-1"))

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    assert repository.exists(workspace_id="workspace-1", dataset_id="dataset-1", document_id="document-1")
    assert not repository.exists(workspace_id="workspace-2", dataset_id="dataset-1", document_id="document-1")
    repository.mark_failed(
        workspace_id="workspace-2", dataset_id="dataset-1", document_id="document-1", error="foreign"
    )
    with sqlite_session_factory() as session:
        document = session.get(Document, "document-1")
        assert document is not None
        assert document.indexing_status == IndexingStatus.COMPLETED
    repository.mark_failed(workspace_id="workspace-1", dataset_id="dataset-1", document_id="document-1", error="failed")
    with sqlite_session_factory() as session:
        document = session.get(Document, "document-1")
        assert document is not None
        assert document.indexing_status == IndexingStatus.ERROR
        assert document.error == "failed"


def test_active_notion_sync_excludes_archived_while_binding_includes_it(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add_all(
            [
                _document("notion"),
                _document("upload", source_type=DataSourceType.UPLOAD_FILE),
                _document("disabled", enabled=False),
                _document("archived", archived=True),
            ]
        )

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    dataset_ref = DatasetRef("workspace-1", "dataset-1")

    assert repository.list_active_notion_refs(dataset_ref) == (dataset_ref.document("notion"),)
    assert repository.list_bound_notion_page_ids(dataset_ref) == frozenset({"page-notion", "page-archived"})


def test_document_records_normalize_invalid_mapping_data_and_ignore_invalid_notion_page_ids(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    invalid_json = _document("invalid-json")
    invalid_json.data_source_info = "{invalid"
    missing_info = _document("missing-info")
    missing_info.data_source_info = None
    invalid_page_id = _document("invalid-page-id")
    invalid_page_id.data_source_info = json.dumps({"notion_page_id": 42})
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add_all([invalid_json, missing_info, invalid_page_id])

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    dataset_ref = DatasetRef("workspace-1", "dataset-1")

    invalid_record = repository.get_estimate_document(dataset_ref.document("invalid-json"))
    missing_record = repository.get_estimate_document(dataset_ref.document("missing-info"))
    assert invalid_record is not None
    assert invalid_record.data_source_info == {}
    assert missing_record is not None
    assert missing_record.data_source_info is None
    assert repository.list_bound_notion_page_ids(dataset_ref) == frozenset()


def test_batch_query_is_scoped_and_keeps_explicit_state_semantics(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _dataset("dataset-2", "workspace-1"),
                _dataset("dataset-3", "workspace-2"),
            ]
        )
        session.add_all(
            [
                _document("available"),
                _document("waiting", status=IndexingStatus.WAITING),
                _document("disabled", enabled=False),
                _document("archived", archived=True),
                _document("other-batch", batch="batch-2"),
                _document("other-dataset", dataset_id="dataset-2"),
                _document("other-workspace", workspace_id="workspace-2", dataset_id="dataset-3"),
            ]
        )

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    dataset_ref = DatasetRef("workspace-1", "dataset-1")

    assert {record.id for record in repository.list_estimate_documents_by_batch(dataset_ref, "batch-1")} == {
        "available",
        "waiting",
        "disabled",
        "archived",
    }


@pytest.mark.parametrize(
    ("tenant", "dataset", "document"),
    [
        ("wrong", "dataset-1", "document-1"),
        ("workspace-1", "wrong", "document-1"),
        ("workspace-1", "dataset-1", "wrong"),
    ],
)
def test_indexing_state_writes_reject_wrong_owner_chain(
    sqlite_session_factory: sessionmaker[Session], tenant: str, dataset: str, document: str
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all([_dataset("dataset-1", "workspace-1"), _document("document-1")])
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(tenant, dataset).document(document)
    for operation in [
        lambda: repository.get_indexing_document(ref),
        lambda: repository.mark_splitting(ref),
        lambda: repository.complete_indexing(ref, tokens=9, latency=1),
    ]:
        with pytest.raises(DocumentIsDeletedPausedError):
            operation()
    repository.fail_indexing(ref, "wrong owner")
    with sqlite_session_factory() as session:
        persisted_document = session.get(Document, "document-1")
        assert persisted_document is not None
        assert persisted_document.error is None


def test_indexing_claim_is_conditional_and_duplicate_delivery_claims_nothing(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant = Tenant(name="Workspace")
    tenant.id = "workspace-1"
    started_at = datetime(2026, 1, 1, 12, 0)
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                tenant,
                _dataset("dataset-1", tenant.id),
                _dataset("dataset-2", tenant.id),
                _document("waiting", status=IndexingStatus.WAITING),
                _document("completed", status=IndexingStatus.COMPLETED),
                _document("parsing", status=IndexingStatus.PARSING),
                _document("foreign-dataset", dataset_id="dataset-2", status=IndexingStatus.WAITING),
            ]
        )
        paused = _document("paused", status=IndexingStatus.WAITING)
        paused.is_paused = True
        session.add(paused)

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    dataset_ref = DatasetRef(tenant.id, "dataset-1")
    requested_ids = ["completed", "waiting", "parsing", "paused", "waiting", "foreign-dataset"]
    admitted_counts: list[int] = []

    def admit(count: int) -> None:
        admitted_counts.append(count)

    claimed = repository.claim_waiting_for_indexing(
        dataset_ref,
        requested_ids,
        processing_started_at=started_at,
        admission_error=admit,
    )
    duplicate = repository.claim_waiting_for_indexing(
        dataset_ref,
        requested_ids,
        processing_started_at=started_at + timedelta(seconds=1),
        admission_error=admit,
    )

    assert claimed == (dataset_ref.document("waiting"),)
    assert duplicate == ()
    assert admitted_counts == [1]
    with sqlite_session_factory() as session:
        waiting = session.get_one(Document, "waiting")
        assert waiting.indexing_status == IndexingStatus.PARSING
        assert waiting.processing_started_at == started_at
        assert session.get_one(Document, "completed").indexing_status == IndexingStatus.COMPLETED
        assert session.get_one(Document, "parsing").indexing_status == IndexingStatus.PARSING
        assert session.get_one(Document, "paused").indexing_status == IndexingStatus.WAITING
        assert session.get_one(Document, "foreign-dataset").indexing_status == IndexingStatus.WAITING


def test_indexing_claim_rejects_only_locked_waiting_documents(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant = Tenant(name="Workspace")
    tenant.id = "workspace-1"
    rejected_at = datetime(2026, 1, 1, 12, 0, 0, 123456)
    dataset_ref = DatasetRef(tenant.id, "dataset-1")
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                tenant,
                _dataset(dataset_ref.dataset_id, tenant.id),
                _dataset("dataset-2", tenant.id),
                _document("waiting", status=IndexingStatus.WAITING),
                _document("completed", status=IndexingStatus.COMPLETED),
                _document("parsing", status=IndexingStatus.PARSING),
                _document("foreign-dataset", dataset_id="dataset-2", status=IndexingStatus.WAITING),
            ]
        )
        paused = _document("paused", status=IndexingStatus.WAITING)
        paused.is_paused = True
        session.add(paused)
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    admitted_counts: list[int] = []

    def reject(count: int) -> str:
        admitted_counts.append(count)
        return "quota exceeded"

    claimed = repository.claim_waiting_for_indexing(
        dataset_ref,
        ["completed", "waiting", "waiting", "parsing", "paused", "foreign-dataset"],
        processing_started_at=rejected_at,
        admission_error=reject,
    )

    assert claimed == ()
    assert admitted_counts == [1]
    with sqlite_session_factory() as session:
        waiting = session.get_one(Document, "waiting")
        assert waiting.indexing_status == IndexingStatus.ERROR
        assert waiting.error == "quota exceeded"
        assert waiting.stopped_at == rejected_at
        assert session.get_one(Document, "completed").indexing_status == IndexingStatus.COMPLETED
        assert session.get_one(Document, "parsing").indexing_status == IndexingStatus.PARSING
        assert session.get_one(Document, "paused").indexing_status == IndexingStatus.WAITING
        assert session.get_one(Document, "foreign-dataset").indexing_status == IndexingStatus.WAITING


def test_notion_sync_claim_fences_duplicate_and_stale_indexing_writes(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant = Tenant(name="Workspace")
    tenant.id = "workspace-1"
    document = _document("document-1")
    document.data_source_info = json.dumps(
        {
            "notion_workspace_id": "notion-workspace",
            "notion_page_id": "notion-page",
            "type": "page",
            "last_edited_time": "2026-01-01T00:00:00Z",
            "credential_id": "credential-1",
        }
    )
    old_segment = DocumentSegment(
        tenant_id=tenant.id,
        dataset_id="dataset-1",
        document_id=document.id,
        position=1,
        content="old",
        word_count=1,
        tokens=1,
        created_by="account-1",
        index_node_id="old-node",
        status=SegmentStatus.COMPLETED,
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, _dataset("dataset-1", tenant.id), document, old_segment])

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(tenant.id, "dataset-1").document(document.id)
    claim = repository.claim_notion_sync(ref, claim_id="claim-1")

    assert claim is not None
    assert repository.claim_notion_sync(ref, claim_id="claim-2") is None
    repository.fail_indexing(ref, "stale unclaimed failure")
    with sqlite_session_factory() as session:
        current = session.get_one(Document, document.id)
        assert current.indexing_status == IndexingStatus.COMPLETED
        assert current.indexing_claim_id == "claim-1"
        assert current.indexing_claim_expires_at is not None
        assert current.indexing_claim_expires_at.microsecond == 0

    repository.prepare_notion_sync(
        claim,
        processing_started_at=datetime(2026, 2, 1, 12, 0),
    )
    with pytest.raises(DocumentIndexingClaimLostError):
        repository.mark_splitting(ref, claim_id="claim-2")
    with sqlite_session_factory() as session:
        current = session.get_one(Document, document.id)
        assert current.indexing_status == IndexingStatus.PARSING
        assert current.indexing_claim_id == "claim-1"
        assert json.loads(current.data_source_info)["last_edited_time"] == "2026-01-01T00:00:00Z"
        assert session.query(DocumentSegment).count() == 1

    repository.complete_indexing(
        ref,
        claim_id="claim-1",
        tokens=3,
        latency=1,
        source_version="2026-02-01T00:00:00Z",
    )
    with sqlite_session_factory() as session:
        current = session.get_one(Document, document.id)
        assert current.indexing_status == IndexingStatus.COMPLETED
        assert current.indexing_claim_id is None
        assert current.indexing_claim_expires_at is None
        assert json.loads(current.data_source_info)["last_edited_time"] == "2026-02-01T00:00:00Z"


def test_notion_sync_claim_allows_missing_previous_edit_time_and_releases_without_state_change(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant = Tenant(name="Workspace")
    tenant.id = "workspace-1"
    document = _document("document-1", status=IndexingStatus.ERROR)
    document.processing_started_at = datetime(2025, 1, 1)
    document.data_source_info = json.dumps(
        {
            "notion_workspace_id": "notion-workspace",
            "notion_page_id": "notion-page",
            "type": "page",
        }
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, _dataset("dataset-1", tenant.id), document])
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    claim = repository.claim_notion_sync(
        DatasetRef(tenant.id, "dataset-1").document(document.id),
        claim_id="claim-1",
    )

    assert claim is not None
    assert claim.page_edited_time is None
    assert claim.recovery_required is True
    assert repository.release_notion_sync(claim)
    with sqlite_session_factory() as session:
        current = session.get_one(Document, document.id)
        assert current.indexing_status == IndexingStatus.ERROR
        assert current.processing_started_at == datetime(2025, 1, 1)
        assert current.indexing_claim_id is None
        assert current.indexing_claim_expires_at is None


def test_expired_notion_sync_lease_is_taken_over_and_fences_the_old_owner(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant = Tenant(name="Workspace")
    tenant.id = "workspace-1"
    document = _document("document-1")
    document.data_source_info = json.dumps(
        {
            "notion_workspace_id": "notion-workspace",
            "notion_page_id": "notion-page",
            "type": "page",
            "last_edited_time": "2026-01-01T00:00:00Z",
        }
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, _dataset("dataset-1", tenant.id), document])

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(tenant.id, "dataset-1").document(document.id)
    old_claim = repository.claim_notion_sync(ref, claim_id="old-claim")
    assert old_claim is not None

    with sqlite_session_factory.begin() as session:
        database_now = session.scalar(select(func.current_timestamp()))
        assert isinstance(database_now, datetime)
        current = session.get_one(Document, document.id)
        current.indexing_status = IndexingStatus.INDEXING
        current.indexing_claim_expires_at = database_now - timedelta(seconds=1)

    new_claim = repository.claim_notion_sync(ref, claim_id="new-claim")

    assert new_claim is not None
    assert new_claim.recovery_required is True
    assert repository.renew_notion_sync(old_claim) is False
    assert repository.release_notion_sync(old_claim) is False
    with pytest.raises(DocumentIndexingClaimLostError):
        repository.prepare_notion_sync(
            old_claim,
            processing_started_at=datetime(2026, 2, 1, 12, 0),
        )
    with pytest.raises(DocumentIndexingClaimLostError):
        repository.complete_indexing(
            ref,
            claim_id=old_claim.claim_id,
            tokens=1,
            latency=1,
            source_version="2026-02-01T00:00:00Z",
        )
    repository.fail_indexing(ref, "stale failure", claim_id=old_claim.claim_id)

    assert repository.renew_notion_sync(new_claim) is True
    with sqlite_session_factory() as session:
        current = session.get_one(Document, document.id)
        database_now = session.scalar(select(func.current_timestamp()))
        assert isinstance(database_now, datetime)
        assert current.indexing_status == IndexingStatus.INDEXING
        assert current.indexing_claim_id == new_claim.claim_id
        assert current.indexing_claim_expires_at is not None
        assert current.indexing_claim_expires_at > database_now
        assert current.error is None
        assert json.loads(current.data_source_info)["last_edited_time"] == "2026-01-01T00:00:00Z"


def test_notion_sync_can_reclaim_a_requestless_owner_after_parent_generation_loss(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant = Tenant(name="Workspace")
    tenant.id = "workspace-1"
    document = _document("document-1", status=IndexingStatus.COMPLETED)
    document.data_source_info = json.dumps(
        {
            "notion_workspace_id": "notion-workspace",
            "notion_page_id": "notion-page",
            "type": "page",
            "last_edited_time": "2026-01-01T00:00:00Z",
        }
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, _dataset("dataset-1", tenant.id), document])

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(tenant.id, "dataset-1").document(document.id)
    old_claim = repository.claim_notion_sync(ref, claim_id="old-claim")
    assert old_claim is not None
    repository.prepare_notion_sync(old_claim, processing_started_at=datetime(2026, 2, 1, 12, 0))
    with sqlite_session_factory.begin() as session:
        session.get_one(Dataset, "dataset-1").indexing_request_id = "dataset-generation-1"

    repository.release_dataset_generation_owner(
        ref,
        indexing_request_id=None,
        claim_id=old_claim.claim_id,
        dataset_generation_id=old_claim.dataset_generation_id,
    )
    successor = repository.claim_notion_sync(ref, claim_id="successor-claim")

    assert successor is not None
    assert successor.recovery_required
    assert successor.dataset_generation_id == "dataset-generation-1"
    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document.id)
        assert persisted.indexing_status == IndexingStatus.WAITING
        assert persisted.indexing_request_id is None
        assert persisted.indexing_claim_id == successor.claim_id
        assert persisted.indexing_dataset_generation_id == successor.dataset_generation_id


def test_failed_notion_sync_keeps_last_successful_version_for_retry(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant = Tenant(name="Workspace")
    tenant.id = "workspace-1"
    document = _document("document-1", status=IndexingStatus.COMPLETED)
    document.data_source_type = DataSourceType.NOTION_IMPORT
    document.data_source_info = json.dumps(
        {
            "notion_workspace_id": "notion-workspace",
            "notion_page_id": "notion-page",
            "type": "page",
            "last_edited_time": "2026-01-01T00:00:00Z",
        }
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, _dataset("dataset-1", tenant.id), document])

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef(tenant.id, "dataset-1").document(document.id)
    failed_claim = repository.claim_notion_sync(ref, claim_id="failed-claim")
    assert failed_claim is not None
    repository.prepare_notion_sync(failed_claim, processing_started_at=datetime(2026, 2, 1, 12, 0))
    repository.fail_indexing(ref, "vector write failed", claim_id=failed_claim.claim_id)

    retry_claim = repository.claim_notion_sync(ref, claim_id="retry-claim")

    assert retry_claim is not None
    assert retry_claim.page_edited_time == "2026-01-01T00:00:00Z"
    with sqlite_session_factory() as session:
        current = session.get_one(Document, document.id)
        assert current.indexing_status == IndexingStatus.ERROR
        assert json.loads(current.data_source_info)["last_edited_time"] == "2026-01-01T00:00:00Z"


def test_indexing_load_validates_dataset_owner_and_process_rule(sqlite_session_factory: sessionmaker[Session]) -> None:
    with sqlite_session_factory.begin() as session:
        document = _document("document-1")
        document.dataset_process_rule_id = "other-rule"
        rule = DatasetProcessRule(
            dataset_id="dataset-2", mode=ProcessRuleMode.AUTOMATIC, rules="{}", created_by="account-1"
        )
        rule.id = "other-rule"
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _dataset("dataset-2", "workspace-2"),
                document,
                rule,
            ]
        )
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    assert repository.get_indexing_document(ref).processing_rule == {}
    with sqlite_session_factory.begin() as session:
        persisted_dataset = session.get(Dataset, "dataset-1")
        assert persisted_dataset is not None
        persisted_dataset.tenant_id = "workspace-2"
    with pytest.raises(DocumentIsDeletedPausedError):
        repository.get_indexing_document(ref)


def test_indexing_transitions_commit_and_preserve_pause(sqlite_session_factory: sessionmaker[Session]) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all([_dataset("dataset-1", "workspace-1"), _document("document-1", status=IndexingStatus.PARSING)])
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    repository.mark_splitting(ref)
    with sqlite_session_factory() as session:
        persisted_document = session.get(Document, "document-1")
        assert persisted_document is not None
        assert persisted_document.parsing_completed_at is not None
    repository.complete_indexing(ref, tokens=37, latency=2)
    with sqlite_session_factory.begin() as session:
        row = session.get(Document, "document-1")
        assert row is not None
        assert (row.indexing_status, row.tokens, row.indexing_latency) == (IndexingStatus.COMPLETED, 37, 2)
        row.is_paused = True
    with pytest.raises(DocumentIsPausedError):
        repository.fail_indexing(ref, "late worker error")
    with sqlite_session_factory() as session:
        persisted_document = session.get(Document, "document-1")
        assert persisted_document is not None
        assert persisted_document.error is None

import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models.dataset import Document, DocumentPipelineExecutionLog
from models.enums import DataSourceType, IndexingStatus
from models.model import DocumentIndexingOutbox
from models.workflow import WorkflowType
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    SQLAlchemyDocumentIndexingOutboxRepository,
    create_document_indexing_intents,
    create_pipeline_document_indexing_intents,
    invalidate_notion_removal_intents,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef
from tests.unit_tests.model_factories import make_dataset, make_document, make_tenant, make_workflow


def _pipeline_execution_log(
    document: Document,
    *,
    log_id: str,
    workflow_id: str = "workflow-1",
    original_document_id: str | None = None,
    actor_id: str = "actor-1",
) -> DocumentPipelineExecutionLog:
    log = DocumentPipelineExecutionLog(
        pipeline_id="pipeline-1",
        document_id=document.id,
        datasource_type="local_file",
        datasource_info=json.dumps({"name": "file"}),
        datasource_node_id="start",
        input_data={"input": "value"},
        created_by=actor_id,
        workflow_id=workflow_id,
        original_document_id=original_document_id,
    )
    log.id = log_id
    return log


def _jobs() -> DocumentIndexingJobs:
    return DocumentIndexingJobs(
        DatasetRef("tenant-1", "dataset-1"),
        created=("created-1", "created-2"),
        duplicated=("duplicate-1",),
        removed_notion=("removed-1",),
        updated="updated-1",
        website_sync="website-1",
    )


def test_create_indexing_intents_rolls_back_with_document_writes(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    intent_ids = create_document_indexing_intents(sqlite_session, _jobs())
    assert len(intent_ids) == 5
    sqlite_session.rollback()

    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert all(repository.get_intent(intent_id) is None for intent_id in intent_ids)


def test_indexing_actions_are_independently_retryable(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        intent_ids = create_document_indexing_intents(session, _jobs())

    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    intents = [repository.get_intent(intent_id) for intent_id in intent_ids]
    assert [(intent.action, intent.document_ids) for intent in intents if intent is not None] == [
        (DocumentIndexingAction.REMOVE_NOTION, ("removed-1",)),
        (DocumentIndexingAction.CREATE, ("created-1", "created-2")),
        (DocumentIndexingAction.DUPLICATE, ("duplicate-1",)),
        (DocumentIndexingAction.UPDATE, ("updated-1",)),
        (DocumentIndexingAction.WEBSITE_SYNC, ("website-1",)),
    ]

    first = intents[0]
    assert first is not None
    repository.complete(first)
    assert repository.get_intent(first.id) is None
    assert all(repository.get_intent(intent_id) is not None for intent_id in intent_ids[1:])


def test_pending_intents_prioritize_new_work_before_attempted_failures(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        (first_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(DatasetRef("tenant-1", "dataset-1"), created=("document-1",)),
        )
    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    repository.mark_attempted(first_id)
    with sqlite_session_factory.begin() as session:
        (second_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(DatasetRef("tenant-1", "dataset-1"), created=("document-2",)),
        )

    assert repository.list_pending_intent_ids(limit=1) == (second_id,)
    with sqlite_session_factory() as session:
        rows = session.scalars(select(DocumentIndexingOutbox)).all()
        assert {row.id for row in rows} == {first_id, second_id}
        assert all(isinstance(json.loads(row.document_ids), list) for row in rows)


def test_indexing_intent_becomes_the_documents_current_request(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
    documents = [
        make_document(document_id=document_id, dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        for document_id in ("document-1", "document-2")
    ]
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id=dataset.tenant_id), dataset, *documents])
        (intent_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(
                DatasetRef(dataset.tenant_id, dataset.id),
                created=tuple(document.id for document in documents),
            ),
        )

    with sqlite_session_factory() as session:
        stored = session.scalars(select(Document).order_by(Document.id)).all()
        assert [document.indexing_request_id for document in stored] == [intent_id, intent_id]

    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    intent = repository.get_intent(intent_id)
    assert intent is not None
    assert repository.request_is_current(intent)
    assert not repository.has_active_worker(intent)


def test_reselection_invalidates_only_notion_removal_requests(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
    removed = make_document(document_id="removed", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
    created = make_document(document_id="created", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id=dataset.tenant_id), dataset, removed, created])
        (removal_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(DatasetRef(dataset.tenant_id, dataset.id), removed_notion=(removed.id,)),
        )
        (creation_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(DatasetRef(dataset.tenant_id, dataset.id), created=(created.id,)),
        )
        invalidated_ids = invalidate_notion_removal_intents(
            session,
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_ids=(removed.id, created.id),
        )
        assert invalidated_ids == (removed.id,)

    with sqlite_session_factory() as session:
        assert session.get(Document, removed.id).indexing_request_id is None  # type: ignore[union-attr]
        assert session.get(Document, removed.id).indexing_status == IndexingStatus.WAITING  # type: ignore[union-attr]
        assert session.get(Document, created.id).indexing_request_id == creation_id  # type: ignore[union-attr]

    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    removal = repository.get_intent(removal_id)
    assert removal is not None
    assert not repository.request_is_current(removal)


def test_recovery_and_retry_intents_persist_generation_and_retry_actor(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
    recovered = make_document(document_id="recovered", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
    retried = make_document(document_id="retried", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id=dataset.tenant_id), dataset, recovered, retried])
        recovery_id, retry_id = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(
                DatasetRef(dataset.tenant_id, dataset.id),
                recovered=recovered.id,
                retried=(retried.id,),
                actor_id="actor-1",
            ),
        )

    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    recovery = repository.get_intent(recovery_id)
    retry = repository.get_intent(retry_id)
    assert recovery is not None
    assert recovery.action == DocumentIndexingAction.RECOVER
    assert recovery.actor_id is None
    assert retry is not None
    assert retry.action == DocumentIndexingAction.RETRY
    assert retry.actor_id == "actor-1"
    with sqlite_session_factory() as session:
        recovered_row = session.get(Document, recovered.id)
        retried_row = session.get(Document, retried.id)
        assert recovered_row is not None
        assert retried_row is not None
        assert recovered_row.indexing_request_id == recovery_id
        assert retried_row.indexing_request_id == retry_id


def test_pipeline_requests_claim_each_document_and_recover_under_the_successor_generation(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1", pipeline_id="pipeline-1")
    dataset.indexing_request_id = "generation-1"
    documents = tuple(
        make_document(document_id=document_id, dataset_id=dataset.id, tenant_id=dataset.tenant_id)
        for document_id in ("document-1", "document-2")
    )
    for document in documents:
        document.indexing_status = IndexingStatus.COMPLETED
    execution_logs = tuple(
        _pipeline_execution_log(
            document,
            log_id=f"00000000-0000-0000-0000-00000000000{index}",
        )
        for index, document in enumerate(documents, start=1)
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id=dataset.tenant_id), dataset, *documents, *execution_logs])
        session.flush()
        requests = create_pipeline_document_indexing_intents(
            session,
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            pipeline_id="pipeline-1",
            pipeline_execution_log_ids={
                document.id: execution_log.id for document, execution_log in zip(documents, execution_logs, strict=True)
            },
            actor_id="actor-1",
        )

    assert set(requests) == {"document-1", "document-2"}
    assert len(set(requests.values())) == 2
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (first_claim,) = repository.claim_indexing_request(
        requests["document-1"],
        expected_action=DocumentIndexingAction.RETRY.value,
        processing_started_at=datetime.now(UTC).replace(tzinfo=None),
        admission_error=lambda _count: None,
    )
    assert first_claim.ref.document_id == "document-1"
    assert first_claim.dataset_generation_id == "generation-1"

    with sqlite_session_factory.begin() as session:
        session.get_one(type(dataset), dataset.id).indexing_request_id = "generation-2"
    repository.release_dataset_generation_owner(
        first_claim.ref,
        indexing_request_id=requests["document-1"],
        claim_id=first_claim.claim_id,
        dataset_generation_id=first_claim.dataset_generation_id,
    )
    (successor_claim,) = repository.claim_indexing_request(
        requests["document-1"],
        expected_action=DocumentIndexingAction.RETRY.value,
        processing_started_at=datetime.now(UTC).replace(tzinfo=None),
        admission_error=lambda _count: None,
    )

    assert successor_claim.recovery_required
    assert successor_claim.dataset_generation_id == "generation-2"
    with sqlite_session_factory() as session:
        untouched = session.get_one(Document, "document-2")
        assert untouched.indexing_status == IndexingStatus.WAITING
        assert untouched.indexing_request_id == requests["document-2"]
        assert untouched.indexing_claim_id is None


def test_pipeline_request_and_status_transition_roll_back_together(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1", pipeline_id="pipeline-1")
    document = make_document(document_id="document-1", dataset_id=dataset.id, tenant_id=dataset.tenant_id)
    document.indexing_status = IndexingStatus.COMPLETED
    execution_log = _pipeline_execution_log(
        document,
        log_id="00000000-0000-0000-0000-000000000001",
    )
    sqlite_session.add_all([make_tenant(tenant_id=dataset.tenant_id), dataset, document, execution_log])
    sqlite_session.commit()

    requests = create_pipeline_document_indexing_intents(
        sqlite_session,
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        pipeline_id="pipeline-1",
        pipeline_execution_log_ids={document.id: execution_log.id},
        actor_id="actor-1",
    )
    sqlite_session.rollback()

    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document.id)
        assert persisted.indexing_status == IndexingStatus.COMPLETED
        assert persisted.indexing_request_id is None
        assert session.get(DocumentIndexingOutbox, requests[document.id]) is None


def test_legacy_website_sync_bridge_reuses_one_waiting_intent(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1")
    document = make_document(
        document_id="document-1",
        dataset_id=dataset.id,
        tenant_id=dataset.tenant_id,
        data_source_type=DataSourceType.WEBSITE_CRAWL,
    )
    document.indexing_status = IndexingStatus.WAITING
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id=dataset.tenant_id), dataset, document])

    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    first = repository.get_or_create_legacy_website_sync_intent(
        dataset_id=dataset.id,
        document_id=document.id,
    )
    second = repository.get_or_create_legacy_website_sync_intent(
        dataset_id=dataset.id,
        document_id=document.id,
    )

    assert first is not None
    assert second == first
    assert first.action == DocumentIndexingAction.WEBSITE_SYNC
    assert first.document_ids == (document.id,)
    with sqlite_session_factory() as session:
        assert session.scalar(select(DocumentIndexingOutbox.id)) == first.id
        stored = session.get_one(Document, document.id)
        assert stored.indexing_request_id == first.id


def test_legacy_pipeline_delivery_bridges_only_its_exact_persisted_source(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1", pipeline_id="pipeline-1")
    document = make_document(
        document_id="document-1",
        dataset_id=dataset.id,
        tenant_id=dataset.tenant_id,
        batch="batch-1",
    )
    document.indexing_status = IndexingStatus.WAITING
    source_log = DocumentPipelineExecutionLog(
        pipeline_id="pipeline-1",
        document_id=document.id,
        datasource_type="local_file",
        datasource_info=json.dumps({"name": "file"}),
        datasource_node_id="start",
        input_data={"input": "value"},
        created_by="actor-1",
    )
    workflow = make_workflow(
        workflow_id="workflow-1",
        tenant_id=dataset.tenant_id,
        app_id="pipeline-1",
        workflow_type=WorkflowType.RAG_PIPELINE,
        version="v1",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id=dataset.tenant_id), dataset, document, source_log, workflow])

    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert (
        outbox.get_or_create_legacy_pipeline_intent(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            pipeline_id="pipeline-1",
            document_id=document.id,
            actor_id="actor-1",
            datasource_type="local_file",
            datasource_info={"name": "different"},
            datasource_node_id="start",
            batch="batch-2",
            workflow_id=workflow.id,
            original_document_id=None,
        )
        is None
    )
    with sqlite_session_factory() as session:
        assert session.scalars(select(DocumentIndexingOutbox)).all() == []
        assert session.get_one(Document, document.id).indexing_request_id is None

    intent = outbox.get_or_create_legacy_pipeline_intent(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        pipeline_id="pipeline-1",
        document_id=document.id,
        actor_id="actor-1",
        datasource_type="local_file",
        datasource_info={"name": "file"},
        datasource_node_id="start",
        batch="batch-1",
        workflow_id=workflow.id,
        original_document_id=None,
    )

    assert intent is not None
    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (claim,) = documents.claim_indexing_request(
        intent.id,
        expected_action=DocumentIndexingAction.RETRY.value,
        processing_started_at=datetime.now(UTC).replace(tzinfo=None),
        admission_error=lambda _count: None,
    )
    assert claim.ref == DatasetRef(dataset.tenant_id, dataset.id).document(document.id)
    assert (
        outbox.get_or_create_legacy_pipeline_intent(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            pipeline_id="pipeline-1",
            document_id=document.id,
            actor_id="actor-1",
            datasource_type="local_file",
            datasource_info={"name": "different"},
            datasource_node_id="start",
            batch="batch-2",
            workflow_id=workflow.id,
            original_document_id=None,
        )
        is None
    )
    with sqlite_session_factory() as session:
        rows = session.scalars(select(DocumentIndexingOutbox)).all()
        assert [row.id for row in rows] == [intent.id]
        assert rows[0].pipeline_execution_log_id == source_log.id
        persisted_log = session.get_one(DocumentPipelineExecutionLog, source_log.id)
        assert persisted_log.workflow_id == workflow.id
        assert persisted_log.original_document_id is None


def test_rolling_pipeline_request_pins_its_embedded_delivery_to_the_exact_source(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset(dataset_id="dataset-1", tenant_id="tenant-1", pipeline_id="pipeline-1")
    document = make_document(
        document_id="document-1",
        dataset_id=dataset.id,
        tenant_id=dataset.tenant_id,
        batch="batch-1",
    )
    document.indexing_status = IndexingStatus.WAITING
    source_log = DocumentPipelineExecutionLog(
        pipeline_id="pipeline-1",
        document_id=document.id,
        datasource_type="local_file",
        datasource_info=json.dumps({"name": "file"}),
        datasource_node_id="start",
        input_data={"content": "v1"},
        created_by="actor-1",
    )
    workflow = make_workflow(
        workflow_id="workflow-1",
        tenant_id=dataset.tenant_id,
        app_id="pipeline-1",
        workflow_type=WorkflowType.RAG_PIPELINE,
        version="v1",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id=dataset.tenant_id), dataset, document, source_log, workflow])
        session.flush()
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(
                DatasetRef(dataset.tenant_id, dataset.id),
                retried=(document.id,),
                actor_id="actor-1",
            ),
        )

    repository = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    rejected = repository.get_pipeline_intent(
        request_id,
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        pipeline_id="pipeline-1",
        document_id=document.id,
        actor_id="actor-1",
        workflow_id=workflow.id,
        datasource_type="local_file",
        datasource_info={"name": "wrong"},
        datasource_node_id="start",
        batch="batch-1",
        original_document_id=None,
    )
    assert rejected is None
    accepted = repository.get_pipeline_intent(
        request_id,
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        pipeline_id="pipeline-1",
        document_id=document.id,
        actor_id="actor-1",
        workflow_id=workflow.id,
        datasource_type="local_file",
        datasource_info={"name": "file"},
        datasource_node_id="start",
        batch="batch-1",
        original_document_id=None,
    )

    assert accepted is not None
    assert accepted.pipeline_execution_log_id == source_log.id
    with sqlite_session_factory() as session:
        row = session.get_one(DocumentIndexingOutbox, request_id)
        assert row.pipeline_execution_log_id == source_log.id
        persisted_log = session.get_one(DocumentPipelineExecutionLog, source_log.id)
        assert persisted_log.workflow_id == workflow.id
        assert persisted_log.original_document_id is None

from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Never, Protocol, cast
from unittest.mock import create_autospec, patch

import pytest
from sqlalchemy import event, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection
from sqlalchemy.orm import ORMExecuteState, Session, sessionmaker
from sqlalchemy.sql.dml import Update

import tasks.summary_vector_cleanup_task as summary_cleanup_task_module
from core.credit_usage import CreditUsageCreatedBy
from core.errors.error import LLMBadRequestError, ProviderTokenNotInitError
from core.model_context import get_credit_usage_metadata
from core.model_manager import ModelManager
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.index_processor.index_processor_base import SummaryIndexSettingDict
from core.rag.index_processor.processor.paragraph_index_processor import ParagraphIndexProcessor
from core.rag.models.document import ChildDocument
from core.rag.models.document import Document as IndexDocument
from extensions.storage.storage_type import StorageType
from graphon.model_runtime.entities.llm_entities import LLMResult, LLMUsage
from graphon.model_runtime.entities.message_entities import AssistantPromptMessage
from libs.helper import generate_text_hash
from libs.pagination import PaginatedResult
from machinery.context import RequestContext
from models.dataset import (
    ChildChunk,
    Dataset,
    Document,
    DocumentSegment,
    DocumentSegmentSummary,
    SegmentAttachmentBinding,
)
from models.enums import CreatorUserRole, DocumentCreatedFrom, IndexingStatus, SegmentStatus, SummaryStatus
from models.model import (
    AttachmentVectorCleanupOutbox,
    ChildVectorCleanupOutbox,
    DatasetIndexingOutbox,
    SegmentStateOutbox,
    SummaryVectorCleanupOutbox,
    UploadFile,
    UploadFileCleanupOutbox,
)
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.child_vector_cleanup_repository import SQLAlchemyChildVectorCleanupRepository
from repositories.knowledge.dataset_indexing_outbox_repository import create_dataset_indexing_intents
from repositories.knowledge.dataset_repository import SQLAlchemyDatasetRepository
from repositories.knowledge.segment_cleanup_repository import add_segment_attachment_bindings
from repositories.knowledge.segment_repository import SegmentIndexingSnapshot, SQLAlchemySegmentRepository
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from repositories.workspace.workspace_repository import WorkspaceRepository
from services.knowledge.dataset_access import DatasetAccessService
from services.knowledge.dataset_service import SegmentService
from services.knowledge.entities.datasets import DatasetIndexingJob, DatasetIndexingRepresentation
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef, SegmentRef
from services.knowledge.entities.segments import (
    ChildChunkUpdateArgs,
    ChildVectorGuard,
    ChildVectorPublication,
    ChildVectorRoute,
    SegmentCreateData,
    SegmentUpdateArgs,
)
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.segments.adapters import (
    CelerySegmentBatchImportDispatcher,
    CelerySegmentStateDispatcher,
    ModelManagerSegmentGuard,
    RedisSegmentIndexingState,
    SegmentPresentationAdapter,
    SQLAlchemySegmentStateTransitions,
)
from services.knowledge.segments.application import (
    ChildChunkIndexingApplicationError,
    ChildChunkState,
    DatasetSegmentApplicationService,
    PreparedChildPublication,
    SegmentBatchImportDispatcher,
    SegmentChildMutation,
    SegmentDatasetRecord,
    SegmentDetail,
    SegmentDocumentRecord,
    SegmentIndex,
    SegmentIndexingState,
    SegmentListFilter,
    SegmentModelGuard,
    SegmentModelProviderError,
    SegmentMutationScope,
    SegmentUpdateState,
    SegmentUploadFileNotFoundError,
)
from tasks.segment_state_outbox_task import (
    drain_segment_state_outbox,
    process_segment_state_intent,
    process_segment_state_intents,
)


@dataclass
class SegmentLimitsStub:
    SINGLE_CHUNK_ATTACHMENT_LIMIT: int = 10


def test_celery_dispatcher_preserves_task_argument_order() -> None:
    calls: list[tuple[str, ...]] = []

    def delay(*args: str) -> None:
        calls.append(args)

    dispatcher = CelerySegmentBatchImportDispatcher(delay=delay)
    dispatcher.dispatch(
        job_id="job-1",
        upload_file_id="file-1",
        dataset_id="dataset-1",
        document_id="document-1",
        workspace_id="workspace-1",
        actor_id="account-1",
    )

    assert calls == [("job-1", "file-1", "dataset-1", "document-1", "workspace-1", "account-1")]


def test_celery_segment_state_dispatcher_publishes_bounded_deduplicated_batches() -> None:
    calls: list[list[str]] = []
    dispatcher = CelerySegmentStateDispatcher(delay=calls.append)
    intent_ids = [f"intent-{position}" for position in range(201)]

    dispatcher.dispatch_many([*intent_ids, "intent-0"])

    assert [len(batch) for batch in calls] == [100, 100, 1]
    assert [intent_id for batch in calls for intent_id in batch] == intent_ids


def _dataset(dataset_id: str, workspace_id: str, *, permission: str = "partial_members") -> Dataset:
    return Dataset(
        id=dataset_id,
        tenant_id=workspace_id,
        name="Dataset",
        description="",
        provider="vendor",
        permission=permission,
        indexing_technique="economy",
        created_by="account-1",
        maintainer="maintainer-1",
    )


def _document(document_id: str, workspace_id: str, dataset_id: str) -> Document:
    return Document(
        id=document_id,
        tenant_id=workspace_id,
        dataset_id=dataset_id,
        position=1,
        data_source_type="upload_file",
        batch="batch-1",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by="account-1",
        doc_form="text_model",
        word_count=0,
    )


def _segment(segment_id: str, workspace_id: str, dataset_id: str, document_id: str) -> DocumentSegment:
    segment = DocumentSegment(
        tenant_id=workspace_id,
        dataset_id=dataset_id,
        document_id=document_id,
        position=1,
        content="content",
        word_count=7,
        tokens=2,
        created_by="account-1",
        status=SegmentStatus.COMPLETED,
    )
    segment.id = segment_id
    return segment


def _upload(file_id: str, workspace_id: str, name: str) -> UploadFile:
    upload = UploadFile(
        tenant_id=workspace_id,
        storage_type=StorageType.OPENDAL,
        key=f"key-{file_id}",
        name=name,
        size=1,
        extension="csv",
        mime_type="text/csv",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="account-1",
        created_at=datetime(2026, 1, 1),
        used=False,
    )
    upload.id = file_id
    return upload


def test_scope_lookup_enforces_owner_chain(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add(_document("document-1", "workspace-1", "dataset-1"))

    store = SQLAlchemyDatasetRepository(session_factory=sqlite_session_factory)
    scope = store.get_segment_scope(
        dataset_ref=DatasetRef("workspace-1", "dataset-1"),
        document_id="document-1",
        actor_id="account-1",
    )

    assert scope.dataset is not None
    assert scope.document is not None
    assert (
        store.get_segment_scope(
            dataset_ref=DatasetRef("workspace-2", "dataset-1"),
            document_id="document-1",
            actor_id="account-1",
        ).dataset
        is None
    )


def test_upload_lookup_is_tenant_scoped(sqlite_session_factory: sessionmaker[Session]) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _upload("file-1", "workspace-1", "one.csv"),
                _upload("file-2", "workspace-2", "two.csv"),
            ]
        )

    store = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)

    assert store.get_file_name(workspace_id="workspace-1", upload_file_id="file-1") == "one.csv"
    assert store.get_file_name(workspace_id="workspace-1", upload_file_id="file-2") is None
    assert store.get_existing_file_ids(
        workspace_id="workspace-1",
        file_ids=("file-1", "file-2", "missing", "file-1"),
    ) == frozenset({"file-1"})


def test_segment_lookup_rejects_segment_from_another_owner_chain(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _dataset("dataset-2", "workspace-2"),
                _document("document-1", "workspace-1", "dataset-1"),
                _document("document-2", "workspace-2", "dataset-2"),
                _segment("segment-1", "workspace-2", "dataset-2", "document-2"),
            ]
        )

    store = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    assert store.get_segment(DatasetRef("workspace-1", "dataset-1").document("document-1").segment("segment-2")) is None


def test_segment_delete_replaces_pending_state_with_durable_cleanup_snapshot(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    child_route = ChildVectorRoute.capture(
        tenant_id="workspace-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="child-collection-a",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        document = _document("document-1", "workspace-1", "dataset-1")
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.enabled = False
        segment.index_node_id = "node-1"
        old_intent = SegmentStateOutbox(
            tenant_id="workspace-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            action="disable",
        )
        segment.indexing_request_id = old_intent.id
        child = ChildChunk(
            tenant_id="workspace-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            position=1,
            content="child",
            word_count=1,
            created_by="account-1",
            index_node_id="child-node-1",
            vector_route=child_route.serialized,
            vector_route_hash=child_route.key,
        )
        session.add_all([dataset, document, segment, child, old_intent])

    store = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    intents = store.delete_segments(
        DatasetRef("workspace-1", "dataset-1").document("document-1"),
        ["segment-1"],
    )

    assert len(intents) == 1
    with sqlite_session_factory() as session:
        assert session.get(DocumentSegment, "segment-1") is None
        assert session.scalar(select(ChildChunk).where(ChildChunk.segment_id == "segment-1")) is None
        assert session.get(SegmentStateOutbox, old_intent.id) is None
        cleanup = session.get(SegmentStateOutbox, intents[0].id)
        assert cleanup is not None
        assert cleanup.action == "delete"
        assert cleanup.index_node_id == "node-1"
        assert cleanup.child_node_ids == "[]"
        assert cleanup.doc_form == "text_model"
        child_cleanup = session.scalar(
            select(ChildVectorCleanupOutbox).where(ChildVectorCleanupOutbox.node_id == "child-node-1")
        )
        assert child_cleanup is not None
        assert child_cleanup.vector_route_hash == child_route.key
        assert ChildVectorRoute.model_validate_json(child_cleanup.vector_route or "") == child_route


def test_segment_state_publish_failure_keeps_and_reuses_committed_intent(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _document("document-1", "workspace-1", "dataset-1"),
                _segment("segment-1", "workspace-1", "dataset-1", "document-1"),
            ]
        )
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    first = transitions.request(
        document_ref=document_ref,
        segment_ids=["segment-1"],
        action="disable",
        actor_id="account-1",
    )
    dispatcher = CelerySegmentStateDispatcher(delay=lambda _intent_id: (_ for _ in ()).throw(RuntimeError("broker")))
    with pytest.raises(RuntimeError, match="broker"):
        dispatcher.dispatch(first[0].id)

    with sqlite_session_factory() as session:
        segment = session.get(DocumentSegment, "segment-1")
        assert segment is not None
        assert not segment.enabled
        assert segment.indexing_request_id == first[0].id
        assert session.get(SegmentStateOutbox, first[0].id) is not None

    repeated = transitions.request(
        document_ref=document_ref,
        segment_ids=["segment-1"],
        action="disable",
        actor_id="account-1",
    )
    assert repeated == first


def test_legacy_reconcile_cannot_reverse_a_newer_opposite_generation(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _document("document-1", "workspace-1", "dataset-1"),
                _segment("segment-1", "workspace-1", "dataset-1", "document-1"),
            ]
        )
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    disable = transitions.request(
        document_ref=document_ref,
        segment_ids=["segment-1"],
        action="disable",
        actor_id="account-1",
    )
    stale_enable = transitions.request(
        document_ref=document_ref,
        segment_ids=["segment-1"],
        action="enable",
        actor_id="account-1",
        reconcile_settled=True,
    )
    assert stale_enable == ()
    with sqlite_session_factory() as session:
        segment = session.get(DocumentSegment, "segment-1")
        assert segment is not None
        assert not segment.enabled
        assert segment.indexing_request_id == disable[0].id


def test_segment_state_drainer_publishes_committed_rows(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        intent = SegmentStateOutbox(
            tenant_id="workspace-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            action="disable",
        )
        session.add(intent)
    published: list[Sequence[str]] = []
    drain_segment_state_outbox(new_session=sqlite_session_factory, publish=published.append)
    assert published == [[intent.id]]
    with sqlite_session_factory() as session:
        refreshed = session.get(SegmentStateOutbox, intent.id)
        assert refreshed is not None
        assert refreshed.last_attempted_at is not None


def test_segment_state_drainer_groups_compatible_rows(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        intents = [
            SegmentStateOutbox(
                tenant_id="workspace-1",
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="segment-1",
                action="disable",
            ),
            SegmentStateOutbox(
                tenant_id="workspace-1",
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="segment-2",
                action="disable",
            ),
            SegmentStateOutbox(
                tenant_id="workspace-1",
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="segment-3",
                action="enable",
            ),
        ]
        session.add_all(intents)
    published: list[Sequence[str]] = []

    drain_segment_state_outbox(new_session=sqlite_session_factory, publish=published.append)

    assert {frozenset(batch) for batch in published} == {
        frozenset([intents[0].id, intents[1].id]),
        frozenset([intents[2].id]),
    }
    with sqlite_session_factory() as session:
        assert all(session.get_one(SegmentStateOutbox, intent.id).last_attempted_at is not None for intent in intents)


def test_economy_segment_enable_keeps_retained_summary_disabled_without_vector_access(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        dataset.indexing_technique = IndexTechniqueType.ECONOMY
        dataset.embedding_model = None
        dataset.embedding_model_provider = None
        document = _document("document-1", "workspace-1", "dataset-1")
        document.indexing_status = IndexingStatus.COMPLETED
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.enabled = False
        segment.index_node_id = "node-1"
        summary = DocumentSegmentSummary(
            dataset_id=dataset.id,
            document_id=document.id,
            chunk_id=segment.id,
            summary_content="retained summary",
            summary_index_node_id="summary-node-1",
            status=SummaryStatus.COMPLETED,
            enabled=False,
        )
        session.add_all([dataset, document, segment, summary])
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    intent = transitions.request(
        document_ref=DatasetRef("workspace-1", "dataset-1").document("document-1"),
        segment_ids=["segment-1"],
        action="enable",
        actor_id="account-1",
    )[0]
    processor = create_autospec(ParagraphIndexProcessor, instance=True)
    with (
        patch("tasks.segment_state_outbox_task.IndexProcessorFactory") as processor_factory,
        patch("tasks.segment_state_outbox_task.Vector") as vector,
    ):
        processor_factory.return_value.init_index_processor.return_value = processor
        assert process_segment_state_intent(intent.id, new_session=sqlite_session_factory) is True
    processor.load.assert_called_once()
    vector.assert_not_called()
    with sqlite_session_factory() as session:
        persisted_segment = session.get_one(DocumentSegment, "segment-1")
        persisted_summary = session.get_one(DocumentSegmentSummary, summary.id)
        assert persisted_segment.indexing_request_id is None
        assert persisted_segment.enabled is True
        assert persisted_summary.enabled is False
        assert session.get(SegmentStateOutbox, intent.id) is None


def test_high_quality_segment_enable_batches_summary_vector_publication(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    old_summary_route = ChildVectorRoute.capture(
        tenant_id="workspace-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="summary-collection-a",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="summary",
    )
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        document = _document("document-1", "workspace-1", "dataset-1")
        document.indexing_status = IndexingStatus.COMPLETED
        session.add_all([dataset, document])
        for position in range(2):
            segment = _segment(f"segment-{position}", "workspace-1", "dataset-1", "document-1")
            segment.enabled = False
            segment.index_node_id = f"node-{position}"
            session.add_all(
                [
                    segment,
                    DocumentSegmentSummary(
                        dataset_id=dataset.id,
                        document_id=document.id,
                        chunk_id=segment.id,
                        summary_content=f"summary-{position}",
                        summary_index_node_id=f"summary-node-{position}",
                        vector_route=old_summary_route.serialized,
                        vector_route_hash=old_summary_route.key,
                        status=SummaryStatus.COMPLETED,
                        # Exact document-enable intents restore the vector even
                        # if a prior remove cleaned it before rolling back this flag.
                        enabled=position == 0,
                    ),
                ]
            )
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    intents = transitions.request(
        document_ref=DatasetRef("workspace-1", "dataset-1").document("document-1"),
        segment_ids=["segment-0", "segment-1"],
        action="enable",
        actor_id="account-1",
    )
    processor = create_autospec(ParagraphIndexProcessor, instance=True)
    with (
        patch("tasks.segment_state_outbox_task.IndexProcessorFactory") as processor_factory,
        patch("tasks.segment_state_outbox_task.Vector") as vector_factory,
        patch("services.knowledge.summaries.adapters.vector_for_route") as vector_for_route,
    ):
        processor_factory.return_value.init_index_processor.return_value = processor
        vector_factory.return_value.get_routing_target.return_value = ("qdrant", "summary-collection-b")
        vector_for_route.return_value = vector_factory.return_value
        results = process_segment_state_intents([intent.id for intent in intents], new_session=sqlite_session_factory)

    assert all(results.values())
    processor.load.assert_called_once()
    # Route publication, both locked phase checks, and each durable summary
    # writer resolve the same physical target independently.
    assert vector_factory.call_count == 5
    summary_documents = [
        document for call in vector_factory.return_value.add_texts.call_args_list for document in call.args[0]
    ]
    assert {document.metadata["original_chunk_id"] for document in summary_documents} == {
        "segment-0",
        "segment-1",
    }
    with sqlite_session_factory() as session:
        summaries = session.scalars(
            select(DocumentSegmentSummary).where(DocumentSegmentSummary.dataset_id == "dataset-1")
        ).all()
        assert len(summaries) == 2
        assert all(summary.enabled for summary in summaries)
        assert {summary.chunk_id: summary.version for summary in summaries} == {"segment-0": 1, "segment-1": 2}
        assert all(summary.vector_route_hash is not None for summary in summaries)
        assert all(summary.vector_route_hash != old_summary_route.key for summary in summaries)
        cleanups = session.scalars(select(SummaryVectorCleanupOutbox)).all()
        assert cleanups == []
    assert {
        node_id for call in vector_for_route.return_value.delete_by_ids.call_args_list for node_id in call.args[0]
    } == {"summary-node-0", "summary-node-1"}


def test_segment_state_worker_batches_current_intents_and_acks_stale_members_independently(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    segment_count = 1000
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        document = _document("document-1", "workspace-1", "dataset-1")
        document.indexing_status = IndexingStatus.COMPLETED
        session.add_all([dataset, document])
        for position in range(segment_count):
            segment = _segment(f"segment-{position:04d}", "workspace-1", "dataset-1", "document-1")
            segment.position = position + 1
            segment.index_node_id = f"node-{position:04d}"
            session.add(segment)
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    original = transitions.request(
        document_ref=DatasetRef("workspace-1", "dataset-1").document("document-1"),
        segment_ids=[f"segment-{position:04d}" for position in range(segment_count)],
        action="disable",
        actor_id="account-1",
    )
    requested_ids: list[str] = []
    with sqlite_session_factory.begin() as session:
        for position, original_intent in enumerate(original):
            requested_ids.append(original_intent.id)
            if position % 10:
                continue
            segment = session.get_one(DocumentSegment, f"segment-{position:04d}")
            current = SegmentStateOutbox(
                tenant_id=segment.tenant_id,
                dataset_id=segment.dataset_id,
                document_id=segment.document_id,
                segment_id=segment.id,
                action="disable",
                actor_id="account-1",
            )
            session.add(current)
            segment.indexing_request_id = current.id
            requested_ids.append(current.id)

    processor = create_autospec(ParagraphIndexProcessor, instance=True)
    with patch("tasks.segment_state_outbox_task.IndexProcessorFactory") as processor_factory:
        processor_factory.return_value.init_index_processor.return_value = processor
        results = process_segment_state_intents(requested_ids, new_session=sqlite_session_factory)

    assert sum(results.values()) == segment_count
    assert processor.clean.call_count <= 11
    cleaned_node_ids = {node_id for call in processor.clean.call_args_list for node_id in call.args[1]}
    assert cleaned_node_ids == {f"node-{position:04d}" for position in range(segment_count)}
    with sqlite_session_factory() as session:
        persisted = session.scalars(
            select(DocumentSegment).where(DocumentSegment.dataset_id == "dataset-1").order_by(DocumentSegment.id)
        ).all()
        assert len(persisted) == segment_count
        assert all(segment.indexing_request_id is None for segment in persisted)
        assert session.scalar(select(SegmentStateOutbox.id)) is None


def test_stale_segment_generation_cannot_override_later_state(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        document = _document("document-1", "workspace-1", "dataset-1")
        document.indexing_status = "completed"
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.enabled = False
        segment.index_node_id = "node-1"
        session.add_all([dataset, document, segment])
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    enable = transitions.request(
        document_ref=document_ref,
        segment_ids=["segment-1"],
        action="enable",
        actor_id="account-1",
    )
    disable = transitions.request(
        document_ref=document_ref,
        segment_ids=["segment-1"],
        action="disable",
        actor_id="account-1",
    )
    processor = create_autospec(ParagraphIndexProcessor, instance=True)
    with patch("tasks.segment_state_outbox_task.IndexProcessorFactory") as processor_factory:
        processor_factory.return_value.init_index_processor.return_value = processor
        assert process_segment_state_intent(enable[0].id, new_session=sqlite_session_factory) is False
        assert process_segment_state_intent(disable[0].id, new_session=sqlite_session_factory) is True
    processor.load.assert_not_called()
    processor.clean.assert_called_once()
    with sqlite_session_factory() as session:
        segment = session.get(DocumentSegment, "segment-1")
        assert segment is not None
        assert not segment.enabled
        assert segment.indexing_request_id is None
        assert session.scalar(select(SegmentStateOutbox.id)) is None


def test_stale_disable_generation_cannot_remove_later_enable(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        document = _document("document-1", "workspace-1", "dataset-1")
        document.indexing_status = "completed"
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.index_node_id = "node-1"
        session.add_all([dataset, document, segment])
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    disable = transitions.request(
        document_ref=document_ref,
        segment_ids=["segment-1"],
        action="disable",
        actor_id="account-1",
    )
    enable = transitions.request(
        document_ref=document_ref,
        segment_ids=["segment-1"],
        action="enable",
        actor_id="account-1",
    )
    processor = create_autospec(ParagraphIndexProcessor, instance=True)
    with patch("tasks.segment_state_outbox_task.IndexProcessorFactory") as processor_factory:
        processor_factory.return_value.init_index_processor.return_value = processor
        assert process_segment_state_intent(disable[0].id, new_session=sqlite_session_factory) is False
        assert process_segment_state_intent(enable[0].id, new_session=sqlite_session_factory) is True
    processor.clean.assert_not_called()
    processor.load.assert_called_once()


def test_segment_state_worker_acks_orphan_without_index_io(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        intent = SegmentStateOutbox(
            tenant_id="missing-tenant",
            dataset_id="missing-dataset",
            document_id="missing-document",
            segment_id="missing-segment",
            action="enable",
        )
        session.add(intent)
    with patch("tasks.segment_state_outbox_task.IndexProcessorFactory") as processor_factory:
        assert process_segment_state_intent(intent.id, new_session=sqlite_session_factory) is False
    processor_factory.assert_not_called()
    with sqlite_session_factory() as session:
        assert session.get(SegmentStateOutbox, intent.id) is None


def test_segment_state_worker_rejects_exact_token_mismatch_without_index_io(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        document = _document("document-1", "workspace-1", "dataset-1")
        document.indexing_status = "completed"
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.index_node_id = "node-1"
        segment.indexing_request_id = "newer-generation"
        stale = SegmentStateOutbox(
            tenant_id="workspace-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            action="enable",
        )
        session.add_all([dataset, document, segment, stale])
    with patch("tasks.segment_state_outbox_task.IndexProcessorFactory") as processor_factory:
        assert process_segment_state_intent(stale.id, new_session=sqlite_session_factory) is False
    processor_factory.assert_not_called()
    with sqlite_session_factory() as session:
        segment = session.get(DocumentSegment, "segment-1")
        assert segment is not None
        assert segment.indexing_request_id == "newer-generation"
        assert session.get(SegmentStateOutbox, stale.id) is None


@pytest.mark.parametrize("owner_field", ["tenant_id", "dataset_id", "document_id"])
def test_segment_update_state_enforces_complete_owner_chain(
    sqlite_session_factory: sessionmaker[Session], owner_field: str
) -> None:
    with sqlite_session_factory.begin() as session:
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        setattr(segment, owner_field, "foreign-owner")
        session.add(segment)
    store = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment("segment-1")

    assert store.get_segment_update_state(ref) is None


def test_empty_segment_list_returns_materialized_page(sqlite_session_factory: sessionmaker[Session]) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add(_document("document-1", "workspace-1", "dataset-1"))

    store = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    with patch("repositories.knowledge.segment_repository.dify_config") as config:
        config.SQLALCHEMY_DATABASE_URI_SCHEME = "mysql"
        result = store.list_segments(
            DatasetRef("workspace-1", "dataset-1").document("document-1"),
            SegmentListFilter(page=1, limit=20, keyword="missing"),
        )

    assert result.items == ()
    assert result.total == 0
    assert result.limit == 20


def test_postgresql_keyword_filter_keeps_json_array_guard(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add(_document("document-1", "workspace-1", "dataset-1"))

    page = PaginatedResult[DocumentSegment](items=[], total=0, page=1, per_page=20)
    store = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    with (
        patch("repositories.knowledge.segment_repository.dify_config") as config,
        patch("repositories.knowledge.segment_repository.paginate_query", return_value=page) as paginate,
    ):
        config.SQLALCHEMY_DATABASE_URI_SCHEME = "postgresql"
        store.list_segments(
            DatasetRef("workspace-1", "dataset-1").document("document-1"),
            SegmentListFilter(keyword="知识"),
        )

    statement = paginate.call_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "jsonb_typeof" in sql
    assert "jsonb_array_elements_text" in sql
    assert "CASE WHEN" in sql


def _application(
    factory: sessionmaker[Session],
    *,
    indexing: SegmentIndex | None = None,
    store: SQLAlchemySegmentRepository | None = None,
) -> DatasetSegmentApplicationService:
    datasets = SQLAlchemyDatasetRepository(session_factory=factory)
    index = create_autospec(SegmentIndex, instance=True, spec_set=True)
    index.count_tokens.return_value = 0
    state = create_autospec(SegmentIndexingState, instance=True, spec_set=True)
    state.is_segment_indexing.return_value = False
    state.is_document_indexing.return_value = False
    segments = store or SQLAlchemySegmentRepository(session_factory=factory)
    return DatasetSegmentApplicationService(
        dataset_access=DatasetAccessService(
            datasets=datasets,
            workspace_roles=WorkspaceRepository(session_factory=factory),
            legacy_permissions_enabled=False,
        ),
        scopes=datasets,
        store=segments,
        presenter=SegmentPresentationAdapter(),
        limits=SegmentLimitsStub(),
        text_hash=generate_text_hash,
        index=indexing or index,
        uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=factory),
        model_guard=create_autospec(SegmentModelGuard, instance=True, spec_set=True),
        indexing_state=state,
        batch_dispatcher=create_autospec(SegmentBatchImportDispatcher, instance=True, spec_set=True),
        state_transitions=SQLAlchemySegmentStateTransitions(session_factory=factory),
        state_dispatcher=CelerySegmentStateDispatcher(delay=lambda _intent_id: None),
        job_id_factory=lambda: "job-1",
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(SegmentUpdateArgs(content="updated content"), id="content"),
        pytest.param(SegmentUpdateArgs(summary="updated summary"), id="summary"),
        pytest.param(SegmentUpdateArgs(attachment_ids=["attachment-1"]), id="attachments"),
    ],
)
def test_ordinary_edit_preserves_concurrent_disable_generation(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    args: SegmentUpdateArgs,
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _document("document-1", "workspace-1", "dataset-1"),
                _segment("segment-1", "workspace-1", "dataset-1", "document-1"),
                _upload("attachment-1", "workspace-1", "attachment.png"),
            ]
        )
    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    index = create_autospec(SegmentIndex, instance=True, spec_set=True)
    index.count_tokens.return_value = 2
    app = _application(sqlite_session_factory, indexing=index, store=repository)
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    original_read = repository.get_segment_update_state
    intent_ids: list[str] = []

    def read_then_disable(segment_ref: SegmentRef) -> SegmentUpdateState | None:
        stale = original_read(segment_ref)
        intents = transitions.request(
            document_ref=document_ref,
            segment_ids=["segment-1"],
            action="disable",
            actor_id="concurrent-editor",
        )
        assert len(intents) == 1
        intent_ids.append(intents[0].id)
        return stale

    monkeypatch.setattr(repository, "get_segment_update_state", read_then_disable)
    app.mutations.update_segment(
        SegmentMutationScope(
            dataset=SegmentDatasetRecord(
                id="dataset-1",
                workspace_id="workspace-1",
                indexing_technique="high_quality",
                embedding_model_provider="provider",
                embedding_model="model",
            ),
            document=SegmentDocumentRecord(
                id="document-1",
                dataset_id="dataset-1",
                workspace_id="workspace-1",
                doc_form="text_model",
            ),
        ),
        actor_id="ordinary-editor",
        segment_id="segment-1",
        args=args,
    )

    assert len(intent_ids) == 1
    with sqlite_session_factory() as session:
        segment = session.get(DocumentSegment, "segment-1")
        assert segment is not None
        assert segment.enabled is False
        assert segment.disabled_at is not None
        assert segment.disabled_by == "concurrent-editor"
        assert segment.indexing_request_id == intent_ids[0]
        assert segment.content == (args.content or "content")
        intent = session.get(SegmentStateOutbox, intent_ids[0])
        assert intent is not None
        assert intent.action == "disable"
    if args.content is not None:
        index.update.assert_called_once_with(
            document_ref.segment("segment-1"),
            keywords=None,
            regenerate_children=False,
        )
    else:
        index.update.assert_not_called()


def test_actual_gateway_skips_disabled_body_update_after_concurrent_state_change(
    indexing_probe: "IndexingProbe",
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, repository, probe = indexing_probe
    with sqlite_session_factory.begin() as session:
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.index_node_id = "node-1"
        segment.index_node_hash = generate_text_hash(segment.content)
        session.add(segment)
    transitions = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    original_read = repository.get_segment_update_state
    original_snapshots = repository.get_indexing_snapshots
    intent_ids: list[str] = []
    snapshot_enabled: list[bool] = []

    def read_then_disable(segment_ref: SegmentRef) -> SegmentUpdateState | None:
        stale = original_read(segment_ref)
        intents = transitions.request(
            document_ref=document_ref,
            segment_ids=[segment_ref.segment_id],
            action="disable",
            actor_id="concurrent-editor",
        )
        assert len(intents) == 1
        intent_ids.append(intents[0].id)
        return stale

    monkeypatch.setattr(repository, "get_segment_update_state", read_then_disable)

    def record_snapshot_enabled(
        scoped_document_ref: DocumentRef, segment_ids: Sequence[str]
    ) -> tuple[SegmentIndexingSnapshot, ...]:
        snapshots = original_snapshots(scoped_document_ref, segment_ids)
        snapshot_enabled.extend(snapshot.segment.enabled for snapshot in snapshots)
        return snapshots

    monkeypatch.setattr(repository, "get_indexing_snapshots", record_snapshot_enabled)
    app.mutations.update_segment(
        SegmentMutationScope(
            SegmentDatasetRecord("dataset-1", "workspace-1", "high_quality", "provider", "model"),
            SegmentDocumentRecord("document-1", "dataset-1", "workspace-1", "text_model"),
        ),
        actor_id="ordinary-editor",
        segment_id="segment-1",
        args=SegmentUpdateArgs(content="updated content"),
    )

    assert snapshot_enabled
    assert not any(snapshot_enabled)
    assert len(intent_ids) == 1
    with sqlite_session_factory() as session:
        segment = session.get_one(DocumentSegment, "segment-1")
        assert segment.content == "updated content"
        assert segment.enabled is False
        assert segment.error is None
        assert segment.status == SegmentStatus.COMPLETED
        assert segment.indexing_request_id == intent_ids[0]
    assert probe.writes == 0
    assert probe.deleted_ids == []


@pytest.mark.parametrize("operation", ["create", "update", "batch", "attachments"])
def test_stale_dataset_generation_discards_body_publication_without_marking_error(
    indexing_probe: "IndexingProbe",
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    app, repository, probe = indexing_probe
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.indexing_request_id = "generation-1"
        if operation in {"update", "attachments"}:
            segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
            segment.index_node_id = "node-1"
            segment.index_node_hash = generate_text_hash(segment.content)
            session.add(segment)
        if operation == "attachments":
            dataset.is_multimodal = True
            session.add(_upload("image-1", "workspace-1", "image.png"))
    if operation == "batch":
        _persist_index_batch(sqlite_session_factory)

    original_snapshots = repository.get_indexing_snapshots
    switched = False
    snapshot_reads = 0

    def read_then_switch_generation(
        document_ref: DocumentRef, segment_ids: Sequence[str]
    ) -> tuple[SegmentIndexingSnapshot, ...]:
        nonlocal snapshot_reads, switched
        snapshots = original_snapshots(document_ref, segment_ids)
        snapshot_reads += 1
        switch_after_read = 2 if operation == "attachments" else 1
        if not switched and snapshot_reads == switch_after_read:
            switched = True
            with sqlite_session_factory.begin() as session:
                dataset = session.get_one(Dataset, "dataset-1")
                assert dataset.indexing_request_id == "generation-1"
                dataset.indexing_request_id = "generation-2"
        return snapshots

    monkeypatch.setattr(repository, "get_indexing_snapshots", read_then_switch_generation)
    scope = SegmentMutationScope(
        SegmentDatasetRecord("dataset-1", "workspace-1", "high_quality", "provider", "model"),
        SegmentDocumentRecord("document-1", "dataset-1", "workspace-1", "text_model"),
    )
    if operation == "create":
        segment_ids = [
            app.mutations.create_segment(
                scope,
                actor_id="editor",
                args=SegmentUpdateArgs(content="created content"),
            ).segment_id
        ]
    elif operation in {"update", "attachments"}:
        app.mutations.update_segment(
            scope,
            actor_id="editor",
            segment_id="segment-1",
            args=(
                SegmentUpdateArgs(content="updated content")
                if operation == "update"
                else SegmentUpdateArgs(attachment_ids=["image-1"])
            ),
        )
        segment_ids = ["segment-1"]
    else:
        segment_ids = ["first", "second"]
        app.mutations.index_segments(scope.document.ref, segment_ids=segment_ids)

    assert switched
    assert probe.writes == 0
    assert probe.batches == []
    assert probe.deleted_ids == []
    with sqlite_session_factory() as session:
        assert session.get_one(Dataset, "dataset-1").indexing_request_id == "generation-2"
        segments = session.scalars(select(DocumentSegment).where(DocumentSegment.id.in_(segment_ids))).all()
        assert len(segments) == len(segment_ids)
        assert all(segment.enabled and segment.status == SegmentStatus.COMPLETED for segment in segments)
        assert all(segment.error is None for segment in segments)


def test_mutations_use_request_actor_without_a_flask_request(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add(_document("document-1", "workspace-1", "dataset-1"))
    app = _application(sqlite_session_factory)
    context = RequestContext("request-1", None, "author-1", "workspace-1")
    created = app.create_segment(
        context, dataset_id="dataset-1", document_id="document-1", values={"content": "original"}
    )
    assert created.data.created_by == "author-1"
    updated = app.update_segment(
        RequestContext("request-2", None, "editor-2", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=created.data.id,
        values={"content": "updated content"},
    )
    assert updated.data.updated_by == "editor-2"
    assert updated.data.content == "updated content"
    assert isinstance(updated.data.created_at, datetime)
    with sqlite_session_factory() as session:
        segment = session.get(DocumentSegment, created.data.id)
        assert segment is not None
        assert segment.created_by == "author-1"
        assert segment.updated_by == "editor-2"
        assert segment.status == SegmentStatus.COMPLETED


def test_list_loads_dataset_and_document_records_once(sqlite_session_factory: sessionmaker[Session]) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add(_document("document-1", "workspace-1", "dataset-1"))
    statements: list[str] = []
    engine = sqlite_session_factory.kw["bind"]

    def record(_conn, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        page = _application(sqlite_session_factory).list_segments(
            RequestContext("request-1", None, "account-1", "workspace-1"),
            dataset_id="dataset-1",
            document_id="document-1",
            query=SegmentListFilter(),
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert page.items == ()
    owner_reads = [statement for statement in statements if "FROM datasets" in statement]
    assert len(owner_reads) == 1
    assert "JOIN documents" in owner_reads[0]
    # The page and its count validate ownership in SQL without loading the owner records again.
    assert len(statements) == 3


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (LLMBadRequestError(), "bad_request"),
        (ProviderTokenNotInitError("token missing"), "token"),
    ],
)
def test_model_guard_translates_provider_errors(error: Exception, kind: str) -> None:
    from core.model_manager import ModelManager

    model = create_autospec(ModelManager, instance=True, spec_set=True)
    model.get_model_instance.side_effect = error
    dataset = SegmentDatasetRecord(
        id="dataset-1",
        workspace_id="workspace-1",
        indexing_technique="high_quality",
        embedding_model_provider="provider",
        embedding_model="model",
    )

    with patch("services.knowledge.segments.adapters.ModelManager.for_tenant", return_value=model):
        with pytest.raises(SegmentModelProviderError) as raised:
            ModelManagerSegmentGuard().check(dataset)

    assert raised.value.kind == kind


def test_redis_state_uses_existing_keys_and_decodes_status() -> None:
    from services.knowledge.segments.adapters import RedisSegmentClient

    redis = create_autospec(RedisSegmentClient, instance=True, spec_set=True)
    redis.get.side_effect = [b"running", b"completed"]
    state = RedisSegmentIndexingState(redis)

    assert state.is_document_indexing("document-1") is True
    state.set_batch_waiting("job-1")
    assert state.get_batch_status("job-1") == "completed"
    redis.get.assert_any_call("document_document-1_indexing")
    redis.setnx.assert_called_once_with("segment_batch_import_job-1", "waiting")


def test_child_chunk_mutations_preserve_explicit_actors(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        document = _document("document-1", "workspace-1", "dataset-1")
        document.doc_form = IndexStructureType.PARENT_CHILD_INDEX
        session.add(document)
        session.add(_segment("segment-1", "workspace-1", "dataset-1", "document-1"))
    app = _application(sqlite_session_factory)
    created = app.create_child_chunk(
        RequestContext("create", None, "creator", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        content="first",
    )
    app.update_child_chunk(
        RequestContext("update", None, "editor", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        child_chunk_id=created.id,
        content="second",
    )
    with sqlite_session_factory() as session:
        chunk = session.get(ChildChunk, created.id)
        assert chunk is not None
        assert chunk.created_by == "creator"
        assert chunk.updated_by == "editor"
    results = app.update_child_chunks(
        RequestContext("batch", None, "batch-editor", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        chunks=[ChildChunkUpdateArgs(id=created.id, content="third"), ChildChunkUpdateArgs(content="new child")],
    )
    with sqlite_session_factory() as session:
        chunks = [session.get(ChildChunk, result.id) for result in results]
        assert len(chunks) == 2
        existing, added = chunks
        assert existing is not None
        assert added is not None
        assert existing.updated_by == "batch-editor"
        assert added.created_by == "batch-editor"


def _persist_child_replacement_rows(
    session_factory: sessionmaker[Session],
) -> tuple[SegmentRef, SegmentRef]:
    target_ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment("target-parent")
    foreign_ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment("foreign-parent")
    with session_factory.begin() as session:
        document = session.get_one(Document, "document-1")
        document.doc_form = IndexStructureType.PARENT_CHILD_INDEX
        target = _segment("target-parent", "workspace-1", "dataset-1", "document-1")
        foreign = _segment("foreign-parent", "workspace-1", "dataset-1", "document-1")
        foreign.position = 2
        children = []
        for child_id, segment_id, position, content in (
            ("old-1", target.id, 1, "first"),
            ("old-2", target.id, 2, "second"),
            ("foreign-child", foreign.id, 1, "foreign"),
        ):
            child = ChildChunk(
                tenant_id="workspace-1",
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id=segment_id,
                position=position,
                content=content,
                word_count=len(content),
                created_by="author",
                index_node_id=f"node-{child_id}",
            )
            child.id = child_id
            children.append(child)
        session.add_all([target, foreign, *children])
    return target_ref, foreign_ref


@pytest.mark.parametrize("case", ["unknown-only", "mixed", "foreign", "duplicate"])
def test_batch_child_replacement_rejects_invalid_explicit_ids_without_side_effects(
    indexing_probe: "IndexingProbe",
    sqlite_session_factory: sessionmaker[Session],
    case: str,
) -> None:
    app, repository, probe = indexing_probe
    target_ref, foreign_ref = _persist_child_replacement_rows(sqlite_session_factory)
    context = RequestContext("request", None, "editor", "workspace-1")
    requests = {
        "unknown-only": [ChildChunkUpdateArgs(id="missing", content="missing")],
        "mixed": [
            ChildChunkUpdateArgs(id="old-1", content="changed"),
            ChildChunkUpdateArgs(content="must not be added"),
            ChildChunkUpdateArgs(id="missing", content="missing"),
        ],
        "foreign": [ChildChunkUpdateArgs(id="foreign-child", content="foreign changed")],
        "duplicate": [
            ChildChunkUpdateArgs(id="old-1", content="first change"),
            ChildChunkUpdateArgs(id="old-1", content="second change"),
        ],
    }
    target_before = repository.get_children(target_ref)
    foreign_before = repository.get_children(foreign_ref)
    probe_before = (
        probe.writes,
        tuple(probe.deleted_ids),
        tuple(probe.batches),
        frozenset(probe.attachment_vector_ids),
    )

    with pytest.raises(ChildChunkIndexingApplicationError):
        app.update_child_chunks(
            context,
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id=target_ref.segment_id,
            chunks=requests[case],
        )

    assert repository.get_children(target_ref) == target_before
    assert repository.get_children(foreign_ref) == foreign_before
    with sqlite_session_factory() as session:
        assert session.scalars(select(SegmentStateOutbox)).all() == []
        assert session.scalars(select(ChildVectorCleanupOutbox)).all() == []
        assert session.get_one(Document, "document-1").word_count == 0
        assert session.get_one(DocumentSegment, target_ref.segment_id).indexing_request_id is None
    assert (
        probe.writes,
        tuple(probe.deleted_ids),
        tuple(probe.batches),
        frozenset(probe.attachment_vector_ids),
    ) == probe_before


def test_batch_child_replacement_accepts_explicit_empty_list(
    indexing_probe: "IndexingProbe",
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, repository, _ = indexing_probe
    target_ref, foreign_ref = _persist_child_replacement_rows(sqlite_session_factory)
    foreign_before = repository.get_children(foreign_ref)

    result = app.update_child_chunks(
        RequestContext("request", None, "editor", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=target_ref.segment_id,
        chunks=[],
    )

    assert result == ()
    assert repository.get_children(target_ref) == ()
    assert repository.get_children(foreign_ref) == foreign_before
    with sqlite_session_factory() as session:
        intents = session.scalars(select(SegmentStateOutbox)).all()
        assert len(intents) == 1
        assert intents[0].segment_id == target_ref.segment_id


def test_child_set_replacement_plans_from_locked_current_rows(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        document = _document("document-1", "workspace-1", "dataset-1")
        document.doc_form = IndexStructureType.PARENT_CHILD_INDEX
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        kept = ChildChunk(
            tenant_id="workspace-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            position=1,
            content="original",
            word_count=len("original"),
            created_by="author",
            index_node_id="kept-node",
        )
        kept.id = "kept-child"
        removed = ChildChunk(
            tenant_id="workspace-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            position=2,
            content="removed",
            word_count=len("removed"),
            created_by="author",
            index_node_id="removed-node",
        )
        removed.id = "removed-child"
        session.add_all([dataset, document, segment, kept, removed])

    outer_repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    competing_repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    outer_index = create_autospec(SegmentIndex, instance=True, spec_set=True)
    competing_index = create_autospec(SegmentIndex, instance=True, spec_set=True)
    outer = _application(sqlite_session_factory, indexing=outer_index, store=outer_repository)
    competing = _application(sqlite_session_factory, indexing=competing_index, store=competing_repository)
    original_child_mutation = outer_repository.child_mutation
    interleaved = False

    @contextmanager
    def interleaved_child_mutation(segment_ref: SegmentRef) -> Generator[SegmentChildMutation]:
        nonlocal interleaved
        if not interleaved:
            interleaved = True
            competing.update_child_chunks(
                RequestContext("competing", None, "editor-b", "workspace-1"),
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="segment-1",
                chunks=[
                    ChildChunkUpdateArgs(id="kept-child", content="updated by B"),
                    ChildChunkUpdateArgs(content="added by B"),
                ],
            )
        with original_child_mutation(segment_ref) as mutation:
            yield mutation

    monkeypatch.setattr(outer_repository, "child_mutation", interleaved_child_mutation)
    result = outer.update_child_chunks(
        RequestContext("outer", None, "editor-a", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        chunks=[
            ChildChunkUpdateArgs(id="kept-child", content="updated by A"),
            ChildChunkUpdateArgs(content="added by A"),
        ],
    )

    assert interleaved
    assert [child.content for child in result] == ["updated by A", "added by A"]
    with sqlite_session_factory() as session:
        children = session.scalars(
            select(ChildChunk)
            .where(
                ChildChunk.tenant_id == "workspace-1",
                ChildChunk.dataset_id == "dataset-1",
                ChildChunk.document_id == "document-1",
                ChildChunk.segment_id == "segment-1",
            )
            .order_by(ChildChunk.position)
        ).all()
        intents = session.scalars(
            select(SegmentStateOutbox).where(
                SegmentStateOutbox.tenant_id == "workspace-1",
                SegmentStateOutbox.dataset_id == "dataset-1",
                SegmentStateOutbox.document_id == "document-1",
                SegmentStateOutbox.segment_id == "segment-1",
            )
        ).all()
    assert [child.content for child in children] == ["updated by A", "added by A"]
    assert children[0].id == "kept-child"
    assert children[1].id not in {"removed-child", "kept-child"}
    assert all(
        (child.tenant_id, child.dataset_id, child.document_id, child.segment_id)
        == ("workspace-1", "dataset-1", "document-1", "segment-1")
        for child in children
    )
    assert len(intents) == 1
    outer_call = outer_index.prepare_child_publication.call_args
    assert [child.data.content for child in outer_call.kwargs["added"]] == ["added by A"]
    assert [child.data.content for child in outer_call.kwargs["updated"]] == ["updated by A"]
    assert [child.data.content for child in outer_call.kwargs["deleted"]] == ["added by B"]


@dataclass
class IndexRedisStub:
    def get(self, name: str | bytes) -> None:
        del name

    def setnx(self, name: str | bytes, value: object) -> bool:
        del name, value
        return True

    def setex(self, name: str, time: int, value: object) -> bool:
        del name, time, value
        return True

    def lock(self, name: str, timeout: int) -> nullcontext[None]:
        del name, timeout
        return nullcontext()


@dataclass
class VectorProbe:
    active_transactions: set[Connection]
    collection_name: str = "child-vector-collection-a"
    fail: bool = False
    fail_after_write_boundary: bool = False
    writes: int = 0
    deleted_ids: list[str] = field(default_factory=list)
    batches: list[tuple[str, ...]] = field(default_factory=list)
    attachment_vector_ids: set[str] = field(default_factory=set)
    child_vector_ids: set[str] = field(default_factory=set)

    def get_routing_target(self) -> tuple[str, str]:
        return "qdrant", self.collection_name

    def add_texts(self, documents: Sequence[IndexDocument], **kwargs: object) -> None:
        # Child mutations deliberately hold the stable parent-row consistency
        # lock across vector I/O so deletion cannot miss a just-published node.
        self.writes += len(documents)
        if "before_write" in kwargs and not any(document.metadata.get("is_summary") for document in documents):
            assert self.active_transactions, "Child publication must hold the Dataset generation lock"
        if self.fail and not self.fail_after_write_boundary:
            raise RuntimeError("index unavailable")
        before_write = kwargs.get("before_write")
        if callable(before_write):
            before_write()
        if self.fail:
            raise RuntimeError("index unavailable")
        self.child_vector_ids.update(str(document.metadata["doc_id"]) for document in documents)

    def create(self, documents: Sequence[IndexDocument], **kwargs: object) -> None:
        self.batches.append(tuple(document.page_content for document in documents))
        self.add_texts(documents, **kwargs)

    def delete_by_ids(self, ids: Sequence[str]) -> None:
        assert ids
        if all(node_id.startswith("image-") for node_id in ids):
            assert self.active_transactions, "Attachment deletion must hold the Dataset ownership lock"
        if self.fail:
            raise RuntimeError("index unavailable")
        self.deleted_ids.extend(ids)
        self.attachment_vector_ids.difference_update(ids)
        self.child_vector_ids.difference_update(ids)

    def create_multimodal(self, documents: Sequence[IndexDocument], **kwargs: object) -> None:
        assert self.active_transactions, "Attachment publication must hold the Dataset ownership lock"
        assert kwargs["upload_files"]
        attachment_ids = tuple(str(document.metadata["doc_id"]) for document in documents)
        before_write = kwargs.get("before_write")
        if callable(before_write):
            before_write(attachment_ids)
        self.writes += len(documents)
        self.attachment_vector_ids.update(attachment_ids)
        after_write = kwargs.get("after_write")
        if callable(after_write):
            after_write(attachment_ids)


type IndexingProbe = tuple[DatasetSegmentApplicationService, SQLAlchemySegmentRepository, VectorProbe]


def _drain_attachment_vector_cleanups(sqlite_session_factory: sessionmaker[Session], probe: VectorProbe) -> None:
    repository = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)
    with sqlite_session_factory() as session:
        cleanup_ids = tuple(session.scalars(select(AttachmentVectorCleanupOutbox.id)))
    for cleanup_id in cleanup_ids:
        claim = repository.claim_pending_cleanup(cleanup_id)
        if claim is None:
            continue
        with repository.guard_cleanup_delete(claim) as guard:
            if not guard.has_live_binding:
                probe.delete_by_ids([claim.attachment_id])
        if not claim.writer_finished:
            repository.release_cleanup(claim)


@pytest.fixture
def indexing_probe(
    sqlite_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> Iterator[IndexingProbe]:
    from services.knowledge.segments import indexing as indexing_module

    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.index_struct = '{"type":"qdrant","vector_store":{"class_prefix":"child-vector-collection-a"}}'
        session.add(dataset)
        session.add(_document("document-1", "workspace-1", "dataset-1"))
    engine = sqlite_session_factory.kw["bind"]
    active: set[Connection] = set()
    begin = active.add
    finish = active.discard
    event.listen(engine, "begin", begin)
    event.listen(engine, "commit", finish)
    event.listen(engine, "rollback", finish)
    probe = VectorProbe(active)

    def vector(dataset: Dataset, *, session: Session | None, vector_type: str) -> VectorProbe:
        assert dataset.tenant_id == "workspace-1"
        assert session is None
        assert vector_type == "qdrant"
        return probe

    class TokenModel:
        def get_text_embedding_num_tokens(self, texts: Sequence[str]) -> list[int]:
            assert not active
            return [len(text) for text in texts]

    class ModelLookup:
        def get_model_instance(self, **kwargs: object) -> TokenModel:
            assert kwargs["tenant_id"] == "workspace-1"
            assert not active
            return TokenModel()

    monkeypatch.setattr(indexing_module, "Vector", vector)
    monkeypatch.setattr(indexing_module.ModelManager, "for_tenant", lambda **_kwargs: ModelLookup())
    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)

    gateway = indexing_module.SegmentIndexingGateway(
        segments=repository,
        uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory),
        attachment_vectors=SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory),
        child_vector_cleanups=SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory),
        redis=IndexRedisStub(),
        new_session=sqlite_session_factory,
    )
    yield _application(sqlite_session_factory, indexing=gateway, store=repository), repository, probe
    event.remove(engine, "begin", begin)
    event.remove(engine, "commit", finish)
    event.remove(engine, "rollback", finish)


class SegmentUpdateEntry(Protocol):
    def __call__(
        self,
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_id: str,
        values: Mapping[str, object],
    ) -> SegmentDetail: ...


@pytest.fixture(params=["console", "service_api"])
def segment_update(
    request: pytest.FixtureRequest, indexing_probe: IndexingProbe, sqlite_session_factory: sessionmaker[Session]
) -> SegmentUpdateEntry:
    app, repository, _ = indexing_probe
    if request.param == "console":
        return app.update_segment

    def update(
        context: RequestContext,
        *,
        dataset_id: str,
        document_id: str,
        segment_id: str,
        values: Mapping[str, object],
    ) -> SegmentDetail:
        with sqlite_session_factory() as session:
            dataset = session.get(Dataset, dataset_id)
            document = session.get(Document, document_id)
            segment = session.get(DocumentSegment, segment_id)
            assert dataset is not None
            assert document is not None
            assert segment is not None
            result = SegmentService.update_segment(
                SegmentUpdateArgs.model_validate(values),
                segment,
                document,
                dataset,
                session,
                actor_id=context.account_id,
                mutations=app.mutations,
            )
            assert result.content == (values.get("content") or segment.content)
        ref = DatasetRef(context.active_workspace_id, dataset_id).document(document_id).segment(segment_id)
        detail = repository.get_segment(ref)
        assert detail is not None
        return SegmentDetail(
            data=SegmentPresentationAdapter().present((detail.data,))[0],
            doc_form=detail.doc_form,
        )

    return update


@pytest.mark.parametrize("segment_update", ["service_api"], indirect=True)
@pytest.mark.parametrize("doc_form", list(IndexStructureType))
def test_enable_only_patch_queues_reindexing_without_replacing_content(
    segment_update: SegmentUpdateEntry,
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    doc_form: IndexStructureType,
) -> None:
    _, _, probe = indexing_probe
    answer = "answer" if doc_form == IndexStructureType.QA_INDEX else None
    word_count = 13 if answer else 7
    with sqlite_session_factory.begin() as session:
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.enabled = False
        segment.disabled_at = datetime(2026, 1, 1)
        segment.disabled_by = "previous-editor"
        segment.answer = answer
        segment.word_count = word_count
        segment.keywords = ["original"]
        segment.index_node_id = "index-1"
        session.add(segment)
        document = session.get(Document, "document-1")
        assert document is not None
        document.doc_form = doc_form
        document.word_count = word_count

    # A disabled segment has no vector to update. Only the asynchronous load path is valid.
    probe.fail = True
    result = segment_update(
        RequestContext("request", None, "editor", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        values={"enabled": True},
    )

    assert result.data.enabled is True
    assert result.data.status == "completed"
    assert result.data.disabled_at is None
    assert result.data.disabled_by is None
    assert result.data.content == "content"
    assert result.data.answer == answer
    assert result.data.keywords == ("original",)
    assert result.data.word_count == word_count
    with sqlite_session_factory() as session:
        refreshed = session.get(DocumentSegment, "segment-1")
        assert refreshed is not None
        assert refreshed.indexing_request_id is not None
        intent = session.get(SegmentStateOutbox, refreshed.indexing_request_id)
        assert intent is not None
        assert intent.action == "enable"
    assert probe.deleted_ids == []
    assert probe.writes == 0
    with sqlite_session_factory() as session:
        document = session.get(Document, "document-1")
        assert document is not None
        assert document.word_count == word_count


@pytest.mark.parametrize("segment_update", ["service_api"], indirect=True)
@pytest.mark.parametrize("values", [{"content": "changed"}, {"keywords": ["replacement"]}])
def test_reenable_patch_with_edits_keeps_synchronous_index_update(
    segment_update: SegmentUpdateEntry,
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    values: dict[str, object],
) -> None:
    _, _, probe = indexing_probe
    with sqlite_session_factory.begin() as session:
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.enabled = False
        segment.disabled_at = datetime(2026, 1, 1)
        segment.disabled_by = "previous-editor"
        segment.index_node_id = "index-1"
        session.add(segment)

    result = segment_update(
        RequestContext("request", None, "editor", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        values={"enabled": True, **values},
    )

    assert result.data.enabled is True
    assert result.data.disabled_at is None
    assert result.data.disabled_by is None
    assert result.data.status == "completed"
    assert result.data.content == values.get("content", "content")
    if "keywords" in values:
        assert result.data.keywords == ("replacement",)
    assert probe.deleted_ids == ["index-1"]
    assert probe.writes == 1


@pytest.mark.parametrize("entry", ["console", "service_api"])
@pytest.mark.parametrize("enabled", [True, False])
def test_delete_segment_with_legacy_scalar_keywords(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    entry: str,
    enabled: bool,
) -> None:
    app, _, probe = indexing_probe
    with sqlite_session_factory.begin() as session:
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.keywords = "legacy-scalar"
        segment.enabled = enabled
        segment.index_node_id = "index-1"
        session.add(segment)
        document = session.get(Document, "document-1")
        assert document is not None
        document.word_count = segment.word_count

    if entry == "console":
        app.delete_segment(
            RequestContext("request", None, "author", "workspace-1"),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
        )
    else:
        with sqlite_session_factory() as session:
            dataset = session.get(Dataset, "dataset-1")
            document = session.get(Document, "document-1")
            segment = session.get(DocumentSegment, "segment-1")
            assert dataset is not None
            assert document is not None
            assert segment is not None
            SegmentService.delete_segment(segment, document, dataset, session, mutations=app.mutations)

    with sqlite_session_factory() as session:
        assert session.get(DocumentSegment, "segment-1") is None
        document = session.get(Document, "document-1")
        assert document is not None
        assert document.word_count == 0
    with sqlite_session_factory() as session:
        intent = session.scalar(select(SegmentStateOutbox).where(SegmentStateOutbox.segment_id == "segment-1"))
        assert intent is not None
        assert intent.action == "delete"
        assert intent.index_node_id == "index-1"


@pytest.mark.parametrize("keywords", ["legacy-scalar", 42, {"invalid": "object"}, [{"invalid": "item"}]])
def test_update_repairs_legacy_keywords_without_validating_presentation_first(
    segment_update: SegmentUpdateEntry,
    sqlite_session_factory: sessionmaker[Session],
    keywords: object,
) -> None:
    with sqlite_session_factory.begin() as session:
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.keywords = keywords
        segment.index_node_id = "index-1"
        segment.index_node_hash = generate_text_hash(segment.content)
        session.add(segment)

    result = segment_update(
        RequestContext("request", None, "editor", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        values={"content": "content", "keywords": ["replacement"]},
    )

    assert result.data.status == "completed"
    assert result.data.keywords == ("replacement",)
    with sqlite_session_factory() as session:
        segment = session.get(DocumentSegment, "segment-1")
        assert segment is not None
        assert segment.keywords == ["replacement"]


@pytest.mark.parametrize("indexing_technique", [IndexTechniqueType.HIGH_QUALITY, IndexTechniqueType.ECONOMY])
def test_content_and_explicit_keywords_update_persist_the_same_keywords_returned_to_the_caller(
    segment_update: SegmentUpdateEntry,
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    indexing_technique: IndexTechniqueType,
) -> None:
    _, _, _ = indexing_probe
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        document = session.get(Document, "document-1")
        assert dataset is not None
        assert document is not None
        dataset.indexing_technique = indexing_technique
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.keywords = ["initial"]
        segment.index_node_id = "index-1"
        session.add(segment)
        document.word_count = segment.word_count

    result = segment_update(
        RequestContext("request", None, "editor", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        values={"content": "changed content", "keywords": ["replacement"]},
    )

    assert result.data.keywords == ("replacement",)
    with sqlite_session_factory() as session:
        persisted = session.get(DocumentSegment, "segment-1")
        assert persisted is not None
        assert persisted.keywords == ["replacement"]


@pytest.mark.parametrize(
    ("values", "expected_keywords"),
    [
        ({"content": "changed content"}, ["initial"]),
        ({"content": "content", "keywords": None}, None),
        ({"content": "content", "keywords": []}, []),
    ],
    ids=["omitted", "explicit-null", "explicit-empty"],
)
def test_high_quality_keyword_patch_distinguishes_omitted_null_and_empty(
    segment_update: SegmentUpdateEntry,
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    values: dict[str, object],
    expected_keywords: list[str] | None,
) -> None:
    _, _, _ = indexing_probe
    with sqlite_session_factory.begin() as session:
        document = session.get(Document, "document-1")
        assert document is not None
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.keywords = ["initial"]
        segment.index_node_id = "index-1"
        session.add(segment)
        document.word_count = segment.word_count

    result = segment_update(
        RequestContext("request", None, "editor", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id="segment-1",
        values=values,
    )

    expected_response = None if expected_keywords is None else tuple(expected_keywords)
    assert result.data.keywords == expected_response
    with sqlite_session_factory() as session:
        persisted = session.get(DocumentSegment, "segment-1")
        assert persisted is not None
        assert persisted.keywords == expected_keywords


@pytest.mark.parametrize("content", ["original", "changed text"])
@pytest.mark.parametrize(
    ("attachment_values", "expected_ids", "removed_original"),
    [
        ({}, ["image-1"], False),
        ({"attachment_ids": None}, ["image-1"], False),
        ({"attachment_ids": []}, [], True),
        ({"attachment_ids": ["image-2"]}, ["image-2"], True),
    ],
    ids=["omitted", "null", "clear", "replace"],
)
def test_segment_update_preserves_or_replaces_attachments(
    segment_update: SegmentUpdateEntry,
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    content: str,
    attachment_values: dict[str, object],
    expected_ids: list[str],
    removed_original: bool,
) -> None:
    app, _, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [_upload("image-1", "workspace-1", "chart.png"), _upload("image-2", "workspace-1", "replacement.png")]
        )
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.is_multimodal = True
    created = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "original", "attachment_ids": ["image-1"]},
    )
    assert [attachment.id for attachment in created.data.attachments] == ["image-1"]

    updated = segment_update(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=created.data.id,
        values={"content": content, **attachment_values},
    )
    _drain_attachment_vector_cleanups(sqlite_session_factory, probe)

    assert updated.data.status == "completed"
    assert [attachment.id for attachment in updated.data.attachments] == expected_ids
    with sqlite_session_factory() as session:
        attachment_ids = list(
            session.scalars(
                select(SegmentAttachmentBinding.attachment_id).where(
                    SegmentAttachmentBinding.segment_id == created.data.id
                )
            )
        )
        original_upload = session.get(UploadFile, "image-1")
        original_cleanup = session.scalar(
            select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == "image-1")
        )
    assert attachment_ids == expected_ids
    assert (original_upload is None) is removed_original
    assert (original_cleanup is not None) is removed_original
    assert ("image-1" in probe.deleted_ids) is removed_original


@pytest.mark.parametrize(
    "replacement_technique",
    [IndexTechniqueType.HIGH_QUALITY, IndexTechniqueType.ECONOMY],
    ids=["new-high-quality-generation", "switch-to-economy"],
)
def test_segment_patch_continues_attachment_replacement_after_body_generation_loss(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    replacement_technique: IndexTechniqueType,
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    with sqlite_session_factory.begin() as session:
        session.add(_upload("image-generation-handoff", "workspace-1", "handoff.png"))
    created = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "original body"},
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    original_lock_index_write = repository.lock_index_write
    generation_switched = False

    @contextmanager
    def switch_generation_before_body_publication(
        segment_ref: SegmentRef,
        *,
        lock_segment: bool = True,
        expected_snapshot: SegmentIndexingSnapshot | None = None,
    ) -> Generator[bool]:
        nonlocal generation_switched
        if not generation_switched:
            generation_switched = True
            with sqlite_session_factory.begin() as session:
                dataset = session.get_one(Dataset, "dataset-1")
                dataset.indexing_request_id = "replacement-generation"
                dataset.indexing_technique = replacement_technique
        with original_lock_index_write(
            segment_ref,
            lock_segment=lock_segment,
            expected_snapshot=expected_snapshot,
        ) as enabled:
            yield enabled

    monkeypatch.setattr(repository, "lock_index_write", switch_generation_before_body_publication)
    probe.deleted_ids.clear()
    updated = app.update_segment(
        context,
        dataset_id=ref.document.dataset.dataset_id,
        document_id=ref.document.document_id,
        segment_id=ref.segment_id,
        values={
            "content": "committed body for replacement generation",
            "attachment_ids": ["image-generation-handoff"],
        },
    )

    assert generation_switched
    assert updated.data.content == "committed body for replacement generation"
    assert [attachment.id for attachment in updated.data.attachments] == ["image-generation-handoff"]
    with sqlite_session_factory() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        segment = session.get_one(DocumentSegment, ref.segment_id)
        attachment_ids = tuple(
            session.scalars(
                select(SegmentAttachmentBinding.attachment_id).where(
                    SegmentAttachmentBinding.tenant_id == "workspace-1",
                    SegmentAttachmentBinding.dataset_id == "dataset-1",
                    SegmentAttachmentBinding.document_id == "document-1",
                    SegmentAttachmentBinding.segment_id == ref.segment_id,
                )
            )
        )
        assert dataset.indexing_request_id == "replacement-generation"
        assert dataset.indexing_technique == replacement_technique
        assert segment.content == "committed body for replacement generation"
        assert segment.status == SegmentStatus.COMPLETED
        assert attachment_ids == ("image-generation-handoff",)
    assert created.data.index_node_id not in probe.deleted_ids


@pytest.mark.parametrize(
    ("requested_ids", "expected_ids", "valid"),
    [
        pytest.param(["missing"], ["image-1"], False, id="nonexistent"),
        pytest.param(["image-2", "missing"], ["image-1"], False, id="mixed"),
        pytest.param(["foreign-image"], ["image-1"], False, id="foreign-tenant"),
        pytest.param([], [], True, id="explicit-empty"),
        pytest.param(["image-2", "image-2"], ["image-2"], True, id="duplicate-valid-id"),
    ],
)
def test_attachment_replacement_validates_complete_request_before_mutation(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    requested_ids: list[str],
    expected_ids: list[str],
    valid: bool,
) -> None:
    app, _, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.is_multimodal = True
        session.add_all(
            [
                _upload("image-1", "workspace-1", "original.png"),
                _upload("image-2", "workspace-1", "replacement.png"),
                _upload("foreign-image", "workspace-2", "foreign.png"),
            ]
        )
    created = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "original", "attachment_ids": ["image-1"]},
    )
    writes_before_update = probe.writes
    probe.deleted_ids.clear()

    update_values = {
        "content": "original" if valid else "must not be persisted",
        "attachment_ids": requested_ids,
    }
    updated: SegmentDetail | None = None
    if valid:
        updated = app.update_segment(
            context,
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id=created.data.id,
            values=update_values,
        )

    else:
        segment_ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
        with pytest.raises(LookupError, match="current workspace"):
            app.mutations._index.update_attachments(  # pyright: ignore[reportPrivateUsage]
                segment_ref,
                requested_ids,
            )
        with pytest.raises(SegmentUploadFileNotFoundError):
            app.update_segment(
                context,
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id=created.data.id,
                values=update_values,
            )

    _drain_attachment_vector_cleanups(sqlite_session_factory, probe)

    with sqlite_session_factory() as session:
        bindings = list(
            session.scalars(
                select(SegmentAttachmentBinding.attachment_id)
                .where(
                    SegmentAttachmentBinding.tenant_id == "workspace-1",
                    SegmentAttachmentBinding.dataset_id == "dataset-1",
                    SegmentAttachmentBinding.document_id == "document-1",
                    SegmentAttachmentBinding.segment_id == created.data.id,
                )
                .order_by(SegmentAttachmentBinding.attachment_id)
            )
        )
        original_upload = session.get(UploadFile, "image-1")
        replacement_upload = session.get(UploadFile, "image-2")
        foreign_upload = session.get(UploadFile, "foreign-image")
        cleanup_ids = set(session.scalars(select(UploadFileCleanupOutbox.upload_file_id)))
        persisted_segment = session.get_one(DocumentSegment, created.data.id)
        state_intents = session.scalars(
            select(SegmentStateOutbox).where(SegmentStateOutbox.segment_id == created.data.id)
        ).all()

    assert bindings == expected_ids
    assert replacement_upload is not None
    assert foreign_upload is not None
    if valid:
        assert updated is not None
        assert updated.data.status == "completed"
        assert original_upload is None
        assert cleanup_ids == {"image-1"}
        assert probe.deleted_ids == ["image-1"]
        expected_new_vectors = 1 if expected_ids else 0
        assert probe.writes == writes_before_update + expected_new_vectors
    else:
        assert persisted_segment.content == "original"
        assert persisted_segment.status == "completed"
        assert persisted_segment.enabled is True
        assert persisted_segment.error is None
        assert persisted_segment.indexing_request_id is None
        assert state_intents == []
        assert original_upload is not None
        assert cleanup_ids == set()
        assert probe.deleted_ids == []
        assert probe.writes == writes_before_update


def test_attachment_generation_guard_reuses_publication_transaction_session(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.knowledge.segments import indexing as indexing_module

    app, repository, probe = indexing_probe
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.is_multimodal = True
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.index_node_id = "node-1"
        segment.index_node_hash = generate_text_hash(segment.content)
        session.add_all([segment, _upload("image-1", "workspace-1", "image.png")])

    guard_sessions: list[Session] = []
    require_snapshot = repository.require_indexing_snapshot

    def record_guard_session(session: Session, snapshot: SegmentIndexingSnapshot) -> None:
        guard_sessions.append(session)
        require_snapshot(session, snapshot)

    monkeypatch.setattr(repository, "require_indexing_snapshot", record_guard_session)
    with patch.object(
        indexing_module,
        "publish_bound_attachment_vectors",
        wraps=indexing_module.publish_bound_attachment_vectors,
    ) as publish:
        app.mutations.update_segment(
            SegmentMutationScope(
                SegmentDatasetRecord("dataset-1", "workspace-1", "high_quality", "provider", "model"),
                SegmentDocumentRecord("document-1", "dataset-1", "workspace-1", "text_model"),
            ),
            actor_id="editor",
            segment_id="segment-1",
            args=SegmentUpdateArgs(attachment_ids=["image-1"]),
        )

    assert publish.call_count == 1
    publication_session = publish.call_args.args[0]
    assert guard_sessions == [publication_session]
    assert probe.attachment_vector_ids == {"image-1"}


@pytest.mark.parametrize("shared_by_other_document", [False, True], ids=["orphaned-y", "shared-y"])
def test_attachment_replacement_cleans_actual_removed_ids_from_stale_snapshot(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    shared_by_other_document: bool,
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.is_multimodal = True
        survivor_document = _document("document-2", "workspace-1", "dataset-1")
        survivor_document.position = 2
        survivor_segment = _segment("segment-2", "workspace-1", "dataset-1", "document-2")
        session.add_all(
            [
                survivor_document,
                survivor_segment,
                _upload("image-x", "workspace-1", "x.png"),
                _upload("image-y", "workspace-1", "y.png"),
                _upload("image-z", "workspace-1", "z.png"),
            ]
        )
    created = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "target", "attachment_ids": ["image-x"]},
    )
    segment_ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    stale_x_snapshot = repository.get_indexing_snapshot(segment_ref)
    gateway = app.mutations._index  # pyright: ignore[reportPrivateUsage]
    probe.deleted_ids.clear()

    gateway.update_attachments(segment_ref, ["image-y"])
    if shared_by_other_document:
        with sqlite_session_factory.begin() as session:
            add_segment_attachment_bindings(
                session,
                tenant_id="workspace-1",
                dataset_id="dataset-1",
                document_id="document-2",
                segment_id="segment-2",
                attachment_ids=("image-y",),
            )

    monkeypatch.setattr(repository, "get_indexing_snapshot", lambda _segment_ref: stale_x_snapshot)
    gateway.update_attachments(segment_ref, ["image-z"])
    _drain_attachment_vector_cleanups(sqlite_session_factory, probe)

    with sqlite_session_factory() as session:
        target_ids = tuple(
            session.scalars(
                select(SegmentAttachmentBinding.attachment_id)
                .where(
                    SegmentAttachmentBinding.tenant_id == "workspace-1",
                    SegmentAttachmentBinding.dataset_id == "dataset-1",
                    SegmentAttachmentBinding.document_id == "document-1",
                    SegmentAttachmentBinding.segment_id == created.data.id,
                )
                .order_by(SegmentAttachmentBinding.attachment_id)
            )
        )
        survivor_ids = tuple(
            session.scalars(
                select(SegmentAttachmentBinding.attachment_id)
                .where(
                    SegmentAttachmentBinding.tenant_id == "workspace-1",
                    SegmentAttachmentBinding.dataset_id == "dataset-1",
                    SegmentAttachmentBinding.document_id == "document-2",
                    SegmentAttachmentBinding.segment_id == "segment-2",
                )
                .order_by(SegmentAttachmentBinding.attachment_id)
            )
        )
        cleanup_ids = set(session.scalars(select(UploadFileCleanupOutbox.upload_file_id)))
        y_upload = session.get(UploadFile, "image-y")
        z_upload = session.get(UploadFile, "image-z")

    assert target_ids == ("image-z",)
    assert survivor_ids == (("image-y",) if shared_by_other_document else ())
    assert z_upload is not None
    if shared_by_other_document:
        assert y_upload is not None
        assert cleanup_ids == {"image-x"}
        assert probe.deleted_ids == ["image-x"]
        assert probe.attachment_vector_ids == {"image-y", "image-z"}
    else:
        assert y_upload is None
        assert cleanup_ids == {"image-x", "image-y"}
        assert set(probe.deleted_ids) == {"image-x", "image-y"}
        assert probe.attachment_vector_ids == {"image-z"}


def test_attachment_replacement_decides_noop_from_locked_current_bindings(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.is_multimodal = True
        survivor_document = _document("document-2", "workspace-1", "dataset-1")
        survivor_document.position = 2
        session.add_all(
            [
                survivor_document,
                _segment("segment-2", "workspace-1", "dataset-1", "document-2"),
                _upload("image-x", "workspace-1", "x.png"),
                _upload("image-y", "workspace-1", "y.png"),
            ]
        )
    created = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "target", "attachment_ids": ["image-x"]},
    )
    with sqlite_session_factory.begin() as session:
        add_segment_attachment_bindings(
            session,
            tenant_id="workspace-1",
            dataset_id="dataset-1",
            document_id="document-2",
            segment_id="segment-2",
            attachment_ids=("image-x",),
        )
    segment_ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    stale_x_snapshot = repository.get_indexing_snapshot(segment_ref)
    gateway = app.mutations._index  # pyright: ignore[reportPrivateUsage]
    probe.deleted_ids.clear()

    gateway.update_attachments(segment_ref, ["image-y"])
    monkeypatch.setattr(repository, "get_indexing_snapshot", lambda _segment_ref: stale_x_snapshot)
    gateway.update_attachments(segment_ref, ["image-x"])
    _drain_attachment_vector_cleanups(sqlite_session_factory, probe)

    with sqlite_session_factory() as session:
        target_ids = tuple(
            session.scalars(
                select(SegmentAttachmentBinding.attachment_id).where(
                    SegmentAttachmentBinding.document_id == "document-1",
                    SegmentAttachmentBinding.segment_id == created.data.id,
                )
            )
        )
        survivor_ids = tuple(
            session.scalars(
                select(SegmentAttachmentBinding.attachment_id).where(
                    SegmentAttachmentBinding.document_id == "document-2",
                    SegmentAttachmentBinding.segment_id == "segment-2",
                )
            )
        )
        cleanup_ids = set(session.scalars(select(UploadFileCleanupOutbox.upload_file_id)))
        x_upload = session.get(UploadFile, "image-x")
        y_upload = session.get(UploadFile, "image-y")

    assert target_ids == ("image-x",)
    assert survivor_ids == ("image-x",)
    assert x_upload is not None
    assert y_upload is None
    assert cleanup_ids == {"image-y"}
    assert probe.deleted_ids == ["image-y"]
    assert probe.attachment_vector_ids == {"image-x"}


def test_segment_attachment_replacement_keeps_vector_shared_by_another_segment(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, _, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.is_multimodal = True
        session.add(_upload("image-1", "workspace-1", "shared.png"))

    first = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "first", "attachment_ids": ["image-1"]},
    )
    second = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "second", "attachment_ids": ["image-1"]},
    )
    probe.deleted_ids.clear()

    app.update_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=first.data.id,
        values={"content": "first", "attachment_ids": []},
    )

    assert "image-1" not in probe.deleted_ids
    with sqlite_session_factory() as session:
        bindings = session.scalars(
            select(SegmentAttachmentBinding).where(SegmentAttachmentBinding.attachment_id == "image-1")
        ).all()
    assert [binding.segment_id for binding in bindings] == [second.data.id]


def test_segment_indexing_and_summary_use_closed_read_sessions(
    segment_update: SegmentUpdateEntry, indexing_probe: IndexingProbe, sqlite_session_factory: sessionmaker[Session]
) -> None:
    from models.dataset import DocumentSegmentSummary

    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "first"})
    updated = segment_update(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=created.data.id,
        values={"content": "second", "summary": "manual summary"},
    )
    assert updated.data.status == "completed"
    assert updated.data.summary == "manual summary"
    with sqlite_session_factory() as session:
        summary = session.scalar(select(DocumentSegmentSummary))
        assert summary is not None
        assert summary.status == "completed"
        document = session.get(Document, "document-1")
        assert document is not None
        assert document.word_count == len("second")
    assert probe.writes >= 3
    probe.fail = True
    failed = segment_update(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=created.data.id,
        values={"content": "failed change"},
    )
    assert failed.data.status == "error"
    assert failed.data.enabled is False
    assert failed.data.error == "index unavailable"


@pytest.mark.parametrize("operation", ["create", "update", "replace"])
def test_child_index_failure_leaves_database_unchanged(indexing_probe: IndexingProbe, operation: str) -> None:
    from services.knowledge.segments.application import (
        ChildChunkDeleteIndexApplicationError,
        ChildChunkIndexingApplicationError,
    )

    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    child = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="original",
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    before = repository.get_children(ref)
    probe.fail = True

    def mutate() -> None:
        match operation:
            case "create":
                app.create_child_chunk(
                    context,
                    dataset_id="dataset-1",
                    document_id="document-1",
                    segment_id=parent.data.id,
                    content="new",
                )
            case "update":
                app.update_child_chunk(
                    context,
                    dataset_id="dataset-1",
                    document_id="document-1",
                    segment_id=parent.data.id,
                    child_chunk_id=child.id,
                    content="changed",
                )
            case "replace":
                app.update_child_chunks(
                    context,
                    dataset_id="dataset-1",
                    document_id="document-1",
                    segment_id=parent.data.id,
                    chunks=[ChildChunkUpdateArgs(content="replacement")],
                )

    with pytest.raises((ChildChunkIndexingApplicationError, ChildChunkDeleteIndexApplicationError)):
        mutate()
    assert repository.get_children(ref) == before


@pytest.mark.parametrize("unknown_result", [False, True], ids=["definite-not-submitted", "submission-unknown"])
def test_child_update_publishes_fresh_node_before_switch_and_retains_unknown_cleanup(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    unknown_result: bool,
) -> None:
    from services.knowledge.segments.application import ChildChunkIndexingApplicationError

    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    created = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="original",
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    before = repository.get_children(ref)
    assert before is not None
    old_node_id = before[0].index_node_id
    assert old_node_id is not None
    probe.fail = True
    probe.fail_after_write_boundary = unknown_result

    with pytest.raises(ChildChunkIndexingApplicationError, match="index unavailable"):
        app.update_child_chunk(
            context,
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id=parent.data.id,
            child_chunk_id=created.id,
            content="changed",
        )

    assert repository.get_children(ref) == before
    assert old_node_id in probe.child_vector_ids
    assert old_node_id not in probe.deleted_ids
    with sqlite_session_factory() as session:
        cleanups = session.scalars(select(ChildVectorCleanupOutbox)).all()
    if unknown_result:
        assert len(cleanups) == 1
        assert cleanups[0].node_id != old_node_id
        assert cleanups[0].writer_finished_at is None
    else:
        assert cleanups == []


def test_child_update_switches_reference_and_retries_old_vector_cleanup(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks import child_vector_cleanup_task

    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    created = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="original",
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    before = repository.get_children(ref)
    assert before is not None
    old_node_id = before[0].index_node_id
    assert old_node_id is not None

    app.update_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        child_chunk_id=created.id,
        content="changed",
    )

    after = repository.get_children(ref)
    assert after is not None
    assert after[0].index_node_id is not None
    assert after[0].index_node_id != old_node_id
    assert {old_node_id, after[0].index_node_id} <= probe.child_vector_ids
    with sqlite_session_factory() as session:
        cleanup = session.scalar(select(ChildVectorCleanupOutbox))
        assert cleanup is not None
        assert cleanup.node_id == old_node_id
        assert cleanup.writer_finished_at is not None
        assert cleanup.vector_route is not None
        cleanup_route = ChildVectorRoute.model_validate_json(cleanup.vector_route)
        assert cleanup_route.collection_name == "child-vector-collection-a"
        persisted_child = session.get_one(ChildChunk, created.id)
        assert persisted_child.vector_route_hash == cleanup_route.key
        cleanup_id = cleanup.id

    class CleanupVector:
        @staticmethod
        def resolve_vector_type(_dataset: Dataset, *, session: Session) -> str:
            del session
            return "qdrant"

        def __new__(cls, *_args: object, **_kwargs: object) -> VectorProbe:
            del cls
            return probe

    monkeypatch.setattr(child_vector_cleanup_task, "Vector", CleanupVector)
    probe.fail = True
    with pytest.raises(RuntimeError, match="index unavailable"):
        child_vector_cleanup_task.process_child_vector_cleanup_task(cleanup_id)
    with sqlite_session_factory() as session:
        assert session.get(ChildVectorCleanupOutbox, cleanup_id) is not None

    probe.fail = False
    child_vector_cleanup_task.process_child_vector_cleanup_task(cleanup_id)
    with sqlite_session_factory() as session:
        assert session.get(ChildVectorCleanupOutbox, cleanup_id) is None
    assert old_node_id not in probe.child_vector_ids
    assert after[0].index_node_id in probe.child_vector_ids


def test_child_update_carries_old_route_into_cleanup_after_collection_switch(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    created = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="original",
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    before = repository.get_children(ref)
    assert before is not None
    old_node_id = before[0].index_node_id
    assert old_node_id is not None

    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.index_struct = '{"type":"qdrant","vector_store":{"class_prefix":"child-vector-collection-b"}}'
    probe.collection_name = "child-vector-collection-b"

    app.update_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        child_chunk_id=created.id,
        content="changed",
    )

    with sqlite_session_factory() as session:
        cleanup = session.scalar(
            select(ChildVectorCleanupOutbox).where(ChildVectorCleanupOutbox.node_id == old_node_id)
        )
        assert cleanup is not None
        assert cleanup.vector_route is not None
        old_route = ChildVectorRoute.model_validate_json(cleanup.vector_route)
        assert old_route.collection_name == "child-vector-collection-a"
        persisted_child = session.get_one(ChildChunk, created.id)
        assert persisted_child.vector_route is not None
        current_route = ChildVectorRoute.model_validate_json(persisted_child.vector_route)
        assert current_route.collection_name == "child-vector-collection-b"


def test_child_publication_commit_outcome_unknown_remains_reconcilable(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.knowledge.segments.application import ChildChunkIndexingApplicationError

    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    created = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="original",
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    before = repository.get_children(ref)
    assert before is not None
    old_node_id = before[0].index_node_id
    assert old_node_id is not None
    original_publish_and_save = app.mutations._publish_and_save_children

    def commit_then_disconnect(
        segment_ref: SegmentRef,
        *,
        expected: Sequence[ChildChunkState],
        expected_vector_guard: ChildVectorGuard,
        added: Sequence[ChildChunkState] = (),
        updated: Sequence[ChildChunkState] = (),
        deleted: Sequence[ChildChunkState] = (),
    ) -> Never:
        original_publish_and_save(
            segment_ref,
            expected=expected,
            expected_vector_guard=expected_vector_guard,
            added=added,
            updated=updated,
            deleted=deleted,
        )
        raise RuntimeError("connection lost after commit")

    monkeypatch.setattr(app.mutations, "_publish_and_save_children", commit_then_disconnect)
    with pytest.raises(ChildChunkIndexingApplicationError, match="connection lost after commit"):
        app.update_child_chunk(
            context,
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id=parent.data.id,
            child_chunk_id=created.id,
            content="changed",
        )

    after = repository.get_children(ref)
    assert after is not None
    assert after[0].data.content == "changed"
    assert after[0].index_node_id is not None
    assert after[0].index_node_id != old_node_id
    assert after[0].index_node_id in probe.child_vector_ids
    with sqlite_session_factory() as session:
        cleanups = session.scalars(select(ChildVectorCleanupOutbox)).all()
    assert [(row.node_id, row.writer_finished_at is not None) for row in cleanups] == [(old_node_id, True)]


def test_child_publication_scope_remains_held_through_row_switch_commit(
    indexing_probe: IndexingProbe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    created = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="original",
    )
    gateway = app.mutations._index
    store = app.mutations._store
    original_prepare = gateway.prepare_child_publication
    original_mutation = store.child_mutation
    events: list[str] = []
    mutation_active = False

    class RecordingPrepared:
        def __init__(self, wrapped: PreparedChildPublication) -> None:
            self._wrapped = wrapped

        @contextmanager
        def submit(self) -> Generator[ChildVectorPublication]:
            events.append("scope-enter")
            with self._wrapped.submit() as publication:
                events.append("vector-published-under-mutation" if mutation_active else "vector-published-unlocked")
                yield publication
            events.append("scope-exit-under-mutation" if mutation_active else "scope-exit-unlocked")

        def discard(self) -> None:
            self._wrapped.discard()

    def recording_prepare(
        segment_ref: SegmentRef,
        *,
        guard: ChildVectorGuard,
        added: Sequence[ChildChunkState] = (),
        updated: Sequence[ChildChunkState] = (),
        deleted: Sequence[ChildChunkState] = (),
    ) -> PreparedChildPublication:
        return RecordingPrepared(
            original_prepare(
                segment_ref,
                guard=guard,
                added=added,
                updated=updated,
                deleted=deleted,
            )
        )

    @contextmanager
    def recording_mutation(segment_ref: SegmentRef) -> Generator[SegmentChildMutation]:
        nonlocal mutation_active
        with original_mutation(segment_ref) as mutation:
            mutation_active = True
            try:
                yield mutation
            finally:
                mutation_active = False
        events.append("row-switch-committed")

    monkeypatch.setattr(gateway, "prepare_child_publication", recording_prepare)
    monkeypatch.setattr(store, "child_mutation", recording_mutation)
    app.update_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        child_chunk_id=created.id,
        content="changed",
    )

    assert events[-4:] == [
        "scope-enter",
        "vector-published-under-mutation",
        "scope-exit-under-mutation",
        "row-switch-committed",
    ]


def test_multi_child_replacement_switches_all_nodes_and_persists_old_cleanups(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    first = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="first",
    )
    second = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="second",
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    before = repository.get_children(ref)
    assert before is not None
    old_nodes = {child.index_node_id for child in before}

    app.update_child_chunks(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        chunks=[
            ChildChunkUpdateArgs(id=first.id, content="first changed"),
            ChildChunkUpdateArgs(content="third"),
        ],
    )

    after = repository.get_children(ref)
    assert after is not None
    assert [child.data.content for child in after] == ["first changed", "third"]
    new_nodes = {child.index_node_id for child in after}
    assert None not in new_nodes
    assert old_nodes.isdisjoint(new_nodes)
    assert new_nodes <= probe.child_vector_ids
    with sqlite_session_factory() as session:
        cleanups = session.scalars(select(ChildVectorCleanupOutbox)).all()
    assert {row.node_id for row in cleanups} == old_nodes
    assert all(row.writer_finished_at is not None for row in cleanups)
    assert second.id not in {child.data.id for child in after}


def test_disabled_child_update_skips_publication_but_durably_retires_old_node(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    created = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="original",
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    before = repository.get_children(ref)
    assert before is not None
    old_node_id = before[0].index_node_id
    assert old_node_id is not None
    with sqlite_session_factory.begin() as session:
        segment = session.get_one(DocumentSegment, parent.data.id)
        segment.enabled = False
    writes_before = probe.writes

    app.update_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        child_chunk_id=created.id,
        content="changed while disabled",
    )

    after = repository.get_children(ref)
    assert after is not None
    assert after[0].data.content == "changed while disabled"
    assert after[0].index_node_id is not None
    assert after[0].index_node_id != old_node_id
    assert probe.writes == writes_before
    assert after[0].index_node_id not in probe.child_vector_ids
    with sqlite_session_factory() as session:
        cleanup = session.scalar(select(ChildVectorCleanupOutbox))
        intent = session.scalar(select(SegmentStateOutbox).where(SegmentStateOutbox.segment_id == parent.data.id))
    assert cleanup is not None
    assert cleanup.node_id == old_node_id
    assert intent is not None
    assert intent.action == "disable"


def test_upload_batch_uses_one_scoped_query(sqlite_session_factory: sessionmaker[Session]) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _upload("file-1", "workspace-1", "one.csv"),
                _upload("file-2", "workspace-1", "two.csv"),
                _upload("foreign", "workspace-2", "foreign.csv"),
            ]
        )
    engine = sqlite_session_factory.kw["bind"]
    statements = []

    def record_sql(_conn, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record_sql)
    try:
        uploads = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory).get_by_ids(
            workspace_id="workspace-1", file_ids=("file-2", "file-1", "foreign", "missing")
        )
    finally:
        event.remove(engine, "before_cursor_execute", record_sql)
    assert set(uploads) == {"file-1", "file-2"}
    assert len(statements) == 1


def test_child_query_escapes_wildcards_and_enforces_complete_scope(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    from services.knowledge.segments.application import ChildChunkListFilter

    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        session.add(_document("document-1", "workspace-1", "dataset-1"))
        session.add(_segment("segment-1", "workspace-1", "dataset-1", "document-1"))
        for position, (workspace, text) in enumerate(
            [("workspace-1", "50%_literal"), ("workspace-1", "5000literal"), ("workspace-2", "50%_literal")], start=1
        ):
            session.add(
                ChildChunk(
                    tenant_id=workspace,
                    dataset_id="dataset-1",
                    document_id="document-1",
                    segment_id="segment-1",
                    position=position,
                    content=text,
                    word_count=len(text),
                    created_by="author",
                )
            )
    repo = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    page = repo.list_child_chunks(
        DatasetRef("workspace-1", "dataset-1").document("document-1").segment("segment-1"),
        ChildChunkListFilter(keyword="%_", limit=1),
    )
    assert page is not None
    assert page.total == 1
    assert [child.content for child in page.items] == ["50%_literal"]


def test_batch_status_and_delete_only_modify_scoped_segments(
    indexing_probe: IndexingProbe, sqlite_session_factory: sessionmaker[Session]
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    first = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "first"})
    second = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "second"})
    with sqlite_session_factory.begin() as session:
        session.add(_segment("foreign", "workspace-2", "dataset-1", "document-1"))
    segment_ids = [first.data.id, second.data.id, "foreign"]
    app.change_segment_status(
        context, dataset_id="dataset-1", document_id="document-1", segment_ids=segment_ids, action="disable"
    )
    with sqlite_session_factory() as session:
        first_row = session.get(DocumentSegment, first.data.id)
        assert first_row is not None
        assert first_row.enabled is False
        foreign_row = session.get(DocumentSegment, "foreign")
        assert foreign_row is not None
        assert foreign_row.enabled is True
        intents = session.scalars(select(SegmentStateOutbox).where(SegmentStateOutbox.action == "disable")).all()
        assert {intent.segment_id for intent in intents} == {first.data.id, second.data.id}
    app.delete_segments(context, dataset_id="dataset-1", document_id="document-1", segment_ids=segment_ids)
    with sqlite_session_factory() as session:
        assert session.get(DocumentSegment, first.data.id) is None
        assert session.get(DocumentSegment, second.data.id) is None
        assert session.get(DocumentSegment, "foreign") is not None
        document = session.get(Document, "document-1")
        assert document is not None
        assert document.word_count == 0


def test_child_regeneration_reuses_splitter_outside_transaction(
    segment_update: SegmentUpdateEntry,
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from core.rag.index_processor.processor.parent_child_index_processor import ParentChildIndexProcessor
    from models.dataset import DatasetProcessRule
    from models.enums import ProcessRuleMode

    app, repository, probe = indexing_probe
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.embedding_model_provider = "provider"
        rule = DatasetProcessRule(
            dataset_id="dataset-1",
            mode=ProcessRuleMode.HIERARCHICAL,
            rules=json.dumps({"parent_mode": "full-doc", "subchunk_segmentation": {"max_tokens": 100}}),
            created_by="author",
        )
        session.add(rule)
        document = session.get(Document, "document-1")
        assert document is not None
        document.doc_form = IndexStructureType.PARENT_CHILD_INDEX
        document.dataset_process_rule_id = rule.id

    def split(_self, document: IndexDocument, _rule, _mode, _model) -> list[ChildDocument]:
        assert not probe.active_transactions
        return [
            ChildDocument(
                page_content=document.page_content, metadata={"doc_id": document.page_content, "doc_hash": "hash"}
            )
        ]

    monkeypatch.setattr(ParentChildIndexProcessor, "split_child_nodes", split)
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "first"})
    assert created.data.status == "completed"
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    original_children = repository.get_children(ref)
    assert original_children is not None
    assert len(original_children) == 1
    old_node_id = original_children[0].index_node_id
    assert old_node_id == "first"
    updated = segment_update(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=created.data.id,
        values={"content": "second", "regenerate_child_chunks": True},
    )
    assert updated.data.status == "completed"
    children = repository.get_children(ref)
    assert children is not None
    assert [child.data.content for child in children] == ["second"]
    assert [child.index_node_id for child in children] == ["second"]
    assert old_node_id not in probe.deleted_ids
    with sqlite_session_factory() as session:
        cleanup = session.scalar(select(ChildVectorCleanupOutbox))
    assert cleanup is not None
    assert cleanup.node_id == old_node_id


@pytest.mark.parametrize(
    ("content", "submitted_summary", "summary_enabled", "expected_summary", "generated"),
    [
        ("new text", "old summary", True, "new summary", True),
        ("new text", None, True, "new summary", True),
        ("old text", "old summary", True, "old summary", False),
        ("new text", "manual summary", True, "manual summary", False),
        ("new text", "", True, None, False),
        ("new text", " \n\t ", True, None, False),
        ("new text", "old summary", False, "old summary", False),
        ("new text", None, False, "old summary", False),
    ],
)
def test_summary_updates_preserve_generation_and_manual_edit_rules(
    segment_update: SegmentUpdateEntry,
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    submitted_summary: str | None,
    summary_enabled: bool,
    expected_summary: str | None,
    generated: bool,
) -> None:
    app, repository, _ = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(
        context, dataset_id="dataset-1", document_id="document-1", values={"content": "old text"}
    )
    reference = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    repository.save_summary(reference, "old summary")
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.summary_index_setting = {
            "enable": summary_enabled,
            "model_name": "llm-model",
            "model_provider_name": "provider",
        }
    calls: list[str] = []

    def generate(_tenant_id: str, text: str, _setting: object, **_kwargs: object) -> tuple[str, LLMUsage]:
        calls.append(text)
        return "new summary", LLMUsage.empty_usage()

    monkeypatch.setattr(ParagraphIndexProcessor, "generate_summary_from_inputs", staticmethod(generate))
    updated = segment_update(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=created.data.id,
        values={"content": content, "summary": submitted_summary},
    )
    assert calls == ([content] if generated else [])
    assert updated.data.summary == expected_summary


def test_manual_summary_update_passes_inherited_enable_owner_to_vector_publication(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "body content"},
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    initial = repository.save_summary(ref, "old summary")
    assert initial.summary is not None
    with sqlite_session_factory.begin() as session:
        segment = session.get_one(DocumentSegment, created.data.id)
        segment.enabled = False
        summary = session.get_one(DocumentSegmentSummary, initial.summary.id)
        summary.status = SummaryStatus.COMPLETED
        summary.summary_index_node_id = "old-summary-node"
        summary.summary_index_node_hash = "old-summary-hash"
        summary.enabled = False

    intent = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory).request(
        document_ref=ref.document,
        segment_ids=[ref.segment_id],
        action="enable",
        actor_id=context.account_id,
    )[0]
    writes_before = probe.writes
    updated = app.update_segment(
        context,
        dataset_id=ref.document.dataset.dataset_id,
        document_id=ref.document.document_id,
        segment_id=ref.segment_id,
        values={"content": "body content", "summary": "new manual summary"},
    )

    assert updated.data.summary == "new manual summary"
    assert probe.writes == writes_before + 1
    with sqlite_session_factory() as session:
        persisted_summary = session.get_one(DocumentSegmentSummary, initial.summary.id)
        persisted_intent = session.get_one(SegmentStateOutbox, intent.id)
        assert persisted_summary.summary_content == "new manual summary"
        assert persisted_summary.status == SummaryStatus.COMPLETED
        assert persisted_summary.enabled is False
        assert persisted_summary.summary_index_node_id != "old-summary-node"
        assert (persisted_intent.summary_id, persisted_intent.summary_version) == (
            persisted_summary.id,
            persisted_summary.version,
        )


def test_manual_summary_update_after_explicit_disable_does_not_publish_an_unowned_node(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "body content"},
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    initial = repository.save_summary(ref, "old summary")
    assert initial.summary is not None
    with sqlite_session_factory.begin() as session:
        segment = session.get_one(DocumentSegment, created.data.id)
        segment.enabled = False
        summary = session.get_one(DocumentSegmentSummary, initial.summary.id)
        summary.status = SummaryStatus.COMPLETED
        summary.summary_index_node_id = "old-summary-node"
        summary.summary_index_node_hash = "old-summary-hash"
        summary.enabled = False

    intent = SQLAlchemySegmentStateTransitions(session_factory=sqlite_session_factory).request(
        document_ref=ref.document,
        segment_ids=[ref.segment_id],
        action="enable",
        actor_id=context.account_id,
    )[0]
    with sqlite_session_factory.begin() as session:
        summary = session.get_one(DocumentSegmentSummary, initial.summary.id)
        summary.version += 1
        summary.enabled = False
        disabled_version = summary.version
    writes_before = probe.writes

    updated = app.update_segment(
        context,
        dataset_id=ref.document.dataset.dataset_id,
        document_id=ref.document.document_id,
        segment_id=ref.segment_id,
        values={"content": "body content", "summary": "manual content after disable"},
    )

    assert updated.data.summary == "manual content after disable"
    assert probe.writes == writes_before
    with sqlite_session_factory() as session:
        persisted_summary = session.get_one(DocumentSegmentSummary, initial.summary.id)
        persisted_intent = session.get_one(SegmentStateOutbox, intent.id)
        assert persisted_summary.version == disabled_version + 1
        assert persisted_summary.enabled is False
        assert persisted_summary.summary_index_node_id == "old-summary-node"
        assert (persisted_intent.summary_id, persisted_intent.summary_version) == (
            initial.summary.id,
            disabled_version - 1,
        )


def test_summary_clear_database_failure_keeps_row_and_remote_vector(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "body content"},
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    saved = repository.save_summary(ref, "summary to clear")
    assert saved.summary is not None
    route = VectorRoute.capture(
        tenant_id=ref.document.dataset.tenant_id,
        dataset_id=ref.document.dataset.dataset_id,
        vector_type="qdrant",
        collection_name=probe.collection_name,
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    node_id = "summary-node-clear-rollback"
    with sqlite_session_factory.begin() as session:
        summary = session.get_one(DocumentSegmentSummary, saved.summary.id)
        summary.summary_index_node_id = node_id
        summary.summary_index_node_hash = "summary-hash-clear-rollback"
        summary.vector_route = route.serialized
        summary.vector_route_hash = route.key
        summary.status = SummaryStatus.COMPLETED
    probe.child_vector_ids.add(node_id)
    engine = sqlite_session_factory.kw["bind"]

    def reject_summary_delete(
        _connection: Connection,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("DELETE FROM DOCUMENT_SEGMENT_SUMMARIES"):
            raise RuntimeError("summary delete rejected")

    event.listen(engine, "before_cursor_execute", reject_summary_delete)
    try:
        updated = app.update_segment(
            context,
            dataset_id=ref.document.dataset.dataset_id,
            document_id=ref.document.document_id,
            segment_id=ref.segment_id,
            values={"content": "body content", "summary": ""},
        )
    finally:
        event.remove(engine, "before_cursor_execute", reject_summary_delete)

    assert updated.data.summary == "summary to clear"
    assert node_id in probe.child_vector_ids
    assert node_id not in probe.deleted_ids
    with sqlite_session_factory() as session:
        persisted = session.get_one(DocumentSegmentSummary, saved.summary.id)
        assert persisted.summary_content == "summary to clear"
        assert persisted.summary_index_node_id == node_id
        assert (
            session.scalar(
                select(SummaryVectorCleanupOutbox).where(
                    SummaryVectorCleanupOutbox.summary_id == saved.summary.id,
                    SummaryVectorCleanupOutbox.node_id == node_id,
                )
            )
            is None
        )


def test_summary_clear_commits_exact_cleanup_before_worker_deletes_vector(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "body content"},
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    saved = repository.save_summary(ref, "summary to clear")
    assert saved.summary is not None
    route = VectorRoute.capture(
        tenant_id=ref.document.dataset.tenant_id,
        dataset_id=ref.document.dataset.dataset_id,
        vector_type="qdrant",
        collection_name=probe.collection_name,
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="summary",
    )
    node_id = "summary-node-clear-success"
    with sqlite_session_factory.begin() as session:
        summary = session.get_one(DocumentSegmentSummary, saved.summary.id)
        summary.summary_index_node_id = node_id
        summary.summary_index_node_hash = "summary-hash-clear-success"
        summary.vector_route = route.serialized
        summary.vector_route_hash = route.key
        summary.status = SummaryStatus.COMPLETED
    probe.child_vector_ids.add(node_id)

    updated = app.update_segment(
        context,
        dataset_id=ref.document.dataset.dataset_id,
        document_id=ref.document.document_id,
        segment_id=ref.segment_id,
        values={"content": "body content", "summary": ""},
    )
    assert updated.data.summary is None
    assert node_id in probe.child_vector_ids
    assert node_id not in probe.deleted_ids
    with sqlite_session_factory() as session:
        assert session.get(DocumentSegmentSummary, saved.summary.id) is None
        cleanup = session.scalar(
            select(SummaryVectorCleanupOutbox).where(
                SummaryVectorCleanupOutbox.summary_id == saved.summary.id,
                SummaryVectorCleanupOutbox.node_id == node_id,
                SummaryVectorCleanupOutbox.vector_route_hash == route.key,
            )
        )
        assert cleanup is not None
        assert cleanup.vector_route == route.serialized
        cleanup_id = cleanup.id

    cleanup_repository = SQLAlchemySummaryVectorCleanupRepository(session_factory=sqlite_session_factory)

    def repository_boundary() -> SQLAlchemySummaryVectorCleanupRepository:
        return cleanup_repository

    def vector_for_exact_route(cleanup_route: VectorRoute) -> Vector:
        assert cleanup_route == route
        return cast(Vector, probe)

    monkeypatch.setattr(summary_cleanup_task_module, "_repository", repository_boundary)
    monkeypatch.setattr(summary_cleanup_task_module, "vector_for_route", vector_for_exact_route)
    summary_cleanup_task_module.process_summary_vector_cleanup_task.run(cleanup_id)

    assert node_id not in probe.child_vector_ids
    assert probe.deleted_ids == [node_id]
    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, cleanup_id) is None


def test_automatic_summary_keeps_knowledge_credit_attribution(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, repository, _ = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(
        context, dataset_id="dataset-1", document_id="document-1", values={"content": "old text"}
    )
    reference = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    repository.save_summary(reference, "old summary")
    setting: SummaryIndexSettingDict = {"enable": True, "model_name": "llm-model", "model_provider_name": "provider"}
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.summary_index_setting = setting
    metadata: list[Mapping[str, object] | None] = []

    class ModelProbe:
        credentials: dict[str, str] = {}

        @property
        def model_type_instance(self) -> "ModelProbe":
            return self

        def get_model_schema(self, _name: str, _credentials: object) -> None:
            return None

        def get_text_embedding_num_tokens(self, texts: Sequence[str]) -> list[int]:
            return [len(text) for text in texts]

        def invoke_llm(self, **_kwargs: object) -> LLMResult:
            metadata.append(get_credit_usage_metadata())
            return LLMResult(
                model="llm-model", message=AssistantPromptMessage(content="new summary"), usage=LLMUsage.empty_usage()
            )

        def get_model_instance(self, **_kwargs: object) -> "ModelProbe":
            return self

    monkeypatch.setattr(ModelManager, "for_tenant", lambda **_kwargs: ModelProbe())
    # Both public summary entry points must preserve the credit-usage context.
    with sqlite_session_factory() as session:
        ParagraphIndexProcessor.generate_summary("workspace-1", "old text", setting, session=session)
    assert metadata.pop() == {"created_by": CreditUsageCreatedBy.KNOWLEDGE_INDEXING}
    updated = app.update_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=created.data.id,
        values={"content": "new text"},
    )
    assert updated.data.summary == "new summary"
    assert metadata == [{"created_by": CreditUsageCreatedBy.KNOWLEDGE_INDEXING}]
    assert get_credit_usage_metadata() is None


@pytest.mark.parametrize("storage_type", ["database", "file"])
def test_economy_index_shares_persistence_with_existing_keyword_calls(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    storage_type: str,
) -> None:
    from core.rag.datasource.keyword.jieba import jieba as jieba_module
    from core.rag.datasource.keyword.keyword_factory import Keyword
    from repositories.knowledge import keyword_table_repository as table_module
    from tests.unit_tests.config_override import apply_config_overrides

    app, repository, probe = indexing_probe
    reads: list[str] = []
    writes: list[str] = []

    class FileStorageProbe:
        def load_once(self, key: str) -> bytes:
            reads.append(key)
            return (tmp_path / key).read_bytes()

        def exists(self, key: str) -> bool:
            return (tmp_path / key).exists()

        def save(self, key: str, data: bytes) -> None:
            writes.append(key)
            path = tmp_path / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        def delete(self, key: str) -> None:
            (tmp_path / key).unlink()

    class KeywordExtractor:
        def extract_keywords(self, text: str, _keyword_number: int = 10) -> set[str]:
            return set(text.split())

    storage = FileStorageProbe()
    monkeypatch.setattr(jieba_module, "storage", storage)
    monkeypatch.setattr(jieba_module, "JiebaKeywordTableHandler", KeywordExtractor)
    apply_config_overrides(monkeypatch, KEYWORD_DATA_SOURCE_TYPE=storage_type)
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.indexing_technique = IndexTechniqueType.ECONOMY
    context = RequestContext("request", None, "author", "workspace-1")
    first = app.create_segment(
        context, dataset_id="dataset-1", document_id="document-1", values={"content": "initial 中文"}
    )
    second = app.create_segment(
        context, dataset_id="dataset-1", document_id="document-1", values={"content": "other", "keywords": ["manual"]}
    )
    updated = app.update_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=first.data.id,
        values={"content": "replacement"},
    )
    assert first.data.status == second.data.status == updated.data.status == "completed"
    assert first.data.index_node_id is not None
    assert second.data.index_node_id is not None
    assert set(first.data.keywords or ()) == {"initial", "中文"}
    assert second.data.keywords == ("manual",)
    assert updated.data.keywords == ("replacement",)
    dataset_ref = DatasetRef("workspace-1", "dataset-1")
    source_type, data = repository.get_keyword_table(dataset_ref)
    payload = table_module.load_keyword_table(
        storage=storage, tenant_id="workspace-1", dataset_id="dataset-1", storage_type=source_type, data=data
    )
    assert payload is not None
    assert payload["__data__"]["table"] == {
        "replacement": {first.data.index_node_id},
        "manual": {second.data.index_node_id},
    }
    if storage_type == "database":
        assert reads == writes == []
    else:
        assert len(writes) == 3
        assert len(set(writes)) == 3
        assert all(key.startswith("keyword_files/workspace-1/dataset-1/") and key.endswith(".txt") for key in writes)

    # The existing session-taking API must consume the exact payload written by
    # the repository-backed path. Only that existing API retains caller sessions.
    class SessionFileStorage:
        def load_once(self, key: str) -> bytes:
            return (tmp_path / key).read_bytes()

        def exists(self, key: str) -> bool:
            return (tmp_path / key).exists()

        def save(self, key: str, data: bytes) -> None:
            (tmp_path / key).write_bytes(data)

        def delete(self, key: str) -> None:
            (tmp_path / key).unlink()

    session_storage = SessionFileStorage()
    monkeypatch.setattr(jieba_module, "storage", session_storage)
    monkeypatch.setattr(jieba_module, "redis_client", IndexRedisStub())
    with sqlite_session_factory() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        keyword = Keyword(dataset)
        assert keyword.text_exists(first.data.index_node_id, session=session)
        assert keyword.text_exists(second.data.index_node_id, session=session)
        keyword.delete_by_ids([first.data.index_node_id], session)
        session.commit()
    source_type, data = repository.get_keyword_table(dataset_ref)
    payload = table_module.load_keyword_table(
        storage=session_storage, tenant_id="workspace-1", dataset_id="dataset-1", storage_type=source_type, data=data
    )
    assert payload is not None
    assert payload["__data__"]["table"] == {"manual": {second.data.index_node_id}}


@pytest.mark.parametrize("generated_summary", [None, "", " \n\t "], ids=["exception", "empty", "whitespace"])
def test_summary_generation_failure_preserves_the_completed_segment(
    segment_update: SegmentUpdateEntry,
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    generated_summary: str | None,
) -> None:
    from models.dataset import DocumentSegmentSummary

    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(
        context, dataset_id="dataset-1", document_id="document-1", values={"content": "original"}
    )
    reference = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(created.data.id)
    repository.save_summary(reference, "original summary")
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.summary_index_setting = {"enable": True, "model_name": "model", "model_provider_name": "provider"}
        summary = session.scalar(select(DocumentSegmentSummary))
        assert summary is not None
        summary.summary_index_node_id = "original-summary-node"

    def generate(*_args: object, **_kwargs: object) -> tuple[str, LLMUsage]:
        assert not probe.active_transactions
        if generated_summary is not None:
            return generated_summary, LLMUsage.empty_usage()
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(ParagraphIndexProcessor, "generate_summary_from_inputs", staticmethod(generate))
    updated = segment_update(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=created.data.id,
        values={"content": "replacement", "summary": "original summary"},
    )
    assert updated.data.content == "replacement"
    assert updated.data.enabled is True
    assert updated.data.status == "completed"
    assert updated.data.summary == "original summary"
    assert "original-summary-node" not in probe.deleted_ids
    with sqlite_session_factory() as session:
        summary = session.scalar(select(DocumentSegmentSummary))
        assert summary is not None
        assert summary.status == "error"
        assert summary.summary_index_node_id == "original-summary-node"
        assert summary.error == ("Generated summary is empty" if generated_summary is not None else "LLM unavailable")


@pytest.mark.parametrize("failed_commits", [1, 2, 3])
def test_summary_commit_retries_use_fresh_sessions_and_persist_outcome(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    failed_commits: int,
) -> None:
    from models.dataset import DocumentSegmentSummary
    from services.knowledge.summaries import adapters as summary_module

    app, _, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    created = app.create_segment(
        context, dataset_id="dataset-1", document_id="document-1", values={"content": "original"}
    )
    attempts: list[Session] = []
    delays: list[float] = []

    def fail_summary_commit(execute_state: ORMExecuteState) -> None:
        statement = execute_state.statement
        completed_update = (
            isinstance(statement, Update) and SummaryStatus.COMPLETED in statement.compile().params.values()
        )
        if completed_update:
            attempts.append(execute_state.session)
            if len(attempts) <= failed_commits:
                raise ConnectionError("connection lost before commit")

    def wait(seconds: float) -> None:
        assert not probe.active_transactions
        delays.append(seconds)

    monkeypatch.setattr(summary_module.time, "sleep", wait)
    event.listen(sqlite_session_factory.class_, "do_orm_execute", fail_summary_commit)
    try:
        updated = app.update_segment(
            context,
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id=created.data.id,
            values={"content": "original", "summary": "manual summary"},
        )
        assert updated.data.status == "completed"
        assert updated.data.enabled is True
        assert not probe.active_transactions
    finally:
        event.remove(sqlite_session_factory.class_, "do_orm_execute", fail_summary_commit)
        for session in attempts:
            session.close()

    assert len(attempts) == min(failed_commits + 1, 3)
    assert len({id(session) for session in attempts}) == len(attempts)
    assert delays == [2.0, 4.0][: min(failed_commits, 2)]
    with sqlite_session_factory() as session:
        summary = session.scalar(select(DocumentSegmentSummary))
        assert summary is not None
        assert summary.summary_content == "manual summary"
        if failed_commits < 3:
            assert summary.status == SummaryStatus.COMPLETED
            assert summary.summary_index_node_id is not None
            assert summary.summary_index_node_hash is not None
            assert summary.error is None
        else:
            assert summary.status == SummaryStatus.ERROR
            assert summary.error == "Vectorization failed: connection lost before commit"


def test_summary_noop_does_not_open_a_database_transaction(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    from services.knowledge.segments.indexing import SegmentIndexingGateway

    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    gateway = SegmentIndexingGateway(
        segments=repository,
        uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory),
        attachment_vectors=SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory),
        child_vector_cleanups=SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory),
        redis=IndexRedisStub(),
        new_session=sqlite_session_factory,
    )
    engine = sqlite_session_factory.kw["bind"]
    connections: list[Connection] = []
    record_connection = connections.append
    event.listen(engine, "begin", record_connection)
    try:
        gateway.update_summary(
            DatasetRef("workspace-1", "dataset-1").document("document-1").segment("segment-1"),
            summary=None,
            content_changed=False,
        )
    finally:
        event.remove(engine, "begin", record_connection)
    assert connections == []


def test_service_api_child_mutations_refresh_rows_and_release_read_transactions(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, repository, _ = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    with sqlite_session_factory() as session:
        dataset = session.get(Dataset, "dataset-1")
        document = session.get(Document, "document-1")
        segment = session.get(DocumentSegment, parent.data.id)
        assert dataset is not None
        assert document is not None
        assert segment is not None
        child = SegmentService.create_child_chunk(
            "first",
            segment,
            document,
            dataset,
            session,
            actor_id="creator",
            mutations=app.mutations,
        )
        child_id = child.id
        assert child.position == 1
        assert child.created_by == "creator"
        updated = SegmentService.update_child_chunk(
            "second",
            child,
            segment,
            document,
            dataset,
            session,
            actor_id="editor",
            mutations=app.mutations,
        )
        assert updated.content == "second"
        assert updated.updated_by == "editor"
        SegmentService.delete_child_chunk(updated, dataset, session, mutations=app.mutations)
        assert session.get(ChildChunk, child_id) is None
        SegmentService.delete_segment(segment, document, dataset, session, mutations=app.mutations)
        assert document.word_count == 0
    assert repository.get_segment(ref) is None


@pytest.mark.parametrize("operation", ["create", "update"])
def test_service_api_child_index_failure_preserves_rows_and_legacy_errors(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    operation: str,
) -> None:
    from services.errors.chunk import ChildChunkIndexingError

    app, repository, probe = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(context, dataset_id="dataset-1", document_id="document-1", values={"content": "parent"})
    created = app.create_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        content="original",
    )
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    before = repository.get_children(ref)
    probe.fail = True
    expected = ChildChunkIndexingError
    with sqlite_session_factory() as session:
        dataset = session.get(Dataset, "dataset-1")
        document = session.get(Document, "document-1")
        segment = session.get(DocumentSegment, parent.data.id)
        child = session.get(ChildChunk, created.id)
        assert dataset is not None
        assert document is not None
        assert segment is not None
        assert child is not None

        def mutate() -> None:
            if operation == "create":
                SegmentService.create_child_chunk(
                    "new",
                    segment,
                    document,
                    dataset,
                    session,
                    actor_id="editor",
                    mutations=app.mutations,
                )
            elif operation == "update":
                SegmentService.update_child_chunk(
                    "changed",
                    child,
                    segment,
                    document,
                    dataset,
                    session,
                    actor_id="editor",
                    mutations=app.mutations,
                )

        with pytest.raises(expected, match="index unavailable"):
            mutate()
    assert repository.get_children(ref) == before


@pytest.mark.parametrize(
    ("values", "expected_content", "expected_answer"),
    [
        ({"keywords": ["updated"]}, "question", "answer"),
        ({"answer": None}, "question", None),
        ({"answer": "new answer"}, "question", "new answer"),
        ({"content": "new question", "answer": "new answer"}, "new question", "new answer"),
    ],
)
def test_service_api_partial_qa_update_uses_shared_word_count_and_indexing(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    values: dict[str, object],
    expected_content: str,
    expected_answer: str | None,
) -> None:
    app, _, _ = indexing_probe
    with sqlite_session_factory.begin() as session:
        document = session.get(Document, "document-1")
        assert document is not None
        document.doc_form = IndexStructureType.QA_INDEX
    created = app.create_segment(
        RequestContext("request", None, "author", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "question", "answer": "answer"},
    )
    with sqlite_session_factory() as session:
        dataset = session.get(Dataset, "dataset-1")
        document = session.get(Document, "document-1")
        segment = session.get(DocumentSegment, created.data.id)
        assert dataset is not None
        assert document is not None
        assert segment is not None
        updated = SegmentService.update_segment(
            SegmentUpdateArgs.model_validate(values),
            segment,
            document,
            dataset,
            session,
            actor_id="editor",
            mutations=app.mutations,
        )
        assert updated.content == expected_content
        assert updated.answer == expected_answer
        assert updated.word_count == len(expected_content) + len(expected_answer or "")
        assert document.word_count == updated.word_count
        if "content" in values or "answer" in values:
            assert updated.tokens == updated.word_count
        if "keywords" in values:
            assert updated.keywords == values["keywords"]


def test_batch_child_creation_continues_after_highest_existing_position(
    indexing_probe: IndexingProbe,
) -> None:
    app, repository, _ = indexing_probe
    context = RequestContext("request", None, "author", "workspace-1")
    parent = app.create_segment(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "parent"},
    )
    segment_ref = DatasetRef("workspace-1", "dataset-1").document("document-1").segment(parent.data.id)
    existing = [
        app.create_child_chunk(
            context,
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id=parent.data.id,
            content=content,
        )
        for content in ("first", "second", "third")
    ]
    app.delete_child_chunk(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        child_chunk_id=existing[1].id,
    )

    added = app.update_child_chunks(
        context,
        dataset_id="dataset-1",
        document_id="document-1",
        segment_id=parent.data.id,
        chunks=[
            ChildChunkUpdateArgs(id=existing[0].id, content=existing[0].content),
            ChildChunkUpdateArgs(id=existing[2].id, content=existing[2].content),
            ChildChunkUpdateArgs(content="fourth"),
            ChildChunkUpdateArgs(content="fifth"),
        ],
    )

    assert [chunk.position for chunk in added] == [4, 5]
    persisted = repository.get_children(segment_ref)
    assert persisted is not None
    assert [(child.data.content, child.data.position) for child in persisted] == [
        ("first", 1),
        ("third", 3),
        ("fourth", 4),
        ("fifth", 5),
    ]


def test_segment_creation_continues_after_highest_existing_position(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, _, _ = indexing_probe
    with sqlite_session_factory.begin() as session:
        for segment_id, position in (("first", 1), ("third", 3)):
            segment = _segment(segment_id, "workspace-1", "dataset-1", "document-1")
            segment.position = position
            session.add(segment)

    created = app.create_segment(
        RequestContext("request", None, "author", "workspace-1"),
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "fourth"},
    )

    assert created.data.position == 4


@pytest.mark.parametrize("owner_field", ["tenant_id", "dataset_id", "document_id"])
def test_legacy_segment_mutation_rejects_mismatched_rows_before_dispatch(owner_field: str) -> None:
    from services.knowledge.segments.application import SegmentMutationService

    dataset = _dataset("dataset-1", "workspace-1")
    document = _document("document-1", "workspace-1", "dataset-1")
    segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
    setattr(segment, owner_field, "foreign-owner")
    mutations = create_autospec(SegmentMutationService, instance=True, spec_set=True)
    session = create_autospec(Session, instance=True, spec_set=True)
    with pytest.raises(ValueError, match="Segment does not belong"):
        SegmentService.update_segment(
            SegmentUpdateArgs(content="new"),
            segment,
            document,
            dataset,
            session,
            actor_id="author",
            mutations=mutations,
        )
    mutations.update_segment.assert_not_called()
    session.commit.assert_not_called()


def _persist_index_batch(
    session_factory: sessionmaker[Session], *, doc_form: IndexStructureType = IndexStructureType.PARAGRAPH_INDEX
) -> None:
    with session_factory.begin() as session:
        document = session.get(Document, "document-1")
        assert document is not None
        document.doc_form = doc_form
        for position, content in enumerate(("first", "second"), start=1):
            segment = _segment(content, "workspace-1", "dataset-1", "document-1")
            segment.position = position
            segment.content = content
            segment.index_node_id = f"node-{content}"
            segment.index_node_hash = f"hash-{content}"
            session.add(segment)


def _install_pending_batch_generation(session_factory: sessionmaker[Session]) -> str:
    with session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        representation = DatasetIndexingRepresentation(
            indexing_technique=str(dataset.indexing_technique),
            index_struct=dataset.index_struct,
            collection_binding_id=dataset.collection_binding_id,
            embedding_model=dataset.embedding_model,
            embedding_model_provider=dataset.embedding_model_provider,
            keyword_number=dataset.keyword_number,
            is_multimodal=dataset.is_multimodal,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            summary_index_setting=None,
        )
        (intent_id,) = create_dataset_indexing_intents(
            session,
            DatasetIndexingJob(DatasetRef("workspace-1", "dataset-1"), representation, representation),
        )
        return intent_id


@pytest.mark.parametrize("doc_form", [IndexStructureType.PARAGRAPH_INDEX, IndexStructureType.QA_INDEX])
def test_batch_index_materializes_ordered_documents_before_one_vector_write(
    indexing_probe: IndexingProbe, sqlite_session_factory: sessionmaker[Session], doc_form: IndexStructureType
) -> None:
    app, _, probe = indexing_probe
    _persist_index_batch(sqlite_session_factory, doc_form=doc_form)

    app.mutations.index_segments(
        DatasetRef("workspace-1", "dataset-1").document("document-1"), segment_ids=["second", "first"]
    )

    assert probe.batches == [("second", "first")]
    assert probe.writes == 2
    assert not probe.active_transactions


@pytest.mark.parametrize(("changed_segment_id", "expected_batch"), [("first", ("second",)), ("second", ("first",))])
def test_batch_index_skips_only_member_changed_after_snapshot(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    changed_segment_id: str,
    expected_batch: tuple[str, ...],
) -> None:
    app, _, probe = indexing_probe
    _persist_index_batch(sqlite_session_factory)
    intent_id = _install_pending_batch_generation(sqlite_session_factory)
    route_reads = 0

    def route_after_concurrent_edit() -> tuple[str, str]:
        nonlocal route_reads
        route_reads += 1
        assert route_reads == 1
        # create_many has already materialized both detached snapshots. This
        # independent committed edit must invalidate only that member.
        with sqlite_session_factory.begin() as writer:
            changed = writer.get_one(DocumentSegment, changed_segment_id)
            changed.content = f"{changed_segment_id}-current"
            changed.word_count = len(changed.content)
            changed.tokens = 17
            changed.index_node_id = f"node-{changed_segment_id}-current"
            changed.index_node_hash = f"hash-{changed_segment_id}-current"
        return "qdrant", probe.collection_name

    monkeypatch.setattr(probe, "get_routing_target", route_after_concurrent_edit)

    app.mutations.index_segments(
        DatasetRef("workspace-1", "dataset-1").document("document-1"), segment_ids=["first", "second"]
    )

    assert probe.batches == [expected_batch]
    assert probe.writes == 1
    with sqlite_session_factory() as observer:
        changed = observer.get_one(DocumentSegment, changed_segment_id)
        assert (changed.content, changed.tokens, changed.index_node_id, changed.index_node_hash) == (
            f"{changed_segment_id}-current",
            17,
            f"node-{changed_segment_id}-current",
            f"hash-{changed_segment_id}-current",
        )
        target = DatasetIndexingRepresentation.model_validate_json(
            observer.get_one(DatasetIndexingOutbox, intent_id).target_representation
        )
        assert [(route.vector_type, route.collection_name) for route in target.cleanup_routes] == [
            ("qdrant", probe.collection_name)
        ]


def test_batch_index_failure_persists_errors_on_every_committed_segment(
    indexing_probe: IndexingProbe, sqlite_session_factory: sessionmaker[Session]
) -> None:
    app, _, probe = indexing_probe
    _persist_index_batch(sqlite_session_factory)
    probe.fail = True

    with pytest.raises(RuntimeError, match="index unavailable"):
        app.mutations.index_segments(
            DatasetRef("workspace-1", "dataset-1").document("document-1"), segment_ids=["first", "second"]
        )

    with sqlite_session_factory() as session:
        segments = session.scalars(select(DocumentSegment)).all()
        assert len(segments) == 2
        assert all(segment.status == "error" and not segment.enabled for segment in segments)
        assert all(segment.error == "index unavailable" and segment.disabled_at is not None for segment in segments)


def test_batch_index_failure_marks_only_still_current_attempted_members(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, probe = indexing_probe
    _persist_index_batch(sqlite_session_factory)

    def route_after_concurrent_edit() -> tuple[str, str]:
        with sqlite_session_factory.begin() as writer:
            first = writer.get_one(DocumentSegment, "first")
            first.content = "first-current"
            first.tokens = 23
            first.index_node_hash = "hash-first-current"
        return "qdrant", probe.collection_name

    monkeypatch.setattr(probe, "get_routing_target", route_after_concurrent_edit)
    probe.fail = True

    with pytest.raises(RuntimeError, match="index unavailable"):
        app.mutations.index_segments(
            DatasetRef("workspace-1", "dataset-1").document("document-1"),
            segment_ids=["first", "second"],
        )

    assert probe.batches == [("second",)]
    with sqlite_session_factory() as observer:
        first = observer.get_one(DocumentSegment, "first")
        second = observer.get_one(DocumentSegment, "second")
        assert (first.content, first.tokens, first.index_node_hash) == (
            "first-current",
            23,
            "hash-first-current",
        )
        assert first.enabled is True
        assert first.status == SegmentStatus.COMPLETED
        assert first.error is None
        assert second.enabled is False
        assert second.status == SegmentStatus.ERROR
        assert second.error == "index unavailable"


def test_batch_route_resolution_failure_cas_preserves_concurrently_edited_member(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, probe = indexing_probe
    _persist_index_batch(sqlite_session_factory)

    def fail_route_after_concurrent_edit() -> tuple[str, str]:
        with sqlite_session_factory.begin() as writer:
            first = writer.get_one(DocumentSegment, "first")
            first.content = "first-current"
            first.tokens = 29
            first.index_node_hash = "hash-first-current"
        raise RuntimeError("route unavailable")

    monkeypatch.setattr(probe, "get_routing_target", fail_route_after_concurrent_edit)

    with pytest.raises(RuntimeError, match="route unavailable"):
        app.mutations.index_segments(
            DatasetRef("workspace-1", "dataset-1").document("document-1"),
            segment_ids=["first", "second"],
        )

    assert probe.batches == []
    with sqlite_session_factory() as observer:
        first = observer.get_one(DocumentSegment, "first")
        second = observer.get_one(DocumentSegment, "second")
        assert (first.content, first.tokens, first.index_node_hash) == (
            "first-current",
            29,
            "hash-first-current",
        )
        assert first.enabled is True
        assert first.status == SegmentStatus.COMPLETED
        assert first.error is None
        assert second.enabled is False
        assert second.status == SegmentStatus.ERROR
        assert second.error == "route unavailable"


@pytest.mark.parametrize("owner_field", ["tenant_id", "dataset_id", "document_id"])
def test_batch_index_validates_every_segment_owner_before_external_io(
    indexing_probe: IndexingProbe, sqlite_session_factory: sessionmaker[Session], owner_field: str
) -> None:
    app, _, probe = indexing_probe
    _persist_index_batch(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        foreign = session.get(DocumentSegment, "second")
        assert foreign is not None
        setattr(foreign, owner_field, "foreign-owner")

    with pytest.raises(LookupError, match="Segment no longer exists"):
        app.mutations.index_segments(
            DatasetRef("workspace-1", "dataset-1").document("document-1"), segment_ids=["first", "second"]
        )

    assert not probe.batches
    with sqlite_session_factory() as session:
        foreign = session.get(DocumentSegment, "second")
        assert foreign is not None
        assert foreign.status == "completed"


def test_batch_index_preserves_owned_attachments(
    indexing_probe: IndexingProbe, sqlite_session_factory: sessionmaker[Session]
) -> None:
    app, _, probe = indexing_probe
    _persist_index_batch(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.is_multimodal = True
        session.add_all([_upload("owned", "workspace-1", "image.png"), _upload("foreign", "workspace-2", "other.png")])
        for file_id in ("owned", "foreign"):
            session.add(
                SegmentAttachmentBinding(
                    tenant_id="workspace-1",
                    dataset_id="dataset-1",
                    document_id="document-1",
                    segment_id="first",
                    attachment_id=file_id,
                )
            )
        session.add(
            SegmentAttachmentBinding(
                tenant_id="workspace-1",
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="second",
                attachment_id="owned",
            )
        )

    app.mutations.index_segments(
        DatasetRef("workspace-1", "dataset-1").document("document-1"), segment_ids=["first", "second"]
    )

    assert probe.writes == 3
    with sqlite_session_factory() as session:
        assert len(session.scalars(select(SegmentAttachmentBinding)).all()) == 3


def test_batch_create_persists_deduplicated_bindings_and_indexing_snapshots(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _document("document-1", "workspace-1", "dataset-1"),
                _upload("image-1", "workspace-1", "one.png"),
                _upload("image-2", "workspace-1", "two.png"),
            ]
        )
    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")

    repository.create_segments(
        document_ref,
        (
            SegmentCreateData(
                id="segment-1",
                content="first",
                word_count=5,
                tokens=1,
                created_by="author",
                status=SegmentStatus.COMPLETED,
                attachment_ids=("image-1", "image-1", "image-2"),
            ),
            SegmentCreateData(
                id="segment-2",
                content="second",
                word_count=6,
                tokens=1,
                created_by="author",
                status=SegmentStatus.COMPLETED,
                attachment_ids=("image-2",),
            ),
        ),
    )

    snapshots = repository.get_indexing_snapshots(document_ref, ("segment-2", "segment-1"))
    assert [snapshot.segment.id for snapshot in snapshots] == ["segment-2", "segment-1"]
    assert [snapshot.attachment_ids for snapshot in snapshots] == [("image-2",), ("image-1", "image-2")]
    with sqlite_session_factory() as session:
        bindings = session.scalars(
            select(SegmentAttachmentBinding).order_by(
                SegmentAttachmentBinding.segment_id, SegmentAttachmentBinding.attachment_id
            )
        ).all()
        document = session.get_one(Document, "document-1")
    assert [(binding.segment_id, binding.attachment_id) for binding in bindings] == [
        ("segment-1", "image-1"),
        ("segment-1", "image-2"),
        ("segment-2", "image-2"),
    ]
    assert document.word_count == 11


def test_application_batch_create_indexes_persisted_attachment_snapshots(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    app, repository, probe = indexing_probe
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.is_multimodal = True
        session.add_all(
            [
                _upload("image-1", "workspace-1", "one.png"),
                _upload("image-2", "workspace-1", "two.png"),
            ]
        )
    scope = SegmentMutationScope(
        SegmentDatasetRecord("dataset-1", "workspace-1", "high_quality", None, None),
        SegmentDocumentRecord("document-1", "dataset-1", "workspace-1", "text_model"),
    )

    refs = app.mutations.create_segments(
        scope,
        actor_id="author",
        args_list=(
            SegmentUpdateArgs(content="first", attachment_ids=["image-1", "image-2", "image-1"]),
            SegmentUpdateArgs(content="second", attachment_ids=["image-2"]),
        ),
        token_counts=(1, 1),
    )
    segment_ids = [ref.segment_id for ref in refs]

    snapshots = repository.get_indexing_snapshots(scope.document.ref, segment_ids)
    assert [snapshot.attachment_ids for snapshot in snapshots] == [
        ("image-1", "image-2"),
        ("image-2",),
    ]
    app.mutations.index_segments(scope.document.ref, segment_ids=segment_ids)
    assert probe.writes == 4  # two body documents plus two unique attachment vectors


@pytest.mark.parametrize("invalid_attachment_id", ["missing", "foreign"])
def test_batch_create_rejects_invalid_attachment_without_partial_rows(
    sqlite_session_factory: sessionmaker[Session], invalid_attachment_id: str
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _document("document-1", "workspace-1", "dataset-1"),
                _upload("owned", "workspace-1", "owned.png"),
                _upload("foreign", "workspace-2", "foreign.png"),
            ]
        )
    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")

    with pytest.raises(LookupError, match="Attachment files no longer exist"):
        repository.create_segments(
            document_ref,
            (
                SegmentCreateData(
                    id="segment-1",
                    content="first",
                    word_count=5,
                    tokens=1,
                    created_by="author",
                    status=SegmentStatus.COMPLETED,
                    attachment_ids=("owned",),
                ),
                SegmentCreateData(
                    id="segment-2",
                    content="second",
                    word_count=6,
                    tokens=1,
                    created_by="author",
                    status=SegmentStatus.COMPLETED,
                    attachment_ids=(invalid_attachment_id,),
                ),
            ),
        )

    with sqlite_session_factory() as session:
        assert session.scalars(select(DocumentSegment)).all() == []
        assert session.scalars(select(SegmentAttachmentBinding)).all() == []
        assert session.get_one(Document, "document-1").word_count == 0


def test_batch_index_keeps_manual_keywords_aligned_with_segments(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.rag.datasource.keyword.jieba import jieba as jieba_module
    from tests.unit_tests.config_override import apply_config_overrides

    app, _, probe = indexing_probe
    _persist_index_batch(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        dataset = session.get(Dataset, "dataset-1")
        assert dataset is not None
        dataset.indexing_technique = IndexTechniqueType.ECONOMY

    class Extractor:
        def extract_keywords(self, text: str, _keyword_number: int = 10) -> set[str]:
            assert probe.active_transactions, "Economy publication must hold the Dataset generation guard"
            return {text}

    monkeypatch.setattr(jieba_module, "JiebaKeywordTableHandler", Extractor)
    apply_config_overrides(monkeypatch, KEYWORD_DATA_SOURCE_TYPE="database")
    app.mutations.index_segments(
        DatasetRef("workspace-1", "dataset-1").document("document-1"),
        segment_ids=["second", "first"],
        keywords_list=[["manual"], None],
    )

    with sqlite_session_factory() as session:
        assert {s.id: s.keywords for s in session.scalars(select(DocumentSegment))} == {
            "first": ["first"],
            "second": ["manual"],
        }
    assert not probe.batches


def test_batch_parent_child_index_uses_shared_splitter_without_open_transactions(
    indexing_probe: IndexingProbe,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.rag.index_processor.processor.parent_child_index_processor import ParentChildIndexProcessor
    from models.dataset import DatasetProcessRule
    from models.enums import ProcessRuleMode

    app, repository, probe = indexing_probe
    _persist_index_batch(sqlite_session_factory, doc_form=IndexStructureType.PARENT_CHILD_INDEX)
    with sqlite_session_factory.begin() as session:
        rule = DatasetProcessRule(
            dataset_id="dataset-1",
            mode=ProcessRuleMode.HIERARCHICAL,
            rules='{"parent_mode":"full-doc","subchunk_segmentation":{"max_tokens":100}}',
            created_by="author",
        )
        session.add(rule)
        document = session.get(Document, "document-1")
        dataset = session.get(Dataset, "dataset-1")
        assert document is not None
        assert dataset is not None
        dataset.embedding_model_provider = "provider"
        document.dataset_process_rule_id = rule.id

    def split(_self, document: IndexDocument, _rule, _mode, _model) -> list[ChildDocument]:
        assert not probe.active_transactions
        return [
            ChildDocument(
                page_content=document.page_content, metadata={"doc_id": document.page_content, "doc_hash": "hash"}
            )
        ]

    monkeypatch.setattr(ParentChildIndexProcessor, "split_child_nodes", split)
    document_ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    app.mutations.index_segments(document_ref, segment_ids=["first", "second"])

    assert probe.child_vector_ids == {"first", "second"}
    for segment_id in ("first", "second"):
        children = repository.get_children(document_ref.segment(segment_id))
        assert children is not None
        assert len(children) == 1
        assert children[0].data.content == segment_id


@pytest.mark.parametrize("entry", ["detail", "list"])
@pytest.mark.parametrize("delete_after_read", [False, True])
def test_segment_presentation_signs_after_repository_connections_are_released(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    delete_after_read: bool,
) -> None:
    from uuid import uuid4

    from repositories.knowledge import segment_read_adapter

    file_id = str(uuid4())
    with sqlite_session_factory.begin() as session:
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.content = f"![image](/files/{file_id}/image-preview?sign=old)"
        session.add_all(
            [
                _dataset("dataset-1", "workspace-1"),
                _document("document-1", "workspace-1", "dataset-1"),
                segment,
                _upload(file_id, "workspace-1", "image.png"),
                SegmentAttachmentBinding(
                    tenant_id="workspace-1",
                    dataset_id="dataset-1",
                    document_id="document-1",
                    segment_id="segment-1",
                    attachment_id=file_id,
                ),
            ]
        )
    engine = sqlite_session_factory.kw["bind"]
    active: set[object] = set()
    signed: list[str] = []
    statements: list[str] = []
    original_sign = segment_read_adapter._sign_upload_file_uri

    def checkout(_connection: object, record: object, _proxy: object) -> None:
        active.add(record)

    def checkin(_connection: object, record: object) -> None:
        active.discard(record)

    def sign(upload_file_id: str, *, preview_kind: Literal["image", "file"] = "file") -> str:
        assert not active
        signed.append(upload_file_id)
        return original_sign(upload_file_id, preview_kind=preview_kind)

    def executed(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        statements.append(statement)

    monkeypatch.setattr(segment_read_adapter, "_sign_upload_file_uri", sign)
    event.listen(engine, "checkout", checkout)
    event.listen(engine, "checkin", checkin)
    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    try:
        if entry == "detail":
            detail = repository.get_segment(ref.segment("segment-1"))
            assert detail is not None
            snapshots = (detail.data,)
        else:
            snapshots = repository.list_segments(ref, SegmentListFilter()).items
        assert not signed
        assert not active
        assert "sign_content" not in snapshots[0].values
        assert not hasattr(snapshots[0].attachments[0], "source_url")
        if delete_after_read:
            with sqlite_session_factory.begin() as session:
                session.delete(session.get(DocumentSegment, "segment-1"))
        event.listen(engine, "before_cursor_execute", executed)
        records = SegmentPresentationAdapter().present(snapshots)
        assert not active
        assert not statements
    finally:
        if event.contains(engine, "before_cursor_execute", executed):
            event.remove(engine, "before_cursor_execute", executed)
        event.remove(engine, "checkout", checkout)
        event.remove(engine, "checkin", checkin)
    assert signed == [file_id, file_id]
    assert "sign=old" not in records[0].sign_content
    assert "?timestamp=" in records[0].sign_content
    assert "?timestamp=" in records[0].attachments[0].source_url


def test_segment_list_batches_content_file_authorization(sqlite_session_factory: sessionmaker[Session]) -> None:
    from uuid import uuid4

    file_ids = [str(uuid4()) for _ in range(100)]
    with sqlite_session_factory.begin() as session:
        session.add_all([_dataset("dataset-1", "workspace-1"), _document("document-1", "workspace-1", "dataset-1")])
        for index, file_id in enumerate(file_ids):
            segment = _segment(f"segment-{index}", "workspace-1", "dataset-1", "document-1")
            segment.position = index
            session.add_all([segment, _upload(file_id, "workspace-1", "image.png")])

    engine = sqlite_session_factory.kw["bind"]
    statements: list[str] = []

    def executed(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    event.listen(engine, "before_cursor_execute", executed)
    try:
        repository.list_segments(ref, SegmentListFilter(limit=100))
        without_links = len(statements)
        with sqlite_session_factory.begin() as session:
            for index, file_id in enumerate(file_ids):
                segment = session.get(DocumentSegment, f"segment-{index}")
                assert segment is not None
                segment.content = f"![image](/files/{file_id}/image-preview?sign=old)"
        statements.clear()
        page = repository.list_segments(ref, SegmentListFilter(limit=100))
        with_links = len(statements)
        records = SegmentPresentationAdapter().present(page.items)
        assert len(statements) == with_links
    finally:
        event.remove(engine, "before_cursor_execute", executed)

    assert with_links == without_links + 1
    assert len(records) == 100
    assert all("?timestamp=" in record.sign_content for record in records)
    assert all(
        snapshot.file_access.uploads == frozenset([file_ids[index]]) for index, snapshot in enumerate(page.items)
    )


@pytest.mark.parametrize(
    ("owner_change", "expected_ids"),
    [
        ("valid", ("segment-1",)),
        ("dataset_tenant", ()),
        ("document_tenant", ()),
        ("document_dataset", ()),
        ("document_status", ()),
        ("document_disabled", ()),
        ("document_archived", ()),
        ("segment_tenant", ()),
        ("segment_dataset", ()),
        ("segment_status", ()),
        ("segment_disabled", ()),
    ],
)
def test_retrieval_snapshot_batch_authorizes_complete_current_owner_chain(
    sqlite_session_factory: sessionmaker[Session],
    owner_change: str,
    expected_ids: tuple[str, ...],
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        document = _document("document-1", "workspace-1", "dataset-1")
        document.indexing_status = IndexingStatus.COMPLETED
        segment = _segment("segment-1", "workspace-1", "dataset-1", "document-1")
        segment.index_node_id = "node-1"
        if owner_change == "dataset_tenant":
            dataset.tenant_id = "other-workspace"
        elif owner_change == "document_tenant":
            document.tenant_id = "other-workspace"
        elif owner_change == "document_dataset":
            document.dataset_id = "other-dataset"
        elif owner_change == "document_status":
            document.indexing_status = IndexingStatus.ERROR
        elif owner_change == "document_disabled":
            document.enabled = False
        elif owner_change == "document_archived":
            document.archived = True
        elif owner_change == "segment_tenant":
            segment.tenant_id = "other-workspace"
        elif owner_change == "segment_dataset":
            segment.dataset_id = "other-dataset"
        elif owner_change == "segment_status":
            segment.status = SegmentStatus.ERROR
        elif owner_change == "segment_disabled":
            segment.enabled = False
        session.add_all([dataset, document, segment])

    snapshot = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory).get_retrieval_snapshot(
        DatasetRef("workspace-1", "dataset-1"),
        node_pairs=(("document-1", "node-1"),),
        summary_pairs=(),
        attachment_pairs=(),
    )

    assert tuple(segment.id for segment in snapshot.segments) == expected_ids
    assert snapshot.children == ()
    assert snapshot.attachments == ()
    assert snapshot.summaries == {}


@pytest.mark.parametrize("owner_change", ["dataset_tenant", "document_tenant", "document_dataset", "deleted_document"])
def test_segment_list_does_not_authorize_rows_with_changed_owners(
    sqlite_session_factory: sessionmaker[Session], owner_change: str
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        document = _document("document-1", "workspace-1", "dataset-1")
        session.add_all([dataset, document, _segment("segment-1", "workspace-1", "dataset-1", "document-1")])
        session.flush()
        if owner_change == "dataset_tenant":
            dataset.tenant_id = "other-workspace"
        elif owner_change == "document_tenant":
            document.tenant_id = "other-workspace"
        elif owner_change == "document_dataset":
            document.dataset_id = "other-dataset"
        else:
            session.delete(document)

    repository = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef("workspace-1", "dataset-1").document("document-1")
    page = repository.list_segments(ref, SegmentListFilter())

    assert page.items == ()
    assert page.total == 0

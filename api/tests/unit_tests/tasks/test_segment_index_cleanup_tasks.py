import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.index_processor.constant.index_type import IndexStructureType
from extensions.storage.storage_type import StorageType
from graphon.file import FileTransferMethod, FileType
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary, SegmentAttachmentBinding
from models.enums import CreatorUserRole, DataSourceType, DocumentCreatedFrom, IndexingStatus, SegmentStatus
from models.model import (
    AttachmentVectorCleanupOutbox,
    MessageFile,
    SummaryVectorCleanupOutbox,
    UploadFile,
    UploadFileCleanupOutbox,
)
from services.knowledge.entities.vector_route import VectorRoute
from tasks.delete_segment_from_index_task import cleanup_segment_index_in_session, delete_segment_from_index_task
from tasks.disable_segment_from_index_task import disable_segment_from_index_task
from tasks.disable_segments_from_index_task import disable_segments_from_index_task


@pytest.fixture
def indexed_segment(sqlite_session: Session) -> tuple[Dataset, Document, DocumentSegment]:
    """Persist the complete owner chain consumed by segment cleanup tasks."""
    tenant_id = str(uuid.uuid4())
    created_by = str(uuid.uuid4())
    dataset = Dataset(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        name="Cleanup dataset",
        data_source_type=DataSourceType.UPLOAD_FILE,
        created_by=created_by,
        is_multimodal=False,
    )
    dataset.index_struct = VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        vector_type="qdrant",
        collection_name=f"attachment-{dataset.id}",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="attachment",
    ).index_struct
    document = Document(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name="document.txt",
        created_from=DocumentCreatedFrom.WEB,
        created_by=created_by,
        indexing_status=IndexingStatus.COMPLETED,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    segment = DocumentSegment(
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="content",
        word_count=1,
        tokens=1,
        created_by=created_by,
        index_node_id="node-1",
        index_node_hash="hash-1",
        disabled_by=created_by,
        status=SegmentStatus.COMPLETED,
    )
    sqlite_session.add_all([dataset, document, segment])
    sqlite_session.commit()
    return dataset, document, segment


class _PersistedAttachmentRouteVector:
    @staticmethod
    def resolve_vector_type(dataset: Dataset, *, session: Session) -> str:
        del session
        index_struct = dataset.index_struct_dict
        assert index_struct is not None
        return str(index_struct["type"])

    def __init__(self, dataset: Dataset, *, session: Session, vector_type: str) -> None:
        del session
        index_struct = dataset.index_struct_dict
        assert index_struct is not None
        self._target = (vector_type, str(index_struct["vector_store"]["class_prefix"]))

    def get_routing_target(self) -> tuple[str, str]:
        return self._target


@contextmanager
def _record_transaction_events(
    sqlite_session_factory: sessionmaker[Session], phase_events: list[str]
) -> Generator[None]:
    """Record real commits made by the task-owned SQLite session."""
    session_type = sqlite_session_factory.class_

    def after_commit(_session: Session) -> None:
        phase_events.append("commit")

    event.listen(session_type, "after_commit", after_commit)
    try:
        yield
    finally:
        event.remove(session_type, "after_commit", after_commit)


def test_disable_segment_commits_index_cleanup(
    indexed_segment: tuple[Dataset, Document, DocumentSegment],
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    del sqlite_session_factory
    _dataset, _document, segment = indexed_segment
    with patch("tasks.disable_segment_from_index_task.enqueue_legacy_segment_state") as enqueue:
        disable_segment_from_index_task.run(segment.id)
    enqueue.assert_called_once_with(action="disable", segment_ids=[segment.id])


def test_disable_segments_commits_index_cleanup(
    indexed_segment: tuple[Dataset, Document, DocumentSegment],
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset, document, segment = indexed_segment
    del sqlite_session_factory
    with patch("tasks.disable_segments_from_index_task.enqueue_legacy_segment_state") as enqueue:
        disable_segments_from_index_task.run([segment.id], dataset.id, document.id)
    enqueue.assert_called_once_with(
        action="disable",
        segment_ids=[segment.id],
        dataset_id=dataset.id,
        document_id=document.id,
    )


def test_delete_segment_commits_index_cleanup_without_attachments(
    indexed_segment: tuple[Dataset, Document, DocumentSegment],
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset, document, segment = indexed_segment
    phase_events: list[str] = []
    processor = MagicMock()
    processor.clean.side_effect = lambda *_args, **_kwargs: phase_events.append("clean")

    with (
        _record_transaction_events(sqlite_session_factory, phase_events),
        patch("tasks.delete_segment_from_index_task.IndexProcessorFactory") as processor_factory,
    ):
        processor_factory.return_value.init_index_processor.return_value = processor
        delete_segment_from_index_task.run(["node-1"], dataset.id, document.id, [segment.id])

    assert phase_events == ["clean", "commit"]


def test_delete_cleanup_with_no_captured_nodes_never_cleans_the_dataset_collection(
    indexed_segment: tuple[Dataset, Document, DocumentSegment],
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset, document, segment = indexed_segment
    processor = MagicMock()
    with (
        sqlite_session_factory.begin() as session,
        patch("tasks.delete_segment_from_index_task.IndexProcessorFactory") as processor_factory,
    ):
        processor_factory.return_value.init_index_processor.return_value = processor
        cleanup_segment_index_in_session(
            session,
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            segment_ids=(segment.id,),
            index_node_ids=(),
            child_node_ids=(),
            doc_form=document.doc_form,
            upload_session_factory=sqlite_session_factory,
        )

    processor.clean.assert_not_called()


@pytest.mark.parametrize("is_multimodal", [False, True])
def test_delete_segment_removes_attachment_blobs_from_storage(
    indexed_segment: tuple[Dataset, Document, DocumentSegment],
    sqlite_session: Session,
    is_multimodal: bool,
) -> None:
    dataset, document, segment = indexed_segment
    dataset.is_multimodal = is_multimodal
    attachment = UploadFile(
        tenant_id=dataset.tenant_id,
        storage_type=StorageType.LOCAL,
        key="attachments/segment-image.png",
        name="segment-image.png",
        size=10,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=segment.created_by,
        created_at=datetime.now(UTC),
        used=True,
    )
    binding = SegmentAttachmentBinding(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        attachment_id=attachment.id,
    )
    sqlite_session.add_all([dataset, attachment, binding])
    sqlite_session.commit()
    attachment_id = attachment.id
    attachment_key = attachment.key
    binding_id = binding.id

    with (
        patch("tasks.delete_segment_from_index_task.IndexProcessorFactory"),
        patch("tasks.delete_segment_from_index_task.Vector", _PersistedAttachmentRouteVector),
        patch("tasks.delete_segment_from_index_task.storage.delete") as storage_delete,
    ):
        delete_segment_from_index_task.run(["node-1"], dataset.id, document.id, [segment.id])

    storage_delete.assert_called_once_with(attachment_key)
    sqlite_session.expire_all()
    assert sqlite_session.get(SegmentAttachmentBinding, binding_id) is None
    assert sqlite_session.get(UploadFile, attachment_id) is None
    assert (
        sqlite_session.scalar(
            select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == attachment_id)
        )
        is None
    )


def test_economy_segment_delete_cleans_attachments_with_retained_summary(
    indexed_segment: tuple[Dataset, Document, DocumentSegment],
    sqlite_session: Session,
) -> None:
    dataset, document, segment = indexed_segment
    dataset.indexing_technique = "economy"
    dataset.embedding_model = None
    dataset.embedding_model_provider = None
    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=segment.id,
        summary_content="Retained from high quality mode",
        summary_index_node_id="old-summary-node",
    )
    attachment = UploadFile(
        tenant_id=dataset.tenant_id,
        storage_type=StorageType.LOCAL,
        key="attachments/economy-segment.png",
        name="economy-segment.png",
        size=10,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=segment.created_by,
        created_at=datetime.now(UTC),
        used=True,
    )
    binding = SegmentAttachmentBinding(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        attachment_id=attachment.id,
    )
    sqlite_session.add_all([dataset, summary, attachment, binding])
    sqlite_session.commit()
    summary_id, attachment_id, binding_id = summary.id, attachment.id, binding.id
    attachment_key = attachment.key

    with (
        patch("tasks.delete_segment_from_index_task.IndexProcessorFactory"),
        patch("tasks.delete_segment_from_index_task.storage.delete") as storage_delete,
    ):
        delete_segment_from_index_task.run(["node-1"], dataset.id, document.id, [segment.id])

    storage_delete.assert_called_once_with(attachment_key)
    sqlite_session.expire_all()
    assert sqlite_session.get(DocumentSegmentSummary, summary_id) is None
    assert sqlite_session.get(SegmentAttachmentBinding, binding_id) is None
    assert sqlite_session.get(UploadFile, attachment_id) is None
    summary_cleanup = sqlite_session.scalar(
        select(SummaryVectorCleanupOutbox).where(SummaryVectorCleanupOutbox.node_id == "old-summary-node")
    )
    assert summary_cleanup is not None
    assert summary_cleanup.vector_route is None
    assert summary_cleanup.writer_finished_at is not None


def test_delete_segment_preserves_attachment_shared_by_another_segment(
    indexed_segment: tuple[Dataset, Document, DocumentSegment],
    sqlite_session: Session,
) -> None:
    dataset, document, segment = indexed_segment
    dataset.is_multimodal = True
    other_segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=2,
        content="other content",
        word_count=1,
        tokens=1,
        created_by=segment.created_by,
        index_node_id="node-2",
        index_node_hash="hash-2",
        disabled_by=segment.disabled_by,
        status=SegmentStatus.COMPLETED,
    )
    attachment = UploadFile(
        tenant_id=dataset.tenant_id,
        storage_type=StorageType.LOCAL,
        key="attachments/shared-segment-image.png",
        name="shared-segment-image.png",
        size=10,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=segment.created_by,
        created_at=datetime.now(UTC),
        used=True,
    )
    binding = SegmentAttachmentBinding(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        attachment_id=attachment.id,
    )
    shared_binding = SegmentAttachmentBinding(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=other_segment.id,
        attachment_id=attachment.id,
    )
    sqlite_session.add_all([dataset, other_segment, attachment, binding, shared_binding])
    sqlite_session.commit()
    attachment_id = attachment.id
    binding_id = binding.id
    shared_binding_id = shared_binding.id

    with (
        patch("tasks.delete_segment_from_index_task.IndexProcessorFactory") as processor_factory,
        patch("tasks.delete_segment_from_index_task.Vector", _PersistedAttachmentRouteVector),
        patch("tasks.delete_segment_from_index_task.storage.delete") as storage_delete,
    ):
        delete_segment_from_index_task.run(["node-1"], dataset.id, document.id, [segment.id])

    sqlite_session.expire_all()
    processor = processor_factory.return_value.init_index_processor.return_value
    assert processor.clean.call_count == 1
    assert processor.clean.call_args.args[1] == ["node-1"]
    assert processor.clean.call_args.kwargs["with_keywords"] is True
    storage_delete.assert_not_called()
    sqlite_session.expire_all()
    assert sqlite_session.get(SegmentAttachmentBinding, binding_id) is None
    assert sqlite_session.get(SegmentAttachmentBinding, shared_binding_id) is not None
    assert sqlite_session.get(UploadFile, attachment_id) is not None


@pytest.mark.parametrize("is_multimodal", [False, True])
def test_delete_segment_uses_global_liveness_before_deleting_attachment_files(
    indexed_segment: tuple[Dataset, Document, DocumentSegment],
    sqlite_session: Session,
    is_multimodal: bool,
) -> None:
    dataset, document, segment = indexed_segment
    dataset.is_multimodal = is_multimodal
    other_dataset = Dataset(
        id=str(uuid.uuid4()),
        tenant_id=dataset.tenant_id,
        name="Other cleanup dataset",
        data_source_type=DataSourceType.UPLOAD_FILE,
        created_by=segment.created_by,
        is_multimodal=True,
    )
    other_document = Document(
        id=str(uuid.uuid4()),
        tenant_id=dataset.tenant_id,
        dataset_id=other_dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-2",
        name="other-document.txt",
        created_from=DocumentCreatedFrom.WEB,
        created_by=segment.created_by,
        indexing_status=IndexingStatus.COMPLETED,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    other_segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=other_dataset.id,
        document_id=other_document.id,
        position=1,
        content="other dataset content",
        word_count=1,
        tokens=1,
        created_by=segment.created_by,
        index_node_id="other-dataset-node",
        index_node_hash="other-dataset-hash",
        disabled_by=segment.disabled_by,
        status=SegmentStatus.COMPLETED,
    )

    def make_attachment(name: str) -> UploadFile:
        return UploadFile(
            tenant_id=dataset.tenant_id,
            storage_type=StorageType.LOCAL,
            key=f"attachments/{name}.png",
            name=f"{name}.png",
            size=10,
            extension="png",
            mime_type="image/png",
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by=segment.created_by,
            created_at=datetime.now(UTC),
            used=True,
        )

    cross_dataset_attachment = make_attachment("cross-dataset")
    message_attachment = make_attachment("message-owned")
    orphan_attachment = make_attachment("orphan")
    current_bindings = [
        SegmentAttachmentBinding(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            segment_id=segment.id,
            attachment_id=attachment.id,
        )
        for attachment in (cross_dataset_attachment, message_attachment, orphan_attachment)
    ]
    other_dataset_binding = SegmentAttachmentBinding(
        tenant_id=dataset.tenant_id,
        dataset_id=other_dataset.id,
        document_id=other_document.id,
        segment_id=other_segment.id,
        attachment_id=cross_dataset_attachment.id,
    )
    message_file = MessageFile(
        message_id=str(uuid.uuid4()),
        type=FileType.IMAGE,
        transfer_method=FileTransferMethod.LOCAL_FILE,
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=segment.created_by,
        upload_file_id=message_attachment.id,
    )
    sqlite_session.add_all(
        [
            dataset,
            other_dataset,
            other_document,
            other_segment,
            cross_dataset_attachment,
            message_attachment,
            orphan_attachment,
            *current_bindings,
            other_dataset_binding,
            message_file,
        ]
    )
    sqlite_session.commit()
    current_binding_ids = [binding.id for binding in current_bindings]
    other_binding_id = other_dataset_binding.id
    message_file_id = message_file.id
    cross_dataset_attachment_id = cross_dataset_attachment.id
    message_attachment_id = message_attachment.id
    orphan_attachment_id = orphan_attachment.id
    orphan_attachment_key = orphan_attachment.key

    with (
        patch("tasks.delete_segment_from_index_task.IndexProcessorFactory") as processor_factory,
        patch("tasks.delete_segment_from_index_task.Vector", _PersistedAttachmentRouteVector),
        patch("tasks.delete_segment_from_index_task.storage.delete") as storage_delete,
    ):
        delete_segment_from_index_task.run(["node-1"], dataset.id, document.id, [segment.id])

    sqlite_session.expire_all()
    processor = processor_factory.return_value.init_index_processor.return_value
    assert processor.clean.call_count == 1
    if is_multimodal:
        assert set(
            sqlite_session.scalars(
                select(AttachmentVectorCleanupOutbox.attachment_id).where(
                    AttachmentVectorCleanupOutbox.dataset_id == dataset.id
                )
            )
        ) == {
            cross_dataset_attachment_id,
            message_attachment_id,
            orphan_attachment_id,
        }
    storage_delete.assert_called_once_with(orphan_attachment_key)

    assert all(sqlite_session.get(SegmentAttachmentBinding, binding_id) is None for binding_id in current_binding_ids)
    assert sqlite_session.get(SegmentAttachmentBinding, other_binding_id) is not None
    assert sqlite_session.get(MessageFile, message_file_id) is not None
    assert sqlite_session.get(UploadFile, cross_dataset_attachment_id) is not None
    assert sqlite_session.get(UploadFile, message_attachment_id) is not None
    assert sqlite_session.get(UploadFile, orphan_attachment_id) is None
    assert (
        sqlite_session.scalar(
            select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == orphan_attachment_id)
        )
        is None
    )


def test_delete_segment_keeps_database_cleanup_when_attachment_storage_delete_fails(
    indexed_segment: tuple[Dataset, Document, DocumentSegment],
    sqlite_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset, document, segment = indexed_segment
    dataset.is_multimodal = True
    attachment = UploadFile(
        tenant_id=dataset.tenant_id,
        storage_type=StorageType.LOCAL,
        key="attachments/failing-segment-image.png",
        name="failing-segment-image.png",
        size=10,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=segment.created_by,
        created_at=datetime.now(UTC),
        used=True,
    )
    binding = SegmentAttachmentBinding(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        attachment_id=attachment.id,
    )
    sqlite_session.add_all([dataset, attachment, binding])
    sqlite_session.commit()
    attachment_id = attachment.id
    attachment_key = attachment.key
    binding_id = binding.id

    with (
        patch("tasks.delete_segment_from_index_task.IndexProcessorFactory"),
        patch("tasks.delete_segment_from_index_task.Vector", _PersistedAttachmentRouteVector),
        patch(
            "tasks.delete_segment_from_index_task.storage.delete",
            side_effect=RuntimeError("storage unavailable"),
        ) as storage_delete,
        caplog.at_level("ERROR", logger="tasks.delete_segment_from_index_task"),
    ):
        delete_segment_from_index_task.run(["node-1"], dataset.id, document.id, [segment.id])

    storage_delete.assert_called_once_with(attachment_key)
    assert "Upload blob cleanup failed; the durable cleanup record will be retried" in caplog.text
    sqlite_session.expire_all()
    assert sqlite_session.get(SegmentAttachmentBinding, binding_id) is None
    assert sqlite_session.get(UploadFile, attachment_id) is None
    cleanup = sqlite_session.scalar(
        select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == attachment_id)
    )
    assert cleanup is not None
    assert cleanup.last_attempted_at is not None

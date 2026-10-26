"""SQLite-backed resilience tests for ``clean_document_task``.

The task must continue PostgreSQL cleanup when vector cleanup fails. Each test
starts from the production incident shape: the caller has already deleted the
``Document`` row while its segments and metadata bindings remain.
"""

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

import tasks.clean_document_task as clean_document_task_module
from core.workflow.file_reference import build_file_reference
from extensions.storage.storage_type import StorageType
from models.dataset import (
    ChildChunk,
    Dataset,
    DatasetMetadataBinding,
    Document,
    DocumentSegment,
    SegmentAttachmentBinding,
)
from models.enums import CreatorUserRole, DataSourceType
from models.model import UploadFile, UploadFileCleanupOutbox
from tasks.clean_document_task import clean_document_task, run_document_cleanup
from tests.unit_tests.model_factories import make_document

SQLITE_MODELS = (
    Dataset,
    Document,
    DocumentSegment,
    ChildChunk,
    SegmentAttachmentBinding,
    UploadFile,
    UploadFileCleanupOutbox,
    DatasetMetadataBinding,
)

pytestmark = pytest.mark.parametrize("sqlite_session", [SQLITE_MODELS], indirect=True)


@pytest.fixture
def document_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def dataset_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def tenant_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def bind_task_sessions(monkeypatch: pytest.MonkeyPatch, sqlite_session: Session) -> None:
    """Bind every short-lived task transaction to the isolated SQLite engine."""
    engine = sqlite_session.get_bind()
    monkeypatch.setattr(
        clean_document_task_module.session_factory,
        "create_session",
        lambda: Session(engine, expire_on_commit=False),
    )


@pytest.fixture
def mock_storage():
    with patch("tasks.batch_clean_document_task.storage", autospec=True) as mock:
        mock.delete.return_value = None
        yield mock


@pytest.fixture
def mock_index_cleanup(tenant_id: str):
    """Keep task resilience tests independent of the external index backend."""
    with patch("tasks.batch_clean_document_task.clean_document_indexes", return_value=tenant_id) as cleanup:
        yield cleanup


def _document(*, document_id: str, dataset_id: str, tenant_id: str, created_by: str) -> Document:
    return make_document(
        document_id=document_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        name=f"{document_id}.txt",
        created_by=created_by,
    )


def _segment(*, segment_id: str, document_id: str, dataset_id: str, tenant_id: str, created_by: str) -> DocumentSegment:
    segment = DocumentSegment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        position=1,
        content="segment content",
        word_count=2,
        tokens=2,
        created_by=created_by,
        index_node_id=f"node-{segment_id}",
    )
    segment.id = segment_id
    return segment


def _upload_file(*, tenant_id: str, created_by: str) -> UploadFile:
    return UploadFile(
        tenant_id=tenant_id,
        storage_type=StorageType.LOCAL,
        key=f"knowledge-images/{uuid.uuid4()}.png",
        name="image.png",
        size=10,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=created_by,
        created_at=datetime.now(UTC),
        used=True,
    )


def _persist_deleted_document_state(
    session: Session,
    *,
    document_id: str,
    dataset_id: str,
    tenant_id: str,
    target_segment_ids: list[str],
) -> tuple[str, str]:
    """Persist target children after deleting their document, plus scoped control rows."""
    created_by = str(uuid.uuid4())
    other_document_id = str(uuid.uuid4())
    survivor_segment_id = str(uuid.uuid4())
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name="Cleanup dataset",
        data_source_type=DataSourceType.UPLOAD_FILE,
        created_by=created_by,
    )
    target_document = _document(
        document_id=document_id,
        dataset_id=dataset_id,
        tenant_id=tenant_id,
        created_by=created_by,
    )
    other_document = _document(
        document_id=other_document_id,
        dataset_id=dataset_id,
        tenant_id=tenant_id,
        created_by=created_by,
    )
    segments = [
        _segment(
            segment_id=segment_id,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            created_by=created_by,
        )
        for segment_id in target_segment_ids
    ]
    segments.append(
        _segment(
            segment_id=survivor_segment_id,
            document_id=other_document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            created_by=created_by,
        )
    )
    metadata_bindings = [
        DatasetMetadataBinding(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            metadata_id=str(uuid.uuid4()),
            document_id=current_document_id,
            created_by=created_by,
        )
        for current_document_id in (document_id, other_document_id)
    ]

    session.add_all([dataset, target_document, other_document, *segments, *metadata_bindings])
    session.commit()
    session.delete(target_document)
    session.commit()
    return other_document_id, survivor_segment_id


def _assert_relational_cleanup(
    session: Session,
    *,
    document_id: str,
    other_document_id: str,
    survivor_segment_id: str,
) -> None:
    session.expire_all()
    assert session.get(Document, document_id) is None
    remaining_segments = session.scalars(select(DocumentSegment)).all()
    assert [(segment.id, segment.document_id) for segment in remaining_segments] == [
        (survivor_segment_id, other_document_id)
    ]
    remaining_binding_document_ids = set(session.scalars(select(DatasetMetadataBinding.document_id)).all())
    assert remaining_binding_document_ids == {other_document_id}


class TestVectorCleanupResilience:
    """Vector/index failures must retain relational evidence for the outbox retry."""

    def test_billing_failure_preserves_relational_cleanup_inputs(
        self,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
        mock_index_cleanup,
    ) -> None:
        """A transient billing failure propagates before segment ownership is removed."""
        other_document_id, survivor_segment_id = _persist_deleted_document_state(
            sqlite_session,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            target_segment_ids=["seg-1", "seg-2"],
        )
        mock_index_cleanup.side_effect = ValueError(
            "Unable to retrieve billing information. Please try again later or contact support."
        )

        with (
            patch("tasks.batch_clean_document_task.schedule_billing_vector_space_refresh") as schedule_refresh,
            pytest.raises(ValueError, match="Unable to retrieve billing information"),
        ):
            clean_document_task(document_id=document_id, dataset_id=dataset_id, doc_form="paragraph", file_id=None)

        mock_index_cleanup.assert_called_once()
        sqlite_session.expire_all()
        assert set(sqlite_session.scalars(select(DocumentSegment.id))) == {"seg-1", "seg-2", survivor_segment_id}
        assert set(sqlite_session.scalars(select(DatasetMetadataBinding.document_id))) == {
            document_id,
            other_document_id,
        }
        schedule_refresh.assert_not_called()


class TestUploadOwnership:
    """Editable segment text must never establish authority to delete an upload."""

    def test_cleanup_without_dataset_uses_captured_tenant_and_retains_failed_blob_outbox(
        self,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
    ) -> None:
        created_by = str(uuid.uuid4())
        segment = _segment(
            segment_id=str(uuid.uuid4()),
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            created_by=created_by,
        )
        source_file = _upload_file(tenant_id=tenant_id, created_by=created_by)
        attachment = _upload_file(tenant_id=tenant_id, created_by=created_by)
        binding = SegmentAttachmentBinding(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment.id,
            attachment_id=attachment.id,
        )
        sqlite_session.add_all([segment, source_file, attachment, binding])
        sqlite_session.commit()
        segment_id = segment.id
        binding_id = binding.id
        source_file_id = source_file.id
        attachment_id = attachment.id
        source_key = source_file.key
        attachment_key = attachment.key
        mock_storage.delete.side_effect = RuntimeError("storage unavailable")

        run_document_cleanup(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            doc_form="paragraph",
            file_id=source_file_id,
        )

        sqlite_session.expire_all()
        assert sqlite_session.get(DocumentSegment, segment_id) is None
        assert sqlite_session.get(SegmentAttachmentBinding, binding_id) is None
        assert sqlite_session.get(UploadFile, source_file_id) is None
        assert sqlite_session.get(UploadFile, attachment_id) is None
        pending = sqlite_session.scalars(select(UploadFileCleanupOutbox)).all()
        assert {(row.upload_file_id, row.key) for row in pending} == {
            (source_file_id, source_key),
            (attachment_id, attachment_key),
        }
        assert {call.args[0] for call in mock_storage.delete.call_args_list} == {source_key, attachment_key}

    def test_foreign_tenant_file_id_in_segment_body_does_not_touch_storage(
        self,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
        mock_index_cleanup,
    ) -> None:
        _persist_deleted_document_state(
            sqlite_session,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            target_segment_ids=["target-segment"],
        )
        foreign_upload = _upload_file(tenant_id=str(uuid.uuid4()), created_by=str(uuid.uuid4()))
        target_segment = sqlite_session.get(DocumentSegment, "target-segment")
        assert target_segment is not None
        target_segment.content = f"![image](https://files.example/files/{foreign_upload.id}/file-preview)"
        sqlite_session.add(foreign_upload)
        sqlite_session.commit()

        with patch("tasks.batch_clean_document_task.schedule_billing_vector_space_refresh"):
            clean_document_task(
                document_id=document_id,
                dataset_id=dataset_id,
                doc_form="paragraph",
                file_id=None,
            )

        sqlite_session.expire_all()
        assert sqlite_session.get(UploadFile, foreign_upload.id) is not None
        mock_storage.delete.assert_not_called()

    def test_same_tenant_file_owned_by_another_document_is_retained(
        self,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
        mock_index_cleanup,
    ) -> None:
        other_document_id, survivor_segment_id = _persist_deleted_document_state(
            sqlite_session,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            target_segment_ids=["target-segment"],
        )
        upload = _upload_file(tenant_id=tenant_id, created_by=str(uuid.uuid4()))
        survivor_binding = SegmentAttachmentBinding(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=other_document_id,
            segment_id=survivor_segment_id,
            attachment_id=upload.id,
        )
        target_segment = sqlite_session.get(DocumentSegment, "target-segment")
        assert target_segment is not None
        target_segment.content = f"![image](https://files.example/files/{upload.id}/file-preview)"
        sqlite_session.add_all([upload, survivor_binding])
        sqlite_session.commit()

        with patch("tasks.batch_clean_document_task.schedule_billing_vector_space_refresh"):
            clean_document_task(
                document_id=document_id,
                dataset_id=dataset_id,
                doc_form="paragraph",
                file_id=None,
            )

        sqlite_session.expire_all()
        assert sqlite_session.get(UploadFile, upload.id) is not None
        assert sqlite_session.get(SegmentAttachmentBinding, survivor_binding.id) is not None
        mock_storage.delete.assert_not_called()

    def test_target_attachment_binding_authorizes_file_cleanup(
        self,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
        mock_index_cleanup,
    ) -> None:
        _persist_deleted_document_state(
            sqlite_session,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            target_segment_ids=["target-segment"],
        )
        upload = _upload_file(tenant_id=tenant_id, created_by=str(uuid.uuid4()))
        binding = SegmentAttachmentBinding(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id="target-segment",
            attachment_id=upload.id,
        )
        sqlite_session.add_all([upload, binding])
        sqlite_session.commit()
        upload_id = upload.id
        upload_key = upload.key
        binding_id = binding.id

        with patch("tasks.batch_clean_document_task.schedule_billing_vector_space_refresh"):
            clean_document_task(
                document_id=document_id,
                dataset_id=dataset_id,
                doc_form="paragraph",
                file_id=None,
            )

        sqlite_session.expire_all()
        assert sqlite_session.get(SegmentAttachmentBinding, binding_id) is None
        assert sqlite_session.get(UploadFile, upload_id) is None
        mock_storage.delete.assert_called_once_with(upload_key)

    @pytest.mark.parametrize("source_schema", ["real_file_id", "reference", "related_id"])
    def test_local_file_document_reference_keeps_shared_upload(
        self,
        source_schema: str,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
        mock_index_cleanup,
    ) -> None:
        other_document_id, _ = _persist_deleted_document_state(
            sqlite_session,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            target_segment_ids=["target-segment"],
        )
        upload = _upload_file(tenant_id=tenant_id, created_by=str(uuid.uuid4()))
        target_binding = SegmentAttachmentBinding(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id="target-segment",
            attachment_id=upload.id,
        )
        surviving_document = sqlite_session.get(Document, other_document_id)
        assert surviving_document is not None
        surviving_document.data_source_type = DataSourceType.LOCAL_FILE
        source_value = build_file_reference(record_id=upload.id) if source_schema == "reference" else upload.id
        surviving_document.data_source_info = json.dumps({source_schema: source_value})
        sqlite_session.add_all([upload, target_binding])
        sqlite_session.commit()
        upload_id = upload.id
        target_binding_id = target_binding.id

        with patch("tasks.batch_clean_document_task.schedule_billing_vector_space_refresh"):
            clean_document_task(
                document_id=document_id,
                dataset_id=dataset_id,
                doc_form="paragraph",
                file_id=None,
            )

        sqlite_session.expire_all()
        assert sqlite_session.get(SegmentAttachmentBinding, target_binding_id) is None
        assert sqlite_session.get(UploadFile, upload_id) is not None
        mock_storage.delete.assert_not_called()

    @pytest.mark.parametrize(
        "reference_template",
        [
            "![diagram](https://files.example/files/{file_id}/file-preview)",
            "![diagram](/files/{file_id}/image-preview)",
        ],
    )
    def test_surviving_markdown_preview_reference_keeps_shared_upload(
        self,
        reference_template: str,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
        mock_index_cleanup,
    ) -> None:
        _, survivor_segment_id = _persist_deleted_document_state(
            sqlite_session,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            target_segment_ids=["target-segment"],
        )
        upload = _upload_file(tenant_id=tenant_id, created_by=str(uuid.uuid4()))
        target_binding = SegmentAttachmentBinding(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id="target-segment",
            attachment_id=upload.id,
        )
        surviving_segment = sqlite_session.get(DocumentSegment, survivor_segment_id)
        assert surviving_segment is not None
        surviving_segment.content = reference_template.format(file_id=upload.id)
        sqlite_session.add_all([upload, target_binding])
        sqlite_session.commit()
        upload_id = upload.id

        with patch("tasks.batch_clean_document_task.schedule_billing_vector_space_refresh"):
            clean_document_task(
                document_id=document_id,
                dataset_id=dataset_id,
                doc_form="paragraph",
                file_id=None,
            )

        sqlite_session.expire_all()
        assert sqlite_session.get(UploadFile, upload_id) is not None
        mock_storage.delete.assert_not_called()

    def test_shared_attachment_binding_keeps_file_for_surviving_document(
        self,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
        mock_index_cleanup,
    ) -> None:
        other_document_id, survivor_segment_id = _persist_deleted_document_state(
            sqlite_session,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            target_segment_ids=["target-segment"],
        )
        upload = _upload_file(tenant_id=tenant_id, created_by=str(uuid.uuid4()))
        target_binding = SegmentAttachmentBinding(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id="target-segment",
            attachment_id=upload.id,
        )
        survivor_binding = SegmentAttachmentBinding(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=other_document_id,
            segment_id=survivor_segment_id,
            attachment_id=upload.id,
        )
        sqlite_session.add_all([upload, target_binding, survivor_binding])
        sqlite_session.commit()
        upload_id = upload.id
        target_binding_id = target_binding.id
        survivor_binding_id = survivor_binding.id

        with patch("tasks.batch_clean_document_task.schedule_billing_vector_space_refresh"):
            clean_document_task(
                document_id=document_id,
                dataset_id=dataset_id,
                doc_form="paragraph",
                file_id=None,
            )

        sqlite_session.expire_all()
        assert sqlite_session.get(SegmentAttachmentBinding, target_binding_id) is None
        assert sqlite_session.get(SegmentAttachmentBinding, survivor_binding_id) is not None
        assert sqlite_session.get(UploadFile, upload_id) is not None
        mock_storage.delete.assert_not_called()


class TestVectorCleanupCompletion:
    def test_vector_cleanup_success_path_remains_unaffected(
        self,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
        mock_index_cleanup,
    ) -> None:
        """The happy path calls the index boundary and completes scoped cleanup."""
        other_document_id, survivor_segment_id = _persist_deleted_document_state(
            sqlite_session,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            target_segment_ids=["seg-1"],
        )

        with patch("tasks.batch_clean_document_task.schedule_billing_vector_space_refresh") as schedule_refresh:
            clean_document_task(
                document_id=document_id,
                dataset_id=dataset_id,
                doc_form="paragraph",
                file_id=None,
            )

        mock_index_cleanup.assert_called_once()
        _, kwargs = mock_index_cleanup.call_args
        new_session = kwargs.pop("new_session")
        with new_session() as cleanup_session:
            assert cleanup_session.get_bind() is sqlite_session.get_bind()
        assert kwargs == {
            "tenant_id": tenant_id,
            "dataset_id": dataset_id,
            "document_ids": (document_id,),
            "doc_form": "paragraph",
        }
        _assert_relational_cleanup(
            sqlite_session,
            document_id=document_id,
            other_document_id=other_document_id,
            survivor_segment_id=survivor_segment_id,
        )
        schedule_refresh.assert_called_once_with(tenant_id)

    def test_no_index_state_does_not_refresh_billing(
        self,
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        sqlite_session: Session,
        bind_task_sessions: None,
        mock_storage,
        mock_index_cleanup,
    ) -> None:
        """Cleanup checks for orphan summaries even when no segments remain."""
        mock_index_cleanup.return_value = None
        other_document_id, survivor_segment_id = _persist_deleted_document_state(
            sqlite_session,
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            target_segment_ids=[],
        )

        with patch("tasks.batch_clean_document_task.schedule_billing_vector_space_refresh") as schedule_refresh:
            clean_document_task(
                document_id=document_id,
                dataset_id=dataset_id,
                doc_form="paragraph",
                file_id=None,
            )

        mock_index_cleanup.assert_called_once()
        _assert_relational_cleanup(
            sqlite_session,
            document_id=document_id,
            other_document_id=other_document_id,
            survivor_segment_id=survivor_segment_id,
        )
        schedule_refresh.assert_not_called()

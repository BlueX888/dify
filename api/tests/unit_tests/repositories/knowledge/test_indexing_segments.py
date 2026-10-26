from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.extractor.entity.extract_setting import CreatedUploadArtifact
from core.rag.index_processor.constant.index_type import IndexStructureType
from core.rag.models.document import ChildDocument
from core.rag.models.document import Document as IndexDocument
from extensions.storage.storage_type import StorageType
from models.dataset import ChildChunk, Dataset, DatasetProcessRule, Document, DocumentSegment
from models.enums import CreatorUserRole, IndexingStatus, ProcessRuleMode, SegmentStatus
from models.model import UploadFile, UploadFileCleanupOutbox
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from repositories.knowledge.segment_cleanup_repository import delete_document_segments_and_claim_uploads
from repositories.knowledge.segment_repository import SQLAlchemySegmentRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from repositories.upload_file_repository import UploadFilePublicationLostError
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.segments import ChildVectorRoute
from services.knowledge.indexing.errors import DatasetIndexingGenerationLostError
from services.knowledge.indexing.execution import IndexingDocument
from tests.unit_tests.repositories.knowledge.test_document_repository import _dataset, _document

IndexingRows = tuple[SQLAlchemySegmentRepository, IndexingDocument, list[IndexDocument]]


@pytest.fixture
def indexing_rows(sqlite_session_factory: sessionmaker[Session]) -> IndexingRows:
    with sqlite_session_factory.begin() as session:
        document = _document("document-1", status=IndexingStatus.SPLITTING)
        document.doc_form = IndexStructureType.PARENT_CHILD_INDEX
        rule = DatasetProcessRule(
            dataset_id="dataset-1",
            mode=ProcessRuleMode.HIERARCHICAL,
            rules='{"parent_mode": "paragraph"}',
            created_by="account-1",
        )
        rule.id = "rule-1"
        document.dataset_process_rule_id = rule.id
        session.add_all([_dataset("dataset-1", "workspace-1"), document, _document("document-2"), rule])
    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    segments = SQLAlchemySegmentRepository(session_factory=sqlite_session_factory)
    job = documents.get_indexing_document(DatasetRef("workspace-1", "dataset-1").document("document-1"))
    chunks = [
        IndexDocument(
            page_content=f"parent-{i}",
            metadata={"doc_id": f"node-{i}", "doc_hash": f"hash-{i}"},
            children=[ChildDocument(page_content="child", metadata={"doc_id": f"child-{i}", "doc_hash": "child-hash"})],
        )
        for i in range(2)
    ]
    return segments, job, chunks


def test_saved_segments_are_visible_before_workers_and_resume_preserves_total(
    sqlite_session_factory: sessionmaker[Session], indexing_rows: IndexingRows
) -> None:
    segments, job, chunks = indexing_rows
    segments.save_for_indexing(job, chunks, [13, 17])
    segments.complete_indexing_segments(job.ref, ["node-0"])
    resumed, tokens = segments.resume_indexing(job)
    assert tokens == 30
    assert [chunk.page_content for chunk in resumed] == ["parent-1"]
    assert resumed[0].children is not None
    assert resumed[0].children[0].page_content == "child"
    with sqlite_session_factory() as session:
        persisted_document = session.get(Document, "document-1")
        assert persisted_document is not None
        assert persisted_document.indexing_status == IndexingStatus.INDEXING
        rows = session.scalars(select(DocumentSegment).order_by(DocumentSegment.position)).all()
        assert [row.status for row in rows] == [SegmentStatus.COMPLETED, SegmentStatus.INDEXING]
    with sqlite_session_factory.begin() as session:
        delete_document_segments_and_claim_uploads(
            session,
            tenant_id=job.ref.dataset.tenant_id,
            dataset_id=job.ref.dataset.dataset_id,
            document_ids=(job.ref.document_id,),
        )
    with sqlite_session_factory() as session:
        assert session.scalars(select(DocumentSegment)).all() == []
        assert session.scalars(select(ChildChunk)).all() == []


def test_completion_persists_the_actual_child_vector_route(
    sqlite_session_factory: sessionmaker[Session], indexing_rows: IndexingRows
) -> None:
    segments, job, chunks = indexing_rows
    segments.save_for_indexing(job, chunks, [13, 17])
    route = ChildVectorRoute.capture(
        tenant_id=job.ref.dataset.tenant_id,
        dataset_id=job.ref.dataset.dataset_id,
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form=IndexStructureType.PARENT_CHILD_INDEX,
    )

    segments.complete_indexing_segments(job.ref, ["node-0"], child_vector_route=route)

    with sqlite_session_factory() as session:
        child = session.scalar(select(ChildChunk).where(ChildChunk.index_node_id == "child-0"))
        untouched = session.scalar(select(ChildChunk).where(ChildChunk.index_node_id == "child-1"))
        assert child is not None
        assert untouched is not None
        assert child.vector_route == route.serialized
        assert child.vector_route_hash == route.key
        assert untouched.vector_route is None


def test_repeated_save_does_not_duplicate_segments_or_children(
    sqlite_session_factory: sessionmaker[Session], indexing_rows: IndexingRows
) -> None:
    segments, job, chunks = indexing_rows
    segments.save_for_indexing(job, chunks, [13, 17])
    segments.save_for_indexing(job, chunks, [19, 23])
    with sqlite_session_factory() as session:
        assert len(session.scalars(select(ChildChunk)).all()) == 2
    assert segments.resume_indexing(job)[1] == 42


def test_segment_publication_consumes_exact_extractor_writer_marker(
    sqlite_session_factory: sessionmaker[Session], indexing_rows: IndexingRows
) -> None:
    segments, job, chunks = indexing_rows
    uploads = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    upload = UploadFile(
        tenant_id="workspace-1",
        storage_type=StorageType.LOCAL,
        key="workspace-1/extracted.png",
        name="extracted.png",
        size=7,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="account-1",
        created_at=datetime.now(UTC),
        used=True,
    )
    artifact = CreatedUploadArtifact(upload.id, upload.key)
    writer_id = uploads.begin_created_upload_artifact_write(workspace_id="workspace-1", artifact=artifact)
    with sqlite_session_factory.begin() as session:
        session.add(upload)
        uploads.finish_created_upload_artifact_writes_in_session(
            session,
            workspace_id="workspace-1",
            writer_ids={upload.id: writer_id},
        )
    chunks[0].page_content += f" ![image](https://files.test/files/{upload.id}/file-preview)"

    segments.save_for_indexing(
        job,
        chunks,
        [13, 17],
        created_upload_file_ids=(upload.id,),
    )

    with sqlite_session_factory() as session:
        assert session.get(UploadFile, upload.id) is not None
        assert session.scalar(select(UploadFileCleanupOutbox)) is None


def test_cleanup_winner_fences_late_segment_publication(
    sqlite_session_factory: sessionmaker[Session], indexing_rows: IndexingRows
) -> None:
    segments, job, chunks = indexing_rows
    uploads = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    upload = UploadFile(
        tenant_id="workspace-1",
        storage_type=StorageType.LOCAL,
        key="workspace-1/fenced.png",
        name="fenced.png",
        size=7,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="account-1",
        created_at=datetime.now(UTC),
        used=True,
    )
    artifact = CreatedUploadArtifact(upload.id, upload.key)
    writer_id = uploads.begin_created_upload_artifact_write(workspace_id="workspace-1", artifact=artifact)
    with sqlite_session_factory.begin() as session:
        session.add(upload)
        uploads.finish_created_upload_artifact_writes_in_session(
            session,
            workspace_id="workspace-1",
            writer_ids={upload.id: writer_id},
        )
    uploads.abandon_created_upload_artifact_writes(
        workspace_id="workspace-1",
        upload_file_ids=(upload.id,),
    )
    assert uploads.claim_unreferenced_files_for_cleanup(
        workspace_id="workspace-1",
        file_ids=(upload.id,),
    )
    chunks[0].page_content += f" ![image](https://files.test/files/{upload.id}/file-preview)"

    with pytest.raises(UploadFilePublicationLostError):
        segments.save_for_indexing(
            job,
            chunks,
            [13, 17],
            created_upload_file_ids=(upload.id,),
        )

    with sqlite_session_factory() as session:
        assert session.scalars(select(DocumentSegment)).all() == []


def test_token_count_mismatch_rolls_back_segments_and_state(
    sqlite_session_factory: sessionmaker[Session], indexing_rows: IndexingRows
) -> None:
    segments, job, chunks = indexing_rows
    with pytest.raises(ValueError, match="zip\\(\\) argument 2 is shorter than argument 1"):
        segments.save_for_indexing(job, chunks, [13])
    with sqlite_session_factory() as session:
        assert session.scalars(select(DocumentSegment)).all() == []
        persisted_document = session.get(Document, "document-1")
        assert persisted_document is not None
        assert persisted_document.indexing_status == IndexingStatus.SPLITTING


def test_node_id_collision_cannot_overwrite_another_document(
    sqlite_session_factory: sessionmaker[Session], indexing_rows: IndexingRows
) -> None:
    segments, job, chunks = indexing_rows
    other = replace(job, source=job.source._replace(document_ref=job.ref.dataset.document("document-2")))
    segments.save_for_indexing(other, chunks, [3, 5])
    segments.save_for_indexing(job, chunks, [13, 17])
    segments.complete_indexing_segments(job.ref, ["node-0", "node-1"])
    with sqlite_session_factory.begin() as session:
        delete_document_segments_and_claim_uploads(
            session,
            tenant_id=job.ref.dataset.tenant_id,
            dataset_id=job.ref.dataset.dataset_id,
            document_ids=(job.ref.document_id,),
        )
    assert segments.resume_indexing(other)[1] == 8
    assert len(segments.resume_indexing(other)[0]) == 2


def test_segment_mutations_reject_foreign_tenant(indexing_rows: IndexingRows) -> None:
    segments, job, chunks = indexing_rows
    foreign = replace(
        job, source=job.source._replace(document_ref=DatasetRef("foreign", "dataset-1").document("document-1"))
    )
    for operation in [
        lambda: segments.save_for_indexing(foreign, chunks, [13, 17]),
        lambda: segments.complete_indexing_segments(foreign.ref, ["node-0"]),
    ]:
        with pytest.raises(DatasetIndexingGenerationLostError):
            operation()


def test_index_write_rejects_snapshot_from_an_older_dataset_generation(
    sqlite_session_factory: sessionmaker[Session], indexing_rows: IndexingRows
) -> None:
    segments, job, chunks = indexing_rows
    segments.save_for_indexing(job, chunks, [13, 17])
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, "dataset-1")
        dataset.indexing_request_id = "generation-1"
        segment_id = session.scalar(select(DocumentSegment.id).where(DocumentSegment.index_node_id == "node-0"))
        assert segment_id is not None
    segment_ref = job.ref.segment(segment_id)
    snapshot = segments.get_indexing_snapshot(segment_ref)
    assert snapshot.dataset.indexing_request_id == "generation-1"
    with sqlite_session_factory.begin() as session:
        session.get_one(Dataset, "dataset-1").indexing_request_id = "generation-2"

    with pytest.raises(DatasetIndexingGenerationLostError, match="older Dataset generation"):
        with segments.lock_index_write(segment_ref, expected_snapshot=snapshot):
            pytest.fail("stale publisher entered its vector write section")

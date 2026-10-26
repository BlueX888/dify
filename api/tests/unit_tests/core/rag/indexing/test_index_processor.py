import uuid
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Never
from unittest.mock import MagicMock, Mock, patch

import pytest
from flask import Flask
from sqlalchemy import event, select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.index_processor import index_processor as index_processor_module
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.index_processor.index_processor import IndexProcessor
from core.rag.index_processor.processor.paragraph_index_processor import ParagraphIndexProcessor
from core.rag.models.document import AttachmentDocument
from core.workflow.nodes.knowledge_index.protocols import Preview, PreviewItem
from extensions.ext_application_services import ApplicationServices
from extensions.ext_login import bind_account_loader
from models.account import Account, Tenant, TenantAccountJoin, TenantAccountRole
from models.dataset import (
    ChildChunk,
    Dataset,
    Document,
    DocumentSegment,
    DocumentSegmentSummary,
    SegmentAttachmentBinding,
)
from models.enums import DataSourceType, DocumentCreatedFrom, SegmentStatus
from models.model import AttachmentVectorCleanupOutbox, UploadFile, UploadFileCleanupOutbox
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from services.knowledge.entities.vector_route import VectorRoute
from tests.unit_tests.model_factories import make_document, make_upload_file


def _index_processor() -> IndexProcessor:
    return IndexProcessor(attachment_vectors=SQLAlchemyAttachmentVectorStore())


class TestIndexingAccountLookup:
    def test_indexes_attachments_with_the_registered_account_loader(
        self, sqlite_session: Session, account_application_services: ApplicationServices
    ) -> None:
        app = Flask(__name__)
        bind_account_loader(app, account_application_services.accounts.identity.load_user)
        dataset, document = _persist_dataset_and_document(sqlite_session)
        account = Account(name="Creator", email="creator@example.com")
        account.id = document.created_by
        tenant = Tenant(name="Creator workspace")
        tenant.id = dataset.tenant_id
        sqlite_session.add_all(
            [
                account,
                tenant,
                TenantAccountJoin(
                    tenant_id=tenant.id, account_id=account.id, role=TenantAccountRole.OWNER, current=True
                ),
            ]
        )
        sqlite_session.commit()
        attachment = AttachmentDocument(page_content="image", metadata={})

        with (
            app.app_context(),
            patch.object(ParagraphIndexProcessor, "_get_content_files", return_value=[attachment]) as content_files,
            patch("core.rag.index_processor.processor.paragraph_index_processor.DatasetDocumentStore", new=Mock()),
            patch("core.rag.index_processor.processor.paragraph_index_processor.Vector", new=Mock()),
            patch.object(index_processor_module, "Vector", autospec=True) as cleanup_vector,
            patch(
                "core.rag.index_processor.processor.paragraph_index_processor.calculate_segment_token_counts",
                return_value=[1],
            ),
        ):
            cleanup_vector.resolve_vector_type.return_value = "pgvector"
            cleanup_vector.return_value.get_routing_target.return_value = ("pgvector", "collection-a")
            result = _index_processor().index_and_clean(
                dataset_id=dataset.id,
                document_id=document.id,
                original_document_id=document.id,
                chunks={"general_chunks": [{"content": "Image content"}]},
                batch="batch-1",
                session=sqlite_session,
            )

        assert result["display_status"] == "completed"
        assert content_files.call_args.kwargs["current_user"].id == account.id
        assert content_files.call_args.kwargs["session"] is sqlite_session


def _persist_dataset_and_document(
    session: Session,
    *,
    indexing_technique: IndexTechniqueType = IndexTechniqueType.HIGH_QUALITY,
    summary_index_setting: dict | None = None,
) -> tuple[Dataset, Document]:
    tenant_id = str(uuid.uuid4())
    created_by = str(uuid.uuid4())
    dataset = Dataset(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        name="Dataset",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=indexing_technique,
        chunk_structure="text_model",
        summary_index_setting=summary_index_setting,
        created_by=created_by,
    )
    document = Document(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name="Document",
        created_from=DocumentCreatedFrom.WEB,
        created_by=created_by,
        doc_language="English",
    )
    session.add_all([dataset, document])
    session.flush()
    return dataset, document


@dataclass
class ReplacementGraph:
    dataset: Dataset
    document: Document
    parent: DocumentSegment
    child: ChildChunk
    summary: DocumentSegmentSummary
    exclusive: UploadFile
    shared: UploadFile
    other_segment: DocumentSegment


@pytest.fixture
def replacement_graph(sqlite_session: Session) -> ReplacementGraph:
    dataset, document = _persist_dataset_and_document(sqlite_session)
    dataset.chunk_structure = IndexStructureType.PARENT_CHILD_INDEX
    other_document = make_document(
        document_id=str(uuid.uuid4()),
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        created_by=dataset.created_by,
    )
    segments = [
        DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=source.id,
            position=1,
            content="old content",
            word_count=3,
            tokens=3,
            created_by=dataset.created_by,
            index_node_id=node_id,
        )
        for source, node_id in ((document, "parent-node"), (other_document, "retained-node"))
    ]
    parent, other_segment = segments
    child = ChildChunk(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=parent.id,
        position=1,
        content="child content",
        word_count=2,
        created_by=dataset.created_by,
        index_node_id="child-node",
    )
    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=parent.id,
        summary_index_node_id="summary-node",
    )
    exclusive, shared = [
        make_upload_file(tenant_id=dataset.tenant_id, key=key, name="image.png", extension="png", mime_type="image/png")
        for key in ("uploads/exclusive.png", "uploads/shared.png")
    ]
    bindings = [
        SegmentAttachmentBinding(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=segment.document_id,
            segment_id=segment.id,
            attachment_id=upload.id,
        )
        for segment, upload in ((parent, exclusive), (parent, shared), (other_segment, shared))
    ]
    sqlite_session.add_all([other_document, *segments, child, summary, exclusive, shared, *bindings])
    sqlite_session.commit()
    return ReplacementGraph(dataset, document, parent, child, summary, exclusive, shared, other_segment)


class TestIndexProcessor:
    def test_format_preview_supports_qa_preview_shape(self) -> None:
        preview = _index_processor().format_preview(
            "qa_model",
            {"qa_chunks": [{"question": "Q1", "answer": "A1"}]},
        )

        assert preview.chunk_structure == "qa_model"
        assert preview.total_segments == 1
        assert len(preview.qa_preview) == 1
        assert preview.qa_preview[0].question == "Q1"
        assert preview.qa_preview[0].answer == "A1"

    def test_index_and_clean_ends_transactions_around_index_io(self, sqlite_session: Session) -> None:
        dataset, document = _persist_dataset_and_document(sqlite_session)
        phase_events: list[str] = []
        statements: list[str] = []
        statements_at_index: list[str] = []
        event.listen(sqlite_session, "after_commit", lambda _session: phase_events.append("commit"))
        bind = sqlite_session.get_bind()

        def record_statement(_connection, _cursor, statement, _parameters, _context, _executemany) -> None:
            statements.append(" ".join(statement.lower().split()))

        event.listen(bind, "before_cursor_execute", record_statement)

        index_processor = MagicMock()

        def record_index(*_args) -> None:
            statements_at_index.extend(statements)
            phase_events.append("index")

        index_processor.index.side_effect = record_index
        processor = _index_processor()
        admission_service = MagicMock()
        chunks = {"general_chunks": ["content"]}

        try:
            with (
                patch(
                    "core.rag.index_processor.index_processor.VectorSpaceAdmissionService",
                    return_value=admission_service,
                ),
                patch("core.rag.index_processor.index_processor.IndexProcessorFactory") as index_processor_factory,
            ):
                index_processor_factory.return_value.init_index_processor.return_value = index_processor
                processor.index_and_clean(
                    dataset_id=dataset.id,
                    document_id=document.id,
                    original_document_id="",
                    chunks=chunks,
                    batch="batch-1",
                    session=sqlite_session,
                )
        finally:
            event.remove(bind, "before_cursor_execute", record_statement)

        assert phase_events == ["commit", "index", "commit"]
        assert " from datasets " in f" {statements_at_index[-2]} "
        assert " from documents " in f" {statements_at_index[-1]} "
        admission_service.ensure_pipeline_can_be_indexed.assert_called_once_with(
            dataset=dataset,
            document_id=document.id,
            chunk_structure=dataset.chunk_structure,
            chunks=chunks,
            include_summaries=False,
            session=sqlite_session,
        )

    def test_index_and_clean_skips_admission_for_replacement_without_existing_vector_points(
        self, sqlite_session: Session
    ) -> None:
        dataset, document = _persist_dataset_and_document(sqlite_session)
        index_processor = MagicMock()
        processor = _index_processor()
        chunks = {"general_chunks": ["content"]}

        with (
            patch("core.rag.index_processor.index_processor.VectorSpaceAdmissionService") as admission_service_class,
            patch("core.rag.index_processor.index_processor.IndexProcessorFactory") as index_processor_factory,
            patch.object(index_processor_module, "Vector", autospec=True) as vector,
        ):
            vector.resolve_vector_type.return_value = "pgvector"
            vector.return_value.get_routing_target.return_value = ("pgvector", "collection-a")
            index_processor_factory.return_value.init_index_processor.return_value = index_processor
            processor.index_and_clean(
                dataset_id=dataset.id,
                document_id=document.id,
                original_document_id=document.id,
                chunks=chunks,
                batch="batch-1",
                session=sqlite_session,
            )

        admission_service_class.assert_not_called()

    def test_index_and_clean_scopes_replacement_queries_to_dataset_owner(self, sqlite_session: Session) -> None:
        dataset, document = _persist_dataset_and_document(sqlite_session)
        original_document = Document(
            id=str(uuid.uuid4()),
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            position=2,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch="batch-1",
            name="Original document",
            created_from=DocumentCreatedFrom.WEB,
            created_by=dataset.created_by,
        )
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=original_document.id,
            position=1,
            content="old content",
            word_count=3,
            tokens=3,
            created_by=dataset.created_by,
            index_node_id="node-1",
            status=SegmentStatus.COMPLETED,
        )
        control_segment = DocumentSegment(
            tenant_id=str(uuid.uuid4()),
            dataset_id=str(uuid.uuid4()),
            document_id=original_document.id,
            position=1,
            content="other tenant content",
            word_count=4,
            tokens=4,
            created_by=str(uuid.uuid4()),
            index_node_id="other-node",
            status=SegmentStatus.COMPLETED,
        )
        sqlite_session.add_all([original_document, segment, control_segment])
        sqlite_session.flush()

        processor = _index_processor()
        with (
            patch("core.rag.index_processor.index_processor.VectorSpaceAdmissionService") as admission_service_class,
            patch("core.rag.index_processor.index_processor.IndexProcessorFactory") as index_processor_factory,
            patch.object(index_processor_module, "Vector", autospec=True) as vector,
        ):
            vector.resolve_vector_type.return_value = "pgvector"
            vector.return_value.get_routing_target.return_value = ("pgvector", "collection-a")
            index_backend = index_processor_factory.return_value.init_index_processor.return_value
            processor.index_and_clean(
                dataset_id=dataset.id,
                document_id=document.id,
                original_document_id=original_document.id,
                chunks={},
                batch="batch-1",
                session=sqlite_session,
            )

        assert sqlite_session.get(DocumentSegment, segment.id) is None
        assert sqlite_session.get(DocumentSegment, control_segment.id) is control_segment
        assert sqlite_session.get(Document, document.id).indexing_status == "completed"
        vector.return_value.delete_by_ids.assert_called_once_with(["node-1"])
        index_backend.clean.assert_not_called()
        index_backend.index.assert_called_once_with(dataset, document, {}, sqlite_session)
        admission_service_class.assert_not_called()

    def test_replacement_cleans_attachment_vectors_and_durably_claims_only_unshared_files(
        self, sqlite_session: Session, replacement_graph: ReplacementGraph
    ) -> None:
        graph = replacement_graph

        deleted_batches: list[tuple[str, ...]] = []

        def delete_vectors(node_ids: Sequence[str]) -> None:
            batch = tuple(node_ids)
            deleted_batches.append(batch)
            if batch == (graph.exclusive.id,):
                assert sqlite_session.in_transaction()
            else:
                assert not sqlite_session.in_transaction()

        with (
            patch.object(index_processor_module, "Vector", autospec=True) as vector,
            patch.object(index_processor_module, "IndexProcessorFactory", autospec=True) as factory,
        ):
            vector.resolve_vector_type.return_value = "pgvector"
            vector.return_value.get_routing_target.return_value = ("pgvector", "collection-a")
            vector.return_value.delete_by_ids.side_effect = delete_vectors
            _index_processor().index_and_clean(
                dataset_id=graph.dataset.id,
                document_id=graph.document.id,
                original_document_id=graph.document.id,
                chunks={},
                batch="batch-1",
                session=sqlite_session,
            )

        assert deleted_batches == [("child-node", "summary-node")]
        attachment_cleanup = sqlite_session.scalars(select(AttachmentVectorCleanupOutbox)).one()
        assert attachment_cleanup.attachment_id == graph.exclusive.id
        assert attachment_cleanup.writer_finished_at is not None
        assert attachment_cleanup.cleanup_started_at is not None
        cleanup_route = VectorRoute.model_validate_json(attachment_cleanup.vector_route)
        assert (cleanup_route.vector_type, cleanup_route.collection_name) == ("pgvector", "collection-a")
        factory.return_value.init_index_processor.return_value.index.assert_called_once()
        assert sqlite_session.get(DocumentSegment, graph.parent.id) is None
        assert sqlite_session.get(ChildChunk, graph.child.id) is None
        assert sqlite_session.get(DocumentSegmentSummary, graph.summary.id) is None
        assert sqlite_session.get(UploadFile, graph.exclusive.id) is None
        assert sqlite_session.get(UploadFile, graph.shared.id) is graph.shared
        assert sqlite_session.get(DocumentSegment, graph.other_segment.id) is graph.other_segment
        bindings = sqlite_session.scalars(select(SegmentAttachmentBinding)).all()
        assert [(binding.segment_id, binding.attachment_id) for binding in bindings] == [
            (graph.other_segment.id, graph.shared.id)
        ]
        pending = sqlite_session.scalars(select(UploadFileCleanupOutbox)).all()
        assert [(row.tenant_id, row.upload_file_id, row.key) for row in pending] == [
            (graph.dataset.tenant_id, graph.exclusive.id, graph.exclusive.key)
        ]

    @pytest.mark.parametrize("failure_phase", ["vector", "sql"])
    def test_failed_replacement_preserves_attachment_ownership_and_files(
        self,
        sqlite_session: Session,
        replacement_graph: ReplacementGraph,
        failure_phase: str,
    ) -> None:
        graph = replacement_graph
        cleanup_rows = index_processor_module.delete_document_segments_and_claim_uploads

        def fail_after_sql(
            session: Session,
            *,
            tenant_id: str,
            dataset_id: str,
            document_ids: Sequence[str],
        ) -> Never:
            cleanup_rows(session, tenant_id=tenant_id, dataset_id=dataset_id, document_ids=document_ids)
            raise RuntimeError("SQL cleanup failed")

        with (
            patch.object(index_processor_module, "Vector", autospec=True) as vector,
            patch.object(index_processor_module, "IndexProcessorFactory", autospec=True) as factory,
            patch.object(index_processor_module, "delete_document_segments_and_claim_uploads") as delete_rows,
        ):
            vector.resolve_vector_type.return_value = "pgvector"
            vector.return_value.get_routing_target.return_value = ("pgvector", "collection-a")
            vector.return_value.delete_by_ids.side_effect = (
                RuntimeError("vector unavailable") if failure_phase == "vector" else None
            )
            delete_rows.side_effect = fail_after_sql if failure_phase == "sql" else cleanup_rows
            with pytest.raises(RuntimeError, match="vector unavailable|SQL cleanup failed"):
                _index_processor().index_and_clean(
                    dataset_id=graph.dataset.id,
                    document_id=graph.document.id,
                    original_document_id=graph.document.id,
                    chunks={},
                    batch="batch-1",
                    session=sqlite_session,
                )
            sqlite_session.rollback()

        factory.return_value.init_index_processor.return_value.index.assert_not_called()
        assert sqlite_session.get(DocumentSegment, graph.parent.id) is not None
        assert sqlite_session.get(ChildChunk, graph.child.id) is not None
        assert sqlite_session.get(DocumentSegmentSummary, graph.summary.id) is not None
        assert sqlite_session.get(UploadFile, graph.exclusive.id) is not None
        assert len(sqlite_session.scalars(select(SegmentAttachmentBinding)).all()) == 3
        assert not sqlite_session.scalars(select(UploadFileCleanupOutbox)).all()

    def test_get_preview_output_scopes_document_to_dataset_owner(self, sqlite_session: Session) -> None:
        dataset, document = _persist_dataset_and_document(sqlite_session, indexing_technique=IndexTechniqueType.ECONOMY)
        processor = _index_processor()
        expected_preview = MagicMock()

        with patch.object(processor, "format_preview", return_value=expected_preview):
            result = processor.get_preview_output(
                chunks={},
                dataset_id=dataset.id,
                document_id=document.id,
                chunk_structure="text_model",
                summary_index_setting=None,
                session=sqlite_session,
            )

        assert result is expected_preview

    def test_preview_summary_workers_use_independent_sessions(
        self, sqlite_session: Session, sqlite_session_factory: sessionmaker[Session]
    ) -> None:
        dataset, _ = _persist_dataset_and_document(
            sqlite_session,
            summary_index_setting={"enable": True},
        )
        phase_events: list[str] = []
        event.listen(sqlite_session, "after_commit", lambda _session: phase_events.append("commit"))
        worker_sessions: list[Session] = []
        preview = Preview(
            chunk_structure="text_model",
            total_segments=2,
            preview=[PreviewItem(content="chunk-1"), PreviewItem(content="chunk-2")],
        )
        flask_app = SimpleNamespace(app_context=lambda: nullcontext())
        processor = _index_processor()

        def generate_summary(*_args, **kwargs):
            phase_events.append("worker")
            worker_sessions.append(kwargs["session"])
            return "summary", None

        with (
            patch.object(processor, "format_preview", return_value=preview),
            patch(
                "core.rag.index_processor.index_processor.current_app",
                SimpleNamespace(_get_current_object=lambda: flask_app),
            ),
            patch.object(index_processor_module.session_factory, "create_session", sqlite_session_factory),
            patch.object(
                index_processor_module.ParagraphIndexProcessor,
                "generate_summary",
                side_effect=generate_summary,
            ) as generate_summary,
        ):
            result = processor.get_preview_output(
                chunks=[],
                dataset_id=dataset.id,
                document_id="",
                chunk_structure="text_model",
                summary_index_setting={"enable": True},
                session=sqlite_session,
            )

        assert all(item.summary == "summary" for item in result.preview)
        assert phase_events == ["commit", "worker", "worker"]
        call_sessions = [call.kwargs["session"] for call in generate_summary.call_args_list]
        assert all(call_session is not sqlite_session for call_session in call_sessions)
        assert call_sessions == worker_sessions

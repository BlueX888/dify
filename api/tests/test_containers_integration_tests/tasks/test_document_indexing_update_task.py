from unittest.mock import patch

import pytest
from faker import Faker
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from models import Account, AccountStatus, Tenant, TenantAccountJoin, TenantAccountRole, TenantStatus
from models.dataset import Dataset, Document, DocumentSegment
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus, SegmentStatus
from services.knowledge.entities.resource_scope import DatasetRef
from tasks.document_indexing_update_task import document_indexing_update_task


class TestDocumentIndexingUpdateTask:
    @pytest.fixture
    def mock_external_dependencies(self):
        """Patch the cleanup and indexing application boundaries."""
        with (
            patch("tasks.document_indexing_update_task.clean_document_for_reindex", autospec=True) as mock_cleanup,
            patch("tasks.document_indexing_update_task.build_document_indexing_service", autospec=True) as mock_runner,
        ):
            runner_instance = mock_runner.return_value
            yield {
                "cleanup": mock_cleanup,
                "runner": mock_runner,
                "runner_instance": runner_instance,
            }

    def _create_dataset_document_with_segments(self, db_session_with_containers: Session, *, segment_count: int = 2):
        fake = Faker()

        # Account and tenant
        account = Account(
            email=fake.email(),
            name=fake.name(),
            interface_language="en-US",
            status=AccountStatus.ACTIVE,
        )
        db_session_with_containers.add(account)
        db_session_with_containers.commit()

        tenant = Tenant(name=fake.company(), status=TenantStatus.NORMAL)
        db_session_with_containers.add(tenant)
        db_session_with_containers.commit()

        join = TenantAccountJoin(
            tenant_id=tenant.id,
            account_id=account.id,
            role=TenantAccountRole.OWNER,
            current=True,
        )
        db_session_with_containers.add(join)
        db_session_with_containers.commit()

        # Dataset and document
        dataset = Dataset(
            tenant_id=tenant.id,
            name=fake.company(),
            description=fake.text(max_nb_chars=64),
            data_source_type=DataSourceType.UPLOAD_FILE,
            indexing_technique=IndexTechniqueType.HIGH_QUALITY,
            created_by=account.id,
        )
        db_session_with_containers.add(dataset)
        db_session_with_containers.commit()

        document = Document(
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            position=0,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch="test_batch",
            name=fake.file_name(),
            created_from=DocumentCreatedFrom.WEB,
            created_by=account.id,
            indexing_status=IndexingStatus.WAITING,
            enabled=True,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
        )
        db_session_with_containers.add(document)
        db_session_with_containers.commit()

        # Segments
        node_ids = []
        for i in range(segment_count):
            node_id = f"node-{i + 1}"
            seg = DocumentSegment(
                tenant_id=tenant.id,
                dataset_id=dataset.id,
                document_id=document.id,
                position=i,
                content=fake.text(max_nb_chars=32),
                answer=None,
                word_count=10,
                tokens=5,
                index_node_id=node_id,
                status=SegmentStatus.COMPLETED,
                created_by=account.id,
            )
            db_session_with_containers.add(seg)
            node_ids.append(node_id)
        db_session_with_containers.commit()

        # Refresh to ensure ORM state
        db_session_with_containers.refresh(dataset)
        db_session_with_containers.refresh(document)

        return dataset, document, node_ids

    def test_delegates_cleanup_and_reindexes(self, db_session_with_containers: Session, mock_external_dependencies):
        dataset, document, _ = self._create_dataset_document_with_segments(db_session_with_containers)

        # Act
        document_indexing_update_task(dataset.id, document.id)

        # Ensure we see committed changes from another session
        db_session_with_containers.expire_all()

        # Assert document status updated before reindex
        updated = db_session_with_containers.scalar(select(Document).where(Document.id == document.id).limit(1))
        assert updated.indexing_status == IndexingStatus.PARSING
        assert updated.processing_started_at is not None

        cleanup_call = mock_external_dependencies["cleanup"].call_args
        assert cleanup_call is not None
        assert cleanup_call.args == ()
        assert cleanup_call.kwargs["tenant_id"] == dataset.tenant_id
        assert cleanup_call.kwargs["dataset_id"] == dataset.id
        assert cleanup_call.kwargs["document_ids"] == (document.id,)
        assert cleanup_call.kwargs["doc_form"] == IndexStructureType.PARAGRAPH_INDEX
        assert callable(cleanup_call.kwargs["new_session"])
        assert callable(cleanup_call.kwargs["delete_blob"])

        # Reindexing receives the detached reference with its complete ownership chain.
        mock_external_dependencies["runner_instance"].run.assert_called_once_with(
            [DatasetRef(dataset.tenant_id, dataset.id).document(document.id)]
        )

    def test_clean_error_marks_document_failed_without_indexing(
        self, db_session_with_containers: Session, mock_external_dependencies
    ):
        dataset, document, _ = self._create_dataset_document_with_segments(db_session_with_containers)

        mock_external_dependencies["cleanup"].side_effect = Exception("boom")

        document_indexing_update_task(dataset.id, document.id)

        # Ensure we see committed changes from another session
        db_session_with_containers.expire_all()

        updated = db_session_with_containers.scalar(select(Document).where(Document.id == document.id).limit(1))
        assert updated is not None
        assert updated.indexing_status == IndexingStatus.ERROR
        assert updated.error == "boom"
        mock_external_dependencies["runner_instance"].run.assert_not_called()

        # Segments should remain (since clean failed before DB delete)
        remaining = db_session_with_containers.scalar(
            select(func.count()).select_from(DocumentSegment).where(DocumentSegment.document_id == document.id)
        )
        assert remaining > 0

    def test_document_not_found_noop(self, db_session_with_containers: Session, mock_external_dependencies):
        fake = Faker()
        # Act with non-existent document id
        document_indexing_update_task(dataset_id=fake.uuid4(), document_id=fake.uuid4())

        # Neither cleanup nor indexing should be called.
        mock_external_dependencies["cleanup"].assert_not_called()
        mock_external_dependencies["runner_instance"].run.assert_not_called()

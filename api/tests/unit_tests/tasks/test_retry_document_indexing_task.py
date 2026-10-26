from dataclasses import dataclass
from unittest.mock import ANY, MagicMock, call, create_autospec, patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from core.app.apps.pipeline.pipeline_generator import PipelineGenerator
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from enums import DeploymentEdition
from models import Account, Tenant, TenantAccountJoin
from models.account import TenantAccountRole
from models.dataset import Dataset, Document, DocumentPipelineExecutionLog
from models.enums import DatasetRuntimeMode, DataSourceType, DocumentCreatedFrom, IndexingStatus
from tasks.retry_document_indexing_task import retry_document_indexing_task
from tests.unit_tests.config_override import config_overrides_context


@dataclass(frozen=True, slots=True)
class _KnowledgeServicesStub:
    pipeline_generator: PipelineGenerator


@dataclass(frozen=True, slots=True)
class _ApplicationServicesStub:
    knowledge: _KnowledgeServicesStub


@dataclass(frozen=True, slots=True)
class _VectorSpaceStub:
    limit: int
    size: int


@dataclass(frozen=True, slots=True)
class _FeaturesStub:
    vector_space: _VectorSpaceStub


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
@pytest.mark.parametrize("runtime_mode", [DatasetRuntimeMode.GENERAL, DatasetRuntimeMode.RAG_PIPELINE])
def test_retry_skips_missing_document_and_uses_dataset_runtime(
    sqlite_session: Session, runtime_mode: DatasetRuntimeMode
) -> None:
    tenant = Tenant(name="Retry tenant")
    user = Account(name="Retry user", email=f"retry-{uuid4()}@example.com")
    membership = TenantAccountJoin(
        tenant_id=tenant.id,
        account_id=user.id,
        current=True,
        role=TenantAccountRole.OWNER,
    )
    dataset = Dataset(
        id=str(uuid4()),
        tenant_id=tenant.id,
        name="Retry dataset",
        created_by=user.id,
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.ECONOMY,
        chunk_structure=IndexStructureType.PARAGRAPH_INDEX,
        runtime_mode=runtime_mode,
    )
    if runtime_mode == DatasetRuntimeMode.RAG_PIPELINE:
        dataset.pipeline_id = str(uuid4())
    document = Document(
        id=str(uuid4()),
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info="{}",
        batch="retry-batch",
        name="Retry document",
        created_from=DocumentCreatedFrom.WEB,
        created_by=user.id,
        indexing_status=IndexingStatus.WAITING,
        enabled=True,
        archived=False,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    records: list[object] = [tenant, user, membership, dataset, document]
    if runtime_mode == DatasetRuntimeMode.RAG_PIPELINE:
        assert dataset.pipeline_id is not None
        records.append(
            DocumentPipelineExecutionLog(
                pipeline_id=dataset.pipeline_id,
                document_id=document.id,
                datasource_type="local_file",
                datasource_info="{}",
                datasource_node_id="start",
                input_data={},
                created_by=user.id,
                workflow_id="workflow-v1",
                original_document_id=document.id,
            )
        )
    sqlite_session.add_all(records)
    sqlite_session.commit()
    features = MagicMock()
    missing_document_id = str(uuid4())
    pipeline_generator = create_autospec(PipelineGenerator, instance=True, spec_set=True)
    services = _ApplicationServicesStub(knowledge=_KnowledgeServicesStub(pipeline_generator=pipeline_generator))

    with (
        patch("tasks.retry_document_indexing_task.FeatureService.get_features", return_value=features),
        patch("tasks.retry_document_indexing_task.clean_document_for_reindex") as clean_for_reindex,
        patch("tasks.retry_document_indexing_task.build_document_indexing_service") as indexing_runner,
        patch("services.rag_pipeline.rag_pipeline.RagPipelineService") as pipeline_service,
        patch("tasks.retry_document_indexing_task.redis_client") as redis,
        patch("extensions.ext_application_services.application_services", return_value=services),
    ):
        retry_document_indexing_task.run(dataset.id, [missing_document_id, document.id], user.id)

    assert redis.delete.call_args_list == [
        call(f"document_{missing_document_id}_is_retried"),
        call(f"document_{document.id}_is_retried"),
    ]
    if runtime_mode == DatasetRuntimeMode.GENERAL:
        assert clean_for_reindex.call_count == 1
        assert clean_for_reindex.call_args.kwargs["tenant_id"] == tenant.id
        assert clean_for_reindex.call_args.kwargs["dataset_id"] == dataset.id
        assert clean_for_reindex.call_args.kwargs["document_ids"] == (document.id,)
        pipeline_service.assert_not_called()
        indexing_runner.assert_called_once_with(session_factory=ANY, enforce_vector_space_admission=True)
        (run_documents,) = indexing_runner.return_value.run.call_args.args
        assert [item.document_id for item in run_documents] == [document.id]
        assert run_documents[0].dataset.tenant_id == dataset.tenant_id
    else:
        clean_for_reindex.assert_not_called()
        indexing_runner.assert_not_called()
        retry = pipeline_service.return_value.retry_error_document
        retry.assert_called_once()
        retry_dataset, retry_document, retry_user = retry.call_args.args
        assert (retry_dataset.id, retry_document.id, retry_user.id) == (dataset.id, document.id, user.id)
        assert retry.call_args.kwargs["generator"] is pipeline_generator


@config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
def test_retry_quota_failure_marks_and_unlocks_every_document(sqlite_session: Session) -> None:
    tenant = Tenant(name="Quota tenant")
    user = Account(name="Quota user", email=f"retry-quota-{uuid4()}@example.com")
    membership = TenantAccountJoin(
        tenant_id=tenant.id,
        account_id=user.id,
        current=True,
        role=TenantAccountRole.OWNER,
    )
    dataset = Dataset(
        id=str(uuid4()),
        tenant_id=tenant.id,
        name="Quota dataset",
        created_by=user.id,
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.ECONOMY,
        chunk_structure=IndexStructureType.PARAGRAPH_INDEX,
        runtime_mode=DatasetRuntimeMode.GENERAL,
    )
    documents = [
        Document(
            id=str(uuid4()),
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            position=position,
            data_source_type=DataSourceType.UPLOAD_FILE,
            data_source_info="{}",
            batch="quota-batch",
            name=f"Quota document {position}",
            created_from=DocumentCreatedFrom.WEB,
            created_by=user.id,
            indexing_status=IndexingStatus.WAITING,
            enabled=True,
            archived=False,
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
        )
        for position in (1, 2)
    ]
    sqlite_session.add_all([tenant, user, membership, dataset, *documents])
    sqlite_session.commit()
    pipeline_generator = create_autospec(PipelineGenerator, instance=True, spec_set=True)
    services = _ApplicationServicesStub(knowledge=_KnowledgeServicesStub(pipeline_generator=pipeline_generator))
    features = _FeaturesStub(vector_space=_VectorSpaceStub(limit=10, size=10))

    with (
        patch("tasks.retry_document_indexing_task.FeatureService.get_features", return_value=features),
        patch("tasks.retry_document_indexing_task.clean_document_for_reindex") as clean_for_reindex,
        patch("tasks.retry_document_indexing_task.build_document_indexing_service") as indexing_runner,
        patch("tasks.retry_document_indexing_task.RagPipelineService") as pipeline_service,
        patch("tasks.retry_document_indexing_task.redis_client") as redis,
        patch("extensions.ext_application_services.application_services", return_value=services),
    ):
        retry_document_indexing_task.run(dataset.id, [document.id for document in documents], user.id)

    sqlite_session.expire_all()
    persisted_documents = [sqlite_session.get(Document, document.id) for document in documents]
    assert all(document is not None for document in persisted_documents)
    assert all(document.indexing_status == IndexingStatus.ERROR for document in persisted_documents if document)
    assert all("over the limit" in (document.error or "") for document in persisted_documents if document)
    assert redis.delete.call_args_list == [
        call(f"document_{documents[0].id}_is_retried"),
        call(f"document_{documents[1].id}_is_retried"),
    ]
    clean_for_reindex.assert_not_called()
    indexing_runner.assert_not_called()
    pipeline_service.assert_not_called()

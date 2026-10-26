from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import create_autospec, patch

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session, sessionmaker

from core.entities.knowledge_entities import PreviewDetail
from core.errors.error import LLMBadRequestError, ProviderTokenNotInitError
from core.plugin.impl.exc import PluginDaemonClientSideError
from core.rag.extractor.entity.datasource_type import DatasourceType
from core.rag.extractor.entity.extract_setting import (
    CreatedUploadArtifact,
    ExtractionResult,
    ExtractSetting,
    NotionInfo,
)
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from core.rag.index_processor.index_processor_base import BaseIndexProcessor
from core.rag.models.document import ChildDocument, Document
from enums import DeploymentEdition
from extensions.storage.storage_type import StorageType
from models.dataset import Dataset, DatasetProcessRule
from models.enums import CreatorUserRole, ProcessRuleMode
from models.model import UploadFile, UploadFileCleanupOutbox
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.estimate import (
    IndexingEstimateAdapter,
    SQLAlchemyProcessRuleReader,
)
from services.knowledge.indexing.estimate import (
    IndexingEstimateExecutionError,
    IndexingEstimateProviderUnavailableError,
)


def _dataset(dataset_id: str, workspace_id: str) -> Dataset:
    return Dataset(
        id=dataset_id,
        tenant_id=workspace_id,
        name=f"Dataset {dataset_id}",
        description="",
        provider="vendor",
        permission="only_me",
        created_by="account-1",
        maintainer="account-1",
        data_source_type="upload_file",
        indexing_technique="economy",
    )


def test_notion_access_token_is_excluded_from_model_serialization_and_repr() -> None:
    notion_info = NotionInfo(
        credential_id="credential-1",
        notion_workspace_id="workspace-1",
        notion_obj_id="page-1",
        notion_page_type="page",
        tenant_id="tenant-1",
        notion_access_token="secret",
    )

    assert "notion_access_token" not in notion_info.model_dump()
    assert "secret" not in repr(notion_info)


def test_process_rule_reader_loads_tenant_scoped_rule(sqlite_session_factory: sessionmaker[Session]) -> None:
    with sqlite_session_factory.begin() as session:
        session.add(_dataset("dataset-1", "workspace-1"))
        rule = DatasetProcessRule(
            dataset_id="dataset-1",
            mode=ProcessRuleMode.CUSTOM,
            rules='{"segmentation": {"separator": "\\n", "max_tokens": 100}}',
            created_by="account-1",
        )
        rule.id = "rule-1"
        session.add(rule)

    reader = SQLAlchemyProcessRuleReader(session_factory=sqlite_session_factory)

    assert reader.get_by_id(dataset_ref=DatasetRef("workspace-1", "dataset-1"), process_rule_id="rule-1") == {
        "mode": "custom",
        "rules": {"segmentation": {"separator": "\n", "max_tokens": 100}},
    }
    assert reader.get_by_id(dataset_ref=DatasetRef("workspace-2", "dataset-1"), process_rule_id="rule-1") is None


@pytest.mark.parametrize("doc_form", ["text_model", "hierarchical_model", "qa_model"])
def test_estimates_segments_and_limits_previews(
    sqlite_session_factory: sessionmaker[Session], doc_form: str, config_overrides: Callable[..., None]
) -> None:
    config_overrides(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    processor = create_autospec(BaseIndexProcessor, instance=True, spec_set=True)
    processor.extract.return_value = ExtractionResult(documents=[Document(page_content="source")])
    processor.transform.return_value = [
        Document(
            page_content=f"question-{i}",
            metadata={"answer": f"answer-{i}"},
            children=[ChildDocument(page_content="child")],
        )
        for i in range(11)
    ]
    with patch("services.knowledge.indexing.adapters.estimate.IndexProcessorFactory") as factory:
        factory.return_value.init_index_processor.return_value = processor
        result = IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
            "workspace-1", [ExtractSetting(datasource_type=DatasourceType.WEBSITE)], {"mode": "automatic"}, doc_form
        )
    if doc_form == "qa_model":
        assert result.total_segments == 220
        assert len(result.qa_preview) == 10
        assert result.qa_preview[-1].answer == "answer-9"
    else:
        assert result.total_segments == 11
        assert len(result.preview) == 10
        assert result.preview[0].child_chunks == ["child"]


def test_estimate_enforces_batch_limit(
    sqlite_session_factory: sessionmaker[Session], config_overrides: Callable[..., None]
) -> None:
    config_overrides(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD, BATCH_UPLOAD_LIMIT=1)
    with pytest.raises(IndexingEstimateExecutionError, match="batch upload limit"):
        IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
            "workspace-1", [ExtractSetting(datasource_type=DatasourceType.WEBSITE)] * 2, {"mode": "automatic"}
        )


@pytest.mark.parametrize("storage_fails", [False, True])
def test_preview_cleanup_commits_before_summary_and_excludes_foreign_files(
    sqlite_session_factory: sessionmaker[Session], storage_fails: bool
) -> None:
    image_ids = []
    with sqlite_session_factory.begin() as session:
        for tenant in ["workspace-1", "foreign"]:
            upload = UploadFile(
                tenant_id=tenant,
                storage_type=StorageType.LOCAL,
                key=f"{tenant}/image.png",
                name="image.png",
                size=10,
                extension="png",
                mime_type="image/png",
                created_by_role=CreatorUserRole.ACCOUNT,
                created_by="user",
                created_at=datetime.now(UTC),
                used=True,
            )
            image_ids.append(upload.id)
            session.add(upload)
    processor = create_autospec(BaseIndexProcessor, instance=True, spec_set=True)
    processor.extract.return_value = ExtractionResult(
        documents=[Document(page_content="source")],
        created_upload_artifacts=(CreatedUploadArtifact(image_ids[0], "workspace-1/image.png"),),
    )
    processor.transform.return_value = [
        Document(page_content=f"preview ![foreign](/files/{image_ids[1]}/file-preview)")
    ]

    def summary(
        _tenant, previews: list[PreviewDetail], _settings, _language, *, session: Session
    ) -> list[PreviewDetail]:
        assert not session.in_transaction()
        with sqlite_session_factory() as observer:
            assert observer.get(UploadFile, image_ids[0]) is None
            assert observer.get(UploadFile, image_ids[1]) is not None
            pending = observer.scalar(
                select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == image_ids[0])
            )
            assert (pending is not None) is storage_fails
        return [PreviewDetail(content=previews[0].content, summary="summary")]

    processor.generate_summary_preview.side_effect = summary
    with (
        patch("services.knowledge.indexing.adapters.estimate.IndexProcessorFactory") as factory,
        patch("services.knowledge.indexing.adapters.estimate.storage") as storage,
    ):
        factory.return_value.init_index_processor.return_value = processor
        if storage_fails:
            storage.delete.side_effect = OSError("storage unavailable")
        result = IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
            "workspace-1",
            [ExtractSetting(datasource_type=DatasourceType.WEBSITE)],
            {"mode": "automatic", "summary_index_setting": {"enable": True}},
            "text_model",
        )
        storage.delete.assert_called_once_with("workspace-1/image.png")
    assert result.preview[0].summary == "summary"


def test_preview_cleanup_runs_when_transform_fails(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    temporary_file_ids: list[str] = []

    def extract(_setting: ExtractSetting, *, session: Session, **_kwargs: object) -> ExtractionResult:
        upload = UploadFile(
            tenant_id="workspace-1",
            storage_type=StorageType.LOCAL,
            key="workspace-1/temp.png",
            name="temp.png",
            size=10,
            extension="png",
            mime_type="image/png",
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by="user",
            created_at=datetime.now(UTC),
            used=True,
        )
        session.add(upload)
        temporary_file_ids.append(upload.id)
        return ExtractionResult(
            documents=[Document(page_content="source")],
            created_upload_artifacts=(CreatedUploadArtifact(upload.id, upload.key),),
        )

    processor = create_autospec(BaseIndexProcessor, instance=True, spec_set=True)
    processor.extract.side_effect = extract
    processor.transform.side_effect = RuntimeError("transform failed")

    with (
        patch("services.knowledge.indexing.adapters.estimate.IndexProcessorFactory") as factory,
        patch("services.knowledge.indexing.adapters.estimate.storage") as storage,
    ):
        factory.return_value.init_index_processor.return_value = processor
        with pytest.raises(IndexingEstimateExecutionError, match="transform failed"):
            IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
                "workspace-1",
                [ExtractSetting(datasource_type=DatasourceType.WEBSITE)],
                {"mode": "automatic"},
            )

    storage.delete.assert_called_once_with("workspace-1/temp.png")
    assert len(temporary_file_ids) == 1
    with sqlite_session_factory() as session:
        assert session.get(UploadFile, temporary_file_ids[0]) is None


def test_preview_owns_and_cleans_files_created_during_transform(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    uploads = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    upload = UploadFile(
        tenant_id="workspace-1",
        storage_type=StorageType.LOCAL,
        key="workspace-1/transform-temp.png",
        name="transform-temp.png",
        size=10,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="user",
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
    processor = create_autospec(BaseIndexProcessor, instance=True, spec_set=True)
    processor.extract.return_value = ExtractionResult(documents=[Document(page_content="source")])

    def transform(*_args: object, **kwargs: object) -> list[Document]:
        callback = kwargs["on_upload_file_created"]
        assert callable(callback)
        callback(upload.id)
        return [Document(page_content="preview")]

    processor.transform.side_effect = transform
    with (
        patch("services.knowledge.indexing.adapters.estimate.IndexProcessorFactory") as factory,
        patch("services.knowledge.indexing.adapters.estimate.storage") as storage,
    ):
        factory.return_value.init_index_processor.return_value = processor
        IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
            "workspace-1",
            [ExtractSetting(datasource_type=DatasourceType.WEBSITE)],
            {"mode": "automatic"},
        )

    storage.delete.assert_called_once_with(upload.key)
    with sqlite_session_factory() as session:
        assert session.get(UploadFile, upload.id) is None
        assert session.scalar(select(UploadFileCleanupOutbox)) is None


def test_preview_commit_failure_keeps_exact_blob_key_for_retry(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    artifact_id: list[str] = []

    def extract(_setting: ExtractSetting, *, session: Session, **kwargs: object) -> ExtractionResult:
        upload = UploadFile(
            tenant_id="workspace-1",
            storage_type=StorageType.LOCAL,
            key="workspace-1/commit-unknown.png",
            name="temp.png",
            size=10,
            extension="png",
            mime_type="image/png",
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by="user",
            created_at=datetime.now(UTC),
            used=True,
        )
        session.add(upload)
        artifact = CreatedUploadArtifact(upload.id, upload.key)
        callback = kwargs["on_upload_artifact_created"]
        assert callable(callback)
        callback(artifact)
        artifact_id.append(upload.id)
        return ExtractionResult(
            documents=[Document(page_content="source")],
            created_upload_artifacts=(artifact,),
        )

    commit_attempts = 0

    def fail_first_commit(_session: Session) -> None:
        nonlocal commit_attempts
        commit_attempts += 1
        # The writer marker commits before storage I/O; fail the following
        # UploadFile publication commit instead.
        if commit_attempts == 2:
            raise RuntimeError("commit result unknown")

    processor = create_autospec(BaseIndexProcessor, instance=True, spec_set=True)
    processor.extract.side_effect = extract
    event.listen(sqlite_session_factory.class_, "before_commit", fail_first_commit)
    try:
        with (
            patch("services.knowledge.indexing.adapters.estimate.IndexProcessorFactory") as factory,
            patch("services.knowledge.indexing.adapters.estimate.storage") as storage,
        ):
            factory.return_value.init_index_processor.return_value = processor
            storage.delete.side_effect = OSError("storage unavailable")
            with pytest.raises(IndexingEstimateExecutionError, match="commit result unknown"):
                IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
                    "workspace-1",
                    [ExtractSetting(datasource_type=DatasourceType.WEBSITE)],
                    {"mode": "automatic"},
                )
    finally:
        event.remove(sqlite_session_factory.class_, "before_commit", fail_first_commit)

    with sqlite_session_factory() as session:
        assert session.get(UploadFile, artifact_id[0]) is None
        pending = session.scalar(
            select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == artifact_id[0])
        )
        assert pending is not None
        assert pending.key == "workspace-1/commit-unknown.png"


@pytest.mark.parametrize("provider", [None, "embedding-provider"])
def test_estimate_selects_dataset_embedding_and_rejects_foreign_tenant(
    sqlite_session_factory: sessionmaker[Session], provider: str | None
) -> None:
    with sqlite_session_factory.begin() as session:
        dataset = _dataset("dataset-1", "workspace-1")
        dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
        dataset.embedding_model_provider = provider
        dataset.embedding_model = "embedding-model"
        session.add(dataset)
    adapter = IndexingEstimateAdapter(session_factory=sqlite_session_factory)
    with (
        patch("services.knowledge.indexing.adapters.estimate.ModelManager.for_tenant") as manager,
        patch("services.knowledge.indexing.adapters.estimate.IndexProcessorFactory"),
    ):
        adapter.run("workspace-1", [], {"mode": "automatic"}, dataset_id="dataset-1")
        selected = (
            manager.return_value.get_model_instance if provider else manager.return_value.get_default_model_instance
        )
        selected.assert_called_once()
        with pytest.raises(IndexingEstimateExecutionError, match="Dataset not found"):
            adapter.run("foreign", [], {"mode": "automatic"}, dataset_id="dataset-1")


@pytest.mark.parametrize(
    ("processor_error", "expected_error"),
    [
        (LLMBadRequestError("provider unavailable"), IndexingEstimateProviderUnavailableError),
        (ProviderTokenNotInitError("provider unavailable"), IndexingEstimateProviderUnavailableError),
        (PluginDaemonClientSideError("provider unavailable"), IndexingEstimateProviderUnavailableError),
        (RuntimeError("processor failed"), IndexingEstimateExecutionError),
    ],
)
def test_estimate_translates_infrastructure_errors(
    sqlite_session_factory: sessionmaker[Session],
    processor_error: Exception,
    expected_error: type[Exception],
) -> None:
    adapter = IndexingEstimateAdapter(session_factory=sqlite_session_factory)
    with (
        patch("services.knowledge.indexing.adapters.estimate.IndexProcessorFactory", side_effect=processor_error),
        pytest.raises(expected_error, match="provider unavailable|processor failed"),
    ):
        adapter.run(
            "workspace-1",
            [ExtractSetting(datasource_type=DatasourceType.WEBSITE)],
            {"mode": "automatic", "rules": {}},
        )

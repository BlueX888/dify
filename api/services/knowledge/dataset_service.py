import copy
import datetime
import json
import logging
import secrets
import time
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import Any, Literal, Protocol, TypedDict, cast

import sqlalchemy as sa
from redis.exceptions import LockNotOwnedError
from sqlalchemy import ColumnElement, case, delete, exists, func, select, tuple_, update
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.exceptions import Forbidden, NotFound

from configs import dify_config
from core.errors.error import LLMBadRequestError, ProviderTokenNotInitError
from core.helper.name_generator import generate_incremental_name
from core.model_manager import ModelManager
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.entities import Rule
from core.rag.extractor.entity.extract_setting import WebsiteJobRef
from core.rag.index_processor.constant.built_in_field import BuiltInField
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from enums import CloudPlan, DeploymentEdition
from extensions.application_services.resource_access_token import build_resource_access_token_cleanup_service
from extensions.ext_redis import redis_client
from graphon.file import helpers as file_helpers
from graphon.model_runtime.entities.model_entities import ModelFeature, ModelType
from graphon.model_runtime.model_providers.base.text_embedding_model import TextEmbeddingModel
from libs import helper
from libs.datetime_utils import naive_utc_now
from libs.login import current_user
from libs.pagination import paginate_query
from models import Account, Tenant, TenantAccountRole
from models.dataset import (
    AppDatasetJoin,
    ChildChunk,
    Dataset,
    DatasetAutoDisableLog,
    DatasetCollectionBinding,
    DatasetPermission,
    DatasetPermissionEnum,
    DatasetProcessRule,
    DatasetQuery,
    Document,
    DocumentPipelineExecutionLog,
    DocumentSegment,
    ExternalKnowledgeBindings,
    Pipeline,
    build_built_in_field_values,
)
from models.enums import (
    DatasetRuntimeMode,
    DataSourceType,
    DocumentCreatedFrom,
    IndexingStatus,
    ProcessRuleMode,
    SegmentStatus,
)
from models.model import DocumentIndexingOutbox, UploadFile
from models.provider_ids import ModelProviderID
from models.source import DataSourceOauthBinding
from models.workflow import Workflow
from repositories.knowledge.dataset_indexing_outbox_repository import create_dataset_indexing_intents
from repositories.knowledge.dataset_read_repository import get_dataset_doc_form
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    create_document_indexing_intents,
    create_pipeline_document_indexing_intents,
    invalidate_notion_removal_intents,
)
from repositories.knowledge.document_repository import has_active_indexing_claim
from repositories.knowledge.metadata_repository import lock_metadata_dataset, lock_metadata_document
from repositories.knowledge.resource_cleanup_repository import (
    create_document_cleanup_intent,
    create_document_cleanup_intents,
    delete_dataset_for_cleanup,
)
from repositories.knowledge.segment_repository import query_child_chunks
from repositories.knowledge.upload_file_repository import (
    get_document_source_upload_file_id,
    register_upload_file_cleanup_candidates_in_session,
)
from services.api_token_service import ApiTokenCache
from services.enterprise import rbac_service as enterprise_rbac_service
from services.entities.feature_entities import FeatureModel
from services.entities.knowledge_entities.rag_pipeline_entities import (
    KnowledgeConfiguration,
    RagPipelineDatasetCreateEntity,
)
from services.errors.base import NoPermissionError
from services.errors.chunk import ChildChunkDeleteIndexError, ChildChunkIndexingError
from services.errors.dataset import DatasetNameDuplicateError
from services.errors.document import DocumentIndexingError
from services.errors.file import FileNotExistsError
from services.feature_service import FeatureService
from services.file_service import FileService
from services.knowledge.dataset_access import DatasetAccessRecord, DatasetAccessSnapshot, can_access_dataset
from services.knowledge.entities.datasets import (
    DatasetIndexingJob,
    DatasetIndexingRepresentation,
)
from services.knowledge.entities.document_creation import DocumentCreationDataset, DocumentIndexingJobs
from services.knowledge.entities.indexing_estimate import normalize_process_rule
from services.knowledge.entities.knowledge_entities import (
    KnowledgeConfig,
    RerankingModel,
    RetrievalModel,
)
from services.knowledge.entities.resource_scope import DatasetRef, SegmentRef
from services.knowledge.entities.segments import SegmentUpdateArgs
from services.knowledge.indexing.vector_route import capture_vector_route
from services.knowledge.indexing.website_sync_hint import (
    WEBSITE_SYNC_HINT_TTL_SECONDS,
    delete_website_sync_hint_if_owned,
    website_sync_hint_key,
)
from services.knowledge.segments.adapters import (
    request_document_disable_intents_locked,
    request_document_enable_intents_locked,
)
from services.knowledge.segments.application import (
    ChildChunkDeleteIndexApplicationError,
    ChildChunkIndexingApplicationError,
    SegmentDatasetRecord,
    SegmentDocumentRecord,
    SegmentMutationScope,
    SegmentMutationService,
    validate_segment_values,
)
from services.rag_pipeline.rag_pipeline import RagPipelineService
from services.rbac import contracts as rbac_contracts
from services.tag_application_service import TagTargetQuery
from tasks.add_document_to_index_task import add_document_to_index_task
from tasks.dataset_indexing_outbox_task import enqueue_dataset_indexing_intent
from tasks.document_indexing_outbox_task import enqueue_document_indexing_intent
from tasks.resource_cleanup_task import enqueue_resource_cleanup_intent
from tasks.segment_state_outbox_task import publish_segment_state_intents

logger = logging.getLogger(__name__)


class ProcessRulesDict(TypedDict):
    mode: ProcessRuleMode
    rules: dict[str, Any]


class AutoDisableLogsDict(TypedDict):
    document_ids: list[str]
    count: int


class ExternalKnowledgeTemplateLookup(Protocol):
    """Validate external templates within the dataset caller's transaction."""

    def require_template(self, tenant_id: str, template_id: str, *, session: Session) -> None: ...


class DatasetService:
    @staticmethod
    def _capture_indexing_representation(dataset: Dataset, *, session: Session) -> DatasetIndexingRepresentation:
        """Snapshot desired logical configuration without claiming it was published."""
        summary_setting = dataset.summary_index_setting
        return DatasetIndexingRepresentation(
            indexing_technique=str(dataset.indexing_technique),
            index_struct=str(dataset.index_struct) if dataset.index_struct is not None else None,
            collection_binding_id=(
                str(dataset.collection_binding_id) if dataset.collection_binding_id is not None else None
            ),
            embedding_model=str(dataset.embedding_model) if dataset.embedding_model is not None else None,
            embedding_model_provider=(
                str(dataset.embedding_model_provider) if dataset.embedding_model_provider is not None else None
            ),
            keyword_number=int(dataset.keyword_number) if dataset.keyword_number is not None else None,
            is_multimodal=bool(dataset.is_multimodal),
            doc_form=get_dataset_doc_form(dataset, session=session),
            summary_index_setting=(
                copy.deepcopy(dict(summary_setting)) if isinstance(summary_setting, Mapping) else None
            ),
        )

    @staticmethod
    def _capture_applied_indexing_representation(
        dataset: Dataset, *, session: Session
    ) -> DatasetIndexingRepresentation:
        """Read the applied receipt or establish a factory-resolved legacy baseline."""
        if dataset.indexing_applied_representation is not None:
            return DatasetIndexingRepresentation.model_validate_json(dataset.indexing_applied_representation)

        representation = DatasetService._capture_indexing_representation(dataset, session=session)
        if representation.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
            return representation
        doc_form = representation.doc_form or IndexStructureType.PARAGRAPH_INDEX
        try:
            vector = Vector(dataset, session=session)
        except ValueError as error:
            # A legacy row without index_struct and without a configured
            # default vector store cannot have a resolvable physical receipt.
            # Keep the baseline explicitly unknown instead of inventing one
            # from the desired logical columns.
            if str(error) != "Vector store must be specified.":
                raise
            return representation
        route = capture_vector_route(dataset, vector, doc_form=doc_form)
        return representation.model_copy(update={"vector_route": route, "cleanup_routes": (route,)})

    @staticmethod
    def _can_manage_all_datasets(tenant_id: str, account_id: str) -> bool:
        if not dify_config.RBAC_ENABLED:
            return False

        permissions = enterprise_rbac_service.RBACService.MyPermissions.get(tenant_id, account_id)
        workspace_permission_keys = getattr(getattr(permissions, "workspace", None), "permission_keys", []) or []
        return "dataset.create_and_management" in workspace_permission_keys

    @staticmethod
    def get_datasets(
        page: int,
        per_page: int,
        session: Session,
        tenant_id=None,
        user=None,
        search=None,
        tag_ids=None,
        include_all=False,
        accessible_dataset_ids: list[str] | None = None,
        include_own_datasets: bool = False,
        *,
        tags: TagTargetQuery,
    ):
        """Return visible datasets for a tenant using the injected read session."""
        query = select(Dataset).where(Dataset.tenant_id == tenant_id).order_by(Dataset.created_at.desc(), Dataset.id)

        if dify_config.RBAC_ENABLED and accessible_dataset_ids is not None:
            accessible_filter: ColumnElement[bool] = Dataset.id.in_(accessible_dataset_ids)
            if include_own_datasets and user:
                accessible_filter = sa.or_(Dataset.maintainer == user.id, accessible_filter)
            query = query.where(accessible_filter)

        if user:
            # get permitted dataset ids
            dataset_permission = session.scalars(
                select(DatasetPermission).where(
                    DatasetPermission.account_id == user.id, DatasetPermission.tenant_id == tenant_id
                )
            ).all()
            permitted_dataset_ids = {dp.dataset_id for dp in dataset_permission} if dataset_permission else None
            if not dify_config.RBAC_ENABLED and user.current_role == TenantAccountRole.DATASET_OPERATOR:
                # only show datasets that the user has permission to access
                # Check if permitted_dataset_ids is not empty to avoid WHERE false condition
                if permitted_dataset_ids and len(permitted_dataset_ids) > 0:
                    query = query.where(Dataset.id.in_(permitted_dataset_ids))
                else:
                    return [], 0
            else:
                if dify_config.RBAC_ENABLED:
                    can_manage_all_datasets = DatasetService._can_manage_all_datasets(str(tenant_id), str(user.id))
                    should_show_all_datasets = include_all and can_manage_all_datasets
                else:
                    should_show_all_datasets = user.current_role == TenantAccountRole.OWNER and include_all

                if not should_show_all_datasets:
                    if dify_config.RBAC_ENABLED:
                        # RBAC mode: show all datasets.  Permission control is enforced
                        # via permission_keys on each item and @rbac_permission_required decorators.
                        pass
                    else:
                        # Keep legacy visibility rules when RBAC is disabled.
                        if permitted_dataset_ids and len(permitted_dataset_ids) > 0:
                            query = query.where(
                                sa.or_(
                                    Dataset.permission == DatasetPermissionEnum.ALL_TEAM,
                                    sa.and_(
                                        Dataset.permission == DatasetPermissionEnum.ONLY_ME,
                                        Dataset.maintainer == user.id,
                                    ),
                                    sa.and_(
                                        Dataset.permission == DatasetPermissionEnum.PARTIAL_TEAM,
                                        Dataset.id.in_(permitted_dataset_ids),
                                    ),
                                )
                            )
                        else:
                            query = query.where(
                                sa.or_(
                                    Dataset.permission == DatasetPermissionEnum.ALL_TEAM,
                                    sa.and_(
                                        Dataset.permission == DatasetPermissionEnum.ONLY_ME,
                                        Dataset.maintainer == user.id,
                                    ),
                                )
                            )
        else:
            if dify_config.RBAC_ENABLED:
                # Without an account we cannot resolve RBAC resource visibility.
                query = query.where(sa.false())
            else:
                # if no user, only show datasets that are shared with all team members
                query = query.where(Dataset.permission == DatasetPermissionEnum.ALL_TEAM)

        if search:
            escaped_search = helper.escape_like_pattern(search)
            query = query.where(Dataset.name.ilike(f"%{escaped_search}%", escape="\\"))

        # Check if tag_ids is not empty to avoid WHERE false condition
        if tag_ids and len(tag_ids) > 0:
            if tenant_id is not None:
                target_ids = tags.find_target_ids(
                    tag_type="knowledge",
                    tenant_id=tenant_id,
                    tag_ids=tag_ids,
                    match_all=True,
                )
            else:
                target_ids = []
            if target_ids and len(target_ids) > 0:
                query = query.where(Dataset.id.in_(target_ids))
            else:
                return [], 0

        datasets = paginate_query(query, session=session, page=page, per_page=per_page, max_per_page=100)

        return datasets.items, datasets.total

    @staticmethod
    def get_process_rules(dataset_id, session: Session) -> ProcessRulesDict:
        # get the latest process rule
        dataset_process_rule = session.execute(
            select(DatasetProcessRule)
            .where(DatasetProcessRule.dataset_id == dataset_id)
            .order_by(DatasetProcessRule.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if dataset_process_rule:
            mode = dataset_process_rule.mode
            rules = dataset_process_rule.rules_dict or {}
        else:
            mode = ProcessRuleMode(DocumentService.DEFAULT_RULES["mode"])
            rules = dict(DocumentService.DEFAULT_RULES.get("rules") or {})
        return {"mode": mode, "rules": rules}

    @staticmethod
    def get_datasets_by_ids(
        ids: list[str] | None,
        tenant_id: str,
        user=None,
        accessible_dataset_ids: list[str] | None = None,
        include_own_datasets: bool = False,
        *,
        session: Session,
    ):
        # Check if ids is not empty to avoid WHERE false condition
        if not ids:
            return [], 0
        stmt = select(Dataset).where(Dataset.id.in_(ids), Dataset.tenant_id == tenant_id)

        if dify_config.RBAC_ENABLED and accessible_dataset_ids is not None:
            requested_dataset_ids = set(ids)
            accessible_dataset_ids = [
                dataset_id for dataset_id in accessible_dataset_ids if dataset_id in requested_dataset_ids
            ]
            accessible_filter: ColumnElement[bool] = Dataset.id.in_(accessible_dataset_ids)
            if include_own_datasets and user:
                accessible_filter = sa.or_(Dataset.maintainer == user.id, accessible_filter)
            stmt = stmt.where(accessible_filter)

        datasets = paginate_query(stmt, session=session, page=1, per_page=len(ids), max_per_page=len(ids))

        return datasets.items, datasets.total

    @staticmethod
    def create_empty_dataset(
        tenant_id: str,
        name: str,
        description: str | None,
        indexing_technique: str | None,
        account: Account,
        permission: str | None = None,
        provider: str = "vendor",
        external_knowledge_api_id: str | None = None,
        external_knowledge_id: str | None = None,
        embedding_model_provider: str | None = None,
        embedding_model_name: str | None = None,
        retrieval_model: RetrievalModel | None = None,
        summary_index_setting: dict[str, Any] | None = None,
        *,
        session: Session,
        external_templates: ExternalKnowledgeTemplateLookup,
    ):
        # check if dataset name already exists
        if session.scalar(select(Dataset).where(Dataset.name == name, Dataset.tenant_id == tenant_id).limit(1)):
            raise DatasetNameDuplicateError(f"Dataset with name {name} already exists.")
        embedding_model = None
        if indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            model_manager = ModelManager.for_tenant(tenant_id=tenant_id)
            if embedding_model_provider and embedding_model_name:
                # check if embedding model setting is valid
                DatasetService.check_embedding_model_setting(tenant_id, embedding_model_provider, embedding_model_name)
                embedding_model = model_manager.get_model_instance(
                    tenant_id=tenant_id,
                    provider=embedding_model_provider,
                    model_type=ModelType.TEXT_EMBEDDING,
                    model=embedding_model_name,
                )
            else:
                embedding_model = model_manager.get_default_model_instance(
                    tenant_id=tenant_id, model_type=ModelType.TEXT_EMBEDDING
                )
            if retrieval_model and retrieval_model.reranking_model:
                if (
                    retrieval_model.reranking_model.reranking_provider_name
                    and retrieval_model.reranking_model.reranking_model_name
                ):
                    # check if reranking model setting is valid
                    DatasetService.check_reranking_model_setting(
                        tenant_id,
                        retrieval_model.reranking_model.reranking_provider_name,
                        retrieval_model.reranking_model.reranking_model_name,
                    )
        dataset = Dataset(
            name=name,
            indexing_technique=IndexTechniqueType(indexing_technique) if indexing_technique else None,
        )
        # dataset = Dataset(name=name, provider=provider, config=config)
        dataset.description = description
        dataset.created_by = account.id
        dataset.maintainer = account.id
        dataset.updated_by = account.id
        dataset.tenant_id = tenant_id
        dataset.embedding_model_provider = embedding_model.provider if embedding_model else None
        dataset.embedding_model = embedding_model.model_name if embedding_model else None
        dataset.retrieval_model = retrieval_model.model_dump() if retrieval_model else None
        dataset.permission = DatasetPermissionEnum(permission) if permission else DatasetPermissionEnum.ONLY_ME
        dataset.provider = provider
        if summary_index_setting is not None:
            dataset.summary_index_setting = summary_index_setting
        session.add(dataset)
        session.flush()

        if provider == "external" and external_knowledge_api_id:
            external_templates.require_template(tenant_id, external_knowledge_api_id, session=session)
            if external_knowledge_id is None:
                raise ValueError("external_knowledge_id is required")
            external_knowledge_binding = ExternalKnowledgeBindings(
                tenant_id=tenant_id,
                dataset_id=dataset.id,
                external_knowledge_api_id=external_knowledge_api_id,
                external_knowledge_id=external_knowledge_id,
                created_by=account.id,
            )
            session.add(external_knowledge_binding)

        session.commit()
        enterprise_rbac_service.try_sync_creator_access_policy_member_bindings(
            tenant_id,
            account.id,
            rbac_contracts.RBACResourceType.DATASET,
            dataset.id,
        )
        return dataset

    @staticmethod
    def create_empty_rag_pipeline_dataset(
        tenant_id: str,
        rag_pipeline_dataset_create_entity: RagPipelineDatasetCreateEntity,
        session: Session,
    ):
        if rag_pipeline_dataset_create_entity.name:
            # check if dataset name already exists
            if session.scalar(
                select(Dataset)
                .where(Dataset.name == rag_pipeline_dataset_create_entity.name, Dataset.tenant_id == tenant_id)
                .limit(1)
            ):
                raise DatasetNameDuplicateError(
                    f"Dataset with name {rag_pipeline_dataset_create_entity.name} already exists."
                )
        else:
            # generate a random name as Untitled 1 2 3 ...
            datasets = session.scalars(select(Dataset).where(Dataset.tenant_id == tenant_id)).all()
            names = [dataset.name for dataset in datasets]
            rag_pipeline_dataset_create_entity.name = generate_incremental_name(
                names,
                "Untitled",
            )
        if not current_user or not current_user.id:
            raise ValueError("Current user or current user id not found")
        pipeline = Pipeline(
            tenant_id=tenant_id,
            name=rag_pipeline_dataset_create_entity.name,
            description=rag_pipeline_dataset_create_entity.description,
            created_by=current_user.id,
        )
        session.add(pipeline)
        session.flush()

        dataset = Dataset(
            tenant_id=tenant_id,
            name=rag_pipeline_dataset_create_entity.name,
            description=rag_pipeline_dataset_create_entity.description,
            permission=rag_pipeline_dataset_create_entity.permission,
            provider="vendor",
            runtime_mode=DatasetRuntimeMode.RAG_PIPELINE,
            icon_info=rag_pipeline_dataset_create_entity.icon_info.model_dump(),
            created_by=current_user.id,
            maintainer=current_user.id,
            pipeline_id=pipeline.id,
        )
        session.add(dataset)
        session.flush()
        dataset_id = dataset.id
        creator_id = current_user.id
        session.commit()
        enterprise_rbac_service.try_sync_creator_access_policy_member_bindings(
            tenant_id,
            creator_id,
            rbac_contracts.RBACResourceType.DATASET,
            dataset_id,
        )
        return dataset

    @staticmethod
    def get_dataset(dataset_id, session: Session) -> Dataset | None:
        dataset: Dataset | None = session.get(Dataset, dataset_id)
        return dataset

    @staticmethod
    def get_dataset_for_tenant(dataset_id: str, tenant_id: str, *, session: Session) -> Dataset | None:
        """Fetch a dataset only when it belongs to the provided tenant."""
        return session.scalar(select(Dataset).where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id).limit(1))

    @staticmethod
    def check_doc_form(dataset: Dataset, doc_form: str, *, session: Session):
        dataset_doc_form = dataset.chunk_structure
        if not dataset_doc_form:
            # A document creation request can wait behind the dataset's
            # creation lock after its REPEATABLE READ snapshot was opened.
            # Use a locking (current) read so the request validates against a
            # document committed by the preceding lock owner.
            dataset_doc_form = session.scalar(
                select(Document.doc_form)
                .where(Document.dataset_id == dataset.id, Document.tenant_id == dataset.tenant_id)
                .limit(1)
                .with_for_update(read=True)
            )
        if dataset_doc_form and doc_form != dataset_doc_form:
            raise ValueError("doc_form is different from the dataset doc_form.")

    @staticmethod
    def check_dataset_model_setting(dataset):
        if dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            try:
                model_manager = ModelManager.for_tenant(tenant_id=dataset.tenant_id)
                model_manager.get_model_instance(
                    tenant_id=dataset.tenant_id,
                    provider=dataset.embedding_model_provider,
                    model_type=ModelType.TEXT_EMBEDDING,
                    model=dataset.embedding_model,
                )
            except LLMBadRequestError as caught_error:
                raise ValueError(
                    "No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider."
                ) from caught_error
            except ProviderTokenNotInitError as ex:
                raise ValueError(f"The dataset is unavailable, due to: {ex.description}") from ex

    @staticmethod
    def check_embedding_model_setting(tenant_id: str, embedding_model_provider: str, embedding_model: str):
        try:
            model_manager = ModelManager.for_tenant(tenant_id=tenant_id)
            model_manager.get_model_instance(
                tenant_id=tenant_id,
                provider=embedding_model_provider,
                model_type=ModelType.TEXT_EMBEDDING,
                model=embedding_model,
            )
        except LLMBadRequestError as caught_error:
            raise ValueError(
                "No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider."
            ) from caught_error
        except ProviderTokenNotInitError as ex:
            raise ValueError(ex.description) from ex

    @staticmethod
    def check_is_multimodal_model(tenant_id: str, model_provider: str, model: str):
        try:
            model_manager = ModelManager.for_tenant(tenant_id=tenant_id)
            model_instance = model_manager.get_model_instance(
                tenant_id=tenant_id,
                provider=model_provider,
                model_type=ModelType.TEXT_EMBEDDING,
                model=model,
            )
            text_embedding_model = cast(TextEmbeddingModel, model_instance.model_type_instance)
            model_schema = text_embedding_model.get_model_schema(model_instance.model_name, model_instance.credentials)
            if not model_schema:
                raise ValueError("Model schema not found")
            return bool(model_schema.features and ModelFeature.VISION in model_schema.features)
        except LLMBadRequestError as caught_error:
            raise ValueError(
                "No Model available. Please configure a valid provider in the Settings -> Model Provider."
            ) from caught_error

    @staticmethod
    def check_reranking_model_setting(tenant_id: str, reranking_model_provider: str, reranking_model: str):
        try:
            model_manager = ModelManager.for_tenant(tenant_id=tenant_id)
            model_manager.get_model_instance(
                tenant_id=tenant_id,
                provider=reranking_model_provider,
                model_type=ModelType.RERANK,
                model=reranking_model,
            )
        except LLMBadRequestError as caught_error:
            raise ValueError(
                "No Rerank Model available. Please configure a valid provider in the Settings -> Model Provider."
            ) from caught_error
        except ProviderTokenNotInitError as ex:
            raise ValueError(ex.description) from ex

    @staticmethod
    def _lock_current_dataset_configuration(
        *,
        session: Session,
        tenant_id: str,
        dataset_id: str,
    ) -> Dataset:
        """Take the Dataset lock without flushing a stale identity-map snapshot."""
        with session.no_autoflush:
            dataset = session.scalar(
                select(Dataset)
                .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        if dataset is None:
            raise ValueError("Dataset not found")
        return dataset

    @staticmethod
    def _has_materialized_index_locked(dataset: Dataset, *, session: Session) -> bool:
        """Decide rebuild ownership from current state while Dataset is locked.

        ``has_published`` is only a caller hint: a concurrent configuration
        transaction may have installed a generation after that hint was
        computed.  Locking reads for legacy document/segment evidence keep the
        fallback current under MySQL REPEATABLE READ as well.
        """
        if dataset.indexing_applied_representation is not None or dataset.indexing_request_id is not None:
            return True
        with session.no_autoflush:
            completed_document_id = session.scalar(
                select(Document.id)
                .where(
                    Document.tenant_id == dataset.tenant_id,
                    Document.dataset_id == dataset.id,
                    Document.indexing_status == IndexingStatus.COMPLETED,
                )
                .limit(1)
                .with_for_update()
            )
            if completed_document_id is not None:
                return True
            return (
                session.scalar(
                    select(DocumentSegment.id)
                    .where(
                        DocumentSegment.tenant_id == dataset.tenant_id,
                        DocumentSegment.dataset_id == dataset.id,
                    )
                    .limit(1)
                    .with_for_update()
                )
                is not None
            )

    @staticmethod
    @contextmanager
    def update_dataset(
        dataset_id, data, user, *, session: Session, external_templates: ExternalKnowledgeTemplateLookup
    ) -> Generator[Dataset]:
        """Commit settings and caller follow-ups together before dispatching index work.

        The caller must finish ACL updates and materialize its response inside
        the context. Any failure rolls back the complete configuration update.
        """
        try:
            tenant_id = user.current_tenant_id
            if not tenant_id:
                raise ValueError("Current user or current tenant not found")
            try:
                dataset = DatasetService._lock_current_dataset_configuration(
                    session=session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                )
            except ValueError:
                # Preserve the public missing-vs-forbidden contract without
                # ever locking or refreshing another tenant's configuration.
                with session.no_autoflush:
                    foreign_dataset_exists = session.scalar(select(Dataset.id).where(Dataset.id == dataset_id))
                if foreign_dataset_exists is not None:
                    raise NoPermissionError("You do not have permission to access this dataset.") from None
                raise
            if data.get("name") and data.get("name") != dataset.name:
                if DatasetService._has_dataset_same_name(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset_id,
                    name=data.get("name", dataset.name),
                    session=session,
                ):
                    raise ValueError("Dataset name already exists")

            DatasetService.check_dataset_permission(dataset, user, session)
            if dataset.provider == "external":
                dataset = DatasetService._update_external_dataset(
                    dataset, data, user, session, external_templates=external_templates
                )
                indexing_job = None
            else:
                dataset, indexing_job = DatasetService._update_internal_dataset(dataset, data, user, session)
            yield dataset
            indexing_intent_ids = create_dataset_indexing_intents(session, indexing_job)
            session.commit()
        except BaseException:
            session.rollback()
            raise
        DatasetService.dispatch_dataset_indexing(indexing_intent_ids)

    @staticmethod
    def _has_dataset_same_name(tenant_id: str, dataset_id: str, name: str, session: Session):
        dataset = session.scalar(
            select(Dataset)
            .where(
                Dataset.id != dataset_id,
                Dataset.name == name,
                Dataset.tenant_id == tenant_id,
            )
            .limit(1)
        )
        return dataset is not None

    @staticmethod
    def _update_external_dataset(
        dataset, data, user, session: Session, *, external_templates: ExternalKnowledgeTemplateLookup
    ):
        """
        Update external dataset configuration.

        Args:
            dataset: The dataset object to update
            data: Update data dictionary
            user: User performing the update

        Returns:
            Dataset: Updated dataset object
        """
        # Update retrieval model if provided
        external_retrieval_model = data.get("external_retrieval_model", None)
        if external_retrieval_model:
            dataset.retrieval_model = external_retrieval_model

        # Update summary index setting if provided
        summary_index_setting = data.get("summary_index_setting", None)
        if summary_index_setting is not None:
            dataset.summary_index_setting = summary_index_setting

        # Update basic dataset properties
        dataset.name = data.get("name", dataset.name)
        dataset.description = data.get("description", dataset.description)

        # Update permission if provided
        permission = data.get("permission")
        if permission:
            dataset.permission = permission

        # Update metadata fields
        dataset.updated_by = user.id if user else None
        dataset.updated_at = naive_utc_now()
        session.add(dataset)

        if "external_knowledge_id" in data or "external_knowledge_api_id" in data:
            DatasetService._patch_external_knowledge_binding(
                dataset,
                data,
                session=session,
                external_templates=external_templates,
            )

        # Flush changes to the database without closing the caller-managed
        # transaction. This helper receives a session opened by the caller
        # (`with Session(...) as session`); calling commit() here closed that
        # context manager early and raised
        # sqlalchemy.exc.InvalidRequestError: Can't operate on closed transaction
        # (#39191).
        session.flush()

        return dataset

    @staticmethod
    def _patch_external_knowledge_binding(
        dataset: Dataset,
        data: Mapping[str, Any],
        *,
        session: Session,
        external_templates: ExternalKnowledgeTemplateLookup,
    ) -> None:
        """Apply provided binding fields against the tenant-owned current binding."""
        external_knowledge_binding = session.scalar(
            select(ExternalKnowledgeBindings)
            .where(
                ExternalKnowledgeBindings.dataset_id == dataset.id,
                ExternalKnowledgeBindings.tenant_id == dataset.tenant_id,
            )
            .limit(1)
        )

        if not external_knowledge_binding:
            raise ValueError("External knowledge binding not found.")

        external_knowledge_id = data.get("external_knowledge_id", external_knowledge_binding.external_knowledge_id)
        external_knowledge_api_id = data.get(
            "external_knowledge_api_id", external_knowledge_binding.external_knowledge_api_id
        )
        if not external_knowledge_id:
            raise ValueError("External knowledge id is required.")
        if not external_knowledge_api_id:
            raise ValueError("External knowledge api id is required.")
        if (
            external_knowledge_binding.external_knowledge_id == external_knowledge_id
            and external_knowledge_binding.external_knowledge_api_id == external_knowledge_api_id
        ):
            return

        external_templates.require_template(dataset.tenant_id, external_knowledge_api_id, session=session)
        external_knowledge_binding.external_knowledge_id = external_knowledge_id
        external_knowledge_binding.external_knowledge_api_id = external_knowledge_api_id
        session.add(external_knowledge_binding)

    @staticmethod
    def _update_internal_dataset(dataset, data, user, session: Session) -> tuple[Dataset, DatasetIndexingJob | None]:
        """
        Update internal dataset configuration.

        Args:
            dataset: The dataset object to update
            data: Update data dictionary
            user: User performing the update

        Returns:
            The updated dataset and index work to dispatch after the caller commits.
        """
        # Remove external-specific fields from update data
        data.pop("partial_member_list", None)
        data.pop("external_knowledge_api_id", None)
        data.pop("external_knowledge_id", None)
        data.pop("external_retrieval_model", None)

        # Filter out None values except for description field
        filtered_data = {k: v for k, v in data.items() if v is not None or k == "description"}

        # Handle indexing technique changes and embedding model updates
        action = DatasetService._handle_indexing_technique_change(dataset, data, filtered_data, session)
        source_representation = (
            DatasetService._capture_applied_indexing_representation(dataset, session=session) if action else None
        )

        # Add metadata fields
        filtered_data["updated_by"] = user.id
        filtered_data["updated_at"] = naive_utc_now()
        # update Retrieval model
        if data.get("retrieval_model"):
            filtered_data["retrieval_model"] = data["retrieval_model"]
        # update summary index setting
        if data.get("summary_index_setting"):
            filtered_data["summary_index_setting"] = data.get("summary_index_setting")
        # update icon info
        if data.get("icon_info"):
            filtered_data["icon_info"] = data.get("icon_info")

        # Update dataset in database. Use flush() rather than commit() so the
        # caller-managed transaction (opened with `with Session(...) as session`)
        # stays open for subsequent operations — _update_pipeline_knowledge_base
        # node data and any caller follow-ups run on the same session. Calling
        # commit() here closed the context manager early and raised
        # sqlalchemy.exc.InvalidRequestError: Can't operate on closed transaction
        # (#39191).
        session.execute(update(Dataset).where(Dataset.id == dataset.id).values(**filtered_data))
        session.flush()

        # Reload dataset to get updated values
        session.refresh(dataset)

        # update pipeline knowledge base node data
        DatasetService._update_pipeline_knowledge_base_node_data(dataset, user.id, session)

        # Note: summary_index_setting changes do not trigger automatic regeneration of existing summaries.
        # The new setting will only apply to:
        # 1. New documents added after the setting change
        # 2. Manual summary generation requests

        job = None
        if action:
            assert source_representation is not None
            job = DatasetIndexingJob(
                DatasetRef(dataset.tenant_id, dataset.id),
                source_representation,
                DatasetService._capture_indexing_representation(dataset, session=session),
            )
        return dataset, job

    @staticmethod
    def dispatch_dataset_indexing(intent_ids: Sequence[str]) -> None:
        """Publish committed rebuild intents; the periodic drain recovers broker failures."""
        for intent_id in intent_ids:
            enqueue_dataset_indexing_intent(intent_id)

    @staticmethod
    def _update_pipeline_knowledge_base_node_data(dataset: Dataset, updata_user_id: str, session: Session):
        """
        Update pipeline knowledge base node data.
        """
        if dataset.runtime_mode != DatasetRuntimeMode.RAG_PIPELINE:
            return

        pipeline = session.get(Pipeline, dataset.pipeline_id)
        if not pipeline:
            return

        try:
            rag_pipeline_service = RagPipelineService(
                session, session_maker=sessionmaker(bind=session.get_bind(), expire_on_commit=False)
            )
            published_workflow = rag_pipeline_service.get_published_workflow(pipeline)
            draft_workflow = rag_pipeline_service.get_draft_workflow(pipeline)

            # update knowledge nodes
            def update_knowledge_nodes(workflow_graph: str) -> str:
                """Update knowledge-index nodes in workflow graph."""
                data: dict[str, Any] = json.loads(workflow_graph)

                nodes = data.get("nodes", [])
                updated = False

                for node in nodes:
                    if node.get("data", {}).get("type") == "knowledge-index":
                        try:
                            knowledge_index_node_data = node.get("data", {})
                            knowledge_index_node_data["embedding_model"] = dataset.embedding_model
                            knowledge_index_node_data["embedding_model_provider"] = dataset.embedding_model_provider
                            knowledge_index_node_data["retrieval_model"] = dataset.retrieval_model
                            knowledge_index_node_data["chunk_structure"] = dataset.chunk_structure
                            knowledge_index_node_data["indexing_technique"] = dataset.indexing_technique
                            knowledge_index_node_data["keyword_number"] = dataset.keyword_number
                            knowledge_index_node_data["summary_index_setting"] = dataset.summary_index_setting
                            node["data"] = knowledge_index_node_data
                            updated = True
                        except Exception:
                            logging.exception("Failed to update knowledge node")
                            continue

                if updated:
                    data["nodes"] = nodes
                    return json.dumps(data)
                return workflow_graph

            # Update published workflow
            if published_workflow:
                updated_graph = update_knowledge_nodes(published_workflow.graph)
                if updated_graph != published_workflow.graph:
                    # Create new workflow version
                    workflow = Workflow.new(
                        tenant_id=pipeline.tenant_id,
                        app_id=pipeline.id,
                        type=published_workflow.type,
                        version=str(datetime.datetime.now(datetime.UTC).replace(tzinfo=None)),
                        graph=updated_graph,
                        features=published_workflow.features,
                        created_by=updata_user_id,
                        environment_variables=published_workflow.environment_variables,
                        conversation_variables=published_workflow.conversation_variables,
                        rag_pipeline_variables=published_workflow.rag_pipeline_variables,
                        marked_name="",
                        marked_comment="",
                    )
                    session.add(workflow)

            # Update draft workflow
            if draft_workflow:
                updated_graph = update_knowledge_nodes(draft_workflow.graph)
                if updated_graph != draft_workflow.graph:
                    draft_workflow.graph = updated_graph
                    session.add(draft_workflow)

            # Dataset settings, permissions, and workflow versions share the
            # caller's transaction; only that owner may commit and dispatch.
            session.flush()

        except Exception:
            logging.exception("Failed to update pipeline knowledge base node data")
            raise

    @staticmethod
    def _handle_indexing_technique_change(dataset, data, filtered_data, session: Session):
        """
        Handle changes in indexing technique and configure embedding models accordingly.

        Args:
            dataset: Current dataset object
            data: Update data dictionary
            filtered_data: Filtered update data
            session: SQLAlchemy session used for embedding collection binding lookups

        Returns:
            str: Action to perform ('add', 'remove', 'update', or None)
        """
        if "indexing_technique" not in data:
            return None
        if dataset.indexing_technique != data["indexing_technique"]:
            if data["indexing_technique"] == IndexTechniqueType.ECONOMY:
                # Remove embedding model configuration for economy mode
                filtered_data["embedding_model"] = None
                filtered_data["embedding_model_provider"] = None
                filtered_data["collection_binding_id"] = None
                return "remove"
            elif data["indexing_technique"] == IndexTechniqueType.HIGH_QUALITY:
                # Configure embedding model for high quality mode
                DatasetService._configure_embedding_model_for_high_quality(data, filtered_data, session)
                return "add"
        else:
            # Handle embedding model updates when indexing technique remains the same
            return DatasetService._handle_embedding_model_update_when_technique_unchanged(
                dataset, data, filtered_data, session
            )
        return None

    @staticmethod
    def _configure_embedding_model_for_high_quality(data, filtered_data, session: Session):
        """
        Configure embedding model settings for high quality indexing.

        Args:
            data: Update data dictionary
            filtered_data: Filtered update data to modify
            session: SQLAlchemy session used for embedding collection binding lookups
        """
        # assert isinstance(current_user, Account) and current_user.current_tenant_id is not None
        try:
            model_manager = ModelManager.for_tenant(tenant_id=current_user.current_tenant_id)
            assert isinstance(current_user, Account)
            assert current_user.current_tenant_id is not None
            embedding_model = model_manager.get_model_instance(
                tenant_id=current_user.current_tenant_id,
                provider=data["embedding_model_provider"],
                model_type=ModelType.TEXT_EMBEDDING,
                model=data["embedding_model"],
            )
            embedding_model_name = embedding_model.model_name
            filtered_data["embedding_model"] = embedding_model_name
            filtered_data["embedding_model_provider"] = embedding_model.provider
            dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                embedding_model.provider,
                embedding_model_name,
                session,
            )
            filtered_data["collection_binding_id"] = dataset_collection_binding.id
        except LLMBadRequestError as caught_error:
            raise ValueError(
                "No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider."
            ) from caught_error
        except ProviderTokenNotInitError as ex:
            raise ValueError(ex.description) from ex

    @staticmethod
    def _handle_embedding_model_update_when_technique_unchanged(dataset, data, filtered_data, session: Session):
        """
        Handle embedding model updates when indexing technique remains the same.

        Args:
            dataset: Current dataset object
            data: Update data dictionary
            filtered_data: Filtered update data to modify
            session: SQLAlchemy session used for embedding collection binding lookups

        Returns:
            str: Action to perform ('update' or None)
        """
        # Skip embedding model checks if not provided in the update request
        if (
            "embedding_model_provider" not in data
            or "embedding_model" not in data
            or not data.get("embedding_model_provider")
            or not data.get("embedding_model")
        ):
            DatasetService._preserve_existing_embedding_settings(dataset, filtered_data)
            return None
        else:
            return DatasetService._update_embedding_model_settings(dataset, data, filtered_data, session)

    @staticmethod
    def _preserve_existing_embedding_settings(dataset, filtered_data):
        """
        Preserve existing embedding model settings when not provided in update.

        Args:
            dataset: Current dataset object
            filtered_data: Filtered update data to modify
        """
        # If the dataset already has embedding model settings, use those
        if dataset.embedding_model_provider and dataset.embedding_model:
            filtered_data["embedding_model_provider"] = dataset.embedding_model_provider
            filtered_data["embedding_model"] = dataset.embedding_model
            # If collection_binding_id exists, keep it too
            if dataset.collection_binding_id:
                filtered_data["collection_binding_id"] = dataset.collection_binding_id
        # Otherwise, don't try to update embedding model settings at all
        # Remove these fields from filtered_data if they exist but are None/empty
        if "embedding_model_provider" in filtered_data and not filtered_data["embedding_model_provider"]:
            del filtered_data["embedding_model_provider"]
        if "embedding_model" in filtered_data and not filtered_data["embedding_model"]:
            del filtered_data["embedding_model"]

    @staticmethod
    def _update_embedding_model_settings(dataset, data, filtered_data, session: Session):
        """
        Update embedding model settings with new values.

        Args:
            dataset: Current dataset object
            data: Update data dictionary
            filtered_data: Filtered update data to modify
            session: SQLAlchemy session used for embedding collection binding lookups

        Returns:
            str: Action to perform ('update' or None)
        """
        try:
            # Compare current and new model provider settings
            current_provider_str = (
                str(ModelProviderID(dataset.embedding_model_provider)) if dataset.embedding_model_provider else None
            )
            new_provider_str = (
                str(ModelProviderID(data["embedding_model_provider"])) if data["embedding_model_provider"] else None
            )

            # Only update if values are different
            if current_provider_str != new_provider_str or data["embedding_model"] != dataset.embedding_model:
                DatasetService._apply_new_embedding_settings(dataset, data, filtered_data, session)
                return "update"
        except LLMBadRequestError as caught_error:
            raise ValueError(
                "No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider."
            ) from caught_error
        except ProviderTokenNotInitError as ex:
            raise ValueError(ex.description) from ex
        return None

    @staticmethod
    def _apply_new_embedding_settings(dataset, data, filtered_data, session: Session):
        """
        Apply new embedding model settings to the dataset.

        Args:
            dataset: Current dataset object
            data: Update data dictionary
            filtered_data: Filtered update data to modify
            session: SQLAlchemy session used for embedding collection binding lookups
        """
        # assert isinstance(current_user, Account) and current_user.current_tenant_id is not None

        model_manager = ModelManager.for_tenant(tenant_id=current_user.current_tenant_id)
        try:
            assert isinstance(current_user, Account)
            assert current_user.current_tenant_id is not None
            embedding_model = model_manager.get_model_instance(
                tenant_id=current_user.current_tenant_id,
                provider=data["embedding_model_provider"],
                model_type=ModelType.TEXT_EMBEDDING,
                model=data["embedding_model"],
            )
        except ProviderTokenNotInitError:
            # If we can't get the embedding model, preserve existing settings
            logger.warning(
                "Failed to initialize embedding model %s/%s, preserving existing settings",
                data["embedding_model_provider"],
                data["embedding_model"],
            )
            if dataset.embedding_model_provider and dataset.embedding_model:
                filtered_data["embedding_model_provider"] = dataset.embedding_model_provider
                filtered_data["embedding_model"] = dataset.embedding_model
                if dataset.collection_binding_id:
                    filtered_data["collection_binding_id"] = dataset.collection_binding_id
            # Skip the rest of the embedding model update
            return

        # Apply new embedding model settings
        embedding_model_name = embedding_model.model_name
        filtered_data["embedding_model"] = embedding_model_name
        filtered_data["embedding_model_provider"] = embedding_model.provider
        dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
            embedding_model.provider,
            embedding_model_name,
            session,
        )
        filtered_data["collection_binding_id"] = dataset_collection_binding.id

    @staticmethod
    def _check_summary_index_setting_model_changed(dataset: Dataset, data: dict[str, Any]) -> bool:
        """
        Check if summary_index_setting model (model_name or model_provider_name) has changed.

        Args:
            dataset: Current dataset object
            data: Update data dictionary

        Returns:
            bool: True if summary model changed, False otherwise
        """
        # Check if summary_index_setting is being updated
        if "summary_index_setting" not in data or data.get("summary_index_setting") is None:
            return False

        new_summary_setting = data.get("summary_index_setting")
        old_summary_setting = dataset.summary_index_setting

        # If new setting is disabled, no need to regenerate
        if not new_summary_setting or not new_summary_setting.get("enable"):
            return False

        # If old setting doesn't exist, no need to regenerate (no existing summaries to regenerate)
        # Note: This task only regenerates existing summaries, not generates new ones
        if not old_summary_setting:
            return False

        # Compare model_name and model_provider_name
        old_model_name = old_summary_setting.get("model_name")
        old_model_provider = old_summary_setting.get("model_provider_name")
        new_model_name = new_summary_setting.get("model_name")
        new_model_provider = new_summary_setting.get("model_provider_name")

        # Check if model changed
        if old_model_name != new_model_name or old_model_provider != new_model_provider:
            logger.info(
                "Summary index setting model changed for dataset %s: old=%s/%s, new=%s/%s",
                dataset.id,
                old_model_provider,
                old_model_name,
                new_model_provider,
                new_model_name,
            )
            return True

        return False

    @staticmethod
    def update_rag_pipeline_dataset_settings(
        dataset: Dataset,
        knowledge_configuration: KnowledgeConfiguration,
        has_published: bool = False,
        *,
        session: Session,
        account: Account | None = None,
    ) -> tuple[str, ...]:
        actor = account or current_user
        if not actor or not actor.current_tenant_id:
            raise ValueError("Current user or current tenant not found")
        # RAG publish and both DSL import paths share this configuration owner.
        # Lock and refresh before source capture or any model/binding comparison;
        # otherwise a stale identity-map A could overwrite a committed B.
        dataset = DatasetService._lock_current_dataset_configuration(
            session=session,
            tenant_id=actor.current_tenant_id,
            dataset_id=dataset.id,
        )
        has_materialized_index = has_published or DatasetService._has_materialized_index_locked(
            dataset,
            session=session,
        )
        source_representation = (
            DatasetService._capture_applied_indexing_representation(dataset, session=session)
            if has_materialized_index
            else DatasetService._capture_indexing_representation(dataset, session=session)
        )
        if not has_materialized_index:
            dataset.chunk_structure = knowledge_configuration.chunk_structure
            dataset.indexing_technique = IndexTechniqueType(knowledge_configuration.indexing_technique)
            if knowledge_configuration.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                model_manager = ModelManager.for_tenant(tenant_id=actor.current_tenant_id)
                embedding_model = model_manager.get_model_instance(
                    tenant_id=actor.current_tenant_id,
                    provider=knowledge_configuration.embedding_model_provider or "",
                    model_type=ModelType.TEXT_EMBEDDING,
                    model=knowledge_configuration.embedding_model or "",
                )
                is_multimodal = DatasetService.check_is_multimodal_model(
                    actor.current_tenant_id,
                    knowledge_configuration.embedding_model_provider,
                    knowledge_configuration.embedding_model,
                )
                dataset.is_multimodal = is_multimodal
                embedding_model_name = embedding_model.model_name
                dataset.embedding_model = embedding_model_name
                dataset.embedding_model_provider = embedding_model.provider
                dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                    embedding_model.provider,
                    embedding_model_name,
                    session,
                )
                dataset.collection_binding_id = dataset_collection_binding.id
            elif knowledge_configuration.indexing_technique == IndexTechniqueType.ECONOMY:
                dataset.keyword_number = knowledge_configuration.keyword_number
            else:
                raise ValueError("Invalid index method")
            dataset.retrieval_model = knowledge_configuration.retrieval_model.model_dump()
            # Update summary_index_setting if provided
            if knowledge_configuration.summary_index_setting is not None:
                dataset.summary_index_setting = knowledge_configuration.summary_index_setting
            session.add(dataset)
            return ()
        else:
            if dataset.chunk_structure and dataset.chunk_structure != knowledge_configuration.chunk_structure:
                raise ValueError("Chunk structure is not allowed to be updated.")
            action = None
            if dataset.indexing_technique != knowledge_configuration.indexing_technique:
                # if update indexing_technique
                if knowledge_configuration.indexing_technique == IndexTechniqueType.ECONOMY:
                    raise ValueError("Knowledge base indexing technique is not allowed to be updated to economy.")
                elif knowledge_configuration.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                    action = "add"
                    # get embedding model setting
                    try:
                        model_manager = ModelManager.for_tenant(tenant_id=actor.current_tenant_id)
                        embedding_model = model_manager.get_model_instance(
                            tenant_id=actor.current_tenant_id,
                            provider=knowledge_configuration.embedding_model_provider,
                            model_type=ModelType.TEXT_EMBEDDING,
                            model=knowledge_configuration.embedding_model,
                        )
                        embedding_model_name = embedding_model.model_name
                        dataset.embedding_model = embedding_model_name
                        dataset.embedding_model_provider = embedding_model.provider
                        dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                            embedding_model.provider,
                            embedding_model_name,
                            session,
                        )
                        is_multimodal = DatasetService.check_is_multimodal_model(
                            actor.current_tenant_id,
                            knowledge_configuration.embedding_model_provider,
                            knowledge_configuration.embedding_model,
                        )
                        dataset.is_multimodal = is_multimodal
                        dataset.collection_binding_id = dataset_collection_binding.id
                        dataset.indexing_technique = IndexTechniqueType(knowledge_configuration.indexing_technique)
                    except LLMBadRequestError as caught_error:
                        raise ValueError(
                            "No Embedding Model available. Please configure a valid provider "
                            "in the Settings -> Model Provider."
                        ) from caught_error
                    except ProviderTokenNotInitError as ex:
                        raise ValueError(ex.description) from ex
            else:
                # add default plugin id to both setting sets, to make sure the plugin model provider is consistent
                # Skip embedding model checks if not provided in the update request
                if dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                    skip_embedding_update = False
                    try:
                        # Handle existing model provider
                        plugin_model_provider = dataset.embedding_model_provider
                        plugin_model_provider_str = None
                        if plugin_model_provider:
                            plugin_model_provider_str = str(ModelProviderID(plugin_model_provider))

                        # Handle new model provider from request
                        new_plugin_model_provider = knowledge_configuration.embedding_model_provider
                        new_plugin_model_provider_str = None
                        if new_plugin_model_provider:
                            new_plugin_model_provider_str = str(ModelProviderID(new_plugin_model_provider))

                        # Only update embedding model if both values are provided and different from current
                        if (
                            plugin_model_provider_str != new_plugin_model_provider_str
                            or knowledge_configuration.embedding_model != dataset.embedding_model
                        ):
                            action = "update"
                            model_manager = ModelManager.for_tenant(tenant_id=actor.current_tenant_id)
                            embedding_model = None
                            try:
                                embedding_model = model_manager.get_model_instance(
                                    tenant_id=actor.current_tenant_id,
                                    provider=knowledge_configuration.embedding_model_provider,
                                    model_type=ModelType.TEXT_EMBEDDING,
                                    model=knowledge_configuration.embedding_model,
                                )
                            except ProviderTokenNotInitError:
                                # If we can't get the embedding model, skip updating it
                                # and keep the existing settings if available
                                # Skip the rest of the embedding model update
                                skip_embedding_update = True
                            if not skip_embedding_update:
                                if embedding_model:
                                    embedding_model_name = embedding_model.model_name
                                    dataset.embedding_model = embedding_model_name
                                    dataset.embedding_model_provider = embedding_model.provider
                                    dataset_collection_binding = (
                                        DatasetCollectionBindingService.get_dataset_collection_binding(
                                            embedding_model.provider,
                                            embedding_model_name,
                                            session,
                                        )
                                    )
                                    dataset.collection_binding_id = dataset_collection_binding.id
                                    is_multimodal = DatasetService.check_is_multimodal_model(
                                        actor.current_tenant_id,
                                        knowledge_configuration.embedding_model_provider,
                                        knowledge_configuration.embedding_model,
                                    )
                                    dataset.is_multimodal = is_multimodal
                    except LLMBadRequestError as caught_error:
                        raise ValueError(
                            "No Embedding Model available. Please configure a valid provider "
                            "in the Settings -> Model Provider."
                        ) from caught_error
                    except ProviderTokenNotInitError as ex:
                        raise ValueError(ex.description) from ex
                elif dataset.indexing_technique == IndexTechniqueType.ECONOMY:
                    if dataset.keyword_number != knowledge_configuration.keyword_number:
                        dataset.keyword_number = knowledge_configuration.keyword_number
                        action = "update"
            dataset.retrieval_model = knowledge_configuration.retrieval_model.model_dump()
            # Update summary_index_setting if provided
            if knowledge_configuration.summary_index_setting is not None:
                dataset.summary_index_setting = knowledge_configuration.summary_index_setting
            session.add(dataset)
            session.flush()
            indexing_job = (
                DatasetIndexingJob(
                    DatasetRef(dataset.tenant_id, dataset.id),
                    source_representation,
                    DatasetService._capture_indexing_representation(dataset, session=session),
                )
                if action
                else None
            )
            return create_dataset_indexing_intents(session, indexing_job)

    @staticmethod
    def delete_dataset(dataset_id, user, session: Session):
        dataset = DatasetService.get_dataset(dataset_id, session)

        if dataset is None:
            return False

        DatasetService.check_dataset_permission(dataset, user, session)
        tenant_id = dataset.tenant_id

        deletion = delete_dataset_for_cleanup(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            resource_relations=build_resource_access_token_cleanup_service(session=session),
        )
        if deletion is None:
            return False
        session.commit()
        for key in deletion.deleted_api_keys:
            ApiTokenCache.delete(key.token, key.type)
        for cleanup_intent_id in deletion.cleanup_intent_ids:
            enqueue_resource_cleanup_intent(cleanup_intent_id)
        return True

    @staticmethod
    def dataset_use_check(dataset_ref: DatasetRef, session: Session) -> bool:
        stmt = select(exists().where(AppDatasetJoin.dataset_id == dataset_ref.dataset_id))
        return session.execute(stmt).scalar_one()

    @staticmethod
    def check_dataset_permission(dataset, user, session: Session):
        """Validate dataset access for a user, using the injected session for partial-member lookups."""
        if dataset.tenant_id != user.current_tenant_id:
            logger.debug("User %s does not have permission to access dataset %s", user.id, dataset.id)
            raise NoPermissionError("You do not have permission to access this dataset.")
        if dify_config.RBAC_ENABLED:
            return
        has_permission = False
        if (
            user.current_role != TenantAccountRole.OWNER
            and dataset.permission == DatasetPermissionEnum.PARTIAL_TEAM
            and dataset.maintainer != user.id
        ):
            has_permission = (
                session.scalar(
                    select(DatasetPermission.id)
                    .where(
                        DatasetPermission.dataset_id == dataset.id,
                        DatasetPermission.account_id == user.id,
                        DatasetPermission.tenant_id == dataset.tenant_id,
                        DatasetPermission.has_permission.is_(True),
                    )
                    .limit(1)
                )
                is not None
            )
        snapshot = DatasetAccessSnapshot(
            dataset=DatasetAccessRecord(
                id=dataset.id,
                workspace_id=dataset.tenant_id,
                maintainer_id=dataset.maintainer,
                permission=dataset.permission,
            ),
            actor_has_partial_access=has_permission,
        )
        if not can_access_dataset(snapshot, actor_id=user.id, workspace_role=user.current_role):
            logger.debug("User %s does not have permission to access dataset %s", user.id, dataset.id)
            raise NoPermissionError("You do not have permission to access this dataset.")

    @staticmethod
    def check_dataset_operator_permission(
        user: Account | None = None, dataset: Dataset | None = None, *, session: Session
    ):
        if not dataset:
            raise ValueError("Dataset not found")

        if not user:
            raise ValueError("User not found")

        if user.current_role != TenantAccountRole.OWNER:
            if dataset.permission == DatasetPermissionEnum.ONLY_ME:
                if dataset.maintainer != user.id:
                    raise NoPermissionError("You do not have permission to access this dataset.")

            elif dataset.permission == DatasetPermissionEnum.PARTIAL_TEAM:
                user_permission = session.scalar(
                    select(DatasetPermission.id)
                    .where(
                        DatasetPermission.dataset_id == dataset.id,
                        DatasetPermission.account_id == user.id,
                        DatasetPermission.tenant_id == dataset.tenant_id,
                    )
                    .limit(1)
                )
                if user_permission is None:
                    raise NoPermissionError("You do not have permission to access this dataset.")

    @staticmethod
    def get_dataset_queries(dataset_id: str, page: int, per_page: int, session: Session):
        stmt = select(DatasetQuery).filter_by(dataset_id=dataset_id).order_by(DatasetQuery.created_at.desc())

        dataset_queries = paginate_query(stmt, page=page, per_page=per_page, max_per_page=100, session=session)

        return dataset_queries.items, dataset_queries.total

    @staticmethod
    def get_related_apps(dataset_id: str, session: Session):
        return session.scalars(
            select(AppDatasetJoin)
            .where(AppDatasetJoin.dataset_id == dataset_id)
            .order_by(AppDatasetJoin.created_at.desc())
        ).all()

    @staticmethod
    def update_dataset_api_status(dataset: Dataset, status: bool, actor: Account, session: Session):
        if not actor.id:
            raise ValueError("Current user or current user id not found")
        dataset.enable_api = status
        dataset.updated_by = actor.id
        dataset.updated_at = naive_utc_now()
        session.flush()

    @staticmethod
    def get_dataset_auto_disable_logs(dataset_ref: DatasetRef, session: Session) -> AutoDisableLogsDict:
        if dify_config.DEPLOYMENT_EDITION != DeploymentEdition.CLOUD:
            return {"document_ids": [], "count": 0}

        features = FeatureService.get_features(dataset_ref.tenant_id, exclude_vector_space=True)
        if features.billing.subscription.plan == CloudPlan.SANDBOX:
            return {
                "document_ids": [],
                "count": 0,
            }
        # get recent 30 days auto disable logs
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        dataset_auto_disable_logs = session.scalars(
            select(DatasetAutoDisableLog).where(
                DatasetAutoDisableLog.tenant_id == dataset_ref.tenant_id,
                DatasetAutoDisableLog.dataset_id == dataset_ref.dataset_id,
                DatasetAutoDisableLog.created_at >= start_date,
            )
        ).all()
        if dataset_auto_disable_logs:
            return {
                "document_ids": [log.document_id for log in dataset_auto_disable_logs],
                "count": len(dataset_auto_disable_logs),
            }
        return {
            "document_ids": [],
            "count": 0,
        }


class DocumentService:
    DEFAULT_RULES: dict[str, Any] = {
        "mode": "custom",
        "rules": {
            "pre_processing_rules": [
                {"id": "remove_extra_spaces", "enabled": True},
                {"id": "remove_urls_emails", "enabled": False},
            ],
            "segmentation": {"delimiter": "\n", "max_tokens": 1024, "chunk_overlap": 50},
        },
        "limits": {
            "indexing_max_segmentation_tokens_length": dify_config.INDEXING_MAX_SEGMENTATION_TOKENS_LENGTH,
        },
    }

    DISPLAY_STATUS_ALIASES: dict[str, str] = {
        "active": "available",
        "enabled": "available",
    }

    _INDEXING_STATUSES: tuple[IndexingStatus, ...] = (
        IndexingStatus.PARSING,
        IndexingStatus.CLEANING,
        IndexingStatus.SPLITTING,
        IndexingStatus.INDEXING,
    )

    DISPLAY_STATUS_FILTERS: dict[str, tuple[Any, ...]] = {
        "queuing": (Document.indexing_status == IndexingStatus.WAITING,),
        "indexing": (
            Document.indexing_status.in_(_INDEXING_STATUSES),
            Document.is_paused.is_not(True),
        ),
        "paused": (
            Document.indexing_status.in_(_INDEXING_STATUSES),
            Document.is_paused.is_(True),
        ),
        "error": (Document.indexing_status == IndexingStatus.ERROR,),
        "available": (
            Document.indexing_status == IndexingStatus.COMPLETED,
            Document.archived.is_(False),
            Document.enabled.is_(True),
        ),
        "disabled": (
            Document.indexing_status == IndexingStatus.COMPLETED,
            Document.archived.is_(False),
            Document.enabled.is_(False),
        ),
        "archived": (
            Document.indexing_status == IndexingStatus.COMPLETED,
            Document.archived.is_(True),
        ),
    }
    DOCUMENT_BATCH_DOWNLOAD_ZIP_FILENAME_EXTENSION = ".zip"

    @classmethod
    def normalize_display_status(cls, status: str | None) -> str | None:
        if not status:
            return None
        normalized = status.lower()
        normalized = cls.DISPLAY_STATUS_ALIASES.get(normalized, normalized)
        return normalized if normalized in cls.DISPLAY_STATUS_FILTERS else None

    @classmethod
    def build_display_status_filters(cls, status: str | None) -> tuple[Any, ...]:
        normalized = cls.normalize_display_status(status)
        if not normalized:
            return ()
        return cls.DISPLAY_STATUS_FILTERS[normalized]

    @classmethod
    def apply_display_status_filter(cls, query, status: str | None):
        filters = cls.build_display_status_filters(status)
        if not filters:
            return query
        return query.where(*filters)

    DOCUMENT_METADATA_SCHEMA: dict[str, Any] = {
        "book": {
            "title": str,
            "language": str,
            "author": str,
            "publisher": str,
            "publication_date": str,
            "isbn": str,
            "category": str,
        },
        "web_page": {
            "title": str,
            "url": str,
            "language": str,
            "publish_date": str,
            "author/publisher": str,
            "topic/keywords": str,
            "description": str,
        },
        "paper": {
            "title": str,
            "language": str,
            "author": str,
            "publish_date": str,
            "journal/conference_name": str,
            "volume/issue/page_numbers": str,
            "doi": str,
            "topic/keywords": str,
            "abstract": str,
        },
        "social_media_post": {
            "platform": str,
            "author/username": str,
            "publish_date": str,
            "post_url": str,
            "topic/tags": str,
        },
        "wikipedia_entry": {
            "title": str,
            "language": str,
            "web_page_url": str,
            "last_edit_date": str,
            "editor/contributor": str,
            "summary/introduction": str,
        },
        "personal_document": {
            "title": str,
            "author": str,
            "creation_date": str,
            "last_modified_date": str,
            "document_type": str,
            "tags/category": str,
        },
        "business_document": {
            "title": str,
            "author": str,
            "creation_date": str,
            "last_modified_date": str,
            "document_type": str,
            "department/team": str,
        },
        "im_chat_log": {
            "chat_platform": str,
            "chat_participants/group_name": str,
            "start_date": str,
            "end_date": str,
            "summary": str,
        },
        "synced_from_notion": {
            "title": str,
            "language": str,
            "author/creator": str,
            "creation_date": str,
            "last_modified_date": str,
            "notion_page_link": str,
            "category/tags": str,
            "description": str,
        },
        "synced_from_github": {
            "repository_name": str,
            "repository_description": str,
            "repository_owner/organization": str,
            "code_filename": str,
            "code_file_path": str,
            "programming_language": str,
            "github_link": str,
            "open_source_license": str,
            "commit_date": str,
            "commit_author": str,
        },
        "others": dict,
    }

    @staticmethod
    def get_document(
        dataset_id: str,
        document_id: str | None = None,
        *,
        session: Session,
        tenant_id: str | None = None,
    ) -> Document | None:
        """Fetch a document by id within a dataset using the caller-provided session."""
        if document_id:
            owner_conditions = [Document.id == document_id, Document.dataset_id == dataset_id]
            statement = select(Document).where(*owner_conditions)
            if tenant_id is not None:
                statement = statement.join(Dataset, Dataset.id == Document.dataset_id).where(
                    Document.tenant_id == tenant_id,
                    Dataset.tenant_id == tenant_id,
                )
            document = session.scalar(statement.limit(1))
            return document
        else:
            return None

    @staticmethod
    def get_documents_by_ids(
        dataset_ref: DatasetRef, document_ids: Sequence[str], session: Session
    ) -> Sequence[Document]:
        """Fetch documents for a dataset in a single batch query."""
        if not document_ids:
            return []
        document_id_list: list[str] = list(document_ids)
        # Fetch all requested documents in one query to avoid N+1 lookups.
        documents: Sequence[Document] = session.scalars(
            select(Document).where(
                Document.tenant_id == dataset_ref.tenant_id,
                Document.dataset_id == dataset_ref.dataset_id,
                Document.id.in_(document_id_list),
            )
        ).all()
        return documents

    @staticmethod
    def update_documents_need_summary(
        dataset_id: str,
        document_ids: Sequence[str],
        session: Session,
        need_summary: bool = True,
    ) -> int:
        """
        Update need_summary field for multiple documents.

        This method handles the case where documents were created when summary_index_setting was disabled,
        and need to be updated when summary_index_setting is later enabled.

        Args:
            dataset_id: Dataset ID
            document_ids: List of document IDs to update
            session: SQLAlchemy session used for the update
            need_summary: Value to set for need_summary field (default: True)

        Returns:
            Number of documents updated
        """
        if not document_ids:
            return 0

        document_id_list: list[str] = list(document_ids)

        result = session.execute(
            update(Document)
            .where(
                Document.id.in_(document_id_list),
                Document.dataset_id == dataset_id,
                Document.doc_form != IndexStructureType.QA_INDEX,  # Skip qa_model documents
            )
            .values(need_summary=need_summary)
            .execution_options(synchronize_session=False)
        )
        updated_count = result.rowcount  # type: ignore[union-attr,attr-defined]
        session.commit()
        logger.info(
            "Updated need_summary to %s for %d documents in dataset %s",
            need_summary,
            updated_count,
            dataset_id,
        )
        return updated_count

    @staticmethod
    def get_document_download_url(document: Document, session: Session) -> str:
        """
        Return a signed download URL for an upload-file document.
        """
        upload_file = DocumentService._get_upload_file_for_upload_file_document(document, session)
        return file_helpers.get_signed_file_url(upload_file_id=upload_file.id, as_attachment=True)

    @staticmethod
    def enrich_documents_with_summary_index_status(
        documents: Sequence[Document],
        dataset: Dataset,
        tenant_id: str,
        session: Session,
    ) -> None:
        """
        Enrich documents with summary_index_status based on dataset summary index settings.

        This method calculates and sets the summary_index_status for each document that needs summary.
        Documents that don't need summary or when summary index is disabled will have status set to None.

        Args:
            documents: List of Document instances to enrich
            dataset: Dataset instance containing summary_index_setting
            tenant_id: Tenant ID for summary status lookup
            session: SQLAlchemy session used to read summary status records
        """
        # Check if dataset has summary index enabled
        has_summary_index = dataset.summary_index_setting and dataset.summary_index_setting.get("enable") is True

        # Filter documents that need summary calculation
        documents_need_summary = [doc for doc in documents if doc.need_summary is True]
        document_ids_need_summary = [str(doc.id) for doc in documents_need_summary]

        # Calculate summary_index_status for documents that need summary (only if dataset summary index is enabled)
        summary_status_map: dict[str, str | None] = {}
        if has_summary_index and document_ids_need_summary:
            from services.knowledge.summaries.adapters import SummaryIndexAdapter

            summary_status_map = SummaryIndexAdapter.get_documents_summary_index_status(
                document_ids=document_ids_need_summary,
                dataset_id=dataset.id,
                tenant_id=tenant_id,
                session=session,
            )

        # Add summary_index_status to each document
        for document in documents:
            if has_summary_index and document.need_summary is True:
                # Get status from map, default to None (not queued yet)
                document.summary_index_status = summary_status_map.get(str(document.id))  # type: ignore[attr-defined]
            else:
                # Return null if summary index is not enabled or document doesn't need summary
                document.summary_index_status = None  # type: ignore[attr-defined]

    @staticmethod
    def prepare_document_batch_download_zip(
        *,
        dataset_id: str,
        document_ids: Sequence[str],
        tenant_id: str,
        current_user: Account,
        session: Session,
    ) -> tuple[list[UploadFile], str]:
        """
        Resolve upload files for batch ZIP downloads and generate a client-visible filename.
        """
        dataset = DatasetService.get_dataset(dataset_id, session)
        if not dataset:
            raise NotFound("Dataset not found.")
        try:
            DatasetService.check_dataset_permission(dataset, current_user, session)
        except NoPermissionError as e:
            raise Forbidden(str(e)) from e

        upload_files_by_document_id = DocumentService._get_upload_files_by_document_id_for_zip_download(
            dataset_id=dataset_id,
            document_ids=document_ids,
            tenant_id=tenant_id,
            session=session,
        )
        upload_files = [upload_files_by_document_id[document_id] for document_id in document_ids]
        download_name = DocumentService._generate_document_batch_download_zip_filename()
        return upload_files, download_name

    @staticmethod
    def _generate_document_batch_download_zip_filename() -> str:
        """
        Generate a random attachment filename for the batch download ZIP.
        """
        return f"{uuid.uuid4().hex}{DocumentService.DOCUMENT_BATCH_DOWNLOAD_ZIP_FILENAME_EXTENSION}"

    @staticmethod
    def _get_upload_file_id_for_upload_file_document(
        document: Document,
        *,
        invalid_source_message: str,
        missing_file_message: str,
    ) -> str:
        """
        Normalize and validate `Document -> UploadFile` linkage for download flows.
        """
        if document.data_source_type != DataSourceType.UPLOAD_FILE:
            raise NotFound(invalid_source_message)

        data_source_info: dict[str, Any] = document.data_source_info_dict or {}
        upload_file_id: str | None = data_source_info.get("upload_file_id")
        if not upload_file_id:
            raise NotFound(missing_file_message)

        return str(upload_file_id)

    @staticmethod
    def _get_upload_file_for_upload_file_document(document: Document, session: Session) -> UploadFile:
        """
        Load the `UploadFile` row for an upload-file document.
        """
        upload_file_id = DocumentService._get_upload_file_id_for_upload_file_document(
            document,
            invalid_source_message="Document does not have an uploaded file to download.",
            missing_file_message="Uploaded file not found.",
        )
        upload_files_by_id = FileService.get_upload_files_by_ids(document.tenant_id, [upload_file_id], session=session)
        upload_file = upload_files_by_id.get(upload_file_id)
        if not upload_file:
            raise NotFound("Uploaded file not found.")
        return upload_file

    @staticmethod
    def _get_upload_files_by_document_id_for_zip_download(
        *,
        dataset_id: str,
        document_ids: Sequence[str],
        tenant_id: str,
        session: Session,
    ) -> dict[str, UploadFile]:
        """
        Batch load upload files keyed by document id for ZIP downloads.
        """
        document_id_list: list[str] = list(document_ids)

        documents = DocumentService.get_documents_by_ids(
            DatasetRef(tenant_id=tenant_id, dataset_id=dataset_id), document_id_list, session
        )
        documents_by_id: dict[str, Document] = {str(document.id): document for document in documents}

        missing_document_ids: set[str] = set(document_id_list) - set(documents_by_id.keys())
        if missing_document_ids:
            raise NotFound("Document not found.")

        upload_file_ids: list[str] = []
        upload_file_ids_by_document_id: dict[str, str] = {}
        for document_id, document in documents_by_id.items():
            upload_file_id = DocumentService._get_upload_file_id_for_upload_file_document(
                document,
                invalid_source_message="Only uploaded-file documents can be downloaded as ZIP.",
                missing_file_message="Only uploaded-file documents can be downloaded as ZIP.",
            )
            upload_file_ids.append(upload_file_id)
            upload_file_ids_by_document_id[document_id] = upload_file_id

        upload_files_by_id = FileService.get_upload_files_by_ids(tenant_id, upload_file_ids, session=session)
        missing_upload_file_ids: set[str] = set(upload_file_ids) - set(upload_files_by_id.keys())
        if missing_upload_file_ids:
            raise NotFound("Only uploaded-file documents can be downloaded as ZIP.")

        return {
            document_id: upload_files_by_id[upload_file_id]
            for document_id, upload_file_id in upload_file_ids_by_document_id.items()
        }

    @staticmethod
    def get_document_by_id(document_id: str, session: Session) -> Document | None:
        """Fetch a document by primary key; callers must authorize its dataset before exposing it."""
        document = session.get(Document, document_id)

        return document

    @staticmethod
    def get_document_by_ids(
        dataset_ref: DatasetRef, document_ids: Sequence[str], session: Session
    ) -> Sequence[Document]:
        documents = session.scalars(
            select(Document).where(
                Document.tenant_id == dataset_ref.tenant_id,
                Document.dataset_id == dataset_ref.dataset_id,
                Document.id.in_(document_ids),
                Document.enabled.is_(True),
                Document.indexing_status == IndexingStatus.COMPLETED,
                Document.archived.is_(False),
            )
        ).all()
        return documents

    @staticmethod
    def get_document_by_dataset_id(dataset_id: str, session: Session) -> Sequence[Document]:
        documents = session.scalars(
            select(Document).where(
                Document.dataset_id == dataset_id,
                Document.enabled.is_(True),
            )
        ).all()

        return documents

    @staticmethod
    def get_working_documents_by_dataset_id(dataset_id: str, session: Session) -> Sequence[Document]:
        documents = session.scalars(
            select(Document).where(
                Document.dataset_id == dataset_id,
                Document.enabled.is_(True),
                Document.indexing_status == IndexingStatus.COMPLETED,
                Document.archived.is_(False),
            )
        ).all()

        return documents

    @staticmethod
    def get_error_documents_by_dataset_ref(dataset_ref: DatasetRef, session: Session) -> Sequence[Document]:
        documents = session.scalars(
            select(Document).where(
                Document.tenant_id == dataset_ref.tenant_id,
                Document.dataset_id == dataset_ref.dataset_id,
                Document.indexing_status.in_([IndexingStatus.ERROR, IndexingStatus.PAUSED]),
            )
        ).all()
        return documents

    @staticmethod
    def get_batch_documents(dataset_id: str, batch: str, session: Session) -> Sequence[Document]:
        assert isinstance(current_user, Account)
        documents = session.scalars(
            select(Document).where(
                Document.batch == batch,
                Document.dataset_id == dataset_id,
                Document.tenant_id == current_user.current_tenant_id,
            )
        ).all()

        return documents

    @staticmethod
    def get_document_segment_counts(
        documents: Sequence[Document],
        session: Session,
    ) -> dict[str, tuple[int, int]]:
        """Get completed and total segment counts for multiple documents in one query."""
        if not documents:
            return {}

        document_owner_keys = {
            (str(document.tenant_id), str(document.dataset_id), str(document.id)) for document in documents
        }

        rows = session.execute(
            select(
                DocumentSegment.document_id,
                func.count(DocumentSegment.id).label("total_segments"),
                func.coalesce(func.sum(case((DocumentSegment.completed_at.isnot(None), 1), else_=0)), 0).label(
                    "completed_segments"
                ),
            )
            .where(
                tuple_(DocumentSegment.tenant_id, DocumentSegment.dataset_id, DocumentSegment.document_id).in_(
                    document_owner_keys
                ),
                DocumentSegment.status != SegmentStatus.RE_SEGMENT,
            )
            .group_by(DocumentSegment.document_id)
        )

        return {
            str(document_id): (int(completed_segments or 0), int(total_segments or 0))
            for document_id, total_segments, completed_segments in rows
        }

    @staticmethod
    def get_document_file_detail(file_id: str, session: Session):
        file_detail = session.get(UploadFile, file_id)
        return file_detail

    @staticmethod
    def check_archived(document):
        return bool(document.archived)

    @staticmethod
    def delete_document(document, session: Session):
        document = DocumentService._lock_document_without_indexing_claim(document, session)
        file_id = get_document_source_upload_file_id(document)
        document_id = document.id
        tenant_id = document.tenant_id
        dataset_id = document.dataset_id
        doc_form = document.doc_form
        cleanup_intent_id = create_document_cleanup_intent(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            doc_form=doc_form,
            source_file_id=file_id,
        )
        session.delete(document)
        session.commit()
        enqueue_resource_cleanup_intent(cleanup_intent_id)

    @staticmethod
    def delete_documents(
        dataset_ref: DatasetRef,
        document_ids: list[str],
        doc_form: str | None,
        session: Session,
    ):
        # Check if document_ids is not empty to avoid WHERE false condition
        if not document_ids or len(document_ids) == 0:
            return
        documents = session.scalars(
            select(Document).where(
                Document.id.in_(document_ids),
                Document.tenant_id == dataset_ref.tenant_id,
                Document.dataset_id == dataset_ref.dataset_id,
            )
        ).all()
        documents = [
            DocumentService._lock_document_without_indexing_claim(document, session)
            for document in sorted(documents, key=lambda item: item.id)
        ]
        deleted_document_ids = [document.id for document in documents]
        source_file_ids = {document.id: get_document_source_upload_file_id(document) for document in documents}
        cleanup_intent_ids = create_document_cleanup_intents(
            session,
            tenant_id=dataset_ref.tenant_id,
            dataset_id=dataset_ref.dataset_id,
            document_ids=deleted_document_ids,
            doc_form=doc_form,
            source_file_ids=source_file_ids,
        )

        # Delete documents first, then dispatch cleanup task after commit
        # to avoid deadlock between main transaction and async task
        for document in documents:
            session.delete(document)
        session.commit()

        for cleanup_intent_id in cleanup_intent_ids:
            enqueue_resource_cleanup_intent(cleanup_intent_id)

    @staticmethod
    def rename_document(dataset_id: str, document_id: str, name: str, session: Session) -> Document:
        assert isinstance(current_user, Account)
        tenant_id = current_user.current_tenant_id
        if tenant_id is None:
            raise ValueError("No permission.")
        dataset_ref = DatasetRef(tenant_id, dataset_id)
        dataset = lock_metadata_dataset(session, dataset_ref)
        if not dataset:
            raise ValueError("Dataset not found.")
        document = lock_metadata_document(session, dataset_ref, document_id)
        if not document:
            raise ValueError("Document not found.")
        if dataset.built_in_field_enabled:
            doc_metadata = copy.deepcopy(document.doc_metadata or {})
            doc_metadata[BuiltInField.document_name] = name
            document.doc_metadata = doc_metadata

        document.name = name
        session.add(document)
        if document.data_source_info_dict and "upload_file_id" in document.data_source_info_dict:
            session.execute(
                update(UploadFile)
                .where(
                    UploadFile.id == document.data_source_info_dict["upload_file_id"],
                    UploadFile.tenant_id == dataset.tenant_id,
                )
                .values(name=name)
            )

        session.flush()

        return document

    @staticmethod
    def pause_document(document, session: Session, *, actor_id: str):
        if document.indexing_status not in {
            IndexingStatus.WAITING,
            IndexingStatus.PARSING,
            IndexingStatus.CLEANING,
            IndexingStatus.SPLITTING,
            IndexingStatus.INDEXING,
        }:
            raise DocumentIndexingError()
        # update document to be paused
        document.is_paused = True
        document.paused_by = actor_id
        document.paused_at = naive_utc_now()

        session.add(document)
        session.commit()
        # set document paused flag
        indexing_cache_key = f"document_{document.id}_is_paused"
        redis_client.setnx(indexing_cache_key, "True")

    @staticmethod
    def _lock_document_without_indexing_claim(document: Document, session: Session) -> Document:
        locked_document = session.scalar(
            select(Document)
            .join(Dataset, Dataset.id == Document.dataset_id)
            .join(Tenant, Tenant.id == Dataset.tenant_id)
            .where(
                Tenant.id == document.tenant_id,
                Dataset.id == document.dataset_id,
                Dataset.tenant_id == document.tenant_id,
                Document.id == document.id,
                Document.dataset_id == document.dataset_id,
                Document.tenant_id == document.tenant_id,
            )
            .with_for_update(of=Document)
            .execution_options(populate_existing=True)
        )
        if locked_document is None:
            raise ValueError("Document not found")
        if has_active_indexing_claim(session, locked_document):
            raise DocumentIndexingError()
        locked_document.indexing_claim_id = None
        locked_document.indexing_claim_expires_at = None
        return locked_document

    @staticmethod
    def recover_document(document, session: Session):
        document = DocumentService._lock_document_without_indexing_claim(document, session)
        if not document.is_paused:
            raise DocumentIndexingError()
        current_retry = None
        if document.indexing_request_id is not None:
            candidate = session.scalar(
                select(DocumentIndexingOutbox).where(
                    DocumentIndexingOutbox.id == document.indexing_request_id,
                    DocumentIndexingOutbox.tenant_id == document.tenant_id,
                    DocumentIndexingOutbox.dataset_id == document.dataset_id,
                    DocumentIndexingOutbox.action == DocumentIndexingAction.RETRY.value,
                    DocumentIndexingOutbox.actor_id.is_not(None),
                )
            )
            if candidate is not None:
                candidate_document_ids = json.loads(candidate.document_ids)
                if isinstance(candidate_document_ids, list) and document.id in candidate_document_ids:
                    current_retry = candidate
        # update document to be recover
        document.is_paused = False
        document.paused_by = None
        document.paused_at = None

        session.add(document)
        if current_retry is not None:
            # Pipeline retries require their original actor and graph executor.
            # Requeue the same fenced generation after making it claimable.
            document.indexing_status = IndexingStatus.WAITING
            intent_id = current_retry.id
        else:
            (intent_id,) = create_document_indexing_intents(
                session,
                DocumentIndexingJobs(
                    DatasetRef(document.tenant_id, document.dataset_id),
                    recovered=document.id,
                ),
            )
        session.commit()
        # delete paused flag
        indexing_cache_key = f"document_{document.id}_is_paused"
        redis_client.delete(indexing_cache_key)
        enqueue_document_indexing_intent(intent_id)

    @staticmethod
    def retry_document(dataset_id: str, documents: list[Document], session: Session, *, actor_id: str):
        """Reserve the whole retry batch before changing any document state.

        Redis lock acquisition is intentionally coupled to this bounded status
        transaction so a concurrent request cannot partially admit the batch.
        """
        if not actor_id:
            raise ValueError("Current user or current user id not found")

        unique_documents = list({document.id: document for document in documents}.values())
        if not unique_documents:
            return
        retry_indexing_cache_keys = [f"document_{document.id}_is_retried" for document in unique_documents]
        acquired_locks: list[Any] = []

        def release_acquired_locks() -> None:
            for retry_lock in acquired_locks:
                try:
                    retry_lock.release()
                except Exception:
                    logger.warning("Failed to release document retry lock", exc_info=True)

        try:
            for retry_indexing_cache_key in retry_indexing_cache_keys:
                retry_lock = redis_client.lock(retry_indexing_cache_key, timeout=600, thread_local=False)
                if not retry_lock.acquire(blocking=False):
                    raise ValueError("Document is being retried, please try again later")
                acquired_locks.append(retry_lock)
        except Exception:
            release_acquired_locks()
            raise

        try:
            tenant_ids = {document.tenant_id for document in unique_documents}
            if len(tenant_ids) != 1 or any(document.dataset_id != dataset_id for document in unique_documents):
                raise ValueError("Document not found")
            tenant_id = next(iter(tenant_ids))
            with session.no_autoflush:
                locked_dataset = session.scalar(
                    select(Dataset)
                    .where(
                        Dataset.id == dataset_id,
                        Dataset.tenant_id == tenant_id,
                    )
                    .with_for_update(read=True)
                    .execution_options(populate_existing=True)
                )
                if locked_dataset is None:
                    raise ValueError("Document not found")
                locked_documents = list(
                    session.scalars(
                        select(Document)
                        .where(
                            Document.id.in_(sorted(document.id for document in unique_documents)),
                            Document.dataset_id == dataset_id,
                            Document.tenant_id == tenant_id,
                        )
                        .order_by(Document.id)
                        .with_for_update()
                        .execution_options(populate_existing=True)
                    ).all()
                )
            if len(locked_documents) != len(unique_documents):
                raise ValueError("Document not found")
            for document in locked_documents:
                if has_active_indexing_claim(session, document):
                    raise DocumentIndexingError()
                document.indexing_claim_id = None
                document.indexing_claim_expires_at = None
            unique_documents = locked_documents
            if locked_dataset.runtime_mode == DatasetRuntimeMode.RAG_PIPELINE:
                if locked_dataset.pipeline_id is None:
                    raise ValueError("Pipeline dataset is not bound to a pipeline")
                current_pipeline = session.scalar(
                    select(Pipeline)
                    .where(
                        Pipeline.id == locked_dataset.pipeline_id,
                        Pipeline.tenant_id == tenant_id,
                        Pipeline.is_published.is_(True),
                    )
                    .with_for_update(read=True)
                    .execution_options(populate_existing=True)
                )
                if current_pipeline is None or current_pipeline.workflow_id is None:
                    raise ValueError("Published pipeline workflow is unavailable")
                current_workflow_id = session.scalar(
                    select(Workflow.id)
                    .where(
                        Workflow.id == current_pipeline.workflow_id,
                        Workflow.tenant_id == tenant_id,
                        Workflow.app_id == current_pipeline.id,
                        Workflow.version != Workflow.VERSION_DRAFT,
                    )
                    .with_for_update(read=True)
                )
                if current_workflow_id is None:
                    raise ValueError("Published pipeline workflow is unavailable")
                pipeline_execution_log_ids: dict[str, str] = {}
                for document in unique_documents:
                    source_log = session.scalar(
                        select(DocumentPipelineExecutionLog)
                        .where(
                            DocumentPipelineExecutionLog.document_id == document.id,
                            DocumentPipelineExecutionLog.pipeline_id == locked_dataset.pipeline_id,
                        )
                        .order_by(
                            DocumentPipelineExecutionLog.created_at.desc(),
                            DocumentPipelineExecutionLog.id.desc(),
                        )
                        .limit(1)
                        .with_for_update()
                    )
                    if source_log is None:
                        raise ValueError("Pipeline indexing source snapshot is unavailable")
                    retry_log = DocumentPipelineExecutionLog(
                        pipeline_id=current_pipeline.id,
                        document_id=document.id,
                        datasource_type=source_log.datasource_type,
                        datasource_info=source_log.datasource_info,
                        datasource_node_id=source_log.datasource_node_id,
                        input_data=copy.deepcopy(source_log.input_data),
                        created_by=actor_id,
                        workflow_id=current_workflow_id,
                        # Preserve first-index versus replacement semantics.
                        # A first-index recovery still has a null original id;
                        # the durable worker's recovery flag selects its cleanup
                        # target without turning projected-capacity admission
                        # into replacement admission.
                        original_document_id=source_log.original_document_id,
                    )
                    session.add(retry_log)
                    session.flush()
                    pipeline_execution_log_ids[document.id] = retry_log.id
                requests = create_pipeline_document_indexing_intents(
                    session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    pipeline_id=locked_dataset.pipeline_id,
                    pipeline_execution_log_ids=pipeline_execution_log_ids,
                    actor_id=actor_id,
                )
                intent_ids = tuple(requests[document.id] for document in unique_documents)
            else:
                for document in unique_documents:
                    document.indexing_status = IndexingStatus.WAITING
                    session.add(document)
                (intent_id,) = create_document_indexing_intents(
                    session,
                    DocumentIndexingJobs(
                        DatasetRef(tenant_id, dataset_id),
                        retried=tuple(document.id for document in unique_documents),
                        actor_id=actor_id,
                    ),
                )
                intent_ids = (intent_id,)
            session.commit()
        except Exception:
            session.rollback()
            release_acquired_locks()
            raise

        for intent_id in intent_ids:
            enqueue_document_indexing_intent(intent_id)

    @staticmethod
    def sync_website_document(dataset: Dataset, document: Document, session: Session):
        if document.tenant_id != dataset.tenant_id or document.dataset_id != dataset.id:
            raise ValueError("Document not found.")
        if document.data_source_type != DataSourceType.WEBSITE_CRAWL:
            raise ValueError("Document is not a website document.")
        document = DocumentService._lock_document_without_indexing_claim(document, session)
        tenant_id = document.tenant_id
        dataset_id = document.dataset_id
        document_id = document.id

        # The locked durable request is the duplicate-submission authority.
        # Redis remains a best-effort observability hint and never blocks by
        # itself, including legacy deployments that stored the constant "1".
        sync_indexing_cache_key = website_sync_hint_key(document_id)
        if document.indexing_request_id is not None:
            pending_sync_id = session.scalar(
                select(DocumentIndexingOutbox.id).where(
                    DocumentIndexingOutbox.id == document.indexing_request_id,
                    DocumentIndexingOutbox.tenant_id == document.tenant_id,
                    DocumentIndexingOutbox.dataset_id == document.dataset_id,
                    DocumentIndexingOutbox.action == DocumentIndexingAction.WEBSITE_SYNC.value,
                )
            )
            if pending_sync_id is not None:
                raise ValueError("Document is being synced, please try again later")
        # sync document indexing
        document.indexing_status = IndexingStatus.WAITING
        data_source_info = document.data_source_info_dict
        if data_source_info:
            data_source_info["mode"] = "scrape"
            document.data_source_info = json.dumps(data_source_info, ensure_ascii=False)
        session.add(document)
        (intent_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(
                DatasetRef(tenant_id, dataset_id),
                website_sync=document_id,
            ),
        )
        session.commit()

        hint_written = False
        try:
            redis_client.setex(sync_indexing_cache_key, WEBSITE_SYNC_HINT_TTL_SECONDS, intent_id)
            hint_written = True
        except Exception:
            # The durable request is authoritative. Redis only reports the
            # current request owner; the outbox owns publication recovery.
            logger.warning("Failed to set website sync hint, document_id=%s", document_id, exc_info=True)

        if hint_written:
            request_is_still_pending = (
                session.scalar(
                    select(DocumentIndexingOutbox.id)
                    .join(Document, Document.indexing_request_id == DocumentIndexingOutbox.id)
                    .where(
                        DocumentIndexingOutbox.id == intent_id,
                        DocumentIndexingOutbox.tenant_id == tenant_id,
                        DocumentIndexingOutbox.dataset_id == dataset_id,
                        DocumentIndexingOutbox.action == DocumentIndexingAction.WEBSITE_SYNC.value,
                        Document.id == document_id,
                        Document.tenant_id == tenant_id,
                        Document.dataset_id == dataset_id,
                    )
                )
                is not None
            )
            # Close the reconciliation read before the Redis compare/delete.
            session.commit()
            if not request_is_still_pending:
                try:
                    delete_website_sync_hint_if_owned(
                        redis_client,
                        document_id=document_id,
                        owner_id=intent_id,
                    )
                except Exception:
                    logger.warning(
                        "Failed to reconcile completed website sync hint, document_id=%s",
                        document_id,
                        exc_info=True,
                    )
        enqueue_document_indexing_intent(intent_id)

    @staticmethod
    def get_documents_position(dataset_id, session: Session):
        document = session.scalar(
            select(Document)
            .where(Document.dataset_id == dataset_id)
            .order_by(Document.position.desc())
            .limit(1)
            .with_for_update(read=True)
        )
        if document:
            return document.position + 1
        else:
            return 1

    @staticmethod
    def _get_latest_dataset_process_rule_for_update(dataset: Dataset, *, session: Session) -> DatasetProcessRule | None:
        """Read the latest rule from the current lock owner's view."""
        return session.scalar(
            select(DatasetProcessRule)
            .where(DatasetProcessRule.dataset_id == dataset.id)
            .order_by(DatasetProcessRule.created_at.desc())
            .limit(1)
            .with_for_update(read=True)
        )

    @staticmethod
    @contextmanager
    def save_document_with_dataset_id(
        dataset: Dataset,
        knowledge_config: KnowledgeConfig,
        account: Account | Any,
        dataset_process_rule: DatasetProcessRule | None = None,
        created_from: str = DocumentCreatedFrom.WEB,
        *,
        session: Session,
        website_job_ref: WebsiteJobRef | None = None,
    ) -> Generator[tuple[list[Document], str]]:
        """Commit and dispatch only after the caller materializes its response."""
        assert isinstance(account, Account)
        assert account.current_tenant_id is not None
        dataset_ref = DatasetRef(account.current_tenant_id, dataset.id)
        config = DocumentService.prepare_document_creation(
            DocumentCreationDataset(
                dataset.tenant_id, dataset.indexing_technique, dataset.embedding_model, dataset.embedding_model_provider
            ),
            knowledge_config,
        )
        operation_error: BaseException | None = None
        lock = (
            nullcontext()
            if config.original_document_id
            else redis_client.lock(f"add_document_lock_dataset_id_{dataset_ref.dataset_id}", timeout=600)
        )
        try:
            with lock:
                try:
                    # Preserve caller-owned work on failure. ``begin_nested``
                    # flushes that work before establishing the savepoint, so
                    # only this document operation and response follow-ups are
                    # rolled back when validation or response materialization
                    # fails. Creation reads below are locking/current reads and
                    # therefore do not need to reset the caller's MySQL snapshot.
                    with session.begin_nested():
                        if not config.original_document_id:
                            reloaded_dataset = session.scalar(
                                select(Dataset)
                                .join(Tenant, Tenant.id == Dataset.tenant_id)
                                .where(
                                    Tenant.id == dataset_ref.tenant_id,
                                    Dataset.tenant_id == dataset_ref.tenant_id,
                                    Dataset.id == dataset_ref.dataset_id,
                                )
                                .with_for_update()
                                .execution_options(populate_existing=True)
                            )
                            if reloaded_dataset is None:
                                raise NotFound("Dataset not found")
                            dataset = reloaded_dataset
                        documents, batch, jobs = DocumentService.save_prepared_documents(
                            dataset,
                            config,
                            account,
                            dataset_process_rule,
                            created_from,
                            session=session,
                            website_job_ref=website_job_ref,
                        )
                        indexing_intent_ids = create_document_indexing_intents(session, jobs)
                        yield documents, batch
                    session.commit()
                    DocumentService.dispatch_document_indexing(indexing_intent_ids)
                except BaseException as error:
                    operation_error = error
        except LockNotOwnedError:
            if operation_error is not None:
                raise operation_error from None
            if config.original_document_id:
                raise
            return
        if operation_error is not None:
            raise operation_error

    @staticmethod
    def generate_document_batch() -> str:
        return time.strftime("%Y%m%d%H%M%S") + str(100000 + secrets.randbelow(exclusive_upper_bound=900000))

    @staticmethod
    def prepare_document_creation(
        dataset: DocumentCreationDataset, knowledge_config: KnowledgeConfig
    ) -> KnowledgeConfig:
        """Resolve remote quotas and models from values; new callers close their read transaction first."""
        if dify_config.DEPLOYMENT_EDITION == DeploymentEdition.CLOUD:
            features = FeatureService.get_features(dataset.tenant_id, exclude_vector_space=True)
            if not knowledge_config.original_document_id:
                count = 0
                if knowledge_config.data_source:
                    if knowledge_config.data_source.info_list.data_source_type == "upload_file":
                        if not knowledge_config.data_source.info_list.file_info_list:
                            raise ValueError("File source info is required")
                        upload_file_list = knowledge_config.data_source.info_list.file_info_list.file_ids
                        count = len(upload_file_list)
                    elif knowledge_config.data_source.info_list.data_source_type == "notion_import":
                        notion_info_list = knowledge_config.data_source.info_list.notion_info_list or []
                        for notion_info in notion_info_list:
                            count = count + len(notion_info.pages)
                    elif knowledge_config.data_source.info_list.data_source_type == "website_crawl":
                        website_info = knowledge_config.data_source.info_list.website_info_list
                        assert website_info
                        count = len(website_info.urls)
                    DocumentService.check_document_creation_limits(count, features)

        config = knowledge_config
        effective_dataset = dataset
        if not dataset.indexing_technique:
            if config.indexing_technique not in Dataset.INDEXING_TECHNIQUE_LIST:
                raise ValueError("Indexing technique is invalid")
            if config.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                if not config.embedding_model or not config.embedding_model_provider:
                    model = ModelManager.for_tenant(tenant_id=dataset.tenant_id).get_default_model_instance(
                        tenant_id=dataset.tenant_id, model_type=ModelType.TEXT_EMBEDDING
                    )
                    config = config.model_copy(
                        update={"embedding_model": model.model_name, "embedding_model_provider": model.provider}
                    )
            effective_dataset = replace(
                dataset,
                indexing_technique=config.indexing_technique,
                embedding_model=config.embedding_model,
                embedding_model_provider=config.embedding_model_provider,
            )
        if config.original_document_id:
            DatasetService.check_dataset_model_setting(effective_dataset)
        return config

    @staticmethod
    def dispatch_document_indexing(intent_ids: tuple[str, ...]) -> None:
        """Best-effort publish of committed outbox identifiers."""
        for intent_id in intent_ids:
            enqueue_document_indexing_intent(intent_id)

    @staticmethod
    def save_prepared_documents(
        dataset: Dataset,
        knowledge_config: KnowledgeConfig,
        account: Account | Any,
        dataset_process_rule: DatasetProcessRule | None = None,
        created_from: str = DocumentCreatedFrom.WEB,
        *,
        session: Session,
        website_job_ref: WebsiteJobRef | None = None,
    ) -> tuple[list[Document], str, DocumentIndexingJobs]:
        """Persist prepared input without remote calls, Redis locks, or task dispatch.

        The caller must resolve quotas/models with prepare_document_creation and
        begin a fresh transaction inside the dataset's creation lock for new documents.
        This owner locks and refreshes Dataset before any other SQL so a prepared
        first-document request cannot overwrite a concurrently installed
        configuration generation. Commit the complete operation in the caller
        and publish the returned jobs after its session closes.
        """
        tenant_id = account.current_tenant_id
        if not tenant_id:
            raise ValueError("Current user or current tenant not found")
        dataset = DatasetService._lock_current_dataset_configuration(
            session=session,
            tenant_id=tenant_id,
            dataset_id=dataset.id,
        )

        # Reprocessing never initializes Dataset configuration: transport
        # defaults on that request are irrelevant, and model validation has
        # already used the locked Dataset's current configuration. New-document
        # requests may be the first initializer, so explicit prepared values
        # must still agree when another owner initialized it first.
        if dataset.indexing_technique and not knowledge_config.original_document_id:
            if (
                knowledge_config.indexing_technique
                and dataset.indexing_technique != knowledge_config.indexing_technique
            ):
                raise ValueError("Prepared indexing technique differs from the current dataset configuration.")
            if dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                if (
                    knowledge_config.embedding_model_provider
                    and knowledge_config.embedding_model_provider != dataset.embedding_model_provider
                ):
                    raise ValueError("Prepared embedding provider differs from the current dataset configuration.")
                if knowledge_config.embedding_model and knowledge_config.embedding_model != dataset.embedding_model:
                    raise ValueError("Prepared embedding model differs from the current dataset configuration.")

        files: list[UploadFile] = []
        if (
            not knowledge_config.original_document_id
            and knowledge_config.data_source
            and knowledge_config.data_source.info_list.data_source_type == DataSourceType.UPLOAD_FILE
        ):
            file_info = knowledge_config.data_source.info_list.file_info_list
            if file_info is None:
                raise ValueError("File source info is required")
            # Revalidate uploads within the caller's fresh creation transaction;
            # retain a shared lock until their document references are committed.
            files = list(
                session.scalars(
                    select(UploadFile)
                    .where(UploadFile.tenant_id == dataset.tenant_id, UploadFile.id.in_(file_info.file_ids))
                    .order_by(UploadFile.id)
                    .with_for_update(read=True)
                ).all()
            )
            if len(files) != len(set(file_info.file_ids)):
                raise FileNotExistsError("One or more files not found.")

        DatasetService.check_doc_form(dataset, knowledge_config.doc_form, session=session)
        # if dataset is empty, update dataset data_source_type
        if not dataset.data_source_type and knowledge_config.data_source:
            dataset.data_source_type = knowledge_config.data_source.info_list.data_source_type

        if not dataset.indexing_technique:
            if knowledge_config.indexing_technique not in Dataset.INDEXING_TECHNIQUE_LIST:
                raise ValueError("Indexing technique is invalid")

            dataset.indexing_technique = IndexTechniqueType(knowledge_config.indexing_technique)
            if knowledge_config.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                assert knowledge_config.embedding_model
                assert knowledge_config.embedding_model_provider
                dataset_embedding_model = knowledge_config.embedding_model
                dataset_embedding_model_provider = knowledge_config.embedding_model_provider
                dataset.embedding_model = dataset_embedding_model
                dataset.embedding_model_provider = dataset_embedding_model_provider
                dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                    dataset_embedding_model_provider, dataset_embedding_model, session
                )
                dataset.collection_binding_id = dataset_collection_binding.id
                if not dataset.retrieval_model:
                    default_retrieval_model = {
                        "search_method": RetrievalMethod.SEMANTIC_SEARCH,
                        "reranking_enable": False,
                        "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
                        "top_k": 4,
                        "score_threshold_enabled": False,
                    }

                    dataset.retrieval_model = (
                        knowledge_config.retrieval_model.model_dump()
                        if knowledge_config.retrieval_model
                        else default_retrieval_model
                    )

        documents: list[Document] = []
        jobs = DocumentIndexingJobs(DatasetRef(dataset.tenant_id, dataset.id))
        if knowledge_config.original_document_id:
            document = DocumentService.update_prepared_document(
                dataset,
                knowledge_config,
                account,
                session=session,
                website_job_ref=website_job_ref,
            )
            documents.append(document)
            batch = document.batch
            jobs = replace(jobs, updated=document.id)
        else:
            # When creating new documents, data_source must be provided
            if not knowledge_config.data_source:
                raise ValueError("Data source is required when creating new documents")

            batch = DocumentService.generate_document_batch()
            # save process rule
            if not dataset_process_rule:
                process_rule = knowledge_config.process_rule
                if process_rule:
                    if process_rule.mode in (ProcessRuleMode.CUSTOM, ProcessRuleMode.HIERARCHICAL):
                        if process_rule.rules:
                            dataset_process_rule = DatasetProcessRule(
                                dataset_id=dataset.id,
                                mode=ProcessRuleMode(process_rule.mode),
                                rules=process_rule.rules.model_dump_json() if process_rule.rules else None,
                                created_by=account.id,
                            )
                        else:
                            dataset_process_rule = DocumentService._get_latest_dataset_process_rule_for_update(
                                dataset, session=session
                            )
                            if not dataset_process_rule:
                                raise ValueError("No process rule found.")
                    elif process_rule.mode == ProcessRuleMode.AUTOMATIC:
                        dataset_process_rule = DatasetProcessRule(
                            dataset_id=dataset.id,
                            mode=ProcessRuleMode.AUTOMATIC,
                            rules=json.dumps(DatasetProcessRule.AUTOMATIC_RULES),
                            created_by=account.id,
                        )
                    else:
                        logger.warning(
                            "Invalid process rule mode: %s, can not find dataset process rule",
                            process_rule.mode,
                        )
                        return [], "", jobs
                    session.add(dataset_process_rule)
                    session.flush()
                else:
                    # Fallback when no process_rule provided in knowledge_config:
                    # 1) reuse the dataset's latest process rule if present
                    # 2) otherwise create an automatic rule
                    dataset_process_rule = DocumentService._get_latest_dataset_process_rule_for_update(
                        dataset, session=session
                    )
                    if not dataset_process_rule:
                        dataset_process_rule = DatasetProcessRule(
                            dataset_id=dataset.id,
                            mode=ProcessRuleMode.AUTOMATIC,
                            rules=json.dumps(DatasetProcessRule.AUTOMATIC_RULES),
                            created_by=account.id,
                        )
                        session.add(dataset_process_rule)
                        session.flush()
            assert dataset_process_rule
            position = DocumentService.get_documents_position(dataset.id, session)
            document_ids = []
            duplicate_document_ids = []
            removed_notion_ids: list[str] = []
            if knowledge_config.data_source.info_list.data_source_type == "upload_file":
                file_names = [file.name for file in files]
                db_documents = list(
                    session.scalars(
                        select(Document)
                        .where(
                            Document.dataset_id == dataset.id,
                            Document.tenant_id == account.current_tenant_id,
                            Document.data_source_type == DataSourceType.UPLOAD_FILE,
                            Document.enabled.is_(True),
                            Document.name.in_(file_names),
                        )
                        .with_for_update()
                    ).all()
                )
                documents_map = {document.name: document for document in db_documents}
                for file in files:
                    data_source_info: dict[str, object] = {
                        "upload_file_id": file.id,
                    }
                    document = documents_map.get(file.name)
                    if knowledge_config.duplicate and document:
                        document = DocumentService._lock_document_without_indexing_claim(document, session)
                        document.dataset_process_rule_id = dataset_process_rule.id
                        document.updated_at = naive_utc_now()
                        document.created_from = created_from
                        document.doc_form = IndexStructureType(knowledge_config.doc_form)
                        document.doc_language = knowledge_config.doc_language
                        document.data_source_info = json.dumps(data_source_info)
                        document.batch = batch
                        document.indexing_status = IndexingStatus.WAITING
                        session.add(document)
                        documents.append(document)
                        duplicate_document_ids.append(document.id)
                        continue
                    else:
                        document = DocumentService.build_document(
                            dataset,
                            dataset_process_rule.id,
                            knowledge_config.data_source.info_list.data_source_type,
                            knowledge_config.doc_form,
                            knowledge_config.doc_language,
                            data_source_info,
                            created_from,
                            position,
                            account,
                            file.name,
                            batch,
                        )
                        session.add(document)
                        session.flush()
                        document_ids.append(document.id)
                        documents.append(document)
                        position += 1
            elif knowledge_config.data_source.info_list.data_source_type == "notion_import":
                notion_info_list = knowledge_config.data_source.info_list.notion_info_list  # type: ignore
                if not notion_info_list:
                    raise ValueError("No notion info list found.")
                exist_page_ids = []
                exist_document = {}
                selected_existing_document_ids: list[str] = []
                documents = list(
                    session.scalars(
                        select(Document)
                        .where(
                            Document.dataset_id == dataset.id,
                            Document.tenant_id == account.current_tenant_id,
                            Document.data_source_type == DataSourceType.NOTION_IMPORT,
                            Document.enabled.is_(True),
                        )
                        .with_for_update()
                    ).all()
                )
                if documents:
                    for document in documents:
                        data_source_info = json.loads(document.data_source_info)
                        exist_page_ids.append(data_source_info["notion_page_id"])
                        exist_document[data_source_info["notion_page_id"]] = document.id
                for notion_info in notion_info_list:
                    workspace_id = notion_info.workspace_id
                    for page in notion_info.pages:
                        if page.page_id not in exist_page_ids:
                            data_source_info = {
                                "credential_id": notion_info.credential_id,
                                "notion_workspace_id": workspace_id,
                                "notion_page_id": page.page_id,
                                "notion_page_icon": page.page_icon.model_dump() if page.page_icon else None,  # type: ignore
                                "type": page.type,
                            }
                            # Truncate page name to 255 characters to prevent DB field length errors
                            truncated_page_name = page.page_name[:255] if page.page_name else "nopagename"
                            document = DocumentService.build_document(
                                dataset,
                                dataset_process_rule.id,
                                knowledge_config.data_source.info_list.data_source_type,
                                knowledge_config.doc_form,
                                knowledge_config.doc_language,
                                data_source_info,
                                created_from,
                                position,
                                account,
                                truncated_page_name,
                                batch,
                            )
                            document.id = str(uuid.uuid4())
                            session.add(document)
                            document_ids.append(document.id)
                            documents.append(document)
                            position += 1
                        else:
                            document_id = exist_document.pop(page.page_id, None)
                            if document_id is not None:
                                selected_existing_document_ids.append(document_id)
                reselected_document_ids = invalidate_notion_removal_intents(
                    session,
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_ids=tuple(selected_existing_document_ids),
                )
                duplicate_document_ids.extend(reselected_document_ids)
                session.flush()
                # delete not selected documents
                if len(exist_document) > 0:
                    removed_notion_ids = list(exist_document.values())
            elif knowledge_config.data_source.info_list.data_source_type == "website_crawl":
                website_info = knowledge_config.data_source.info_list.website_info_list
                if not website_info:
                    raise ValueError("No website info list found.")
                urls = website_info.urls
                for url in urls:
                    data_source_info = DocumentService._website_data_source_info(
                        tenant_id=dataset.tenant_id,
                        provider=website_info.provider,
                        job_id=website_info.job_id,
                        url=url,
                        only_main_content=website_info.only_main_content,
                        job_ref=website_job_ref,
                    )
                    document_name = url[:200] + "..." if len(url) > 255 else url
                    document = DocumentService.build_document(
                        dataset,
                        dataset_process_rule.id,
                        knowledge_config.data_source.info_list.data_source_type,
                        knowledge_config.doc_form,
                        knowledge_config.doc_language,
                        data_source_info,
                        created_from,
                        position,
                        account,
                        document_name,
                        batch,
                    )
                    session.add(document)
                    session.flush()
                    document_ids.append(document.id)
                    documents.append(document)
                    position += 1
            session.flush()

            jobs = replace(
                jobs,
                created=tuple(document_ids),
                duplicated=tuple(duplicate_document_ids),
                removed_notion=tuple(removed_notion_ids),
            )
        return documents, batch, jobs

    # @staticmethod
    # def save_document_with_dataset_id(
    #     dataset: Dataset,
    #     knowledge_config: KnowledgeConfig,
    #     account: Account | Any,
    #     dataset_process_rule: Optional[DatasetProcessRule] = None,
    #     created_from: str = "web",
    # ):
    #     # check document limit
    #     features = FeatureService.get_features(current_user.current_tenant_id)

    #     if dify_config.DEPLOYMENT_EDITION == DeploymentEdition.CLOUD:
    #         if not knowledge_config.original_document_id:
    #             count = 0
    #             if knowledge_config.data_source:
    #                 if knowledge_config.data_source.info_list.data_source_type == "upload_file":
    #                     upload_file_list = knowledge_config.data_source.info_list.file_info_list.file_ids
    #
    #                     count = len(upload_file_list)
    #                 elif knowledge_config.data_source.info_list.data_source_type == "notion_import":
    #                     notion_info_list = knowledge_config.data_source.info_list.notion_info_list
    #                     for notion_info in notion_info_list:
    #                         count = count + len(notion_info.pages)
    #                 elif knowledge_config.data_source.info_list.data_source_type == "website_crawl":
    #                     website_info = knowledge_config.data_source.info_list.website_info_list
    #                     count = len(website_info.urls)
    #                 batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)

    #                 if features.billing.subscription.plan == CloudPlan.SANDBOX and count > 1:
    #                     raise ValueError("Your current plan does not support batch upload, please upgrade your plan.")
    #                 if count > batch_upload_limit:
    #                     raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")

    #                 DocumentService.check_documents_upload_quota(count, features)

    #     # if dataset is empty, update dataset data_source_type
    #     if not dataset.data_source_type:
    #         dataset.data_source_type = knowledge_config.data_source.info_list.data_source_type

    #     if not dataset.indexing_technique:
    #         if knowledge_config.indexing_technique not in Dataset.INDEXING_TECHNIQUE_LIST:
    #             raise ValueError("Indexing technique is invalid")

    #         dataset.indexing_technique = knowledge_config.indexing_technique
    #         if knowledge_config.indexing_technique == "high_quality":
    #             model_manager = ModelManager.for_tenant(tenant_id=current_user.current_tenant_id)
    #             if knowledge_config.embedding_model and knowledge_config.embedding_model_provider:
    #                 dataset_embedding_model = knowledge_config.embedding_model
    #                 dataset_embedding_model_provider = knowledge_config.embedding_model_provider
    #             else:
    #                 embedding_model = model_manager.get_default_model_instance(
    #                     tenant_id=current_user.current_tenant_id, model_type=ModelType.TEXT_EMBEDDING
    #                 )
    #                 dataset_embedding_model = embedding_model.model
    #                 dataset_embedding_model_provider = embedding_model.provider
    #             dataset.embedding_model = dataset_embedding_model
    #             dataset.embedding_model_provider = dataset_embedding_model_provider
    #             dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
    #                 dataset_embedding_model_provider, dataset_embedding_model
    #             )
    #             dataset.collection_binding_id = dataset_collection_binding.id
    #             if not dataset.retrieval_model:
    #                 default_retrieval_model = {
    #                     "search_method": RetrievalMethod.SEMANTIC_SEARCH,
    #                     "reranking_enable": False,
    #                     "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
    #                     "top_k": 2,
    #                     "score_threshold_enabled": False,
    #                 }

    #                 dataset.retrieval_model = (
    #                     knowledge_config.retrieval_model.model_dump()
    #                     if knowledge_config.retrieval_model
    #                     else default_retrieval_model
    #                 )

    #     documents = []
    #     if knowledge_config.original_document_id:
    #         document = DocumentService.update_prepared_document(dataset, knowledge_config, account)
    #         documents.append(document)
    #         batch = document.batch
    #     else:
    #         batch = time.strftime("%Y%m%d%H%M%S") + str(random.randint(100000, 999999))
    #         # save process rule
    #         if not dataset_process_rule:
    #             process_rule = knowledge_config.process_rule
    #             if process_rule:
    #                 if process_rule.mode in ("custom", "hierarchical"):
    #                     dataset_process_rule = DatasetProcessRule(
    #                         dataset_id=dataset.id,
    #                         mode=process_rule.mode,
    #                         rules=process_rule.rules.model_dump_json() if process_rule.rules else None,
    #                         created_by=account.id,
    #                     )
    #                 elif process_rule.mode == "automatic":
    #                     dataset_process_rule = DatasetProcessRule(
    #                         dataset_id=dataset.id,
    #                         mode=process_rule.mode,
    #                         rules=json.dumps(DatasetProcessRule.AUTOMATIC_RULES),
    #                         created_by=account.id,
    #                     )
    #                 else:
    #                     logging.warn(
    #                         f"Invalid process rule mode: {process_rule.mode}, can not find dataset process rule"
    #                     )
    #                     return
    #                 session.add(dataset_process_rule)
    #                 session.commit()
    #         lock_name = "add_document_lock_dataset_id_{}".format(dataset.id)
    #         with redis_client.lock(lock_name, timeout=600):
    #             position = DocumentService.get_documents_position(dataset.id)
    #             document_ids = []
    #             duplicate_document_ids = []
    #             if knowledge_config.data_source.info_list.data_source_type == "upload_file":
    #                 upload_file_list = knowledge_config.data_source.info_list.file_info_list.file_ids
    #                 for file_id in upload_file_list:
    #                     file = (
    #                         session.query(UploadFile)
    #                         .filter(UploadFile.tenant_id == dataset.tenant_id, UploadFile.id == file_id)
    #                         .first()
    #                     )

    #                     # raise error if file not found
    #                     if not file:
    #                         raise FileNotExistsError()

    #                     file_name = file.name
    #                     data_source_info = {
    #                         "upload_file_id": file_id,
    #                     }
    #                     # check duplicate
    #                     if knowledge_config.duplicate:
    #                         document = Document.query.filter_by(
    #                             dataset_id=dataset.id,
    #                             tenant_id=current_user.current_tenant_id,
    #                             data_source_type="upload_file",
    #                             enabled=True,
    #                             name=file_name,
    #                         ).first()
    #                         if document:
    #                             document.dataset_process_rule_id = dataset_process_rule.id
    #                             document.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    #                             document.created_from = created_from
    #                             document.doc_form = knowledge_config.doc_form
    #                             document.doc_language = knowledge_config.doc_language
    #                             document.data_source_info = json.dumps(data_source_info)
    #                             document.batch = batch
    #                             document.indexing_status = "waiting"
    #                             session.add(document)
    #                             documents.append(document)
    #                             duplicate_document_ids.append(document.id)
    #                             continue
    #                     document = DocumentService.build_document(
    #                         dataset,
    #                         dataset_process_rule.id,
    #                         knowledge_config.data_source.info_list.data_source_type,
    #                         knowledge_config.doc_form,
    #                         knowledge_config.doc_language,
    #                         data_source_info,
    #                         created_from,
    #                         position,
    #                         account,
    #                         file_name,
    #                         batch,
    #                     )
    #                     session.add(document)
    #                     session.flush()
    #                     document_ids.append(document.id)
    #                     documents.append(document)
    #                     position += 1
    #             elif knowledge_config.data_source.info_list.data_source_type == "notion_import":
    #                 notion_info_list = knowledge_config.data_source.info_list.notion_info_list
    #                 if not notion_info_list:
    #                     raise ValueError("No notion info list found.")
    #                 exist_page_ids = []
    #                 exist_document = {}
    #                 documents = Document.query.filter_by(
    #                     dataset_id=dataset.id,
    #                     tenant_id=current_user.current_tenant_id,
    #                     data_source_type="notion_import",
    #                     enabled=True,
    #                 ).all()
    #                 if documents:
    #                     for document in documents:
    #                         data_source_info = json.loads(document.data_source_info)
    #                         exist_page_ids.append(data_source_info["notion_page_id"])
    #                         exist_document[data_source_info["notion_page_id"]] = document.id
    #                 for notion_info in notion_info_list:
    #                     workspace_id = notion_info.workspace_id
    #                     data_source_binding = DataSourceOauthBinding.query.filter(
    #                         sa.and_(
    #                             DataSourceOauthBinding.tenant_id == current_user.current_tenant_id,
    #                             DataSourceOauthBinding.provider == "notion",
    #                             DataSourceOauthBinding.disabled == False,
    #                             DataSourceOauthBinding.source_info["workspace_id"] == f'"{workspace_id}"',
    #                         )
    #                     ).first()
    #                     if not data_source_binding:
    #                         raise ValueError("Data source binding not found.")
    #                     for page in notion_info.pages:
    #                         if page.page_id not in exist_page_ids:
    #                             data_source_info = {
    #                                 "notion_workspace_id": workspace_id,
    #                                 "notion_page_id": page.page_id,
    #                                 "notion_page_icon": page.page_icon.model_dump() if page.page_icon else None,
    #                                 "type": page.type,
    #                             }
    #                             # Truncate page name to 255 characters to prevent DB field length errors
    #                             truncated_page_name = page.page_name[:255] if page.page_name else "nopagename"
    #                             document = DocumentService.build_document(
    #                                 dataset,
    #                                 dataset_process_rule.id,
    #                                 knowledge_config.data_source.info_list.data_source_type,
    #                                 knowledge_config.doc_form,
    #                                 knowledge_config.doc_language,
    #                                 data_source_info,
    #                                 created_from,
    #                                 position,
    #                                 account,
    #                                 truncated_page_name,
    #                                 batch,
    #                             )
    #                             session.add(document)
    #                             session.flush()
    #                             document_ids.append(document.id)
    #                             documents.append(document)
    #                             position += 1
    #                         else:
    #                             exist_document.pop(page.page_id)
    #                 # delete not selected documents
    #                 if len(exist_document) > 0:
    #                     clean_notion_document_task.delay(list(exist_document.values()), dataset.id)
    #             elif knowledge_config.data_source.info_list.data_source_type == "website_crawl":
    #                 website_info = knowledge_config.data_source.info_list.website_info_list
    #                 if not website_info:
    #                     raise ValueError("No website info list found.")
    #                 urls = website_info.urls
    #                 for url in urls:
    #                     data_source_info = {
    #                         "url": url,
    #                         "provider": website_info.provider,
    #                         "job_id": website_info.job_id,
    #                         "only_main_content": website_info.only_main_content,
    #                         "mode": "crawl",
    #                     }
    #                     if len(url) > 255:
    #                         document_name = url[:200] + "..."
    #                     else:
    #                         document_name = url
    #                     document = DocumentService.build_document(
    #                         dataset,
    #                         dataset_process_rule.id,
    #                         knowledge_config.data_source.info_list.data_source_type,
    #                         knowledge_config.doc_form,
    #                         knowledge_config.doc_language,
    #                         data_source_info,
    #                         created_from,
    #                         position,
    #                         account,
    #                         document_name,
    #                         batch,
    #                     )
    #                     session.add(document)
    #                     session.flush()
    #                     document_ids.append(document.id)
    #                     documents.append(document)
    #                     position += 1
    #             session.commit()

    #             # trigger async task
    #             if document_ids:
    #                 document_indexing_task.delay(dataset.id, document_ids)
    #             if duplicate_document_ids:
    #                 duplicate_document_indexing_task.delay(dataset.id, duplicate_document_ids)

    #     return documents, batch

    @staticmethod
    def check_documents_upload_quota(count: int, features: FeatureModel):
        can_upload_size = features.documents_upload_quota.limit - features.documents_upload_quota.size
        if count > can_upload_size:
            raise ValueError(
                f"You have reached the limit of your subscription. Only {can_upload_size} documents can be uploaded."
            )

    @staticmethod
    def check_document_creation_limits(count: int, features: FeatureModel):
        """Validate billing-backed document creation limits before document rows are created."""
        if dify_config.DEPLOYMENT_EDITION != DeploymentEdition.CLOUD:
            return

        if features.billing.subscription.plan == CloudPlan.SANDBOX and count > 1:
            raise ValueError("Your current plan does not support batch upload, please upgrade your plan.")

        batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)
        if count > batch_upload_limit:
            raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")

        DocumentService.check_documents_upload_quota(count, features)

    @staticmethod
    def _website_data_source_info(
        *,
        tenant_id: str,
        provider: str,
        job_id: str,
        url: str,
        only_main_content: bool,
        job_ref: WebsiteJobRef | None,
    ) -> dict[str, object]:
        if job_ref is None:
            raise ValueError("Website source is missing a stable provider instance")
        if job_ref.tenant_id != tenant_id or job_ref.provider != provider or job_ref.job_id != job_id:
            raise ValueError("Website source job reference does not match the document source")
        return {
            "url": url,
            "provider": provider,
            "provider_instance_id": job_ref.provider_instance_id,
            "job_id": job_id,
            "only_main_content": only_main_content,
            "mode": "crawl",
        }

    @staticmethod
    def build_document(
        dataset: Dataset,
        process_rule_id: str | None,
        data_source_type: str,
        document_form: str,
        document_language: str,
        data_source_info: dict[str, Any],
        created_from: str,
        position: int,
        account: Account,
        name: str,
        batch: str,
    ):
        # Set need_summary based on dataset's summary_index_setting
        need_summary = False
        if dataset.summary_index_setting and dataset.summary_index_setting.get("enable") is True:
            need_summary = True

        document = Document(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            position=position,
            data_source_type=data_source_type,
            data_source_info=json.dumps(data_source_info),
            dataset_process_rule_id=process_rule_id,
            batch=batch,
            name=name,
            created_from=created_from,
            created_by=account.id,
            doc_form=document_form,
            doc_language=document_language,
            need_summary=need_summary,
        )
        if dataset.built_in_field_enabled:
            now = naive_utc_now()
            document.doc_metadata = build_built_in_field_values(
                document_name=name,
                uploader=account.name,
                upload_date=now,
                last_update_date=now,
                data_source_type=data_source_type,
            )
        return document

    @staticmethod
    def get_tenant_documents_count(session: Session):
        assert isinstance(current_user, Account)

        documents_count = (
            session.scalar(
                select(func.count(Document.id)).where(
                    Document.completed_at.isnot(None),
                    Document.enabled.is_(True),
                    Document.archived.is_(False),
                    Document.tenant_id == current_user.current_tenant_id,
                )
            )
            or 0
        )
        return documents_count

    @staticmethod
    def update_prepared_document(
        dataset: Dataset,
        document_data: KnowledgeConfig,
        account: Account,
        dataset_process_rule: DatasetProcessRule | None = None,
        created_from: str = DocumentCreatedFrom.WEB,
        *,
        session: Session,
        website_job_ref: WebsiteJobRef | None = None,
    ):
        assert isinstance(account, Account)

        document = DocumentService.get_document(dataset.id, document_data.original_document_id, session=session)
        if document is None:
            raise NotFound("Document not found")
        upload_file_list: list[str] = []
        locked_upload_files: dict[str, UploadFile] = {}
        if (
            document_data.data_source
            and document_data.data_source.info_list.data_source_type == DataSourceType.UPLOAD_FILE
        ):
            file_info_list = document_data.data_source.info_list.file_info_list
            if not file_info_list:
                raise ValueError("No file info list found.")
            upload_file_list = file_info_list.file_ids
            locked_upload_files = {
                file.id: file
                for file in session.scalars(
                    select(UploadFile)
                    .where(
                        UploadFile.tenant_id == dataset.tenant_id,
                        UploadFile.id.in_(set(upload_file_list)),
                    )
                    .order_by(UploadFile.id)
                    .with_for_update(read=True)
                )
            }
            if len(locked_upload_files) != len(set(upload_file_list)):
                raise FileNotExistsError()
        document = DocumentService._lock_document_without_indexing_claim(document, session)
        if document.display_status != "available":
            raise ValueError("Document is not available")
        previous_source_upload_file_id = get_document_source_upload_file_id(document)
        # save process rule
        if document_data.process_rule:
            process_rule = document_data.process_rule
            if process_rule.mode in {ProcessRuleMode.CUSTOM, ProcessRuleMode.HIERARCHICAL}:
                dataset_process_rule = DatasetProcessRule(
                    dataset_id=dataset.id,
                    mode=ProcessRuleMode(process_rule.mode),
                    rules=process_rule.rules.model_dump_json() if process_rule.rules else None,
                    created_by=account.id,
                )
            elif process_rule.mode == ProcessRuleMode.AUTOMATIC:
                dataset_process_rule = DatasetProcessRule(
                    dataset_id=dataset.id,
                    mode=ProcessRuleMode.AUTOMATIC,
                    rules=json.dumps(DatasetProcessRule.AUTOMATIC_RULES),
                    created_by=account.id,
                )
            if dataset_process_rule is not None:
                session.add(dataset_process_rule)
                session.flush()
                document.dataset_process_rule_id = dataset_process_rule.id
        # update document data source
        if document_data.data_source:
            file_name = ""
            data_source_info: dict[str, object] = {}
            if document_data.data_source.info_list.data_source_type == "upload_file":
                for file_id in upload_file_list:
                    file = locked_upload_files[file_id]

                    file_name = file.name
                    data_source_info = {
                        "upload_file_id": file_id,
                    }
            elif document_data.data_source.info_list.data_source_type == "notion_import":
                if not document_data.data_source.info_list.notion_info_list:
                    raise ValueError("No notion info list found.")
                notion_info_list = document_data.data_source.info_list.notion_info_list
                for notion_info in notion_info_list:
                    workspace_id = notion_info.workspace_id
                    data_source_binding = session.scalar(
                        select(DataSourceOauthBinding)
                        .where(
                            sa.and_(
                                DataSourceOauthBinding.tenant_id == account.current_tenant_id,
                                DataSourceOauthBinding.provider == "notion",
                                DataSourceOauthBinding.disabled.is_(False),
                                DataSourceOauthBinding.source_info["workspace_id"] == f'"{workspace_id}"',
                            )
                        )
                        .limit(1)
                    )
                    if not data_source_binding:
                        raise ValueError("Data source binding not found.")
                    for page in notion_info.pages:
                        data_source_info = {
                            "credential_id": notion_info.credential_id,
                            "notion_workspace_id": workspace_id,
                            "notion_page_id": page.page_id,
                            "notion_page_icon": page.page_icon.model_dump() if page.page_icon else None,  # type: ignore
                            "type": page.type,
                        }
            elif document_data.data_source.info_list.data_source_type == "website_crawl":
                website_info = document_data.data_source.info_list.website_info_list
                if website_info:
                    urls = website_info.urls
                    for url in urls:
                        data_source_info = DocumentService._website_data_source_info(
                            tenant_id=dataset.tenant_id,
                            provider=website_info.provider,
                            job_id=website_info.job_id,
                            url=url,
                            only_main_content=website_info.only_main_content,
                            job_ref=website_job_ref,
                        )
            document.data_source_type = document_data.data_source.info_list.data_source_type
            document.data_source_info = json.dumps(data_source_info)
            document.name = file_name

        # update document name
        if document_data.name:
            document.name = document_data.name
        # update document to be waiting
        document.indexing_status = IndexingStatus.WAITING
        document.completed_at = None
        document.processing_started_at = None
        document.parsing_completed_at = None
        document.cleaning_completed_at = None
        document.splitting_completed_at = None
        document.updated_at = naive_utc_now()
        document.created_from = created_from
        document.doc_form = IndexStructureType(document_data.doc_form)
        if "doc_language" in document_data.model_fields_set:
            document.doc_language = document_data.doc_language
        session.add(document)
        session.flush()
        current_source_upload_file_id = get_document_source_upload_file_id(document)
        if previous_source_upload_file_id and previous_source_upload_file_id != current_source_upload_file_id:
            # The file domain owns cross-domain liveness and durable blob
            # cleanup. Persist its decision in this source-swap transaction;
            # the periodic drainer performs external storage I/O after commit.
            register_upload_file_cleanup_candidates_in_session(
                session,
                workspace_id=dataset.tenant_id,
                file_ids=(previous_source_upload_file_id,),
            )
        # update document segment

        session.execute(
            update(DocumentSegment)
            .where(DocumentSegment.document_id == document.id)
            .values(status=SegmentStatus.RE_SEGMENT)
        )
        session.flush()
        return document

    @staticmethod
    def build_dataset_for_documents(
        ref: DatasetRef, knowledge_config: KnowledgeConfig, account: Account, session: Session
    ) -> Dataset:
        """Create a dataset inside the caller's SQL-only persistence phase."""
        assert knowledge_config.data_source
        dataset_collection_binding_id = None
        retrieval_model = None
        if knowledge_config.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            assert knowledge_config.embedding_model_provider
            assert knowledge_config.embedding_model
            dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                knowledge_config.embedding_model_provider,
                knowledge_config.embedding_model,
                session,
            )
            dataset_collection_binding_id = dataset_collection_binding.id
        if knowledge_config.retrieval_model:
            retrieval_model = knowledge_config.retrieval_model
        else:
            retrieval_model = RetrievalModel(
                search_method=RetrievalMethod.SEMANTIC_SEARCH,
                reranking_enable=False,
                reranking_model=RerankingModel(reranking_provider_name="", reranking_model_name=""),
                top_k=4,
                score_threshold_enabled=False,
            )
        # save dataset
        dataset = Dataset(
            id=ref.dataset_id,
            tenant_id=ref.tenant_id,
            name="",
            data_source_type=knowledge_config.data_source.info_list.data_source_type,
            indexing_technique=IndexTechniqueType(knowledge_config.indexing_technique),
            created_by=account.id,
            maintainer=account.id,
            embedding_model=knowledge_config.embedding_model,
            embedding_model_provider=knowledge_config.embedding_model_provider,
            collection_binding_id=dataset_collection_binding_id,
            retrieval_model=retrieval_model.model_dump() if retrieval_model else None,
            summary_index_setting=knowledge_config.summary_index_setting,
            is_multimodal=knowledge_config.is_multimodal,
        )

        session.add(dataset)
        session.flush()

        return dataset

    @classmethod
    def document_create_args_validate(cls, knowledge_config: KnowledgeConfig):
        if not knowledge_config.data_source and not knowledge_config.process_rule:
            raise ValueError("Data source or Process rule is required")
        else:
            if knowledge_config.data_source:
                DocumentService.data_source_args_validate(knowledge_config)
            if knowledge_config.process_rule:
                DocumentService.process_rule_args_validate(knowledge_config)

    @classmethod
    def data_source_args_validate(cls, knowledge_config: KnowledgeConfig):
        if not knowledge_config.data_source:
            raise ValueError("Data source is required")

        if knowledge_config.data_source.info_list.data_source_type not in Document.DATA_SOURCES:
            raise ValueError("Data source type is invalid")

        if not knowledge_config.data_source.info_list:
            raise ValueError("Data source info is required")

        if knowledge_config.data_source.info_list.data_source_type == "upload_file":
            if not knowledge_config.data_source.info_list.file_info_list:
                raise ValueError("File source info is required")
        if knowledge_config.data_source.info_list.data_source_type == "notion_import":
            if not knowledge_config.data_source.info_list.notion_info_list:
                raise ValueError("Notion source info is required")
        if knowledge_config.data_source.info_list.data_source_type == "website_crawl":
            if not knowledge_config.data_source.info_list.website_info_list:
                raise ValueError("Website source info is required")

    @classmethod
    def process_rule_args_validate(cls, knowledge_config: KnowledgeConfig):
        if not knowledge_config.process_rule:
            raise ValueError("Process rule is required")

        if not knowledge_config.process_rule.mode:
            raise ValueError("Process rule mode is required")

        if knowledge_config.process_rule.mode not in DatasetProcessRule.MODES:
            raise ValueError("Process rule mode is invalid")

        process_rule = knowledge_config.process_rule
        normalized = normalize_process_rule(process_rule.model_dump(mode="python", exclude_unset=True))
        if process_rule.mode == ProcessRuleMode.AUTOMATIC:
            knowledge_config.process_rule.rules = None
            return
        normalized_rules = normalized.get("rules")
        if not isinstance(normalized_rules, Mapping):
            raise ValueError("Process rule rules is required")
        knowledge_config.process_rule.rules = Rule.model_validate(normalized_rules)

    @staticmethod
    def batch_update_document_status(
        dataset: Dataset,
        document_ids: list[str],
        action: Literal["enable", "disable", "archive", "un_archive"],
        user,
        session: Session,
    ):
        """
        Batch update document status.

        Args:
            dataset (Dataset): The dataset object
            document_ids (list[str]): List of document IDs to update
            action (Literal["enable", "disable", "archive", "un_archive"]): Action to perform
            user: Current user performing the action

        Raises:
            DocumentIndexingError: If document is being indexed or not in correct state
            ValueError: If action is invalid

        Once the document and SegmentStateOutbox rows commit, the transition
        is accepted. Immediate broker publication is best-effort because the
        periodic outbox drainer owns recovery.
        """
        if not document_ids:
            return

        # Early validation of action parameter
        valid_actions = ["enable", "disable", "archive", "un_archive"]
        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {valid_actions}")

        locked_dataset = session.scalar(
            select(Dataset)
            .where(Dataset.id == dataset.id, Dataset.tenant_id == dataset.tenant_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if locked_dataset is None:
            raise ValueError("Dataset not found.")
        dataset = locked_dataset

        documents = list(
            session.scalars(
                select(Document).where(
                    Document.id.in_(document_ids),
                    Document.tenant_id == dataset.tenant_id,
                    Document.dataset_id == dataset.id,
                )
            ).all()
        )
        # Redis is only a legacy worker hint. Check it before taking database
        # row locks; the persisted indexing claim is the authoritative owner.
        for document in documents:
            # Check if document is being indexed
            indexing_cache_key = f"document_{document.id}_indexing"
            cache_result = redis_client.get(indexing_cache_key)
            if cache_result is not None:
                raise DocumentIndexingError(f"Document:{document.name} is being indexed, please try again later")

        locked_documents = [
            DocumentService._lock_document_without_indexing_claim(document, session)
            for document in sorted(documents, key=lambda item: item.id)
        ]
        documents_to_update = []

        # First pass: validate all documents and prepare updates
        for document in locked_documents:
            # Prepare update based on action
            update_info = DocumentService._prepare_document_status_update(document, action, user)
            if update_info:
                documents_to_update.append(update_info)

        requires_model = any(
            update_info["async_task"] is not None
            and update_info["async_task"]["function"] is add_document_to_index_task
            and bool(update_info["updates"].get("enabled", update_info["document"].enabled))
            and not bool(update_info["updates"].get("archived", update_info["document"].archived))
            for update_info in documents_to_update
        )
        if requires_model:
            DatasetService.check_dataset_model_setting(dataset)

        # Second pass: apply all updates in a single transaction
        if documents_to_update:
            segment_intent_ids: list[str] = []
            try:
                for update_info in documents_to_update:
                    document = update_info["document"]
                    updates = update_info["updates"]

                    # Apply updates to the document
                    for field, value in updates.items():
                        setattr(document, field, value)

                    session.add(document)
                    async_task = update_info["async_task"]
                    segment_action = update_info["segment_action"]
                    if (async_task is not None and async_task["function"] is add_document_to_index_task) or (
                        segment_action == "disable"
                    ):
                        segment_query = select(DocumentSegment).where(
                            DocumentSegment.tenant_id == dataset.tenant_id,
                            DocumentSegment.dataset_id == dataset.id,
                            DocumentSegment.document_id == document.id,
                        )
                        if segment_action != "disable":
                            segment_query = segment_query.where(DocumentSegment.status == SegmentStatus.COMPLETED)
                        segments = session.scalars(
                            segment_query.order_by(DocumentSegment.id)
                            .with_for_update()
                            .execution_options(populate_existing=True)
                        ).all()
                    else:
                        segments = []
                    if async_task is not None and async_task["function"] is add_document_to_index_task:
                        request_document_enable_intents_locked(
                            session,
                            dataset=dataset,
                            document=document,
                            segments=segments,
                            actor_id=user.id,
                        )
                    elif segment_action == "disable":
                        intents = request_document_disable_intents_locked(
                            session,
                            dataset=dataset,
                            document=document,
                            segments=segments,
                            actor_id=user.id,
                        )
                        segment_intent_ids.extend(intent.id for intent in intents)

                # Batch commit all changes
                session.commit()
            except Exception as e:
                # Rollback on any error
                session.rollback()
                raise e
            # Execute async tasks and set Redis cache after successful commit
            if segment_intent_ids:
                try:
                    publish_segment_state_intents(segment_intent_ids)
                except Exception:
                    # The committed rows remain visible to the periodic drainer.
                    logger.exception("Error publishing document segment state intents %s", segment_intent_ids)
            for update_info in documents_to_update:
                try:
                    # Execute async tasks after successful commit
                    if update_info["async_task"]:
                        task_info = update_info["async_task"]
                        task_func = task_info["function"]
                        task_args = task_info["args"]
                        task_func.delay(*task_args)
                except Exception:
                    # The committed SegmentStateOutbox rows remain visible to
                    # the periodic drainer. Broker publication is a latency
                    # optimization, not part of the accepted state transition.
                    logger.exception("Error executing async task for document %s", update_info["document"].id)
                try:
                    # Set Redis cache if needed after successful commit
                    if update_info["set_cache"]:
                        document = update_info["document"]
                        indexing_cache_key = f"document_{document.id}_indexing"
                        redis_client.setex(indexing_cache_key, 600, 1)
                except Exception:
                    # Log the error but do not rollback the transaction
                    logger.exception("Error setting cache for document %s", update_info["document"].id)

    @staticmethod
    def _prepare_document_status_update(
        document: Document, action: Literal["enable", "disable", "archive", "un_archive"], user
    ):
        """Prepare document status update information.

        Args:
            document: Document object to update
            action: Action to perform
            user: Current user

        Returns:
            dict: Update information or None if no update needed
        """
        now = naive_utc_now()

        match action:
            case "enable":
                return DocumentService._prepare_enable_update(document, now)
            case "disable":
                return DocumentService._prepare_disable_update(document, user, now)
            case "archive":
                return DocumentService._prepare_archive_update(document, user, now)
            case "un_archive":
                return DocumentService._prepare_unarchive_update(document, now)

        return None

    @staticmethod
    def _prepare_enable_update(document, now):
        """Prepare updates for enabling a document."""
        if document.enabled:
            return None

        return {
            "document": document,
            "updates": {"enabled": True, "disabled_at": None, "disabled_by": None, "updated_at": now},
            "async_task": {"function": add_document_to_index_task, "args": [document.id]},
            "segment_action": None,
            # SegmentStateOutbox is the durable authority. A legacy document
            # cache marker has no owner to clear if broker publication fails.
            "set_cache": False,
        }

    @staticmethod
    def _prepare_disable_update(document, user, now):
        """Prepare updates for disabling a document."""
        if not document.completed_at or document.indexing_status != IndexingStatus.COMPLETED:
            raise DocumentIndexingError(f"Document: {document.name} is not completed.")

        if not document.enabled:
            return None

        return {
            "document": document,
            "updates": {"enabled": False, "disabled_at": now, "disabled_by": user.id, "updated_at": now},
            "async_task": None,
            "segment_action": "disable",
            "set_cache": False,
        }

    @staticmethod
    def _prepare_archive_update(document, user, now):
        """Prepare updates for archiving a document."""
        if document.archived:
            return None

        update_info: dict[str, Any] = {
            "document": document,
            "updates": {"archived": True, "archived_at": now, "archived_by": user.id, "updated_at": now},
            "async_task": None,
            "segment_action": None,
            "set_cache": False,
        }

        # The current owner lock proved no indexing worker is active. Even an
        # incomplete archived document may have ERROR segments or summary nodes
        # from an earlier attempt that require durable cleanup.
        if document.enabled:
            update_info["segment_action"] = "disable"

        return update_info

    @staticmethod
    def _prepare_unarchive_update(document, now):
        """Prepare updates for unarchiving a document."""
        if not document.archived:
            return None

        update_info = {
            "document": document,
            "updates": {"archived": False, "archived_at": None, "archived_by": None, "updated_at": now},
            "async_task": None,
            "segment_action": None,
            "set_cache": False,
        }

        # Only re-index if the document is currently enabled
        if document.enabled:
            # pyrefly: ignore [bad-assignment]
            update_info["async_task"] = {"function": add_document_to_index_task, "args": [document.id]}
            update_info["set_cache"] = False

        return update_info


def _segment_mutation_scope(dataset: Dataset, document: Document) -> SegmentMutationScope:
    """Materialize already-authorized rows before ending their read transaction."""
    if document.tenant_id != dataset.tenant_id or document.dataset_id != dataset.id:
        raise ValueError("Document does not belong to the dataset")
    return SegmentMutationScope(
        dataset=SegmentDatasetRecord(
            id=dataset.id,
            workspace_id=dataset.tenant_id,
            indexing_technique=dataset.indexing_technique or "economy",
            embedding_model_provider=dataset.embedding_model_provider,
            embedding_model=dataset.embedding_model,
        ),
        document=SegmentDocumentRecord(
            id=document.id,
            dataset_id=document.dataset_id,
            workspace_id=document.tenant_id,
            doc_form=document.doc_form,
        ),
    )


def _segment_ref_for_mutation(scope: SegmentMutationScope, segment: DocumentSegment) -> SegmentRef:
    document_ref = scope.document.ref
    if (
        segment.tenant_id != document_ref.dataset.tenant_id
        or segment.dataset_id != document_ref.dataset.dataset_id
        or segment.document_id != document_ref.document_id
    ):
        raise ValueError("Segment does not belong to the document")
    return document_ref.segment(segment.id)


def _check_child_chunk_scope(child: ChildChunk, segment_ref: SegmentRef) -> None:
    if (
        child.tenant_id != segment_ref.document.dataset.tenant_id
        or child.dataset_id != segment_ref.document.dataset.dataset_id
        or child.document_id != segment_ref.document.document_id
        or child.segment_id != segment_ref.segment_id
    ):
        raise ValueError("Child chunk does not belong to the segment")


class SegmentService:
    @classmethod
    def segment_create_args_validate(cls, args: dict[str, Any], document: Document):
        validate_segment_values(
            args, doc_form=document.doc_form, attachment_limit=dify_config.SINGLE_CHUNK_ATTACHMENT_LIMIT
        )

    @classmethod
    def multi_create_segment(
        cls,
        segments: list,
        document: Document,
        dataset: Dataset,
        session: Session,
        *,
        mutations: SegmentMutationService,
    ):
        """Commit a Service API batch before dispatching its scoped index update."""
        assert isinstance(current_user, Account)
        assert current_user.current_tenant_id is not None
        scope = _segment_mutation_scope(dataset, document)
        if current_user.current_tenant_id != scope.dataset.workspace_id:
            raise ValueError("Dataset does not belong to the current workspace")
        actor_id = current_user.id
        document_ref = scope.document.ref
        session.commit()
        args_list = [
            validate_segment_values(
                item,
                doc_form=scope.document.doc_form,
                attachment_limit=dify_config.SINGLE_CHUNK_ATTACHMENT_LIMIT,
            )
            for item in segments
        ]
        embedding_model = None
        if scope.dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            model_manager = ModelManager.for_tenant(tenant_id=scope.dataset.workspace_id)
            embedding_model = model_manager.get_model_instance(
                tenant_id=scope.dataset.workspace_id,
                provider=scope.dataset.embedding_model_provider or "",
                model_type=ModelType.TEXT_EMBEDDING,
                model=scope.dataset.embedding_model or "",
            )
        token_counts = [
            embedding_model.get_text_embedding_num_tokens(
                texts=[
                    (args.content or "") + (args.answer or "")
                    if scope.document.doc_form == IndexStructureType.QA_INDEX
                    else (args.content or "")
                ]
            )[0]
            if embedding_model
            else 0
            for args in args_list
        ]
        segment_refs = mutations.create_segments(
            scope,
            actor_id=actor_id,
            args_list=args_list,
            token_counts=token_counts,
        )
        segment_ids = [ref.segment_id for ref in segment_refs]
        keywords_list = [args.keywords if "keywords" in args.model_fields_set else None for args in args_list]
        try:
            mutations.index_segments(document_ref, segment_ids=segment_ids, keywords_list=keywords_list)
        except Exception:
            logger.exception("create segment index failed")
        session.expire_all()
        persisted = session.scalars(
            select(DocumentSegment).where(
                DocumentSegment.tenant_id == document_ref.dataset.tenant_id,
                DocumentSegment.dataset_id == document_ref.dataset.dataset_id,
                DocumentSegment.document_id == document_ref.document_id,
                DocumentSegment.id.in_(segment_ids),
            )
        ).all()
        by_id = {segment.id: segment for segment in persisted}
        return [by_id[segment_id] for segment_id in segment_ids if segment_id in by_id]

    @classmethod
    def update_segment(
        cls,
        args: SegmentUpdateArgs,
        segment: DocumentSegment,
        document: Document,
        dataset: Dataset,
        session: Session,
        *,
        actor_id: str,
        mutations: SegmentMutationService,
    ) -> DocumentSegment:
        """Delegate an authorized Service API patch to the shared mutation owner."""
        scope = _segment_mutation_scope(dataset, document)
        segment_ref = _segment_ref_for_mutation(scope, segment)
        # Finish the caller's read transaction before the mutation owner performs
        # external indexing I/O in between its own bounded write transactions.
        session.commit()
        mutations.update_segment(scope, actor_id=actor_id, segment_id=segment_ref.segment_id, args=args)
        session.expire_all()
        updated = cls.get_segment_by_ref(segment_ref, session=session)
        if updated is None:
            raise ValueError("new_segment is not found")
        return updated

    @classmethod
    def delete_segment(
        cls,
        segment: DocumentSegment,
        document: Document,
        dataset: Dataset,
        session: Session,
        *,
        mutations: SegmentMutationService,
    ) -> None:
        segment_ref = _segment_ref_for_mutation(_segment_mutation_scope(dataset, document), segment)
        session.commit()
        mutations.delete_segment(segment_ref)
        session.expire_all()

    @classmethod
    def create_child_chunk(
        cls,
        content: str,
        segment: DocumentSegment,
        document: Document,
        dataset: Dataset,
        session: Session,
        *,
        actor_id: str,
        mutations: SegmentMutationService,
    ) -> ChildChunk:
        segment_ref = _segment_ref_for_mutation(_segment_mutation_scope(dataset, document), segment)
        session.commit()
        try:
            created = mutations.create_child_chunk(segment_ref, content=content, actor_id=actor_id)
        except ChildChunkIndexingApplicationError as error:
            raise ChildChunkIndexingError(str(error)) from error
        session.expire_all()
        child = cls.get_child_chunk_by_segment_ref(created.id, segment_ref, session=session)
        if child is None:
            raise ValueError("Child chunk no longer exists")
        return child

    @classmethod
    def update_child_chunk(
        cls,
        content: str,
        child_chunk: ChildChunk,
        segment: DocumentSegment,
        document: Document,
        dataset: Dataset,
        session: Session,
        *,
        actor_id: str,
        mutations: SegmentMutationService,
    ) -> ChildChunk:
        segment_ref = _segment_ref_for_mutation(_segment_mutation_scope(dataset, document), segment)
        _check_child_chunk_scope(child_chunk, segment_ref)
        child_id = child_chunk.id
        session.commit()
        try:
            mutations.update_child_chunk(segment_ref, child_chunk_id=child_id, content=content, actor_id=actor_id)
        except ChildChunkIndexingApplicationError as error:
            raise ChildChunkIndexingError(str(error)) from error
        session.expire_all()
        updated = cls.get_child_chunk_by_segment_ref(child_id, segment_ref, session=session)
        if updated is None:
            raise ValueError("Child chunk no longer exists")
        return updated

    @classmethod
    def delete_child_chunk(
        cls,
        child_chunk: ChildChunk,
        dataset: Dataset,
        session: Session,
        *,
        mutations: SegmentMutationService,
    ) -> None:
        segment_ref = (
            DatasetRef(tenant_id=dataset.tenant_id, dataset_id=dataset.id)
            .document(child_chunk.document_id)
            .segment(child_chunk.segment_id)
        )
        _check_child_chunk_scope(child_chunk, segment_ref)
        child_id = child_chunk.id
        session.commit()
        try:
            mutations.delete_child_chunk(segment_ref, child_chunk_id=child_id)
        except ChildChunkDeleteIndexApplicationError as error:
            raise ChildChunkDeleteIndexError(str(error)) from error
        session.expire_all()

    @classmethod
    def get_child_chunks(
        cls,
        segment_id: str,
        document_id: str,
        dataset_id: str,
        page: int,
        limit: int,
        keyword: str | None = None,
        *,
        session: Session,
    ):
        assert isinstance(current_user, Account)

        assert current_user.current_tenant_id is not None
        return query_child_chunks(
            session,
            DatasetRef(current_user.current_tenant_id, dataset_id).document(document_id).segment(segment_id),
            page=page,
            limit=limit,
            keyword=keyword,
        )

    @classmethod
    def get_child_chunk_by_id(cls, child_chunk_id: str, tenant_id: str, session: Session) -> ChildChunk | None:
        """Get a child chunk by its ID."""
        result = session.scalar(
            select(ChildChunk).where(ChildChunk.id == child_chunk_id, ChildChunk.tenant_id == tenant_id).limit(1)
        )
        return result if isinstance(result, ChildChunk) else None

    @classmethod
    def get_child_chunk_by_segment_ref(
        cls, child_chunk_id: str, segment_ref: SegmentRef, session: Session
    ) -> ChildChunk | None:
        """Get a child chunk through the full tenant/dataset/document/segment chain."""
        result = session.scalar(
            select(ChildChunk)
            .where(
                ChildChunk.id == child_chunk_id,
                ChildChunk.tenant_id == segment_ref.document.dataset.tenant_id,
                ChildChunk.dataset_id == segment_ref.document.dataset.dataset_id,
                ChildChunk.document_id == segment_ref.document.document_id,
                ChildChunk.segment_id == segment_ref.segment_id,
            )
            .limit(1)
        )
        return result if isinstance(result, ChildChunk) else None

    @classmethod
    def get_segments(
        cls,
        document_id: str,
        tenant_id: str,
        status_list: list[str] | None = None,
        keyword: str | None = None,
        page: int = 1,
        limit: int = 20,
        *,
        session: Session,
    ):
        """Get segments for a document with optional filtering."""
        query = select(DocumentSegment).where(
            DocumentSegment.document_id == document_id, DocumentSegment.tenant_id == tenant_id
        )

        # Check if status_list is not empty to avoid WHERE false condition
        if status_list and len(status_list) > 0:
            query = query.where(DocumentSegment.status.in_(status_list))

        if keyword:
            escaped_keyword = helper.escape_like_pattern(keyword)
            query = query.where(DocumentSegment.content.ilike(f"%{escaped_keyword}%", escape="\\"))

        query = query.order_by(DocumentSegment.position.asc(), DocumentSegment.id.asc())
        paginated_segments = paginate_query(query, session=session, page=page, per_page=limit, max_per_page=100)

        return paginated_segments.items, paginated_segments.total

    @classmethod
    def get_segment_by_id(cls, segment_id: str, tenant_id: str, session: Session) -> DocumentSegment | None:
        """Get a segment by its ID."""
        result = session.scalar(
            select(DocumentSegment)
            .where(DocumentSegment.id == segment_id, DocumentSegment.tenant_id == tenant_id)
            .limit(1)
        )
        return result if isinstance(result, DocumentSegment) else None

    @classmethod
    def get_segment_by_ref(cls, segment_ref: SegmentRef, session: Session) -> DocumentSegment | None:
        """Get a segment through the full tenant/dataset/document ownership chain."""
        result = session.scalar(
            select(DocumentSegment)
            .where(
                DocumentSegment.id == segment_ref.segment_id,
                DocumentSegment.tenant_id == segment_ref.document.dataset.tenant_id,
                DocumentSegment.dataset_id == segment_ref.document.dataset.dataset_id,
                DocumentSegment.document_id == segment_ref.document.document_id,
            )
            .limit(1)
        )
        return result if isinstance(result, DocumentSegment) else None

    @classmethod
    def get_segments_by_document_and_dataset(
        cls,
        document_id: str,
        dataset_id: str,
        session: Session,
        status: str | None = None,
        enabled: bool | None = None,
        *,
        tenant_id: str,
    ) -> Sequence[DocumentSegment]:
        """
        Get segments through their tenant, dataset, and document owner chain.

        Args:
            document_id: Document ID
            dataset_id: Dataset ID
            tenant_id: Owning tenant ID
            status: Optional status filter (e.g., "completed")
            enabled: Optional enabled filter (True/False)

        Returns:
            Sequence of DocumentSegment instances
        """
        query = select(DocumentSegment).where(
            DocumentSegment.tenant_id == tenant_id,
            DocumentSegment.document_id == document_id,
            DocumentSegment.dataset_id == dataset_id,
        )

        if status is not None:
            query = query.where(DocumentSegment.status == status)

        if enabled is not None:
            query = query.where(DocumentSegment.enabled == enabled)

        return session.scalars(query).all()


class DatasetCollectionBindingService:
    @classmethod
    def get_dataset_collection_binding(
        cls, provider_name: str, model_name: str, session: Session, collection_type: str = "dataset"
    ) -> DatasetCollectionBinding:
        dataset_collection_binding = session.scalar(
            select(DatasetCollectionBinding)
            .where(
                DatasetCollectionBinding.provider_name == provider_name,
                DatasetCollectionBinding.model_name == model_name,
                DatasetCollectionBinding.type == collection_type,
            )
            .order_by(DatasetCollectionBinding.created_at)
            .limit(1)
        )

        if not dataset_collection_binding:
            dataset_collection_binding = DatasetCollectionBinding(
                provider_name=provider_name,
                model_name=model_name,
                collection_name=Dataset.gen_collection_name_by_id(str(uuid.uuid4())),
                type=collection_type,
            )
            session.add(dataset_collection_binding)
            session.flush()
        return dataset_collection_binding

    @classmethod
    def get_dataset_collection_binding_by_id_and_type(
        cls, collection_binding_id: str, session: Session, collection_type: str = "dataset"
    ) -> DatasetCollectionBinding:
        dataset_collection_binding = session.scalar(
            select(DatasetCollectionBinding)
            .where(
                DatasetCollectionBinding.id == collection_binding_id, DatasetCollectionBinding.type == collection_type
            )
            .order_by(DatasetCollectionBinding.created_at)
            .limit(1)
        )
        if not dataset_collection_binding:
            raise ValueError("Dataset collection binding not found")

        return dataset_collection_binding


class DatasetPermissionService:
    @classmethod
    def get_dataset_partial_member_list(cls, dataset_id, session: Session):
        user_list_query = session.scalars(
            select(
                DatasetPermission.account_id,
            ).where(DatasetPermission.dataset_id == dataset_id)
        ).all()

        return user_list_query

    @classmethod
    def update_partial_member_list(cls, tenant_id, dataset_id, user_list, session: Session):
        session.execute(delete(DatasetPermission).where(DatasetPermission.dataset_id == dataset_id))
        permissions = []
        for user in user_list:
            permission = DatasetPermission(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                account_id=user["user_id"],
            )
            permissions.append(permission)

        session.add_all(permissions)
        session.flush()

    @classmethod
    def check_permission(cls, user, dataset, requested_permission, requested_partial_member_list, *, session: Session):
        if not user.is_dataset_editor:
            raise NoPermissionError("User does not have permission to edit this dataset.")

        if user.is_dataset_operator and dataset.permission != requested_permission:
            raise NoPermissionError("Dataset operators cannot change the dataset permissions.")

        if user.is_dataset_operator and requested_permission == "partial_members":
            if not requested_partial_member_list:
                raise ValueError("Partial member list is required when setting to partial members.")

            local_member_list = cls.get_dataset_partial_member_list(dataset.id, session)
            request_member_list = [user["user_id"] for user in requested_partial_member_list]
            if set(local_member_list) != set(request_member_list):
                raise ValueError("Dataset operators cannot change the dataset permissions.")

    @classmethod
    def clear_partial_member_list(cls, dataset_id, session: Session):
        session.execute(delete(DatasetPermission).where(DatasetPermission.dataset_id == dataset_id))
        session.flush()

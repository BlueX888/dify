"""Session-owning adapters for document queries and existing document operations."""

import copy
import json
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, Literal, Protocol
from uuid import uuid4

from redis.exceptions import LockNotOwnedError
from sqlalchemy import asc, desc, func, select, update
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.exceptions import NotFound

from configs import dify_config
from core.errors.error import ModelCurrentlyNotSupportError, ProviderTokenNotInitError, QuotaExceededError
from core.model_manager import ModelManager
from core.rag.extractor.entity.extract_setting import WebsiteJobRef
from core.rag.index_processor.constant.built_in_field import BuiltInField
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from extensions.ext_redis import redis_client
from graphon.model_runtime.entities.model_entities import ModelType
from graphon.model_runtime.errors.invoke import InvokeAuthorizationError
from libs.pagination import paginate_query
from machinery.context import RequestContext
from models import Account, Dataset, Document, DocumentSegment, UploadFile
from models.dataset import DatasetPermissionEnum, DocumentPipelineExecutionLog
from models.enums import IndexingStatus
from repositories.knowledge.dataset_read_repository import (
    get_dataset_doc_form,
    get_document_hit_count,
    get_document_process_rule,
    get_document_segment_count,
    get_latest_dataset_process_rule,
)
from repositories.knowledge.dataset_repository import _get_dataset
from repositories.knowledge.document_indexing_outbox_repository import create_document_indexing_intents
from repositories.knowledge.document_repository import _get_document
from repositories.knowledge.metadata_repository import (
    lock_metadata_dataset,
    lock_metadata_document,
    lock_metadata_document_bindings,
)
from services.data_source.credential_gateway import (
    ActorDatasourceCredentialResolver,
    DatasourceCredentialError,
    DatasourceCredentialNotFoundError,
)
from services.enterprise import rbac_service
from services.errors.base import NoPermissionError
from services.errors.dataset import DatasetAccessDeniedError, DatasetNotFoundError
from services.errors.document import DocumentIndexingError
from services.file_service import FileService
from services.knowledge.dataset_read_service import (
    get_document_metadata_details,
    get_document_source_detail,
    load_document_detail,
    load_document_details,
)
from services.knowledge.dataset_service import DatasetService, DocumentService
from services.knowledge.documents.application import (
    DocumentIndexingStateError,
    DocumentInvalidActionError,
    DocumentListFilter,
    DocumentNotFoundError,
    DocumentProviderError,
    DocumentState,
    DocumentZip,
)
from services.knowledge.entities.document_creation import DocumentCreationDataset
from services.knowledge.entities.knowledge_entities import KnowledgeConfig
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef
from services.knowledge.summaries.adapters import SummaryIndexAdapter
from services.rbac import contracts as rbac_contracts
from services.vector_space_admission_service import get_vector_space_admission_error_fields
from tasks.generate_summary_index_task import generate_summary_index_task


@contextmanager
def _translate_errors() -> Generator[None]:
    try:
        yield
    except NotFound as error:
        raise DocumentNotFoundError(error.description) from error
    except NoPermissionError as error:
        raise DatasetAccessDeniedError() from error
    except DocumentIndexingError as error:
        raise DocumentIndexingStateError(str(error)) from error
    except (ProviderTokenNotInitError, InvokeAuthorizationError, DatasourceCredentialError) as error:
        raise DocumentProviderError(str(error), kind="uninitialized") from error
    except QuotaExceededError as error:
        raise DocumentProviderError(str(error), kind="quota") from error
    except ModelCurrentlyNotSupportError as error:
        raise DocumentProviderError(str(error), kind="unsupported") from error


def _require_dataset(session: Session, ref: DatasetRef) -> Dataset:
    dataset = _get_dataset(session, ref)
    if dataset is None:
        raise DatasetNotFoundError()
    return dataset


def _require_document(session: Session, ref: DocumentRef) -> Document:
    document = _get_document(session, ref)
    if document is None:
        raise DocumentNotFoundError("Document not found.")
    return document


def _state(document: Document) -> DocumentState:
    return DocumentState(
        id=document.id,
        archived=bool(document.archived),
        indexing_status=str(document.indexing_status),
        is_paused=bool(document.is_paused),
        data_source_type=str(document.data_source_type),
        doc_form=str(document.doc_form),
        need_summary=bool(document.need_summary),
    )


def _created_response(dataset: Dataset, documents: Sequence[Document], batch: str, session: Session) -> dict[str, Any]:
    return {
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "permission": dataset.permission,
            "data_source_type": dataset.data_source_type,
            "indexing_technique": dataset.indexing_technique,
            "created_by": dataset.created_by,
            "created_at": dataset.created_at,
        },
        "documents": load_document_details(documents, session=session),
        "batch": batch,
    }


def _indexing_status(document: Document, counts: tuple[int, int]) -> dict[str, Any]:
    return {
        "id": document.id,
        "processing_started_at": document.processing_started_at,
        "parsing_completed_at": document.parsing_completed_at,
        "cleaning_completed_at": document.cleaning_completed_at,
        "splitting_completed_at": document.splitting_completed_at,
        "completed_at": document.completed_at,
        "paused_at": document.paused_at,
        "error": document.error,
        "stopped_at": document.stopped_at,
        "indexing_status": IndexingStatus.PAUSED if document.is_paused else document.indexing_status,
        **get_vector_space_admission_error_fields(document.error),
        "completed_segments": counts[0],
        "total_segments": counts[1],
    }


class WebsiteJobResolver(Protocol):
    def resolve_job_ref(
        self, *, tenant_id: str, provider: str, job_id: str, source_handle: str | None = None
    ) -> WebsiteJobRef: ...


class SQLAlchemyDocumentOperations:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        actor_credentials: ActorDatasourceCredentialResolver,
        website_jobs: WebsiteJobResolver | None = None,
    ) -> None:
        self._sessions = session_factory
        self._actor_credentials = actor_credentials
        self._website_jobs = website_jobs

    def find_document(self, *, workspace_id: str, document_id: str) -> DocumentRef | None:
        with self._sessions() as session:
            document = session.scalar(
                select(Document).where(Document.id == document_id, Document.tenant_id == workspace_id)
            )
            return DatasetRef(workspace_id, document.dataset_id).document(document.id) if document else None

    def get_state(self, ref: DocumentRef) -> DocumentState | None:
        with self._sessions() as session:
            document = _get_document(session, ref)
            return _state(document) if document else None

    def get_states(self, ref: DatasetRef, document_ids: Sequence[str]) -> Sequence[DocumentState]:
        with self._sessions() as session:
            return tuple(
                _state(document) for document in DocumentService.get_documents_by_ids(ref, document_ids, session)
            )

    def get_process_rule(self, ref: DatasetRef | None) -> dict[str, Any]:
        result = copy.deepcopy(DocumentService.DEFAULT_RULES)
        if ref is not None:
            with self._sessions() as session:
                rule = get_latest_dataset_process_rule(_require_dataset(session, ref), session=session)
                if rule is not None:
                    result.update(mode=rule.mode, rules=rule.rules_dict)
        return result

    def list_documents(self, ref: DatasetRef, query: DocumentListFilter) -> dict[str, Any]:
        with self._sessions() as session:
            dataset = _require_dataset(session, ref)
            statement = select(Document).where(
                Document.dataset_id == ref.dataset_id, Document.tenant_id == ref.tenant_id
            )
            statement = DocumentService.apply_display_status_filter(statement, query.status)
            if query.search:
                statement = statement.where(Document.name.like(f"%{query.search}%"))
            direction = desc if query.sort.startswith("-") else asc
            sort = query.sort.removeprefix("-")
            if sort == "hit_count":
                hits = (
                    select(DocumentSegment.document_id, func.sum(DocumentSegment.hit_count).label("total_hit_count"))
                    .where(DocumentSegment.dataset_id == ref.dataset_id, DocumentSegment.tenant_id == ref.tenant_id)
                    .group_by(DocumentSegment.document_id)
                    .subquery()
                )
                statement = statement.outerjoin(hits, hits.c.document_id == Document.id).order_by(
                    direction(func.coalesce(hits.c.total_hit_count, 0)), direction(Document.position)
                )
            elif sort == "created_at":
                statement = statement.order_by(direction(Document.created_at), direction(Document.position))
            else:
                statement = statement.order_by(desc(Document.created_at), desc(Document.position))
            page = paginate_query(statement, session=session, page=query.page, per_page=query.limit, max_per_page=100)
            DocumentService.enrich_documents_with_summary_index_status(
                documents=page.items, dataset=dataset, tenant_id=ref.tenant_id, session=session
            )
            if query.fetch:
                counts = DocumentService.get_document_segment_counts(page.items, session=session)
                for document in page.items:
                    document.completed_segments, document.total_segments = counts.get(document.id, (0, 0))
            return {
                "data": load_document_details(page.items, session=session),
                "has_more": page.has_next,
                "limit": page.per_page,
                "total": page.total,
                "page": page.page,
            }

    def get_detail(self, ref: DocumentRef, *, metadata_only: bool) -> dict[str, Any]:
        with self._sessions() as session:
            document = _require_document(session, ref)
            result = {
                "id": document.id,
                "doc_type": document.doc_type,
                "doc_metadata": get_document_metadata_details(document, session=session),
            }
            if metadata_only:
                return result
            rule = get_document_process_rule(document, session=session)
            segment_count = get_document_segment_count(document, session=session)
            result.update(
                {
                    "position": document.position,
                    "data_source_type": document.data_source_type,
                    "dataset_process_rule_id": document.dataset_process_rule_id,
                    "name": document.name,
                    "created_from": document.created_from,
                    "created_by": document.created_by,
                    "tokens": document.tokens,
                    "indexing_status": document.indexing_status,
                    "indexing_latency": document.indexing_latency,
                    "error": document.error,
                    "enabled": document.enabled,
                    "disabled_by": document.disabled_by,
                    "archived": document.archived,
                    "display_status": document.display_status,
                    "doc_form": document.doc_form,
                    "doc_language": document.doc_language,
                }
            )
            result.update(
                {
                    "created_at": int(document.created_at.timestamp()) if document.created_at else None,
                    "completed_at": int(document.completed_at.timestamp()) if document.completed_at else None,
                    "updated_at": int(document.updated_at.timestamp()) if document.updated_at else None,
                    "disabled_at": int(document.disabled_at.timestamp()) if document.disabled_at else None,
                }
            )
            result.update(
                data_source_info=document.data_source_info_dict,
                data_source_detail_dict=get_document_source_detail(document, session=session),
                dataset_process_rule=DatasetService.get_process_rules(ref.dataset.dataset_id, session),
                document_process_rule=dict(rule.to_dict()) if rule else None,
                segment_count=segment_count,
                average_segment_length=(document.word_count or 0) // segment_count if segment_count else 0,
                hit_count=get_document_hit_count(document, session=session),
                need_summary=bool(document.need_summary),
            )
            return result

    def get_indexing_status(self, ref: DocumentRef) -> dict[str, Any]:
        with self._sessions() as session:
            document = _require_document(session, ref)
            counts = DocumentService.get_document_segment_counts([document], session=session)
            return _indexing_status(document, counts.get(document.id, (0, 0)))

    def get_batch_indexing_status(self, ref: DatasetRef, batch: str) -> dict[str, Any]:
        with self._sessions() as session:
            documents = session.scalars(
                select(Document).where(
                    Document.tenant_id == ref.tenant_id, Document.dataset_id == ref.dataset_id, Document.batch == batch
                )
            ).all()
            if not documents:
                raise DocumentNotFoundError("Documents not found.")
            counts = DocumentService.get_document_segment_counts(documents, session=session)
            return {"data": [_indexing_status(document, counts.get(document.id, (0, 0))) for document in documents]}

    def get_execution_log(self, ref: DocumentRef) -> dict[str, Any]:
        with self._sessions() as session:
            _require_document(session, ref)
            log = session.scalar(
                select(DocumentPipelineExecutionLog)
                .where(DocumentPipelineExecutionLog.document_id == ref.document_id)
                .order_by(DocumentPipelineExecutionLog.created_at.desc())
                .limit(1)
            )
            if log is None:
                return {}
            return {
                "datasource_info": json.loads(log.datasource_info),
                "datasource_type": log.datasource_type,
                "input_data": log.input_data,
                "datasource_node_id": log.datasource_node_id,
            }

    def get_summary_status(self, ref: DocumentRef) -> dict[str, Any]:
        with self._sessions() as session:
            _require_document(session, ref)
            return dict(
                SummaryIndexAdapter.get_document_summary_status_detail(
                    tenant_id=ref.dataset.tenant_id,
                    document_id=ref.document_id,
                    dataset_id=ref.dataset.dataset_id,
                    session=session,
                )
            )

    def get_download_url(self, ref: DocumentRef) -> str:
        with _translate_errors(), self._sessions() as session:
            return DocumentService.get_document_download_url(_require_document(session, ref), session)

    @contextmanager
    def build_download_zip(self, ref: DatasetRef, document_ids: Sequence[str]) -> Generator[DocumentZip]:
        with _translate_errors(), self._sessions() as session:
            files_by_document = DocumentService._get_upload_files_by_document_id_for_zip_download(
                dataset_id=ref.dataset_id, document_ids=document_ids, tenant_id=ref.tenant_id, session=session
            )
            files = [files_by_document[document_id] for document_id in document_ids]
            filename = DocumentService._generate_document_batch_download_zip_filename()
        # The read transaction is closed before reading storage or streaming the response.
        with FileService.build_upload_files_zip_tempfile(upload_files=files) as path:
            yield DocumentZip(path=path, filename=filename)

    @staticmethod
    def _actor(session: Session, context: RequestContext) -> Account:
        account = session.get(Account, context.account_id)
        if account is None:
            raise DatasetAccessDeniedError()
        account.set_tenant_id_with_session(context.active_workspace_id, session=session)
        return account

    def _creation_dataset(
        self, context: RequestContext, ref: DatasetRef, config: KnowledgeConfig
    ) -> DocumentCreationDataset:
        with self._sessions() as session:
            dataset = _require_dataset(session, ref)
            if not dataset.indexing_technique and not config.indexing_technique:
                raise ValueError("indexing_technique is required.")
            self._actor(session, context)
            DatasetService.check_doc_form(dataset, config.doc_form, session=session)
            return DocumentCreationDataset(
                dataset.tenant_id, dataset.indexing_technique, dataset.embedding_model, dataset.embedding_model_provider
            )

    def _resolve_website_job_ref(self, *, workspace_id: str, config: KnowledgeConfig) -> WebsiteJobRef | None:
        data_source = config.data_source
        if data_source is None or data_source.info_list.data_source_type != "website_crawl":
            return None
        website_info = data_source.info_list.website_info_list
        if website_info is None:
            raise ValueError("No website info list found.")
        if self._website_jobs is None:
            raise ValueError("Website source resolver is unavailable")
        return self._website_jobs.resolve_job_ref(
            tenant_id=workspace_id,
            provider=website_info.provider,
            job_id=website_info.job_id,
            source_handle=website_info.source_handle,
        )

    def _validate_actor_notion_credentials(self, *, context: RequestContext, config: KnowledgeConfig) -> None:
        """Authorize every actor-selected Notion credential before source persistence."""
        data_source = config.data_source
        if data_source is None or data_source.info_list.data_source_type != "notion_import":
            return
        notion_info_list = data_source.info_list.notion_info_list
        if not notion_info_list:
            raise ValueError("No notion info list found.")
        credential_ids = {notion_info.credential_id for notion_info in notion_info_list}
        if "" in credential_ids:
            raise DatasourceCredentialNotFoundError()
        for credential_id in sorted(credential_ids):
            self._actor_credentials.resolve(
                workspace_id=context.active_workspace_id,
                actor_id=context.account_id,
                credential_id=credential_id,
                provider="notion_datasource",
                plugin_id="langgenius/notion_datasource",
            )

    def _persist_creation(
        self,
        context: RequestContext,
        ref: DatasetRef,
        config: KnowledgeConfig,
        *,
        initialize: bool,
        website_job_ref: WebsiteJobRef | None,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Materialize the response and commit all document writes as one operation."""
        with self._sessions() as session:
            account = self._actor(session, context)
            dataset = (
                DocumentService.build_dataset_for_documents(ref, config, account, session)
                if initialize
                else _require_dataset(session, ref)
            )
            documents, batch, jobs = DocumentService.save_prepared_documents(
                dataset,
                config,
                account,
                session=session,
                website_job_ref=website_job_ref,
            )
            indexing_intent_ids = create_document_indexing_intents(session, jobs)
            if initialize:
                dataset.name = documents[0].name[:18] + "..."
                dataset.description = "useful for when you want to answer queries about the " + documents[0].name
                dataset.permission = (
                    DatasetPermissionEnum.ALL_TEAM if dify_config.RBAC_ENABLED else DatasetPermissionEnum.ONLY_ME
                )
                session.flush()
            result = _created_response(dataset, documents, batch, session)
            session.commit()
            return result, indexing_intent_ids

    def _run_creation(
        self,
        context: RequestContext,
        ref: DatasetRef,
        config: KnowledgeConfig,
        *,
        initialize: bool,
        website_job_ref: WebsiteJobRef | None,
    ) -> dict[str, Any]:
        lock = (
            nullcontext()
            if config.original_document_id
            else redis_client.lock(f"add_document_lock_dataset_id_{ref.dataset_id}", timeout=600)
        )
        result = None
        try:
            with lock:
                result, indexing_intent_ids = self._persist_creation(
                    context,
                    ref,
                    config,
                    initialize=initialize,
                    website_job_ref=website_job_ref,
                )
                # No session or ORM entity crosses the task boundary. Keep the
                # creation lock until committed documents have been dispatched.
                DocumentService.dispatch_document_indexing(indexing_intent_ids)
        except LockNotOwnedError:
            if config.original_document_id:
                raise
            # Preserve successful creation when the Redis lease expires on release.
            if result is None:
                if initialize:
                    raise
                with self._sessions() as session:
                    result = _created_response(
                        _require_dataset(session, ref), [], DocumentService.generate_document_batch(), session
                    )
        assert result is not None
        return result

    def create_documents(self, context: RequestContext, ref: DatasetRef, settings: Mapping[str, Any]) -> dict[str, Any]:
        config = KnowledgeConfig.model_validate(settings)
        DocumentService.document_create_args_validate(config)
        with _translate_errors():
            snapshot = self._creation_dataset(context, ref, config)
            config = DocumentService.prepare_document_creation(snapshot, config)
            self._validate_actor_notion_credentials(context=context, config=config)
            website_job_ref = self._resolve_website_job_ref(
                workspace_id=context.active_workspace_id,
                config=config,
            )
            return self._run_creation(
                context,
                ref,
                config,
                initialize=False,
                website_job_ref=website_job_ref,
            )

    def initialize_dataset(self, context: RequestContext, settings: Mapping[str, Any]) -> dict[str, Any]:
        config = KnowledgeConfig.model_validate(settings)
        with _translate_errors():
            if config.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                if config.embedding_model is None or config.embedding_model_provider is None:
                    raise ValueError(
                        "embedding model and embedding model provider are required for high quality indexing."
                    )
                ModelManager.for_tenant(tenant_id=context.active_workspace_id).get_model_instance(
                    tenant_id=context.active_workspace_id,
                    provider=config.embedding_model_provider,
                    model_type=ModelType.TEXT_EMBEDDING,
                    model=config.embedding_model,
                )
                config = config.model_copy(
                    update={
                        "is_multimodal": DatasetService.check_is_multimodal_model(
                            context.active_workspace_id, config.embedding_model_provider, config.embedding_model
                        )
                    }
                )
            DocumentService.document_create_args_validate(config)
            with self._sessions() as session:
                self._actor(session, context)
            config = DocumentService.prepare_document_creation(
                DocumentCreationDataset(
                    context.active_workspace_id,
                    config.indexing_technique,
                    config.embedding_model,
                    config.embedding_model_provider,
                ),
                config,
            )
            self._validate_actor_notion_credentials(context=context, config=config)
            website_job_ref = self._resolve_website_job_ref(
                workspace_id=context.active_workspace_id,
                config=config,
            )
            dataset_id = str(uuid4())
            result = self._run_creation(
                context,
                DatasetRef(context.active_workspace_id, dataset_id),
                config,
                initialize=True,
                website_job_ref=website_job_ref,
            )
            if dify_config.RBAC_ENABLED:
                rbac_service.RBACService.DatasetAccess.replace_whitelist(
                    context.active_workspace_id,
                    context.account_id,
                    dataset_id,
                    rbac_contracts.ReplaceMemberBindings(automatic_include_workspace_members=False),
                )
                rbac_service.try_sync_creator_access_policy_member_bindings(
                    context.active_workspace_id,
                    context.account_id,
                    rbac_contracts.RBACResourceType.DATASET,
                    dataset_id,
                )
            return result

    def delete_document(self, ref: DocumentRef) -> None:
        with _translate_errors(), self._sessions() as session:
            DocumentService.delete_document(_require_document(session, ref), session)

    def delete_documents(self, ref: DatasetRef, document_ids: Sequence[str]) -> None:
        with _translate_errors(), self._sessions() as session:
            dataset = _require_dataset(session, ref)
            DocumentService.delete_documents(
                ref, list(document_ids), get_dataset_doc_form(dataset, session=session), session
            )

    def update_metadata(self, ref: DocumentRef, *, doc_type: str, doc_metadata: Mapping[str, object]) -> None:
        """Replace document-type metadata without overwriting other metadata owners.

        Dataset metadata schema/binding writers and built-in metadata toggles
        serialize on the dataset lock. The document lock then protects the JSON
        projection. Only values owned by this endpoint are replaced; custom
        values with a current binding and enabled built-in values survive.
        """

        with self._sessions.begin() as session:
            dataset = lock_metadata_dataset(session, ref.dataset)
            if dataset is None:
                raise DatasetNotFoundError()
            document = lock_metadata_document(session, ref.dataset, ref.document_id)
            if document is None:
                raise DocumentNotFoundError("Document not found.")

            current = dict(document.doc_metadata or {})
            bound = lock_metadata_document_bindings(session, ref.dataset, ref.document_id)
            custom_values = {field.name: current[field.name] for _binding, field in bound if field.name in current}

            values = dict(doc_metadata)
            # Built-in metadata is exclusively maintained by the metadata
            # repository. A stale free-form document-type payload cannot revive
            # fields after the dataset-wide built-in toggle removes them.
            for field in BuiltInField:
                values.pop(field, None)
            if dataset.built_in_field_enabled:
                values.update({field: current[field] for field in BuiltInField if field in current})

            # A bound custom field wins a same-name document-type field. Custom
            # values may only be edited through the ID-based metadata API, which
            # keeps their JSON projection and binding in one transaction.
            values.update(custom_values)
            document.doc_type = doc_type
            document.doc_metadata = values

    def rename_document(self, ref: DocumentRef, name: str) -> dict[str, Any]:
        with self._sessions.begin() as session:
            dataset = lock_metadata_dataset(session, ref.dataset)
            if dataset is None:
                raise DatasetNotFoundError()
            document = lock_metadata_document(session, ref.dataset, ref.document_id)
            if document is None:
                raise DocumentNotFoundError("Document not found.")
            if dataset.built_in_field_enabled:
                document.doc_metadata = {**(document.doc_metadata or {}), BuiltInField.document_name: name}
            document.name = name
            if document.data_source_info_dict and (file_id := document.data_source_info_dict.get("upload_file_id")):
                session.execute(
                    update(UploadFile)
                    .where(UploadFile.id == file_id, UploadFile.tenant_id == ref.dataset.tenant_id)
                    .values(name=name)
                )
            session.flush()
            return load_document_detail(document, session=session)

    def pause_document(self, ref: DocumentRef, *, actor_id: str) -> None:
        with _translate_errors(), self._sessions() as session:
            DocumentService.pause_document(_require_document(session, ref), session, actor_id=actor_id)

    def recover_document(self, ref: DocumentRef) -> None:
        with _translate_errors(), self._sessions() as session:
            DocumentService.recover_document(_require_document(session, ref), session)

    def retry_documents(self, ref: DatasetRef, document_ids: Sequence[str], *, actor_id: str) -> None:
        with _translate_errors(), self._sessions() as session:
            documents = DocumentService.get_documents_by_ids(ref, document_ids, session)
            DocumentService.retry_document(ref.dataset_id, list(documents), session, actor_id=actor_id)

    def change_status(
        self,
        context: RequestContext,
        ref: DatasetRef,
        document_ids: Sequence[str],
        action: Literal["enable", "disable", "archive", "un_archive"],
    ) -> None:
        with _translate_errors(), self._sessions() as session:
            dataset = _require_dataset(session, ref)
            account = self._actor(session, context)
            documents = DocumentService.get_documents_by_ids(ref, document_ids, session)
            try:
                DocumentService.batch_update_document_status(
                    dataset, [doc.id for doc in documents], action, account, session
                )
            except (ValueError, DocumentIndexingError) as error:
                raise DocumentInvalidActionError(str(error)) from error

    def sync_website(self, ref: DocumentRef) -> None:
        with _translate_errors(), self._sessions() as session:
            DocumentService.sync_website_document(
                _require_dataset(session, ref.dataset), _require_document(session, ref), session
            )

    def require_summary_enabled(self, ref: DatasetRef) -> None:
        with self._sessions() as session:
            dataset = _require_dataset(session, ref)
            if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
                raise ValueError(
                    "Summary generation is only available for 'high_quality' indexing technique. "
                    f"Current indexing technique: {dataset.indexing_technique}"
                )
            if not dataset.summary_index_setting or not dataset.summary_index_setting.get("enable"):
                raise ValueError(
                    "Summary index is not enabled for this dataset. Please enable it in the dataset settings."
                )

    def enable_summary(self, ref: DatasetRef, document_ids: Sequence[str]) -> None:
        if document_ids:
            with self._sessions.begin() as session:
                session.execute(
                    update(Document)
                    .where(
                        Document.tenant_id == ref.tenant_id,
                        Document.dataset_id == ref.dataset_id,
                        Document.id.in_(document_ids),
                    )
                    .values(need_summary=True)
                )

    def dispatch_summary(self, ref: DocumentRef) -> None:
        with self._sessions() as session:
            dataset = _require_dataset(session, ref.dataset)
            parent_generation_id = dataset.indexing_request_id
        generate_summary_index_task.delay(ref.dataset.dataset_id, ref.document_id, None, parent_generation_id)

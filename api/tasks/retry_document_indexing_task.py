import logging
import time
from contextlib import suppress
from functools import partial

import click
from celery import shared_task
from sqlalchemy import select
from sqlalchemy.orm import Session

from configs import dify_config
from core.db.session_factory import session_factory
from enums import DeploymentEdition
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from libs.datetime_utils import naive_utc_now
from models import Account, Tenant
from models.dataset import Dataset, Document
from models.enums import IndexingStatus
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    SQLAlchemyDocumentIndexingOutboxRepository,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository, require_indexing_document
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.feature_service import FeatureService
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.cleanup import clean_document_for_reindex
from services.knowledge.indexing.adapters.execution import build_document_indexing_service
from services.knowledge.indexing.errors import (
    DatasetIndexingGenerationLostError,
    DocumentIndexingClaimLostError,
    DocumentIsPausedError,
)
from services.rag_pipeline.rag_pipeline import (
    PipelineIndexingSnapshotUnavailableError,
    RagPipelineService,
    pipeline_indexing_admission_error,
    run_durable_pipeline_indexing,
)

logger = logging.getLogger(__name__)


def _guard_retry_cleanup(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    indexing_request_id: str | None = None,
    claim_id: str | None = None,
) -> None:
    if claim_id is not None:
        dataset = session.scalar(
            select(Dataset)
            .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        if dataset is None:
            raise DatasetIndexingGenerationLostError(dataset_id)
        document = require_indexing_document(
            session,
            DatasetRef(tenant_id, dataset_id).document(document_id),
            claim_id=claim_id,
            lock=True,
        )
        if document.indexing_dataset_generation_id != dataset.indexing_request_id:
            raise DatasetIndexingGenerationLostError(dataset_id)
        if document.indexing_request_id != indexing_request_id or document.indexing_status != IndexingStatus.PARSING:
            raise DocumentIndexingClaimLostError()
        return

    request_condition = (
        Document.indexing_request_id == indexing_request_id
        if indexing_request_id is not None
        else Document.indexing_request_id.is_(None)
    )
    claim_condition = (
        Document.indexing_claim_id == claim_id if claim_id is not None else Document.indexing_claim_id.is_(None)
    )
    owned_document_id = session.scalar(
        select(Document.id)
        .where(
            Document.id == document_id,
            Document.dataset_id == dataset_id,
            Document.tenant_id == tenant_id,
            Document.indexing_status == IndexingStatus.PARSING,
            request_condition,
            claim_condition,
        )
        .with_for_update()
    )
    if owned_document_id is None:
        raise DocumentIndexingClaimLostError()


def _durable_retry_document_indexing(
    dataset_id: str,
    document_ids: list[str],
    user_id: str,
    *,
    indexing_request_id: str,
) -> None:
    from extensions.ext_application_services import application_services

    database_client = session_factory.get_session_maker()
    documents = SQLAlchemyDocumentRepository(session_factory=database_client)
    pipeline_generator = application_services().knowledge.pipeline_generator

    dataset: Dataset | None = None
    user: Account | None = None
    tenant_id: str | None = None
    context_error: str | None = None
    with session_factory.create_session() as session:
        dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
        if dataset is None:
            context_error = "Dataset not found"
        else:
            user = session.scalar(select(Account).where(Account.id == user_id).limit(1))
            tenant = session.scalar(select(Tenant).where(Tenant.id == dataset.tenant_id).limit(1))
            if user is None:
                context_error = "User not found"
            elif tenant is None:
                context_error = "Tenant not found"
            else:
                user.set_current_tenant_with_session(tenant, session=session)
                tenant_id = tenant.id
    if context_error is None:
        assert tenant_id is not None
        context_error = pipeline_indexing_admission_error(tenant_id)

    def admission_error(_count: int) -> str | None:
        return context_error

    try:
        while True:
            claims = documents.claim_indexing_request(
                indexing_request_id,
                expected_action=DocumentIndexingAction.RETRY.value,
                processing_started_at=naive_utc_now(),
                admission_error=admission_error,
            )
            if not claims:
                return
            (claim,) = claims
            assert dataset is not None
            assert user is not None
            try:
                if dataset.runtime_mode == "rag_pipeline":
                    run_durable_pipeline_indexing(
                        documents=documents,
                        claim=claim,
                        indexing_request_id=indexing_request_id,
                        user=user,
                        pipeline_generator=pipeline_generator,
                    )
                else:
                    build_document_indexing_service(
                        session_factory=database_client,
                        enforce_vector_space_admission=True,
                    ).restart_claimed(claim.ref, claim_id=claim.claim_id)
            except DatasetIndexingGenerationLostError:
                with suppress(DocumentIndexingClaimLostError):
                    documents.release_dataset_generation_owner(
                        claim.ref,
                        indexing_request_id=indexing_request_id,
                        claim_id=claim.claim_id,
                        dataset_generation_id=claim.dataset_generation_id,
                    )
                logger.info("Retry indexing parent generation was superseded: %s", claim.ref)
                return
            except DocumentIndexingClaimLostError:
                logger.info("Retry indexing claim was superseded: %s", claim.ref)
                return
            except DocumentIsPausedError:
                try:
                    documents.release_indexing_claim_for_pause(claim.ref, claim_id=claim.claim_id)
                except DocumentIndexingClaimLostError:
                    logger.info("Retry indexing pause arrived after ownership changed: %s", claim.ref)
                    return
            except PipelineIndexingSnapshotUnavailableError:
                with suppress(DocumentIndexingClaimLostError):
                    documents.release_dataset_generation_owner(
                        claim.ref,
                        indexing_request_id=indexing_request_id,
                        claim_id=claim.claim_id,
                        dataset_generation_id=claim.dataset_generation_id,
                    )
                logger.warning("Retry indexing snapshot is unavailable: %s", claim.ref)
                return
            except Exception as error:
                logger.exception("Durable retry failed, document_id: %s", claim.ref.document_id)
                try:
                    documents.fail_indexing(
                        claim.ref,
                        str(error),
                        claim_id=claim.claim_id,
                        dataset_generation_id=claim.dataset_generation_id,
                    )
                except DatasetIndexingGenerationLostError:
                    with suppress(DocumentIndexingClaimLostError):
                        documents.release_dataset_generation_owner(
                            claim.ref,
                            indexing_request_id=indexing_request_id,
                            claim_id=claim.claim_id,
                            dataset_generation_id=claim.dataset_generation_id,
                        )
                    return
                except DocumentIndexingClaimLostError:
                    logger.info("Retry indexing claim was superseded while recording failure: %s", claim.ref)
                    return
    finally:
        for document_id in document_ids:
            redis_client.delete(f"document_{document_id}_is_retried")


def _legacy_retry_document_indexing(dataset_id: str, document_ids: list[str], user_id: str) -> None:
    """
    Async process document
    :param dataset_id:
    :param document_ids:
    :param user_id:

    This path remains for callers that predate durable indexing requests.
    """
    from extensions.ext_application_services import application_services

    pipeline_generator = application_services().knowledge.pipeline_generator

    start_at = time.perf_counter()
    with session_factory.create_session() as session:
        try:
            dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
            if not dataset:
                logger.info(click.style(f"Dataset not found: {dataset_id}", fg="red"))
                return
            user = session.scalar(select(Account).where(Account.id == user_id).limit(1))
            if not user:
                logger.info(click.style(f"User not found: {user_id}", fg="red"))
                return
            tenant = session.scalar(select(Tenant).where(Tenant.id == dataset.tenant_id).limit(1))
            if not tenant:
                raise ValueError("Tenant not found")
            user.set_current_tenant_with_session(tenant, session=session)

            if dataset.runtime_mode == "rag_pipeline":
                if dataset.pipeline_id is None:
                    raise ValueError("Pipeline dataset is not bound to a pipeline")
                tenant_id = tenant.id
                pipeline_id = dataset.pipeline_id
                session.commit()
                outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=session_factory.get_session_maker())
                requests: dict[str, str] = {}
                for document_id in document_ids:
                    intent = outbox.get_or_create_pipeline_retry_intent(
                        tenant_id=tenant_id,
                        dataset_id=dataset_id,
                        pipeline_id=pipeline_id,
                        document_id=document_id,
                        actor_id=user_id,
                    )
                    if intent is not None:
                        requests[document_id] = intent.id
                    else:
                        redis_client.delete(f"document_{document_id}_is_retried")
                for document_id, request_id in requests.items():
                    _durable_retry_document_indexing(
                        dataset_id,
                        [document_id],
                        user_id,
                        indexing_request_id=request_id,
                    )
                return

            for document_id in document_ids:
                retry_indexing_cache_key = f"document_{document_id}_is_retried"
                try:
                    document = session.scalar(
                        select(Document)
                        .where(
                            Document.id == document_id,
                            Document.dataset_id == dataset_id,
                            Document.tenant_id == tenant.id,
                            Document.indexing_status == IndexingStatus.WAITING,
                            Document.indexing_request_id.is_(None),
                            Document.indexing_claim_id.is_(None),
                        )
                        .limit(1)
                    )
                    session.commit()
                    if not document:
                        logger.info(click.style(f"Document is not eligible for retry: {document_id}", fg="yellow"))
                        continue
                    # check document limit
                    if dify_config.DEPLOYMENT_EDITION == DeploymentEdition.CLOUD:
                        features = FeatureService.get_features(tenant.id)
                        try:
                            vector_space = features.vector_space
                            assert vector_space is not None
                            if 0 < vector_space.limit <= vector_space.size:
                                raise ValueError(
                                    "Your total number of documents plus the number of uploads have over the limit of "
                                    "your subscription."
                                )
                        except Exception as e:
                            current_document = session.scalar(
                                select(Document)
                                .where(
                                    Document.id == document_id,
                                    Document.dataset_id == dataset_id,
                                    Document.tenant_id == tenant.id,
                                    Document.indexing_status == IndexingStatus.WAITING,
                                    Document.indexing_request_id.is_(None),
                                    Document.indexing_claim_id.is_(None),
                                )
                                .with_for_update(of=Document)
                                .execution_options(populate_existing=True)
                                .limit(1)
                            )
                            if current_document:
                                current_document.indexing_status = IndexingStatus.ERROR
                                current_document.error = str(e)
                                current_document.stopped_at = naive_utc_now()
                                session.add(current_document)
                                session.commit()
                            continue

                    logger.info(click.style(f"Start retry document: {document_id}", fg="green"))
                    document = session.scalar(
                        select(Document)
                        .where(
                            Document.id == document_id,
                            Document.dataset_id == dataset_id,
                            Document.tenant_id == tenant.id,
                            Document.indexing_status == IndexingStatus.WAITING,
                            Document.indexing_request_id.is_(None),
                            Document.indexing_claim_id.is_(None),
                        )
                        .with_for_update(of=Document)
                        .limit(1)
                    )
                    if not document:
                        logger.info(click.style(f"Document not found: {document_id}", fg="yellow"))
                        continue
                    try:
                        doc_form = document.doc_form
                        document.indexing_status = IndexingStatus.PARSING
                        document.processing_started_at = naive_utc_now()
                        document_ref = DatasetRef(document.tenant_id, document.dataset_id).document(document.id)
                        runtime_mode = dataset.runtime_mode
                        # Claim this retry before vector I/O and release the row lock.
                        session.commit()

                        clean_document_for_reindex(
                            tenant_id=tenant.id,
                            dataset_id=dataset_id,
                            document_ids=(document_id,),
                            doc_form=doc_form,
                            new_session=session_factory.create_session,
                            uploads=SQLAlchemyKnowledgeUploadRepository(
                                session_factory=session_factory.get_session_maker()
                            ),
                            delete_blob=storage.delete,
                            logger=logger,
                            mutation_guard=partial(
                                _guard_retry_cleanup,
                                tenant_id=tenant.id,
                                dataset_id=dataset_id,
                                document_id=document_id,
                            ),
                        )

                        if runtime_mode == "rag_pipeline":
                            with session_factory.create_session() as rag_session:
                                rag_pipeline_service = RagPipelineService(rag_session)
                                rag_pipeline_service.retry_error_document(
                                    dataset,
                                    document,
                                    user,
                                    generator=pipeline_generator,
                                )
                        else:
                            indexing_service = build_document_indexing_service(
                                session_factory=session_factory.get_session_maker(),
                                enforce_vector_space_admission=True,
                            )
                            indexing_service.run([document_ref])
                    except Exception as ex:
                        session.rollback()
                        document = session.scalar(
                            select(Document)
                            .where(
                                Document.id == document_id,
                                Document.dataset_id == dataset_id,
                                Document.tenant_id == tenant.id,
                                Document.indexing_status == IndexingStatus.PARSING,
                                Document.indexing_request_id.is_(None),
                                Document.indexing_claim_id.is_(None),
                            )
                            .with_for_update(of=Document)
                            .execution_options(populate_existing=True)
                            .limit(1)
                        )
                        if document:
                            document.indexing_status = IndexingStatus.ERROR
                            document.error = str(ex)
                            document.stopped_at = naive_utc_now()
                            session.add(document)
                            session.commit()
                        logger.info(click.style(str(ex), fg="yellow"))
                        logger.exception("retry_document_indexing_task failed, document_id: %s", document_id)
                finally:
                    redis_client.delete(retry_indexing_cache_key)
            end_at = time.perf_counter()
            logger.info(click.style(f"Retry dataset: {dataset_id} latency: {end_at - start_at}", fg="green"))
        except Exception as e:
            logger.exception(
                "retry_document_indexing_task failed, dataset_id: %s, document_ids: %s", dataset_id, document_ids
            )
            raise e


@shared_task(queue="dataset")
def retry_document_indexing_task(
    dataset_id: str,
    document_ids: list[str],
    user_id: str,
    *,
    indexing_request_id: str | None = None,
) -> None:
    """Retry documents, using exact durable ownership when a request id is supplied."""
    if indexing_request_id is None:
        _legacy_retry_document_indexing(dataset_id, document_ids, user_id)
        return
    _durable_retry_document_indexing(
        dataset_id,
        document_ids,
        user_id,
        indexing_request_id=indexing_request_id,
    )

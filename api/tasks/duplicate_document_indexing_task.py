import logging
import time
from collections.abc import Callable, Sequence

import click
from celery import shared_task
from sqlalchemy import select

from configs import dify_config
from core.db.session_factory import session_factory
from core.entities.document_task import DocumentTask
from core.rag.pipeline.queue import TenantIsolatedTaskQueue
from enums import CloudPlan, DeploymentEdition
from extensions.ext_storage import storage
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset, Document
from repositories.knowledge.document_indexing_outbox_repository import DocumentIndexingAction
from repositories.knowledge.document_repository import ClaimedIndexingRequest, SQLAlchemyDocumentRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.feature_service import FeatureService
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.cleanup import clean_document_for_reindex
from services.knowledge.indexing.adapters.execution import build_document_indexing_service
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError, DocumentIsPausedError

logger = logging.getLogger(__name__)


@shared_task(queue="dataset")
def duplicate_document_indexing_task(dataset_id: str, document_ids: list):
    """
    Async process document
    :param dataset_id:
    :param document_ids:

    .. warning:: TO BE DEPRECATED
        This function will be deprecated and removed in a future version.
        Use normal_duplicate_document_indexing_task or priority_duplicate_document_indexing_task instead.

    Usage: duplicate_document_indexing_task.delay(dataset_id, document_ids)
    """
    logger.warning("duplicate document indexing task received: %s - %s", dataset_id, document_ids)
    _duplicate_document_indexing_task(dataset_id, document_ids)


def _duplicate_document_indexing_task_with_tenant_queue(
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    task_func: Callable[..., None],
    *,
    indexing_request_id: str | None = None,
):
    try:
        if indexing_request_id is None:
            _duplicate_document_indexing_task(dataset_id, document_ids)
        else:
            _duplicate_document_indexing_task(
                dataset_id,
                document_ids,
                indexing_request_id=indexing_request_id,
            )
    except Exception:
        logger.exception(
            "Error processing duplicate document indexing %s for tenant %s: %s",
            dataset_id,
            tenant_id,
            document_ids,
            exc_info=True,
        )
    finally:
        tenant_isolated_task_queue = TenantIsolatedTaskQueue(tenant_id, "duplicate_document_indexing")
        if indexing_request_id is not None:
            tenant_isolated_task_queue.clear_request_marker(indexing_request_id)

        # Check if there are waiting tasks in the queue
        # Use rpop to get the next task from the queue (FIFO order)
        next_tasks = tenant_isolated_task_queue.pull_tasks(count=dify_config.TENANT_ISOLATED_TASK_CONCURRENCY)

        logger.info("duplicate document indexing tenant isolation queue %s next tasks: %s", tenant_id, next_tasks)

        if next_tasks:
            for next_task in next_tasks:
                document_task = DocumentTask(**next_task)
                # Process the next waiting task
                # Keep the flag set to indicate a task is running
                tenant_isolated_task_queue.set_task_waiting_time()
                task_kwargs: dict[str, object] = {
                    "tenant_id": document_task.tenant_id,
                    "dataset_id": document_task.dataset_id,
                    "document_ids": document_task.document_ids,
                }
                if document_task.indexing_request_id is not None:
                    task_kwargs["indexing_request_id"] = document_task.indexing_request_id
                task_func.delay(**task_kwargs)  # type: ignore
        else:
            # No more waiting tasks, clear the flag
            tenant_isolated_task_queue.delete_task_key()


def _fail_claimed_requests(
    documents: SQLAlchemyDocumentRepository,
    claims: Sequence[ClaimedIndexingRequest],
    error: BaseException,
) -> None:
    for claim in claims:
        try:
            documents.fail_indexing(
                claim.ref,
                str(error),
                claim_id=claim.claim_id,
                dataset_generation_id=claim.dataset_generation_id,
            )
        except (DocumentIndexingClaimLostError, DocumentIsPausedError):
            logger.info("Duplicate indexing claim is no longer owned while recording failure: %s", claim.ref)


def _release_paused_claims(
    documents: SQLAlchemyDocumentRepository,
    claims: Sequence[ClaimedIndexingRequest],
) -> None:
    for claim in claims:
        try:
            documents.release_indexing_claim_for_pause(claim.ref, claim_id=claim.claim_id)
        except DocumentIndexingClaimLostError:
            logger.info("Duplicate indexing pause arrived after ownership changed: %s", claim.ref)


def _duplicate_document_indexing_task(
    dataset_id: str,
    document_ids: Sequence[str],
    *,
    indexing_request_id: str | None = None,
):
    start_at = time.perf_counter()
    claims: tuple[ClaimedIndexingRequest, ...] = ()
    documents_repository: SQLAlchemyDocumentRepository | None = None
    try:
        with session_factory.create_session() as session:
            dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
            if dataset is None:
                logger.info(click.style(f"Dataset not found: {dataset_id}", fg="red"))
                return
            dataset_ref = DatasetRef(dataset.tenant_id, dataset.id)

        def admission_error(_count: int) -> str | None:
            return None

        if dify_config.DEPLOYMENT_EDITION == DeploymentEdition.CLOUD:
            try:
                features = FeatureService.get_features(dataset.tenant_id)
                vector_space = features.vector_space
                assert vector_space is not None
                batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)

                def check_admission(count: int) -> str | None:
                    if features.billing.subscription.plan == CloudPlan.SANDBOX and count > 1:
                        return "Your current plan does not support batch upload, please upgrade your plan."
                    if count > batch_upload_limit:
                        return f"You have reached the batch upload limit of {batch_upload_limit}."
                    current = int(getattr(vector_space, "size", 0) or 0)
                    limit = int(getattr(vector_space, "limit", 0) or 0)
                    if limit > 0 and (current + count) > limit:
                        return (
                            "Your total number of documents plus the number of uploads have exceeded the limit of "
                            "your subscription."
                        )
                    return None

                admission_error = check_admission
            except Exception as error:
                quota_error = str(error)

                def reject_admission(_count: int) -> str:
                    return quota_error

                admission_error = reject_admission

        database_client = session_factory.get_session_maker()
        documents_repository = SQLAlchemyDocumentRepository(session_factory=database_client)

        def claim_next() -> tuple[ClaimedIndexingRequest, ...]:
            assert indexing_request_id is not None
            return documents_repository.claim_indexing_request(
                indexing_request_id,
                expected_action=DocumentIndexingAction.DUPLICATE.value,
                processing_started_at=naive_utc_now(),
                admission_error=admission_error,
            )

        if indexing_request_id is None:
            document_refs = documents_repository.claim_waiting_for_indexing(
                dataset_ref,
                document_ids,
                processing_started_at=naive_utc_now(),
                admission_error=admission_error,
            )
        else:
            claims = claim_next()
            document_refs = tuple(claim.ref for claim in claims)
            if claims:
                dataset_id = claims[0].ref.dataset.dataset_id
        if not document_refs:
            logger.info("No waiting duplicate documents admitted for dataset %s", dataset_id)
            return

        indexing_service = build_document_indexing_service(session_factory=database_client)
        if indexing_request_id is not None:
            while claims:
                (claim,) = claims
                indexing_service.restart_claimed(claim.ref, claim_id=claim.claim_id)
                claims = claim_next()
            end_at = time.perf_counter()
            logger.info(click.style(f"Processed dataset: {dataset_id} latency: {end_at - start_at}", fg="green"))
            return

        claimed_ids = [ref.document_id for ref in document_refs]

        with session_factory.create_session() as session:
            documents = list(
                session.scalars(
                    select(Document).where(
                        Document.id.in_(claimed_ids),
                        Document.dataset_id == dataset_id,
                        Document.tenant_id == dataset.tenant_id,
                    )
                ).all()
            )

            documents_by_form: dict[str | None, list[str]] = {}
            for document in documents:
                logger.info(click.style(f"Start process document: {document.id}", fg="green"))
                documents_by_form.setdefault(document.doc_form, []).append(document.id)

        uploads = SQLAlchemyKnowledgeUploadRepository(session_factory=database_client)
        for doc_form, owned_document_ids in documents_by_form.items():
            clean_document_for_reindex(
                tenant_id=dataset.tenant_id,
                dataset_id=dataset_id,
                document_ids=owned_document_ids,
                doc_form=doc_form,
                new_session=session_factory.create_session,
                uploads=uploads,
                delete_blob=storage.delete,
                logger=logger,
            )

        indexing_service.run(document_refs)
        end_at = time.perf_counter()
        logger.info(click.style(f"Processed dataset: {dataset_id} latency: {end_at - start_at}", fg="green"))
    except DocumentIsPausedError as ex:
        logger.info(click.style(str(ex), fg="yellow"))
        if documents_repository is not None:
            _release_paused_claims(documents_repository, claims)
    except DocumentIndexingClaimLostError:
        logger.info("Duplicate document indexing claim was superseded, dataset_id: %s", dataset_id)
    except Exception as error:
        logger.exception("duplicate_document_indexing_task failed, dataset_id: %s", dataset_id)
        if documents_repository is not None:
            _fail_claimed_requests(documents_repository, claims, error)


@shared_task(queue="dataset")
def normal_duplicate_document_indexing_task(
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    *,
    indexing_request_id: str | None = None,
):
    """
    Async process duplicate documents
    :param tenant_id:
    :param dataset_id:
    :param document_ids:

    Usage: normal_duplicate_document_indexing_task.delay(tenant_id, dataset_id, document_ids)
    """
    logger.info("normal duplicate document indexing task received: %s - %s - %s", tenant_id, dataset_id, document_ids)
    if indexing_request_id is None:
        _duplicate_document_indexing_task_with_tenant_queue(
            tenant_id, dataset_id, document_ids, normal_duplicate_document_indexing_task
        )
    else:
        _duplicate_document_indexing_task_with_tenant_queue(
            tenant_id,
            dataset_id,
            document_ids,
            normal_duplicate_document_indexing_task,
            indexing_request_id=indexing_request_id,
        )


@shared_task(queue="priority_dataset")
def priority_duplicate_document_indexing_task(
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    *,
    indexing_request_id: str | None = None,
):
    """
    Async process duplicate documents
    :param tenant_id:
    :param dataset_id:
    :param document_ids:

    Usage: priority_duplicate_document_indexing_task.delay(tenant_id, dataset_id, document_ids)
    """
    logger.info("priority duplicate document indexing task received: %s - %s - %s", tenant_id, dataset_id, document_ids)
    if indexing_request_id is None:
        _duplicate_document_indexing_task_with_tenant_queue(
            tenant_id, dataset_id, document_ids, priority_duplicate_document_indexing_task
        )
    else:
        _duplicate_document_indexing_task_with_tenant_queue(
            tenant_id,
            dataset_id,
            document_ids,
            priority_duplicate_document_indexing_task,
            indexing_request_id=indexing_request_id,
        )

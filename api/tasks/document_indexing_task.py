import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import click
from celery import current_app, shared_task
from sqlalchemy import select

from configs import dify_config
from core.db.session_factory import session_factory
from core.entities.document_task import DocumentTask
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.pipeline.queue import TenantIsolatedTaskQueue
from enums import CloudPlan, DeploymentEdition
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset, Document
from models.enums import IndexingStatus
from repositories.knowledge.document_indexing_outbox_repository import DocumentIndexingAction
from repositories.knowledge.document_repository import ClaimedIndexingRequest, SQLAlchemyDocumentRepository
from services.feature_service import FeatureService
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.execution import build_document_indexing_service
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError, DocumentIsPausedError
from tasks.generate_summary_index_task import generate_summary_index_task

logger = logging.getLogger(__name__)


def _allow_indexing(_count: int) -> None:
    return None


class CeleryTaskLike(Protocol):
    def delay(self, *args: Any, **kwargs: Any) -> Any: ...

    def apply_async(self, *args: Any, **kwargs: Any) -> Any: ...


@shared_task(queue="dataset")
def document_indexing_task(dataset_id: str, document_ids: list):
    """
    Async process document
    :param dataset_id:
    :param document_ids:

    .. warning:: TO BE DEPRECATED
        This function will be deprecated and removed in a future version.
        Use normal_document_indexing_task or priority_document_indexing_task instead.

    Usage: document_indexing_task.delay(dataset_id, document_ids)
    """
    logger.warning("document indexing legacy mode received: %s - %s", dataset_id, document_ids)
    _document_indexing(dataset_id, document_ids)


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
            logger.info("Indexing claim is no longer owned while recording failure: %s", claim.ref)


def _release_paused_claims(
    documents: SQLAlchemyDocumentRepository,
    claims: Sequence[ClaimedIndexingRequest],
) -> None:
    for claim in claims:
        try:
            documents.release_indexing_claim_for_pause(claim.ref, claim_id=claim.claim_id)
        except DocumentIndexingClaimLostError:
            logger.info("Indexing pause arrived after ownership changed: %s", claim.ref)


def _document_indexing(
    dataset_id: str,
    document_ids: Sequence[str],
    *,
    indexing_request_id: str | None = None,
):
    """
    Process document for tasks
    :param dataset_id:
    :param document_ids:

    Usage: _document_indexing(dataset_id, document_ids)
    """
    start_at = time.perf_counter()

    with session_factory.create_session() as session:
        dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
        if not dataset:
            logger.info(click.style(f"Dataset is not found: {dataset_id}", fg="yellow"))
            return
        dataset_ref = DatasetRef(dataset.tenant_id, dataset.id)
        legacy_dataset_generation_id = dataset.indexing_request_id

    database_client = session_factory.get_session_maker()
    documents = SQLAlchemyDocumentRepository(session_factory=database_client)
    admission_error: Callable[[int], str | None] = _allow_indexing

    # Resolve remote quota state before opening the short claim transaction.
    # The repository evaluates these limits against only the waiting rows it locks.
    if dify_config.DEPLOYMENT_EDITION == DeploymentEdition.CLOUD:
        try:
            features = FeatureService.get_features(dataset_ref.tenant_id)
            vector_space = features.vector_space
            assert vector_space is not None
            batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)

            def check_admission(count: int) -> str | None:
                if features.billing.subscription.plan == CloudPlan.SANDBOX and count > 1:
                    return "Your current plan does not support batch upload, please upgrade your plan."
                if count > batch_upload_limit:
                    return f"You have reached the batch upload limit of {batch_upload_limit}."
                if 0 < vector_space.limit <= vector_space.size:
                    return (
                        "Your total number of documents plus the number of uploads have over the limit of "
                        "your subscription."
                    )
                return None

            admission_error = check_admission
        except Exception as error:
            quota_error = str(error)

            def reject_admission(_count: int) -> str:
                return quota_error

            admission_error = reject_admission

    processing_started_at = naive_utc_now()
    claims: tuple[ClaimedIndexingRequest, ...] = ()

    def claim_next() -> tuple[ClaimedIndexingRequest, ...]:
        assert indexing_request_id is not None
        return documents.claim_indexing_request(
            indexing_request_id,
            expected_action=DocumentIndexingAction.CREATE.value,
            processing_started_at=naive_utc_now(),
            admission_error=admission_error,
        )

    if indexing_request_id is None:
        document_refs = documents.claim_waiting_for_indexing(
            dataset_ref,
            document_ids,
            processing_started_at=processing_started_at,
            admission_error=admission_error,
        )
    else:
        claims = claim_next()
        document_refs = tuple(claim.ref for claim in claims)
        if claims:
            dataset_id = claims[0].ref.dataset.dataset_id
    if not document_refs:
        logger.info("No waiting documents admitted for dataset %s", dataset_id)
        return
    claimed_document_ids: list[str] = []
    claimed_document_generations: dict[str, str | None] = {}

    # Execute indexing after the claim transaction has committed.
    has_error = False
    try:
        indexing_service = build_document_indexing_service(
            session_factory=database_client, enforce_vector_space_admission=True
        )
        if indexing_request_id is None:
            indexing_service.run(document_refs)
            claimed_document_ids.extend(ref.document_id for ref in document_refs)
            claimed_document_generations.update(
                (ref.document_id, legacy_dataset_generation_id) for ref in document_refs
            )
        else:
            while claims:
                (claim,) = claims
                claimed_document_ids.append(claim.ref.document_id)
                claimed_document_generations[claim.ref.document_id] = claim.dataset_generation_id
                if claim.recovery_required:
                    indexing_service.restart_claimed(claim.ref, claim_id=claim.claim_id)
                else:
                    indexing_service.run_claimed(claim.ref, claim_id=claim.claim_id)
                claims = claim_next()

        end_at = time.perf_counter()
        logger.info(click.style(f"Processed dataset: {dataset_id} latency: {end_at - start_at}", fg="green"))
    except DocumentIsPausedError as ex:
        logger.info(click.style(str(ex), fg="yellow"))
        _release_paused_claims(documents, claims)
        has_error = True
    except DocumentIndexingClaimLostError:
        logger.info("Document indexing claim was superseded, dataset_id: %s", dataset_id)
        has_error = True
    except Exception as error:
        logger.exception("Document indexing task failed, dataset_id: %s", dataset_id)
        _fail_claimed_requests(documents, claims, error)
        has_error = True

    if not has_error:
        with session_factory.create_session() as session:
            dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
            if not dataset:
                logger.warning("Dataset %s not found after indexing", dataset_id)
                return

            if dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
                summary_index_setting = dataset.summary_index_setting
                if summary_index_setting and summary_index_setting.get("enable"):
                    # Check each document's indexing status and trigger summary generation if completed

                    indexed_documents = list(
                        session.scalars(
                            select(Document).where(
                                Document.id.in_(claimed_document_ids), Document.dataset_id == dataset_id
                            )
                        ).all()
                    )

                    for document in indexed_documents:
                        if document:
                            logger.info(
                                "Checking document %s for summary generation: status=%s, doc_form=%s, need_summary=%s",
                                document.id,
                                document.indexing_status,
                                document.doc_form,
                                document.need_summary,
                            )
                            if (
                                document.indexing_status == IndexingStatus.COMPLETED
                                and document.doc_form != IndexStructureType.QA_INDEX
                                and document.need_summary is True
                            ):
                                try:
                                    generate_summary_index_task.delay(
                                        dataset.id,
                                        document.id,
                                        None,
                                        claimed_document_generations[document.id],
                                    )
                                    logger.info(
                                        "Queued summary index generation task for document %s in dataset %s "
                                        "after indexing completed",
                                        document.id,
                                        dataset.id,
                                    )
                                except Exception:
                                    logger.exception(
                                        "Failed to queue summary index generation task for document %s",
                                        document.id,
                                    )
                                    # Don't fail the entire indexing process if summary task queuing fails
                            else:
                                logger.info(
                                    "Skipping summary generation for document %s: "
                                    "status=%s, doc_form=%s, need_summary=%s",
                                    document.id,
                                    document.indexing_status,
                                    document.doc_form,
                                    document.need_summary,
                                )
                        else:
                            logger.warning("Document %s not found after indexing", document.id)
            else:
                logger.info(
                    "Summary index generation skipped for dataset %s: indexing_technique=%s (not 'high_quality')",
                    dataset.id,
                    dataset.indexing_technique,
                )


def _document_indexing_with_tenant_queue(
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    task_func: CeleryTaskLike,
    *,
    indexing_request_id: str | None = None,
) -> None:
    try:
        if indexing_request_id is None:
            _document_indexing(dataset_id, document_ids)
        else:
            _document_indexing(dataset_id, document_ids, indexing_request_id=indexing_request_id)
    except Exception:
        logger.exception(
            "Error processing document indexing %s for tenant %s: %s",
            dataset_id,
            tenant_id,
            document_ids,
            exc_info=True,
        )
    finally:
        tenant_isolated_task_queue = TenantIsolatedTaskQueue(tenant_id, "document_indexing")
        if indexing_request_id is not None:
            tenant_isolated_task_queue.clear_request_marker(indexing_request_id)

        # Check if there are waiting tasks in the queue
        # Use rpop to get the next task from the queue (FIFO order)
        next_tasks = tenant_isolated_task_queue.pull_tasks(count=dify_config.TENANT_ISOLATED_TASK_CONCURRENCY)

        logger.info("document indexing tenant isolation queue %s next tasks: %s", tenant_id, next_tasks)

        if next_tasks:
            with current_app.producer_or_acquire() as producer:  # type: ignore
                for next_task in next_tasks:
                    document_task = DocumentTask(**next_task)
                    # Keep the flag set to indicate a task is running
                    tenant_isolated_task_queue.set_task_waiting_time()
                    task_kwargs: dict[str, object] = {
                        "tenant_id": document_task.tenant_id,
                        "dataset_id": document_task.dataset_id,
                        "document_ids": document_task.document_ids,
                    }
                    if document_task.indexing_request_id is not None:
                        task_kwargs["indexing_request_id"] = document_task.indexing_request_id
                    task_func.apply_async(kwargs=task_kwargs, producer=producer)

        else:
            # No more waiting tasks, clear the flag
            tenant_isolated_task_queue.delete_task_key()


@shared_task(queue="dataset")
def normal_document_indexing_task(
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    *,
    indexing_request_id: str | None = None,
):
    """
    Async process document
    :param tenant_id:
    :param dataset_id:
    :param document_ids:

    Usage: normal_document_indexing_task.delay(tenant_id, dataset_id, document_ids)
    """
    logger.info("normal document indexing task received: %s - %s - %s", tenant_id, dataset_id, document_ids)
    if indexing_request_id is None:
        _document_indexing_with_tenant_queue(tenant_id, dataset_id, document_ids, normal_document_indexing_task)
    else:
        _document_indexing_with_tenant_queue(
            tenant_id,
            dataset_id,
            document_ids,
            normal_document_indexing_task,
            indexing_request_id=indexing_request_id,
        )


@shared_task(queue="priority_dataset")
def priority_document_indexing_task(
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    *,
    indexing_request_id: str | None = None,
):
    """
    Priority async process document
    :param tenant_id:
    :param dataset_id:
    :param document_ids:

    Usage: priority_document_indexing_task.delay(tenant_id, dataset_id, document_ids)
    """
    logger.info("priority document indexing task received: %s - %s - %s", tenant_id, dataset_id, document_ids)
    if indexing_request_id is None:
        _document_indexing_with_tenant_queue(tenant_id, dataset_id, document_ids, priority_document_indexing_task)
    else:
        _document_indexing_with_tenant_queue(
            tenant_id,
            dataset_id,
            document_ids,
            priority_document_indexing_task,
            indexing_request_id=indexing_request_id,
        )

import logging
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from itertools import islice
from typing import Any

import click
from celery import group, shared_task
from flask import current_app
from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from configs import dify_config
from core.app.apps.pipeline.pipeline_generator import PipelineGenerator
from core.app.entities.app_invoke_entities import RagPipelineGenerateEntity
from core.app.entities.rag_pipeline_invoke_entities import RagPipelineInvokeEntity
from core.db.session_factory import session_factory
from core.rag.pipeline.queue import TenantIsolatedTaskQueue
from extensions.ext_database import db
from libs.datetime_utils import naive_utc_now
from models import Account, Tenant
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    DocumentIndexingIntent,
    SQLAlchemyDocumentIndexingOutboxRepository,
)
from repositories.knowledge.document_repository import ClaimedIndexingRequest, SQLAlchemyDocumentRepository
from services.file_service import FileService
from services.knowledge.indexing.errors import (
    DatasetIndexingGenerationLostError,
    DocumentIndexingClaimLostError,
    DocumentIsPausedError,
)
from services.rag_pipeline.rag_pipeline import (
    PipelineIndexingSnapshotUnavailableError,
    pipeline_indexing_admission_error,
    run_durable_pipeline_indexing,
)

logger = logging.getLogger(__name__)

# Each entry is validated into RagPipelineInvokeEntity inside the worker.
_INVOKE_ENTITIES_ADAPTER = TypeAdapter(list[dict[str, Any]])


@dataclass(frozen=True, slots=True)
class _OwnedPipelineEntity:
    request_id: str
    claim: ClaimedIndexingRequest
    documents: SQLAlchemyDocumentRepository


def _claim_pipeline_entity(
    invoke_entity: RagPipelineInvokeEntity,
    entity: RagPipelineGenerateEntity,
) -> _OwnedPipelineEntity | None:
    """Resolve a trusted single-document RETRY intent and acquire its lease."""
    if entity.indexing_claim_id is not None:
        raise ValueError("Async pipeline payload must not carry a pre-issued claim")
    document_id = entity.document_id
    if document_id is None:
        raise ValueError("Published pipeline document is required")
    if entity.start_node_id is None:
        raise ValueError("Published pipeline start node is required")
    database_client = session_factory.get_session_maker()
    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=database_client)
    request_id = entity.indexing_request_id
    intent: DocumentIndexingIntent | None
    if request_id is None:
        intent = outbox.get_or_create_legacy_pipeline_intent(
            tenant_id=invoke_entity.tenant_id,
            dataset_id=entity.dataset_id,
            pipeline_id=invoke_entity.pipeline_id,
            document_id=document_id,
            actor_id=invoke_entity.user_id,
            datasource_type=str(entity.datasource_type),
            datasource_info=dict(entity.datasource_info),
            datasource_node_id=entity.start_node_id,
            batch=entity.batch,
            workflow_id=invoke_entity.workflow_id,
            original_document_id=entity.original_document_id,
        )
        if intent is None:
            logger.info("Skipped an unowned legacy pipeline delivery: document_id=%s", document_id)
            return None
        request_id = intent.id
    else:
        intent = outbox.get_pipeline_intent(
            request_id,
            tenant_id=invoke_entity.tenant_id,
            dataset_id=entity.dataset_id,
            pipeline_id=invoke_entity.pipeline_id,
            document_id=document_id,
            actor_id=invoke_entity.user_id,
            workflow_id=invoke_entity.workflow_id,
            datasource_type=str(entity.datasource_type),
            datasource_info=dict(entity.datasource_info),
            datasource_node_id=entity.start_node_id,
            batch=entity.batch,
            original_document_id=entity.original_document_id,
        )
    if (
        intent is None
        or intent.action != DocumentIndexingAction.RETRY
        or intent.tenant_id != invoke_entity.tenant_id
        or intent.dataset_id != entity.dataset_id
        or intent.document_ids != (document_id,)
        or intent.actor_id != invoke_entity.user_id
    ):
        raise ValueError("Pipeline indexing request does not own this delivery")

    documents = SQLAlchemyDocumentRepository(session_factory=database_client)
    admission_error = pipeline_indexing_admission_error(invoke_entity.tenant_id)
    claims = documents.claim_indexing_request(
        request_id,
        expected_action=DocumentIndexingAction.RETRY.value,
        processing_started_at=naive_utc_now(),
        admission_error=lambda _count: admission_error,
    )
    if not claims:
        return None
    (claim,) = claims
    if (
        claim.ref.dataset.tenant_id != invoke_entity.tenant_id
        or claim.ref.dataset.dataset_id != entity.dataset_id
        or claim.ref.document_id != document_id
    ):
        raise DocumentIndexingClaimLostError()
    return _OwnedPipelineEntity(request_id=request_id, claim=claim, documents=documents)


def _release_pipeline_entity(owner: _OwnedPipelineEntity) -> None:
    with suppress(DocumentIndexingClaimLostError):
        owner.documents.release_dataset_generation_owner(
            owner.claim.ref,
            indexing_request_id=owner.request_id,
            claim_id=owner.claim.claim_id,
            dataset_generation_id=owner.claim.dataset_generation_id,
        )


def _fail_pipeline_entity(owner: _OwnedPipelineEntity, error: Exception) -> None:
    try:
        owner.documents.fail_indexing(
            owner.claim.ref,
            str(error),
            claim_id=owner.claim.claim_id,
            dataset_generation_id=owner.claim.dataset_generation_id,
        )
    except DatasetIndexingGenerationLostError:
        _release_pipeline_entity(owner)
    except DocumentIndexingClaimLostError:
        pass


def chunked(iterable: Sequence, size: int):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])


@shared_task(queue="pipeline")
def rag_pipeline_run_task(
    rag_pipeline_invoke_entities_file_id: str,
    tenant_id: str,
):
    """
    Async Run rag pipeline task using regular priority queue.

    :param rag_pipeline_invoke_entities_file_id: File ID containing serialized RAG pipeline invoke entities
    :param tenant_id: Tenant ID for the pipeline execution
    """
    from extensions.ext_application_services import application_services

    # run with threading, thread pool size is 10

    try:
        pipeline_generator = application_services().knowledge.pipeline_generator
        start_at = time.perf_counter()
        rag_pipeline_invoke_entities_content = FileService(db.engine).get_file_content(
            rag_pipeline_invoke_entities_file_id
        )
        rag_pipeline_invoke_entities = _INVOKE_ENTITIES_ADAPTER.validate_json(rag_pipeline_invoke_entities_content)

        logger.info("tenant %s received %d rag pipeline invoke entities", tenant_id, len(rag_pipeline_invoke_entities))

        # Get Flask app object for thread context
        flask_app = current_app._get_current_object()  # type: ignore

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for rag_pipeline_invoke_entity in rag_pipeline_invoke_entities:
                # Submit task to thread pool with Flask app
                future = executor.submit(
                    run_single_rag_pipeline_task,
                    rag_pipeline_invoke_entity,
                    flask_app,
                    pipeline_generator=pipeline_generator,
                )
                futures.append(future)

            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()  # This will raise any exceptions that occurred in the thread
                except Exception:
                    logging.exception("Error in pipeline task")
        end_at = time.perf_counter()
        logging.info(
            click.style(
                f"tenant_id: {tenant_id}, Rag pipeline run completed. Latency: {end_at - start_at}s", fg="green"
            )
        )
    except Exception:
        logging.exception(click.style(f"Error running rag pipeline, tenant_id: {tenant_id}", fg="red"))
        raise
    finally:
        tenant_isolated_task_queue = TenantIsolatedTaskQueue(tenant_id, "pipeline")

        # Check if there are waiting tasks in the queue
        # Use rpop to get the next task from the queue (FIFO order)
        next_file_ids = tenant_isolated_task_queue.pull_tasks(count=dify_config.TENANT_ISOLATED_TASK_CONCURRENCY)
        logger.info("rag pipeline tenant isolation queue %s next files: %s", tenant_id, next_file_ids)

        if next_file_ids:
            for batch in chunked(next_file_ids, 100):
                jobs = []
                for next_file_id in batch:
                    tenant_isolated_task_queue.set_task_waiting_time()

                    file_id = (
                        next_file_id.decode("utf-8") if isinstance(next_file_id, (bytes, bytearray)) else next_file_id
                    )

                    jobs.append(
                        rag_pipeline_run_task.s(
                            rag_pipeline_invoke_entities_file_id=file_id,
                            tenant_id=tenant_id,
                        )
                    )

                if jobs:
                    group(jobs).apply_async()
        else:
            # No more waiting tasks, clear the flag
            tenant_isolated_task_queue.delete_task_key()
        file_service = FileService(db.engine)
        file_service.delete_file(rag_pipeline_invoke_entities_file_id)
        db.session.close()


def run_single_rag_pipeline_task(
    rag_pipeline_invoke_entity: Mapping[str, Any], flask_app, *, pipeline_generator: PipelineGenerator
):
    """Claim and replay one persisted pipeline source through its durable owner."""
    with flask_app.app_context():
        owner: _OwnedPipelineEntity | None = None
        try:
            invoke_entity = RagPipelineInvokeEntity.model_validate(rag_pipeline_invoke_entity)
            entity = RagPipelineGenerateEntity.model_validate(invoke_entity.application_generate_entity)
            owner = _claim_pipeline_entity(invoke_entity, entity)
            if owner is None:
                return
            with sessionmaker(db.engine, expire_on_commit=False).begin() as session:
                account = session.scalar(select(Account).where(Account.id == invoke_entity.user_id).limit(1))
                tenant = session.scalar(select(Tenant).where(Tenant.id == invoke_entity.tenant_id).limit(1))
                if account is None:
                    raise ValueError(f"Account {invoke_entity.user_id} not found")
                if tenant is None:
                    raise ValueError(f"Tenant {invoke_entity.tenant_id} not found")
                account.set_current_tenant_with_session(tenant, session=session)

            run_durable_pipeline_indexing(
                documents=owner.documents,
                claim=owner.claim,
                indexing_request_id=owner.request_id,
                user=account,
                pipeline_generator=pipeline_generator,
            )
        except DatasetIndexingGenerationLostError:
            assert owner is not None
            _release_pipeline_entity(owner)
        except DocumentIndexingClaimLostError:
            logger.info("Pipeline indexing claim was superseded: %s", owner.claim.ref if owner else "unclaimed")
        except DocumentIsPausedError:
            assert owner is not None
            with suppress(DocumentIndexingClaimLostError):
                owner.documents.release_indexing_claim_for_pause(owner.claim.ref, claim_id=owner.claim.claim_id)
        except PipelineIndexingSnapshotUnavailableError:
            assert owner is not None
            _release_pipeline_entity(owner)
            logger.warning("Pipeline indexing snapshot is unavailable: %s", owner.claim.ref)
        except Exception as error:
            if owner is not None:
                _fail_pipeline_entity(owner, error)
            logging.exception("Error in pipeline task")
            raise

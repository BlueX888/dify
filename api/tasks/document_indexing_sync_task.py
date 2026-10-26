import logging
import threading
import time
from types import TracebackType
from typing import Protocol, Self
from uuid import uuid4

import click
import httpx
from celery import shared_task
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db.session_factory import get_session_maker, session_factory
from core.rag.extractor.notion_extractor import NotionExtractor
from extensions.application_services.data_sources import build_data_source_credentials
from extensions.ext_storage import storage
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository, renew_indexing_claim
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.document_sync import (
    NOTION_SYNC_HEARTBEAT_SECONDS,
    NOTION_SYNC_LEASE_SECONDS,
    ClaimedNotionDocument,
)
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.cleanup import clean_document_for_reindex
from services.knowledge.indexing.adapters.execution import build_document_indexing_service
from services.knowledge.indexing.adapters.sources import NotionSourceResolver
from services.knowledge.indexing.errors import (
    DocumentIndexingClaimLostError,
    DocumentIsPausedError,
)
from services.knowledge.indexing.estimate import StoredSource

logger = logging.getLogger(__name__)

_NOTION_SYNC_RETRY_DELAY_SECONDS = NOTION_SYNC_LEASE_SECONDS + 5
_NOTION_SYNC_MAX_RETRIES = 3


class _NotionSyncRetryableError(RuntimeError):
    """A transient, pre-publication sync failure safe for lease-based retry."""


class _NotionSyncClaimReleaseError(_NotionSyncRetryableError):
    """An unchanged sync could not durably release its otherwise healthy claim."""


class _NotionSyncPreflightTransientError(_NotionSyncRetryableError):
    """A read-only Notion version probe failed before index cleanup started."""


class _NotionSyncLeaseRenewer(Protocol):
    def renew_notion_sync(self, claim: ClaimedNotionDocument) -> bool: ...


class _NotionSyncLeaseHeartbeat:
    """Keep one persisted claim alive without holding a transaction across I/O."""

    def __init__(
        self,
        repository: _NotionSyncLeaseRenewer,
        claim: ClaimedNotionDocument,
        *,
        interval_seconds: float = NOTION_SYNC_HEARTBEAT_SECONDS,
    ) -> None:
        self._repository = repository
        self._claim = claim
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"notion-sync-lease-{claim.ref.document_id}",
            daemon=True,
        )

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join(timeout=min(self._interval_seconds, 1.0))
        if self._thread.is_alive():
            logger.warning("Notion sync lease heartbeat did not stop before task exit: %s", self._claim.ref.document_id)

    def ensure_owned(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise DocumentIndexingClaimLostError() from failure

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                if not self._repository.renew_notion_sync(self._claim):
                    raise DocumentIndexingClaimLostError()
            except DocumentIndexingClaimLostError as error:
                with self._failure_lock:
                    self._failure = error
                return
            except Exception:
                # A transient database failure is not proof that another
                # worker owns the claim. Keep retrying; every destructive or
                # publishing phase still validates the persisted claim.
                logger.exception("Notion sync lease heartbeat failed: %s", self._claim.ref.document_id)


@shared_task(
    queue="dataset",
    autoretry_for=(_NotionSyncRetryableError,),
    retry_kwargs={
        "countdown": _NOTION_SYNC_RETRY_DELAY_SECONDS,
        "max_retries": _NOTION_SYNC_MAX_RETRIES,
    },
)
def document_indexing_sync_task(dataset_id: str, document_id: str):
    """
    Async update document
    :param dataset_id:
    :param document_id:

    Usage: document_indexing_sync_task.delay(dataset_id, document_id)
    """
    logger.info(click.style(f"Start sync document: {document_id}", fg="green"))
    start_at = time.perf_counter()
    with session_factory.create_session() as session:
        dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
        if not dataset:
            logger.info(click.style(f"Dataset not found: {dataset_id}", fg="red"))
            return
        dataset_ref = DatasetRef(dataset.tenant_id, dataset.id)

    database_client = get_session_maker()
    documents = SQLAlchemyDocumentRepository(session_factory=database_client)
    claim = documents.claim_notion_sync(
        dataset_ref.document(document_id),
        claim_id=str(uuid4()),
    )
    if claim is None:
        logger.info(click.style(f"Document {document_id} is not eligible for sync, skipping", fg="yellow"))
        return

    try:
        with _NotionSyncLeaseHeartbeat(documents, claim) as heartbeat:
            credentials = build_data_source_credentials(database_client=database_client)
            source = NotionSourceResolver(
                actor_credentials=credentials.actor,
                stored_credentials=credentials.stored,
            ).resolve(
                StoredSource(
                    document_ref=claim.ref,
                    source_type="notion_import",
                    source_info=claim.source_info,
                    document_model=claim.index_type,
                )
            )
            heartbeat.ensure_owned()
            notion_info = source.notion_info
            assert notion_info is not None
            notion_token = notion_info.notion_access_token
            assert notion_token is not None

            loader = NotionExtractor(
                notion_token_loader=lambda: notion_token,
                notion_workspace_id=notion_info.notion_workspace_id or "",
                notion_obj_id=notion_info.notion_obj_id,
                notion_page_type=notion_info.notion_page_type,
                notion_access_token=notion_token,
                tenant_id=notion_info.tenant_id,
            )
            try:
                last_edited_time = loader.get_notion_last_edited_time()
            except httpx.RequestError as error:
                # A transport failure while reading the remote version says
                # nothing about the last successfully published index. Keep
                # it healthy and retry after this claim's lease can expire.
                raise _NotionSyncPreflightTransientError(
                    f"Failed to read Notion source version for document {document_id}"
                ) from error
            heartbeat.ensure_owned()
            if last_edited_time == claim.page_edited_time and not claim.recovery_required:
                try:
                    documents.release_notion_sync(claim)
                except Exception as error:
                    # Releasing an unchanged page is lease bookkeeping, not an
                    # indexing failure. Keep the last healthy index published
                    # and let a durable retry reclaim the expired lease.
                    raise _NotionSyncClaimReleaseError(
                        f"Failed to release unchanged Notion sync claim for document {document_id}"
                    ) from error
                logger.info(click.style(f"Document {document_id} content unchanged, skipping sync", fg="yellow"))
                return

            logger.info(click.style(f"Document {document_id} content changed, starting sync", fg="green"))
            dataset, _ = documents.get_indexing_models(claim.ref, claim_id=claim.claim_id)

            def guard_claim(session: Session) -> None:
                renew_indexing_claim(session, claim.ref, claim_id=claim.claim_id)

            # Persist the resumable phase before the first destructive side
            # effect. From this point onward failures describe a potentially
            # changed index and must follow the normal ERROR/recovery path.
            heartbeat.ensure_owned()
            documents.prepare_notion_sync(
                claim,
                processing_started_at=naive_utc_now(),
            )
            clean_document_for_reindex(
                tenant_id=claim.ref.dataset.tenant_id,
                dataset_id=dataset.id,
                document_ids=[claim.ref.document_id],
                doc_form=claim.index_type,
                new_session=session_factory.create_session,
                uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=database_client),
                delete_blob=storage.delete,
                logger=logger,
                mutation_guard=guard_claim,
            )
            heartbeat.ensure_owned()
            logger.info(click.style(f"Cleaned vector index for document {document_id}", fg="green"))

            indexing_service = build_document_indexing_service(session_factory=database_client)
            indexing_service.run_claimed(claim.ref, claim_id=claim.claim_id)
            end_at = time.perf_counter()
            logger.info(
                click.style(f"Sync completed for document {document_id} latency: {end_at - start_at}", fg="green")
            )
    except DocumentIsPausedError as ex:
        documents.release_notion_sync(claim)
        logger.info(click.style(str(ex), fg="yellow"))
    except DocumentIndexingClaimLostError:
        logger.info("Document sync claim was lost before completion: %s", document_id)
    except _NotionSyncRetryableError:
        logger.exception("Retryable Notion sync preflight failed for document_id: %s", document_id)
        raise
    except Exception as error:
        logger.exception("document_indexing_sync_task failed for document_id: %s", document_id)
        documents.fail_indexing(
            claim.ref,
            str(error),
            claim_id=claim.claim_id,
            dataset_generation_id=claim.dataset_generation_id,
        )

import logging
import time

import click
from celery import shared_task

from configs import dify_config
from core.db.session_factory import session_factory
from enums import DeploymentEdition
from extensions.ext_redis import redis_client
from libs.datetime_utils import naive_utc_now
from repositories.knowledge.document_indexing_outbox_repository import (
    DocumentIndexingAction,
    SQLAlchemyDocumentIndexingOutboxRepository,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.feature_service import FeatureService
from services.knowledge.indexing.adapters.execution import build_document_indexing_service
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError, DocumentIsPausedError
from services.knowledge.indexing.website_sync_hint import delete_website_sync_hint_if_owned

logger = logging.getLogger(__name__)


def _clear_website_sync_hint(*, document_id: str, indexing_request_id: str) -> None:
    try:
        delete_website_sync_hint_if_owned(
            redis_client,
            document_id=document_id,
            owner_id=indexing_request_id,
        )
    except Exception:
        # The database request and claim are authoritative; a cache outage must
        # not make completed indexing retry or overwrite a newer hint owner.
        logger.warning("Failed to clear website sync hint, document_id=%s", document_id, exc_info=True)


def _website_sync_admission_error(tenant_id: str) -> str | None:
    if dify_config.DEPLOYMENT_EDITION != DeploymentEdition.CLOUD:
        return None
    try:
        vector_space = FeatureService.get_features(tenant_id).vector_space
        if vector_space is None:
            raise ValueError("Vector space entitlement is unavailable.")
        if 0 < vector_space.limit <= vector_space.size:
            raise ValueError(
                "Your total number of documents plus the number of uploads have over the limit of your subscription."
            )
    except Exception as error:
        return str(error)
    return None


@shared_task(queue="dataset")
def sync_website_document_indexing_task(
    dataset_id: str,
    document_id: str,
    *,
    indexing_request_id: str | None = None,
) -> None:
    """Rebuild one website document while owning its exact durable generation.

    A rolling-deployment message without a request token may create or reuse a
    durable request only after the repository locks and validates the complete
    persisted website-document owner chain.
    """
    started_at = time.perf_counter()
    database_client = session_factory.get_session_maker()
    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=database_client)
    if indexing_request_id is None:
        intent = outbox.get_or_create_legacy_website_sync_intent(
            dataset_id=dataset_id,
            document_id=document_id,
        )
        if intent is None:
            logger.info(
                "Legacy website sync message is not eligible for bridging, dataset_id=%s document_id=%s",
                dataset_id,
                document_id,
            )
            return
        indexing_request_id = intent.id
    else:
        intent = outbox.get_intent(indexing_request_id)
    if intent is None or intent.action != DocumentIndexingAction.WEBSITE_SYNC:
        logger.info("No website sync intent found, indexing_request_id=%s", indexing_request_id)
        return
    if len(intent.document_ids) != 1:
        logger.error("Invalid website sync intent payload, indexing_request_id=%s", indexing_request_id)
        return
    (owned_document_id,) = intent.document_ids
    if not outbox.request_is_current(intent):
        outbox.complete(intent)
        _clear_website_sync_hint(document_id=owned_document_id, indexing_request_id=indexing_request_id)
        logger.info("Website sync request was superseded, indexing_request_id=%s", indexing_request_id)
        return

    admission_error = _website_sync_admission_error(intent.tenant_id)
    documents = SQLAlchemyDocumentRepository(session_factory=database_client)
    claims = documents.claim_indexing_request(
        indexing_request_id,
        expected_action=DocumentIndexingAction.WEBSITE_SYNC.value,
        processing_started_at=naive_utc_now(),
        admission_error=lambda _count: admission_error,
    )
    if not claims:
        # An admission failure or a terminal document consumes the request.
        # Do not clear the hint while a duplicate delivery observes the active
        # owner of this same request.
        if outbox.get_intent(indexing_request_id) is None:
            _clear_website_sync_hint(document_id=owned_document_id, indexing_request_id=indexing_request_id)
        return

    (claim,) = claims
    logger.info(click.style(f"Start sync website document: {claim.ref.document_id}", fg="green"))
    try:
        build_document_indexing_service(session_factory=database_client).restart_claimed(
            claim.ref,
            claim_id=claim.claim_id,
        )
    except DocumentIndexingClaimLostError:
        logger.info("Website sync indexing claim was superseded: %s", claim.ref)
    except DocumentIsPausedError:
        try:
            documents.release_indexing_claim_for_pause(claim.ref, claim_id=claim.claim_id)
        except DocumentIndexingClaimLostError:
            logger.info("Website sync claim was superseded while recording pause: %s", claim.ref)
    except Exception as error:
        logger.exception("Website sync failed, document_id=%s", claim.ref.document_id)
        try:
            documents.fail_indexing(
                claim.ref,
                str(error),
                claim_id=claim.claim_id,
                dataset_generation_id=claim.dataset_generation_id,
            )
        except DocumentIndexingClaimLostError:
            logger.info("Website sync claim was superseded while recording failure: %s", claim.ref)
    finally:
        _clear_website_sync_hint(
            document_id=claim.ref.document_id,
            indexing_request_id=indexing_request_id,
        )
        logger.info(
            click.style(
                f"Sync document: {claim.ref.document_id} latency: {time.perf_counter() - started_at}",
                fg="green",
            )
        )

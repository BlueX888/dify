import logging
import time

import click
from celery import shared_task
from sqlalchemy import select

from core.db.session_factory import session_factory
from libs.datetime_utils import naive_utc_now
from models.dataset import Document
from repositories.knowledge.document_indexing_outbox_repository import DocumentIndexingAction
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.execution import build_document_indexing_service
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError, DocumentIsPausedError

logger = logging.getLogger(__name__)


@shared_task(queue="dataset")
def recover_document_indexing_task(dataset_id: str, document_id: str, *, indexing_request_id: str | None = None):
    """
    Async recover document
    :param dataset_id:
    :param document_id:

    Usage: recover_document_indexing_task.delay(dataset_id, document_id)
    """
    logger.info(click.style(f"Recover document: {document_id}", fg="green"))
    start_at = time.perf_counter()

    if indexing_request_id is not None:
        documents = SQLAlchemyDocumentRepository(session_factory=session_factory.get_session_maker())
        claims = documents.claim_indexing_request(
            indexing_request_id,
            expected_action=DocumentIndexingAction.RECOVER.value,
            processing_started_at=naive_utc_now(),
            admission_error=lambda _count: None,
        )
        for claim in claims:
            try:
                indexing_service = build_document_indexing_service(session_factory=session_factory.get_session_maker())
                if claim.recovery_required or claim.resume_status in {"parsing", "cleaning", "splitting"}:
                    indexing_service.restart_claimed(claim.ref, claim_id=claim.claim_id)
                elif claim.resume_status == "indexing":
                    indexing_service.run_in_indexing_status(claim.ref, claim_id=claim.claim_id)
                else:
                    indexing_service.run_claimed(claim.ref, claim_id=claim.claim_id)
            except DocumentIsPausedError:
                try:
                    documents.release_indexing_claim_for_pause(claim.ref, claim_id=claim.claim_id)
                except DocumentIndexingClaimLostError:
                    logger.info("Recovery pause arrived after ownership changed, document_id=%s", claim.ref.document_id)
            except DocumentIndexingClaimLostError:
                logger.info("Recovery interrupted or superseded, document_id=%s", claim.ref.document_id)
            except Exception:
                logger.exception("Recovery failed, document_id=%s", claim.ref.document_id)
                raise
        return

    with session_factory.create_session() as session:
        document = session.scalar(
            select(Document)
            .where(
                Document.id == document_id,
                Document.dataset_id == dataset_id,
                Document.indexing_request_id.is_(None),
                Document.indexing_claim_id.is_(None),
            )
            .limit(1)
        )

        if not document:
            logger.info(click.style(f"Document not found: {document_id}", fg="red"))
            return

        try:
            indexing_service = build_document_indexing_service(session_factory=session_factory.get_session_maker())
            document_ref = DatasetRef(document.tenant_id, document.dataset_id).document(document.id)
            status = document.indexing_status
            session.commit()
            if status in {"waiting", "parsing"}:
                indexing_service.run([document_ref])
            elif status in {"cleaning", "splitting"}:
                indexing_service.run_in_splitting_status(document_ref)
            elif status == "indexing":
                indexing_service.run_in_indexing_status(document_ref)
            end_at = time.perf_counter()
            logger.info(click.style(f"Processed document: {document.id} latency: {end_at - start_at}", fg="green"))
        except DocumentIsPausedError as ex:
            logger.info(click.style(str(ex), fg="yellow"))
        except Exception:
            logger.exception("recover_document_indexing_task failed, document_id: %s", document_id)

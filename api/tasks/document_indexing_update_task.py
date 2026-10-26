import logging
import time

import click
from celery import shared_task
from sqlalchemy import select

from core.db.session_factory import session_factory
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from extensions.ext_storage import storage
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset, Document
from models.enums import IndexingStatus
from repositories.knowledge.document_indexing_outbox_repository import DocumentIndexingAction
from repositories.knowledge.document_repository import ClaimedIndexingRequest, SQLAlchemyDocumentRepository
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.cleanup import clean_document_for_reindex
from services.knowledge.indexing.adapters.execution import build_document_indexing_service
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError, DocumentIsPausedError
from tasks.generate_summary_index_task import generate_summary_index_task

logger = logging.getLogger(__name__)


@shared_task(queue="dataset")
def document_indexing_update_task(
    dataset_id: str,
    document_id: str,
    *,
    indexing_request_id: str | None = None,
):
    """
    Async update document
    :param dataset_id:
    :param document_id:

    Usage: document_indexing_update_task.delay(dataset_id, document_id)
    """
    logger.info(click.style(f"Start update document: {document_id}", fg="green"))
    start_at = time.perf_counter()

    has_error = False
    tenant_id: str | None = None
    dataset_generation_id: str | None = None
    claim: ClaimedIndexingRequest | None = None
    documents_repository: SQLAlchemyDocumentRepository | None = None
    try:
        database_client = session_factory.get_session_maker()
        if indexing_request_id is not None:
            documents_repository = SQLAlchemyDocumentRepository(session_factory=database_client)
            claims = documents_repository.claim_indexing_request(
                indexing_request_id,
                expected_action=DocumentIndexingAction.UPDATE.value,
                processing_started_at=naive_utc_now(),
                admission_error=lambda _count: None,
            )
            if not claims:
                logger.info("No current update request admitted for document %s", document_id)
                return
            (claim,) = claims
            tenant_id = claim.ref.dataset.tenant_id
            dataset_generation_id = claim.dataset_generation_id
            dataset_id = claim.ref.dataset.dataset_id
            document_id = claim.ref.document_id
            indexing_service = build_document_indexing_service(session_factory=database_client)
            indexing_service.restart_claimed(claim.ref, claim_id=claim.claim_id)
        else:
            with session_factory.create_session() as session:
                dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
                if not dataset:
                    return
                tenant_id = dataset.tenant_id
                dataset_generation_id = dataset.indexing_request_id
                document = session.scalar(
                    select(Document)
                    .where(
                        Document.id == document_id,
                        Document.dataset_id == dataset_id,
                        Document.tenant_id == dataset.tenant_id,
                        Document.indexing_status == IndexingStatus.WAITING,
                        Document.indexing_claim_id.is_(None),
                        Document.indexing_request_id.is_(None),
                    )
                    .with_for_update(of=Document)
                    .limit(1)
                )

                if not document:
                    logger.info(click.style(f"Document not found: {document_id}", fg="red"))
                    return

                document.indexing_status = IndexingStatus.PARSING
                document.processing_started_at = naive_utc_now()
                document_ref = DatasetRef(document.tenant_id, document.dataset_id).document(document.id)
                doc_form = document.doc_form
                session.commit()

                clean_document_for_reindex(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset_id,
                    document_ids=(document_id,),
                    doc_form=doc_form,
                    new_session=session_factory.create_session,
                    uploads=SQLAlchemyKnowledgeUploadRepository(session_factory=database_client),
                    delete_blob=storage.delete,
                    logger=logger,
                )

                indexing_service = build_document_indexing_service(session_factory=database_client)
                indexing_service.run([document_ref])

        end_at = time.perf_counter()
        logger.info(click.style(f"update document: {document_id} latency: {end_at - start_at}", fg="green"))
    except DocumentIsPausedError as ex:
        logger.info(click.style(str(ex), fg="yellow"))
        if claim is not None and documents_repository is not None:
            try:
                documents_repository.release_indexing_claim_for_pause(claim.ref, claim_id=claim.claim_id)
            except DocumentIndexingClaimLostError:
                logger.info("Document update indexing claim was superseded while recording pause: %s", claim.ref)
        has_error = True
    except DocumentIndexingClaimLostError:
        logger.info("Document update indexing claim was superseded, document_id: %s", document_id)
        has_error = True
    except Exception as error:
        logger.exception("document_indexing_update_task failed, document_id: %s", document_id)
        if claim is not None and documents_repository is not None:
            try:
                documents_repository.fail_indexing(
                    claim.ref,
                    str(error),
                    claim_id=claim.claim_id,
                    dataset_generation_id=claim.dataset_generation_id,
                )
            except DocumentIndexingClaimLostError:
                logger.info("Document update indexing claim was superseded while recording failure: %s", claim.ref)
        elif tenant_id is not None:
            with session_factory.create_session() as session:
                failed_document = session.scalar(
                    select(Document).where(
                        Document.id == document_id,
                        Document.dataset_id == dataset_id,
                        Document.tenant_id == tenant_id,
                        Document.indexing_status == IndexingStatus.PARSING,
                    )
                )
                if failed_document is not None:
                    failed_document.indexing_status = IndexingStatus.ERROR
                    failed_document.error = str(error)
                    failed_document.stopped_at = naive_utc_now()
                    session.commit()
        has_error = True

    if has_error:
        return

    # Trigger summary index generation for the updated document if enabled.
    # Only generate for high_quality indexing technique and when summary_index_setting is enabled.
    with session_factory.create_session() as session:
        dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).limit(1))
        if not dataset:
            logger.warning("Dataset %s not found after update indexing", dataset_id)
            return
        document = session.scalar(
            select(Document)
            .where(
                Document.id == document_id,
                Document.dataset_id == dataset_id,
                Document.tenant_id == dataset.tenant_id,
            )
            .limit(1)
        )

        if dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            summary_index_setting = dataset.summary_index_setting
            if summary_index_setting and summary_index_setting.get("enable"):
                if (
                    document
                    and document.indexing_status == IndexingStatus.COMPLETED
                    and document.doc_form != IndexStructureType.QA_INDEX
                    and document.need_summary is True
                ):
                    try:
                        generate_summary_index_task.delay(dataset.id, document.id, None, dataset_generation_id)
                        logger.info(
                            "Queued summary index generation task for document %s in dataset %s "
                            "after update indexing completed",
                            document.id,
                            dataset.id,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to queue summary index generation task for document %s after update",
                            document.id,
                        )

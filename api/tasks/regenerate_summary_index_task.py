"""Task for regenerating summary indexes when dataset settings change."""

import logging
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager

import click
from celery import shared_task
from sqlalchemy import select

from core.credit_usage import CreditUsageCreatedBy
from core.db.session_factory import session_factory
from core.model_context import with_credit_usage_created_by
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from models.dataset import Dataset, DocumentSegment, DocumentSegmentSummary
from models.dataset import Document as DatasetDocument
from models.enums import SegmentStatus, SummaryStatus
from repositories.knowledge.dataset_indexing_outbox_repository import (
    SQLAlchemyDatasetIndexingOutboxRepository,
    lock_dataset_indexing_execution_shared,
    lock_dataset_indexing_generation_shared,
)
from services.knowledge.summaries.adapters import SummaryIndexAdapter

logger = logging.getLogger(__name__)


@contextmanager
def _lock_current_dataset_generation(
    dataset_id: str,
    indexing_request_id: str | None,
    *,
    indexing_intent_id: str | None = None,
    indexing_claim_id: str | None = None,
    tenant_id: str | None = None,
) -> Generator[Dataset | None]:
    """Hold a shared generation lock across one summary's external work.

    Summary writer registration takes its own Dataset shared lock, so keeping
    this guard in a separate session avoids releasing it when the adapter
    commits its caller session and remains compatible with the writer lock.
    """
    with session_factory.create_session() as guard_session:
        execution_values = (indexing_intent_id, indexing_claim_id, tenant_id)
        if any(value is not None for value in execution_values):
            if indexing_request_id is None or any(value is None for value in execution_values):
                raise ValueError("A durable summary rebuild requires generation, intent, claim, and tenant ownership")
            assert indexing_intent_id is not None
            assert indexing_claim_id is not None
            assert tenant_id is not None
            yield lock_dataset_indexing_execution_shared(
                guard_session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                generation_id=indexing_request_id,
                intent_id=indexing_intent_id,
                claim_id=indexing_claim_id,
            )
            return
        yield lock_dataset_indexing_generation_shared(
            guard_session,
            dataset_id=dataset_id,
            generation_id=indexing_request_id,
        )


@shared_task(queue="dataset_summary")
@with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
def regenerate_summary_index_task(
    dataset_id: str,
    regenerate_reason: str = "summary_model_changed",
    regenerate_vectors_only: bool = False,
    indexing_request_id: str | None = None,
    *,
    indexing_intent_id: str | None = None,
    indexing_claim_id: str | None = None,
    tenant_id: str | None = None,
) -> bool:
    """
    Regenerate summary indexes for all documents in a dataset.

    This task is triggered when:
    1. summary_index_setting model changes (regenerate_reason="summary_model_changed")
       - Regenerates summary content and vectors for all existing summaries
    2. embedding_model changes (regenerate_reason="embedding_model_changed")
       - Keeps valid summary text and regenerates only its vector
       - When automatic summaries are enabled, also takes over missing/pending text
         for documents whose ``need_summary`` intent remains enabled

    Args:
        dataset_id: Dataset ID
        regenerate_reason: Reason for regeneration ("summary_model_changed" or "embedding_model_changed")
        regenerate_vectors_only: If True, retain valid text while converging all required summary vectors and,
            when automatic summaries and the document intent are enabled, generate missing/pending text before
            success.
        indexing_request_id: Exact durable configuration generation for embedding-model rebuilds
    """
    if indexing_intent_id is None and indexing_claim_id is None and tenant_id is None:
        repository = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=session_factory.get_session_maker())
        intent_id = repository.find_current_intent_id(
            dataset_id=dataset_id,
            generation_id=indexing_request_id,
        )
        if intent_id is not None:
            from tasks.dataset_indexing_outbox_task import enqueue_dataset_indexing_intent

            enqueue_dataset_indexing_intent(intent_id)
            return False
        if indexing_request_id is not None:
            return False

    logger.info(
        click.style(
            f"Start regenerate summary index for dataset {dataset_id}, reason: {regenerate_reason}",
            fg="green",
        )
    )
    start_at = time.perf_counter()

    try:
        with session_factory.create_session() as session:
            with _lock_current_dataset_generation(
                dataset_id,
                indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
                tenant_id=tenant_id,
            ) as dataset:
                if dataset is None:
                    logger.info(
                        "Skipping unowned summary rebuild, dataset_id=%s, request_id=%s",
                        dataset_id,
                        indexing_request_id,
                    )
                    return False

                # Only regenerate summary index for high_quality indexing technique
                if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
                    logger.info(
                        click.style(
                            f"Skipping summary regeneration for dataset {dataset_id}: "
                            f"indexing_technique is {dataset.indexing_technique}, not 'high_quality'",
                            fg="cyan",
                        )
                    )
                    return True

                # Check if summary index is enabled (only for summary_model change)
                # For embedding_model change, we still re-vectorize existing summaries even if setting is disabled
                dataset_tenant_id = dataset.tenant_id
                summary_index_setting = dataset.summary_index_setting
                if not regenerate_vectors_only:
                    # For summary_model change, require summary_index_setting to be enabled
                    if not summary_index_setting or not summary_index_setting.get("enable"):
                        logger.info(
                            click.style(
                                f"Summary index is disabled for dataset {dataset_id}",
                                fg="cyan",
                            )
                        )
                        return True

            total_segments_processed = 0
            total_segments_failed = 0

            if regenerate_vectors_only:
                # A Dataset representation rebuild owns both parts of summary
                # convergence. Existing text keeps its content revision and only
                # publishes a new vector generation. When automatic summaries
                # and the current Document.need_summary intent are enabled, this
                # owner also fills missing/pending text before the Dataset intent
                # can be acknowledged.
                auto_generate_missing = bool(summary_index_setting and summary_index_setting.get("enable"))
                segment_summary_pairs = session.execute(
                    select(DocumentSegment, DocumentSegmentSummary)
                    .outerjoin(
                        DocumentSegmentSummary,
                        (DocumentSegment.id == DocumentSegmentSummary.chunk_id)
                        & (DocumentSegmentSummary.dataset_id == dataset_id),
                    )
                    .join(
                        DatasetDocument,
                        DocumentSegment.document_id == DatasetDocument.id,
                    )
                    .where(
                        DocumentSegment.dataset_id == dataset_id,
                        DocumentSegment.tenant_id == dataset_tenant_id,
                        DocumentSegment.status == "completed",  # Segment must be completed
                        DocumentSegment.enabled.is_(True),
                        DatasetDocument.enabled.is_(True),  # Document must be enabled
                        DatasetDocument.tenant_id == dataset_tenant_id,
                        DatasetDocument.archived.is_(False),  # Document must not be archived
                        DatasetDocument.doc_form != IndexStructureType.QA_INDEX,  # Skip qa_model documents
                    )
                    .order_by(DocumentSegment.document_id.asc(), DocumentSegment.position.asc())
                ).all()

                if not segment_summary_pairs:
                    logger.info(
                        click.style(
                            f"No eligible segments found for summary convergence in dataset {dataset_id}",
                            fg="cyan",
                        )
                    )
                    return True

                logger.info(
                    "Found %s segments for summary convergence in dataset %s, auto_generate_missing=%s",
                    len(segment_summary_pairs),
                    dataset_id,
                    auto_generate_missing,
                )

                # Group by document for logging
                segments_by_document = defaultdict(list)
                for segment, summary_record in segment_summary_pairs:
                    segments_by_document[segment.document_id].append((segment, summary_record))

                logger.info(
                    "Segments grouped into %s documents for summary convergence",
                    len(segments_by_document),
                )

                for document_id, segment_summary_pairs in segments_by_document.items():
                    logger.info(
                        "Converging summaries for %s segments in document %s",
                        len(segment_summary_pairs),
                        document_id,
                    )

                    for segment, summary_record in segment_summary_pairs:
                        try:
                            action: str | None = None
                            generation_dataset: Dataset | None = None
                            generation_segment: DocumentSegment | None = None
                            with _lock_current_dataset_generation(
                                dataset_id,
                                indexing_request_id,
                                indexing_intent_id=indexing_intent_id,
                                indexing_claim_id=indexing_claim_id,
                                tenant_id=tenant_id,
                            ) as current_dataset:
                                if current_dataset is None:
                                    return False
                                generation_dataset = current_dataset
                                current_document = session.scalar(
                                    select(DatasetDocument)
                                    .where(
                                        DatasetDocument.id == segment.document_id,
                                        DatasetDocument.dataset_id == current_dataset.id,
                                        DatasetDocument.tenant_id == current_dataset.tenant_id,
                                        DatasetDocument.enabled.is_(True),
                                        DatasetDocument.archived.is_(False),
                                        DatasetDocument.doc_form != IndexStructureType.QA_INDEX,
                                    )
                                    .with_for_update()
                                    .execution_options(populate_existing=True)
                                )
                                if current_document is None:
                                    session.rollback()
                                    continue
                                current_segment = session.scalar(
                                    select(DocumentSegment)
                                    .where(
                                        DocumentSegment.id == segment.id,
                                        DocumentSegment.dataset_id == current_dataset.id,
                                        DocumentSegment.tenant_id == current_dataset.tenant_id,
                                        DocumentSegment.document_id == current_document.id,
                                        DocumentSegment.status == SegmentStatus.COMPLETED,
                                        DocumentSegment.enabled.is_(True),
                                    )
                                    .with_for_update()
                                    .execution_options(populate_existing=True)
                                )
                                if current_segment is None:
                                    session.rollback()
                                    continue
                                generation_segment = current_segment
                                current_summary = session.scalar(
                                    select(DocumentSegmentSummary)
                                    .where(
                                        DocumentSegmentSummary.chunk_id == current_segment.id,
                                        DocumentSegmentSummary.dataset_id == current_dataset.id,
                                    )
                                    .limit(1)
                                    .with_for_update()
                                    .execution_options(populate_existing=True)
                                )

                                if current_summary is not None and not current_summary.enabled:
                                    # Explicit disable is stronger than automatic
                                    # summary generation or an embedding rebuild.
                                    session.rollback()
                                    continue

                                if current_summary is not None and (current_summary.summary_content or "").strip():
                                    # The text revision remains valid. The node,
                                    # durable writer, and parent generation own
                                    # the new vector publication independently.
                                    summary_record = current_summary
                                    summary_record.status = SummaryStatus.GENERATING
                                    summary_record.error = None
                                    session.commit()
                                    action = "vectorize"
                                elif auto_generate_missing and current_document.need_summary:
                                    # The adapter owns row creation, content
                                    # revision, LLM CAS, and vector publication.
                                    # Retained content intentionally bypasses the
                                    # document intent and remains vector-only.
                                    session.rollback()
                                    action = "generate"
                                else:
                                    session.rollback()
                                    continue
                            # The adapter registers its durable writer first, then
                            # takes its own generation guard across remote I/O. Do
                            # not retain this outer SHARE lock: a queued MySQL
                            # UPDATE would otherwise sit between the two SHAREs.
                            assert generation_dataset is not None
                            assert generation_segment is not None
                            if action == "vectorize":
                                assert summary_record is not None
                                completed = SummaryIndexAdapter.vectorize_summary(
                                    summary_record,
                                    generation_segment,
                                    generation_dataset,
                                    session=session,
                                    indexing_request_id=indexing_request_id,
                                    indexing_intent_id=indexing_intent_id,
                                    indexing_claim_id=indexing_claim_id,
                                )
                            elif action == "generate":
                                completed = (
                                    SummaryIndexAdapter.generate_and_vectorize_summary(
                                        generation_segment,
                                        generation_dataset,
                                        summary_index_setting or {},
                                        session=session,
                                        indexing_request_id=indexing_request_id,
                                        indexing_intent_id=indexing_intent_id,
                                        indexing_claim_id=indexing_claim_id,
                                    )
                                    is not None
                                )
                            else:
                                raise AssertionError("Unreachable summary convergence action")
                            if not completed:
                                if indexing_intent_id is not None:
                                    raise RuntimeError("Summary convergence lost rebuild ownership")
                                continue
                            session.commit()
                            total_segments_processed += 1

                        except Exception as e:
                            logger.error(
                                "Failed to converge summary for segment %s: %s",
                                segment.id,
                                str(e),
                                exc_info=True,
                            )
                            total_segments_failed += 1
                            session.rollback()
                            continue

            else:
                # For summary_model change: require document indexing_status == "completed"
                # Get all documents with completed indexing status
                dataset_documents = session.scalars(
                    select(DatasetDocument).where(
                        DatasetDocument.dataset_id == dataset_id,
                        DatasetDocument.indexing_status == "completed",
                        DatasetDocument.enabled.is_(True),
                        DatasetDocument.archived.is_(False),
                    )
                ).all()

                if not dataset_documents:
                    logger.info(
                        click.style(
                            f"No documents found for summary regeneration in dataset {dataset_id}",
                            fg="cyan",
                        )
                    )
                    return True

                logger.info(
                    "Found %s documents for summary regeneration in dataset %s",
                    len(dataset_documents),
                    dataset_id,
                )

                for dataset_document in dataset_documents:
                    # Skip qa_model documents
                    if dataset_document.doc_form == IndexStructureType.QA_INDEX:
                        continue

                    try:
                        # Get all segments with existing summaries
                        segments = session.scalars(
                            select(DocumentSegment)
                            .join(
                                DocumentSegmentSummary,
                                DocumentSegment.id == DocumentSegmentSummary.chunk_id,
                            )
                            .where(
                                DocumentSegment.document_id == dataset_document.id,
                                DocumentSegment.dataset_id == dataset_id,
                                DocumentSegment.status == "completed",
                                DocumentSegment.enabled.is_(True),
                                DocumentSegmentSummary.dataset_id == dataset_id,
                            )
                            .order_by(DocumentSegment.position.asc())
                        ).all()

                        if not segments:
                            continue

                        logger.info(
                            "Regenerating summaries for %s segments in document %s",
                            len(segments),
                            dataset_document.id,
                        )

                        for segment in segments:
                            existing_summary_record: DocumentSegmentSummary | None = None
                            try:
                                with _lock_current_dataset_generation(
                                    dataset_id,
                                    indexing_request_id,
                                    indexing_intent_id=indexing_intent_id,
                                    indexing_claim_id=indexing_claim_id,
                                    tenant_id=tenant_id,
                                ) as current_dataset:
                                    if current_dataset is None:
                                        return False
                                    # Get existing summary record
                                    existing_summary_record = session.scalar(
                                        select(DocumentSegmentSummary)
                                        .where(
                                            DocumentSegmentSummary.chunk_id == segment.id,
                                            DocumentSegmentSummary.dataset_id == dataset_id,
                                        )
                                        .limit(1)
                                    )

                                    if not existing_summary_record:
                                        logger.warning("Summary record not found for segment %s, skipping", segment.id)
                                        continue
                                    generation_dataset = current_dataset

                                # Regenerate both summary content and vectors (for summary_model change)
                                generated_summary = SummaryIndexAdapter.generate_and_vectorize_summary(
                                    segment,
                                    generation_dataset,
                                    summary_index_setting,
                                    session=session,
                                    indexing_request_id=indexing_request_id,
                                    indexing_intent_id=indexing_intent_id,
                                    indexing_claim_id=indexing_claim_id,
                                )
                                if generated_summary is None and indexing_intent_id is not None:
                                    raise RuntimeError("Summary generation lost rebuild ownership")
                                total_segments_processed += 1

                            except Exception as e:
                                logger.error(
                                    "Failed to regenerate summary for segment %s: %s",
                                    segment.id,
                                    str(e),
                                    exc_info=True,
                                )
                                total_segments_failed += 1
                                session.rollback()
                                continue

                    except Exception as e:
                        logger.error(
                            "Failed to process document %s for summary regeneration: %s",
                            dataset_document.id,
                            str(e),
                            exc_info=True,
                        )
                        total_segments_failed += 1
                        continue

            end_at = time.perf_counter()
            if regenerate_vectors_only:
                logger.info(
                    click.style(
                        f"Summary re-vectorization completed for dataset {dataset_id}: "
                        f"{total_segments_processed} segments processed successfully, "
                        f"{total_segments_failed} segments failed, "
                        f"latency: {end_at - start_at:.2f}s",
                        fg="green",
                    )
                )
            else:
                logger.info(
                    click.style(
                        f"Summary index regeneration completed for dataset {dataset_id}: "
                        f"{total_segments_processed} segments processed successfully, "
                        f"{total_segments_failed} segments failed, "
                        f"latency: {end_at - start_at:.2f}s",
                        fg="green",
                    )
                )

            if total_segments_failed:
                raise RuntimeError(f"{total_segments_failed} summary rebuild operations failed")
            return True

    except Exception:
        logger.exception("Regenerate summary index failed for dataset %s", dataset_id)
        if indexing_intent_id is not None:
            raise
        return False

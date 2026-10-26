"""Adapters for existing ORM-based summary callers and indexing infrastructure."""

import logging
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import TypedDict, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import set_committed_value

from core.credit_usage import CreditUsageCreatedBy
from core.db.session_factory import session_factory
from core.model_context import with_credit_usage_created_by
from core.model_manager import ModelInstance, ModelManager
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.constant.doc_type import DocType
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.index_processor.index_processor_base import SummaryIndexSettingDict
from core.rag.models.document import Document
from graphon.model_runtime.entities.llm_entities import LLMUsage
from graphon.model_runtime.entities.model_entities import ModelType
from libs import helper
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset, DocumentSegment, DocumentSegmentSummary
from models.dataset import Document as DatasetDocument
from models.enums import SegmentStatus, SummaryStatus
from repositories.knowledge.dataset_indexing_outbox_repository import (
    lock_dataset_indexing_execution_shared,
    lock_dataset_indexing_generation_shared,
)
from repositories.knowledge.dataset_read_repository import get_segment_document
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.summaries import (
    SummaryVectorCleanupClaim,
    SummaryVectorGenerationLostError,
    SummaryVectorInput,
    SummaryVectorResult,
)
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.vector_route import capture_vector_route, vector_for_route
from services.knowledge.summaries.application import (
    SummaryIndexService,
    SummaryVectorWriteNotSubmittedError,
)

logger = logging.getLogger(__name__)


def _lock_summary_dataset_owner(
    session: Session,
    *,
    ref: DatasetRef,
    indexing_request_id: str | None,
    indexing_intent_id: str | None,
    indexing_claim_id: str | None,
) -> Dataset | None:
    execution_values = (indexing_intent_id, indexing_claim_id)
    if any(value is not None for value in execution_values):
        if indexing_request_id is None or any(value is None for value in execution_values):
            raise ValueError("A durable summary rebuild requires generation, intent, and claim ownership")
        assert indexing_intent_id is not None
        assert indexing_claim_id is not None
        return lock_dataset_indexing_execution_shared(
            session,
            tenant_id=ref.tenant_id,
            dataset_id=ref.dataset_id,
            generation_id=indexing_request_id,
            intent_id=indexing_intent_id,
            claim_id=indexing_claim_id,
        )
    return lock_dataset_indexing_generation_shared(
        session,
        dataset_id=ref.dataset_id,
        generation_id=indexing_request_id,
    )


def _require_summary_dataset_owner(
    session: Session,
    *,
    ref: DatasetRef,
    indexing_request_id: str | None,
    indexing_intent_id: str | None,
    indexing_claim_id: str | None,
) -> Dataset:
    """Lock one immutable task generation or report that it was superseded."""

    dataset = _lock_summary_dataset_owner(
        session,
        ref=ref,
        indexing_request_id=indexing_request_id,
        indexing_intent_id=indexing_intent_id,
        indexing_claim_id=indexing_claim_id,
    )
    if dataset is None or dataset.tenant_id != ref.tenant_id:
        raise SummaryVectorGenerationLostError()
    return dataset


def _lock_eligible_summary_segments(
    session: Session,
    dataset: Dataset,
    segment_owners: dict[str, str],
    *,
    read: bool = False,
) -> dict[str, DocumentSegment]:
    """Lock and return current summary-eligible segments in owner order.

    The Dataset generation lock must already be held. Documents are locked
    before segments so archive/disable and segment state transitions cannot
    cross an authoritative summary mutation. ``read=True`` keeps the vector
    publication guard compatible with the cleanup repository's separate
    summary-row transaction while still excluding Document/Segment writers.
    """

    if not segment_owners:
        return {}
    document_ids = sorted(set(segment_owners.values()))
    documents = session.scalars(
        select(DatasetDocument)
        .where(
            DatasetDocument.id.in_(document_ids),
            DatasetDocument.dataset_id == dataset.id,
            DatasetDocument.tenant_id == dataset.tenant_id,
        )
        .order_by(DatasetDocument.id)
        .with_for_update(read=read)
        .execution_options(populate_existing=True)
    ).all()
    documents_by_id = {document.id: document for document in documents}
    segment_ids = sorted(segment_owners)
    segments = session.scalars(
        select(DocumentSegment)
        .where(
            DocumentSegment.id.in_(segment_ids),
            DocumentSegment.dataset_id == dataset.id,
            DocumentSegment.tenant_id == dataset.tenant_id,
        )
        .order_by(DocumentSegment.id)
        .with_for_update(read=read)
        .execution_options(populate_existing=True)
    ).all()
    eligible: dict[str, DocumentSegment] = {}
    for segment in segments:
        expected_document_id = segment_owners.get(segment.id)
        document = documents_by_id.get(segment.document_id)
        if (
            expected_document_id != segment.document_id
            or document is None
            or not document.enabled
            or document.archived
            or document.doc_form == IndexStructureType.QA_INDEX
            or not segment.enabled
            or segment.status != SegmentStatus.COMPLETED
        ):
            continue
        eligible[segment.id] = segment
    return eligible


def _require_eligible_summary_segments(
    session: Session,
    dataset: Dataset,
    segment_owners: dict[str, str],
    *,
    read: bool = False,
) -> dict[str, DocumentSegment]:
    eligible = _lock_eligible_summary_segments(session, dataset, segment_owners, read=read)
    if set(eligible) != set(segment_owners):
        raise SummaryVectorGenerationLostError()
    return eligible


class SummaryEntryDict(TypedDict):
    segment_id: str
    segment_position: int
    status: str
    summary_preview: str | None
    error: str | None
    created_at: int | None
    updated_at: int | None


class DocumentSummaryStatusDetailDict(TypedDict):
    total_segments: int
    summary_status: dict[str, int]
    summaries: list[SummaryEntryDict]


class SummaryIndexAdapter:
    """Adapt existing indexing callers to summary generation, storage and vectors."""

    @staticmethod
    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def generate_summary_for_segment(
        segment: DocumentSegment,
        dataset: Dataset,
        summary_index_setting: SummaryIndexSettingDict,
        *,
        session: Session,
    ) -> tuple[str, LLMUsage]:
        """
        Generate summary for a single segment.

        Args:
            segment: DocumentSegment to generate summary for
            dataset: Dataset containing the segment
            summary_index_setting: Summary index configuration

        Keyword Args:
            session: Caller session, committed after loading generation inputs and
                before invoking the model. Image reads use a separate bounded session.

        Returns:
            Tuple of (summary_content, llm_usage) where llm_usage is LLMUsage object

        Raises:
            ValueError: If summary_index_setting is invalid or generation fails
        """
        # Lazy import avoids the index processor's circular dependency on this adapter.
        from core.rag.index_processor.processor.paragraph_index_processor import ParagraphIndexProcessor

        # Get document language to ensure summary is generated in the correct language
        # This is especially important for image-only chunks where text is empty or minimal
        document_language = None
        document = get_segment_document(segment, session=session)
        if document and document.doc_language:
            document_language = document.doc_language

        tenant_id, text, segment_id = dataset.tenant_id, segment.content, segment.id
        if segment.tenant_id != tenant_id or segment.dataset_id != dataset.id:
            raise ValueError("Segment does not belong to the dataset")
        session.commit()

        def load_images():
            with session_factory.create_session() as image_session:
                images = ParagraphIndexProcessor._extract_images_from_segment_attachments(
                    tenant_id, segment_id, image_session
                )
                return images or ParagraphIndexProcessor._extract_images_from_text(tenant_id, text, image_session)

        summary_content, usage = ParagraphIndexProcessor.generate_summary_from_inputs(
            tenant_id=tenant_id,
            text=text,
            summary_index_setting=summary_index_setting,
            document_language=document_language,
            image_loader=load_images,
        )

        if not summary_content:
            raise ValueError("Generated summary is empty")

        return summary_content, usage

    @staticmethod
    def create_summary_record(
        segment: DocumentSegment,
        dataset: Dataset,
        summary_content: str,
        status: SummaryStatus = SummaryStatus.GENERATING,
        *,
        session: Session,
    ) -> DocumentSegmentSummary:
        """
        Create or update a DocumentSegmentSummary record.
        If a summary record already exists for this segment, it will be updated instead of creating a new one.

        Args:
            segment: DocumentSegment to create summary for
            dataset: Dataset containing the segment
            summary_content: Generated summary content
            status: Summary status (default: SummaryStatus.GENERATING)

        Keyword Args:
            session: SQLAlchemy session used for the summary record.

        Returns:
            Created or updated DocumentSegmentSummary instance
        """
        current_segment = _require_eligible_summary_segments(
            session,
            dataset,
            {segment.id: segment.document_id},
        )[segment.id]
        # Check if summary record already exists
        existing_summary = session.scalar(
            select(DocumentSegmentSummary)
            .where(
                DocumentSegmentSummary.chunk_id == segment.id,
                DocumentSegmentSummary.dataset_id == dataset.id,
            )
            .limit(1)
            .with_for_update()
            .execution_options(populate_existing=True)
        )

        if existing_summary:
            # Update existing record
            existing_summary.version += 1
            existing_summary.summary_content = summary_content
            existing_summary.status = status
            existing_summary.error = None  # Clear any previous errors
            # Re-enable if it was disabled
            if not existing_summary.enabled:
                existing_summary.enabled = True
                existing_summary.disabled_at = None
                existing_summary.disabled_by = None
            session.add(existing_summary)
            session.flush()
            return existing_summary
        else:
            # Create new record (enabled by default)
            summary_record = DocumentSegmentSummary(
                dataset_id=dataset.id,
                document_id=current_segment.document_id,
                chunk_id=current_segment.id,
                summary_content=summary_content,
                status=status,
                enabled=True,  # Explicitly set enabled to True
            )
            session.add(summary_record)
            session.flush()
            return summary_record

    @staticmethod
    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def vectorize_summary(
        summary_record: DocumentSegmentSummary,
        segment: DocumentSegment,
        dataset: Dataset,
        session: Session | None = None,
        *,
        new_session: Callable[[], Session] | None = None,
        vector_factory: Callable[[], Vector] | None = None,
        indexing_request_id: str | None = None,
        indexing_intent_id: str | None = None,
        indexing_claim_id: str | None = None,
        reuse_existing_node: bool = False,
        expected_segment_state_intent_id: str | None = None,
    ) -> bool:
        """Bridge already-authorized ORM callers to the detached summary use case.

        Pending summary writes are committed before external I/O. Each outcome
        attempt then owns a separate transaction, including retries and failures.
        Vector submission and publication hold shared locks on the current
        eligible Document/Segment owner chain; losing eligibility returns
        ``False`` without publishing ERROR.
        """
        if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
            return False
        ref = DatasetRef(dataset.tenant_id, dataset.id).document(segment.document_id).segment(segment.id)
        if segment.dataset_id != dataset.id or segment.tenant_id != dataset.tenant_id:
            raise ValueError("Segment does not belong to the dataset")
        if (
            summary_record.dataset_id != dataset.id
            or summary_record.document_id != segment.document_id
            or summary_record.chunk_id != segment.id
        ):
            raise ValueError("Summary does not belong to the segment")
        # ``None`` is the exact legacy generation, not an instruction to read
        # whichever generation the possibly-expired ORM Dataset has now.
        effective_generation_id = indexing_request_id
        command = SummaryVectorInput(
            ref,
            summary_record.id,
            summary_record.version,
            summary_record.summary_content or "",
            summary_record.summary_index_node_id,
            effective_generation_id,
        )
        if not command.content.strip():
            raise ValueError(f"Summary content is empty for segment {segment.id}, cannot vectorize")
        if session is not None:
            session.commit()
        create_session = new_session or session_factory.create_session
        with create_session() as read_session:
            owned_dataset = _lock_summary_dataset_owner(
                read_session,
                ref=ref.document.dataset,
                indexing_request_id=effective_generation_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
            )
            if owned_dataset is None or owned_dataset.tenant_id != ref.document.dataset.tenant_id:
                raise LookupError("Summary segment no longer exists in this dataset")
            try:
                eligible = _require_eligible_summary_segments(
                    read_session,
                    owned_dataset,
                    {ref.segment_id: ref.document.document_id},
                    read=True,
                )
                if (
                    expected_segment_state_intent_id is not None
                    and eligible[ref.segment_id].indexing_request_id != expected_segment_state_intent_id
                ):
                    raise SummaryVectorGenerationLostError()
            except SummaryVectorGenerationLostError:
                read_session.rollback()
                return False
            vector_type = (
                Vector.resolve_vector_type(owned_dataset, session=read_session) if vector_factory is None else None
            )
        factory = vector_factory or (lambda: Vector(owned_dataset, session=None, vector_type=vector_type))

        @contextmanager
        def generation_guard():
            # begin_write commits the durable writer before this lock is taken.
            # No second Dataset lock is requested while the guard is held, so a
            # queued configuration UPDATE cannot create a MySQL lock-queue cycle.
            with create_session() as guard_session:
                current_dataset = _lock_summary_dataset_owner(
                    guard_session,
                    ref=ref.document.dataset,
                    indexing_request_id=effective_generation_id,
                    indexing_intent_id=indexing_intent_id,
                    indexing_claim_id=indexing_claim_id,
                )
                if current_dataset is None or current_dataset.tenant_id != ref.document.dataset.tenant_id:
                    raise SummaryVectorGenerationLostError()
                eligible = _require_eligible_summary_segments(
                    guard_session,
                    current_dataset,
                    {ref.segment_id: ref.document.document_id},
                    read=True,
                )
                if (
                    expected_segment_state_intent_id is not None
                    and eligible[ref.segment_id].indexing_request_id != expected_segment_state_intent_id
                ):
                    raise SummaryVectorGenerationLostError()
                yield

        service = SummaryIndexService(
            store=SQLAlchemySummaryVectorCleanupRepository(session_factory=create_session),
            vectors=SummaryVectorGateway(owned_dataset, factory),
            new_id=(
                (lambda: command.previous_node_id or str(uuid.uuid4()))
                if reuse_existing_node
                else lambda: str(uuid.uuid4())
            ),
            text_hash=helper.generate_text_hash,
            sleep=time.sleep,
        )
        try:
            result = service.vectorize(command, mutation_guard=generation_guard)
        except Exception as error:
            set_committed_value(summary_record, "status", SummaryStatus.ERROR)
            set_committed_value(summary_record, "error", f"Vectorization failed: {error}")
            raise
        if result is None:
            return False
        set_committed_value(summary_record, "summary_index_node_id", result.node_id)
        set_committed_value(summary_record, "summary_index_node_hash", result.content_hash)
        set_committed_value(summary_record, "tokens", result.tokens)
        set_committed_value(summary_record, "status", SummaryStatus.COMPLETED)
        set_committed_value(summary_record, "error", None)
        return True

    @staticmethod
    def batch_create_summary_records(
        segments: list[DocumentSegment],
        dataset: Dataset,
        status: SummaryStatus = SummaryStatus.NOT_STARTED,
        *,
        indexing_request_id: str | None,
        indexing_intent_id: str | None = None,
        indexing_claim_id: str | None = None,
    ) -> bool:
        """Reserve summary rows for one immutable dataset generation.

        ``True`` means the exact parent generation committed the whole batch.
        ``False`` means that generation was superseded before the transaction
        acquired its Dataset lock, so no summary row was changed. Other errors
        are raised for the task retry policy to handle. Segments that are no
        longer enabled/completed, or whose document is disabled, archived, or
        QA-formatted, are skipped without reserving a summary row.

        Args:
            segments: List of DocumentSegment instances
            dataset: Dataset containing the segments
            status: Initial status for the records (default: SummaryStatus.NOT_STARTED)
        """
        segment_owners = [(segment.id, segment.document_id) for segment in segments]
        segment_ids = [segment_id for segment_id, _document_id in segment_owners]
        if not segment_ids:
            return True
        ref = DatasetRef(dataset.tenant_id, dataset.id)

        with session_factory.create_session() as session:
            try:
                owned_dataset = _require_summary_dataset_owner(
                    session,
                    ref=ref,
                    indexing_request_id=indexing_request_id,
                    indexing_intent_id=indexing_intent_id,
                    indexing_claim_id=indexing_claim_id,
                )
            except SummaryVectorGenerationLostError:
                session.rollback()
                return False
            eligible_segments = _lock_eligible_summary_segments(session, owned_dataset, dict(segment_owners))
            eligible_owners = [
                (segment_id, eligible_segments[segment_id].document_id)
                for segment_id, _document_id in segment_owners
                if segment_id in eligible_segments
            ]
            eligible_ids = [segment_id for segment_id, _document_id in eligible_owners]
            if not eligible_ids:
                session.commit()
                return True
            # Query existing summary records
            existing_summaries = session.scalars(
                select(DocumentSegmentSummary)
                .where(
                    DocumentSegmentSummary.chunk_id.in_(eligible_ids),
                    DocumentSegmentSummary.dataset_id == ref.dataset_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            ).all()
            existing_summary_map = {summary.chunk_id: summary for summary in existing_summaries}

            # Create or update records
            for segment_id, document_id in eligible_owners:
                existing_summary = existing_summary_map.get(segment_id)
                if existing_summary:
                    # Update existing record
                    existing_summary.version += 1
                    existing_summary.status = status
                    existing_summary.error = None  # Clear any previous errors
                    if not existing_summary.enabled:
                        existing_summary.enabled = True
                        existing_summary.disabled_at = None
                        existing_summary.disabled_by = None
                    session.add(existing_summary)
                else:
                    # Create new record
                    summary_record = DocumentSegmentSummary(
                        dataset_id=ref.dataset_id,
                        document_id=document_id,
                        chunk_id=segment_id,
                        summary_content=None,  # Will be filled later
                        status=status,
                        enabled=True,
                    )
                    session.add(summary_record)

            # Commit the batch created records
            session.commit()
            return True

    @staticmethod
    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def generate_and_vectorize_summary(
        segment: DocumentSegment,
        dataset: Dataset,
        summary_index_setting: SummaryIndexSettingDict,
        *,
        session: Session,
        indexing_request_id: str | None,
        indexing_intent_id: str | None = None,
        indexing_claim_id: str | None = None,
    ) -> DocumentSegmentSummary | None:
        """Generate and publish one summary under an immutable parent generation.

        A successful call commits the reserved version, generated content, and
        vector publication for the supplied owner. If that owner is superseded
        or its Document/Segment becomes ineligible, the current transition is
        rolled back and ``None`` is returned without publishing content, vector,
        or ERROR. Same-generation failures persist ERROR on the reserved version
        only while the owner chain remains eligible, then are re-raised so the
        task may retry.

        Args:
            segment: DocumentSegment to generate summary for
            dataset: Dataset containing the segment
            summary_index_setting: Summary index configuration

        Keyword Args:
            session: SQLAlchemy session used for summary record updates.

        Returns:
            Created DocumentSegmentSummary instance

        Raises:
            ValueError: If summary generation fails
        """
        ref = DatasetRef(dataset.tenant_id, dataset.id)
        summary_id: str | None = None
        generation_version: int | None = None
        try:
            owned_dataset = _require_summary_dataset_owner(
                session,
                ref=ref,
                indexing_request_id=indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
            )
            current_segment = _require_eligible_summary_segments(
                session,
                owned_dataset,
                {segment.id: segment.document_id},
            )[segment.id]
            # Get or refresh summary record in this session
            summary_record_in_session = session.scalar(
                select(DocumentSegmentSummary)
                .where(
                    DocumentSegmentSummary.chunk_id == current_segment.id,
                    DocumentSegmentSummary.dataset_id == dataset.id,
                )
                .limit(1)
                .with_for_update()
                .execution_options(populate_existing=True)
            )

            if not summary_record_in_session:
                # If not found, create one
                logger.warning("Summary record not found for segment %s, creating one", current_segment.id)
                summary_record_in_session = DocumentSegmentSummary(
                    dataset_id=dataset.id,
                    document_id=current_segment.document_id,
                    chunk_id=current_segment.id,
                    summary_content="",
                    status=SummaryStatus.GENERATING,
                    enabled=True,
                )
                session.add(summary_record_in_session)
                session.flush()
            else:
                # Reserve a distinct generation before model I/O. Any later edit,
                # regeneration, or clear invalidates this worker's completion.
                summary_record_in_session.version += 1

            # Update status to "generating"
            summary_record_in_session.status = SummaryStatus.GENERATING
            summary_record_in_session.error = None
            session.add(summary_record_in_session)
            summary_id = summary_record_in_session.id
            generation_version = summary_record_in_session.version
            # Persist GENERATING and release the write transaction before LLM I/O.
            session.commit()

            # Refresh the complete current segment immediately before loading
            # model inputs. The reservation commit released its owner locks, so
            # a same-generation content edit or state transition may have won
            # in between.
            owned_dataset = _require_summary_dataset_owner(
                session,
                ref=ref,
                indexing_request_id=indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
            )
            current_segment = _require_eligible_summary_segments(
                session,
                owned_dataset,
                {segment.id: segment.document_id},
            )[segment.id]
            generated_from_content = current_segment.content
            # Generate summary (returns summary_content and llm_usage)
            summary_content, llm_usage = SummaryIndexAdapter.generate_summary_for_segment(
                current_segment, owned_dataset, summary_index_setting, session=session
            )

            # Reacquire the same immutable parent generation before publishing
            # model output. A newer Dataset generation must never receive old
            # content, even when the summary version itself is unchanged.
            owned_dataset = _require_summary_dataset_owner(
                session,
                ref=ref,
                indexing_request_id=indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
            )
            current_segment = _require_eligible_summary_segments(
                session,
                owned_dataset,
                {segment.id: segment.document_id},
            )[segment.id]
            if current_segment.content != generated_from_content:
                session.rollback()
                logger.info("Discarded summary generated from stale content for segment %s", segment.id)
                return None
            cursor = session.execute(
                update(DocumentSegmentSummary)
                .where(
                    DocumentSegmentSummary.id == summary_id,
                    DocumentSegmentSummary.version == generation_version,
                    DocumentSegmentSummary.dataset_id == dataset.id,
                    DocumentSegmentSummary.document_id == current_segment.document_id,
                    DocumentSegmentSummary.chunk_id == current_segment.id,
                )
                .values(summary_content=summary_content, updated_at=naive_utc_now())
            )
            if cast(CursorResult[object], cursor).rowcount != 1:
                session.rollback()
                logger.info("Discarded superseded summary generation for segment %s", segment.id)
                return None
            session.commit()
            summary_record_in_session = session.scalar(
                select(DocumentSegmentSummary).where(
                    DocumentSegmentSummary.id == summary_id,
                    DocumentSegmentSummary.version == generation_version,
                )
            )
            if summary_record_in_session is None:
                return None

            # Log LLM usage for summary generation
            if llm_usage and llm_usage.total_tokens > 0:
                logger.info(
                    "Summary generation for segment %s used %s tokens (prompt: %s, completion: %s)",
                    segment.id,
                    llm_usage.total_tokens,
                    llm_usage.prompt_tokens,
                    llm_usage.completion_tokens,
                )

            if not SummaryIndexAdapter.vectorize_summary(
                summary_record_in_session,
                current_segment,
                owned_dataset,
                session=session,
                indexing_request_id=indexing_request_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
            ):
                logger.info("Discarded superseded summary vector for segment %s", segment.id)
                return None
            # vectorize_summary mutates status and token fields; refresh before returning the ORM object.
            summary_record_in_session = session.scalar(
                select(DocumentSegmentSummary).where(
                    DocumentSegmentSummary.id == summary_id,
                    DocumentSegmentSummary.version == generation_version,
                )
            )
            if summary_record_in_session is None:
                return None
            logger.info("Successfully generated and vectorized summary for segment %s", segment.id)
            return summary_record_in_session

        except SummaryVectorGenerationLostError:
            session.rollback()
            logger.info(
                "Discarded summary generation after losing dataset generation, dataset_id=%s, segment_id=%s",
                ref.dataset_id,
                segment.id,
            )
            return None
        except Exception as e:
            logger.exception("Failed to generate summary for segment %s", segment.id)
            session.rollback()
            if summary_id is not None and generation_version is not None:
                try:
                    owned_dataset = _require_summary_dataset_owner(
                        session,
                        ref=ref,
                        indexing_request_id=indexing_request_id,
                        indexing_intent_id=indexing_intent_id,
                        indexing_claim_id=indexing_claim_id,
                    )
                    _require_eligible_summary_segments(
                        session,
                        owned_dataset,
                        {segment.id: segment.document_id},
                    )
                except SummaryVectorGenerationLostError:
                    session.rollback()
                    logger.info(
                        "Discarded summary error after losing dataset generation, dataset_id=%s, segment_id=%s",
                        ref.dataset_id,
                        segment.id,
                    )
                    return None
                session.execute(
                    update(DocumentSegmentSummary)
                    .where(
                        DocumentSegmentSummary.id == summary_id,
                        DocumentSegmentSummary.version == generation_version,
                        DocumentSegmentSummary.dataset_id == ref.dataset_id,
                        DocumentSegmentSummary.document_id == segment.document_id,
                        DocumentSegmentSummary.chunk_id == segment.id,
                    )
                    .values(status=SummaryStatus.ERROR, error=str(e), updated_at=naive_utc_now())
                )
                session.commit()
            raise

    @staticmethod
    def generate_summaries_for_document(
        dataset: Dataset,
        document: DatasetDocument,
        summary_index_setting: SummaryIndexSettingDict,
        segment_ids: list[str] | None = None,
        only_parent_chunks: bool = False,
        *,
        indexing_request_id: str | None,
        indexing_intent_id: str | None = None,
        indexing_claim_id: str | None = None,
    ) -> list[DocumentSegmentSummary]:
        """Generate a document's summaries for one captured Dataset generation.

        The parent identity is a primitive task token and is reused unchanged
        for batch reservation and every segment. Losing it stops the remaining
        work without adopting a newer ORM Dataset snapshot. Same-generation
        failures remain retryable per segment.

        Args:
            dataset: Dataset containing the document
            document: DatasetDocument to generate summaries for
            summary_index_setting: Summary index configuration
            segment_ids: Optional list of specific segment IDs to process
            only_parent_chunks: If True, only process parent chunks (for parent-child mode)

        Returns:
            List of created DocumentSegmentSummary instances
        """
        # Capture only primitives before any commit can expire the caller's ORM
        # snapshot. Every segment below reuses this exact task identity.
        tenant_id = dataset.tenant_id
        dataset_id = dataset.id
        document_id = document.id
        parent_generation_id = indexing_request_id

        # Only generate summary index for high_quality indexing technique
        if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
            logger.info(
                "Skipping summary generation for dataset %s: indexing_technique is %s, not 'high_quality'",
                dataset.id,
                dataset.indexing_technique,
            )
            return []

        if not summary_index_setting or not summary_index_setting.get("enable"):
            logger.info("Summary index is disabled for dataset %s", dataset.id)
            return []

        # Skip qa_model documents
        if document.doc_form == "qa_model":
            logger.info("Skipping summary generation for qa_model document %s", document.id)
            return []

        logger.info(
            "Starting summary generation for document %s in dataset %s, segment_ids: %s, only_parent_chunks: %s",
            document.id,
            dataset.id,
            len(segment_ids) if segment_ids else "all",
            only_parent_chunks,
        )

        with session_factory.create_session() as session:
            # Query segments (only enabled segments)
            stmt = select(DocumentSegment).where(
                DocumentSegment.dataset_id == dataset_id,
                DocumentSegment.document_id == document_id,
                DocumentSegment.tenant_id == tenant_id,
                DocumentSegment.status == "completed",
                DocumentSegment.enabled.is_(True),  # Only generate summaries for enabled segments
            )

            if segment_ids:
                stmt = stmt.where(DocumentSegment.id.in_(segment_ids))

            segments = list(session.scalars(stmt).all())

            if not segments:
                logger.info("No segments found for document %s", document.id)
                return []

            session.commit()
            # Batch create summary records with "not_started" status before processing
            # This ensures all records exist upfront, allowing status tracking
            reserved = SummaryIndexAdapter.batch_create_summary_records(
                segments=segments,
                dataset=dataset,
                status=SummaryStatus.NOT_STARTED,
                indexing_request_id=parent_generation_id,
                indexing_intent_id=indexing_intent_id,
                indexing_claim_id=indexing_claim_id,
            )
            if not reserved:
                logger.info(
                    "Discarded summary batch after losing dataset generation, dataset_id=%s, document_id=%s",
                    dataset_id,
                    document_id,
                )
                return []

            summary_records = []

            for segment in segments:
                # For parent-child mode, only process parent chunks
                # In parent-child mode, all DocumentSegments are parent chunks,
                # so we process all of them. Child chunks are stored in ChildChunk table
                # and are not DocumentSegments, so they won't be in the segments list.
                # This check is mainly for clarity and future-proofing.
                if only_parent_chunks:
                    # In parent-child mode, all segments in the query are parent chunks
                    # Child chunks are not DocumentSegments, so they won't appear here
                    # We can process all segments
                    pass

                try:
                    summary_record = SummaryIndexAdapter.generate_and_vectorize_summary(
                        segment,
                        dataset,
                        summary_index_setting,
                        session=session,
                        indexing_request_id=parent_generation_id,
                        indexing_intent_id=indexing_intent_id,
                        indexing_claim_id=indexing_claim_id,
                    )
                    if summary_record is not None:
                        summary_records.append(summary_record)
                    else:
                        session.rollback()
                        try:
                            _require_summary_dataset_owner(
                                session,
                                ref=DatasetRef(tenant_id, dataset_id),
                                indexing_request_id=parent_generation_id,
                                indexing_intent_id=indexing_intent_id,
                                indexing_claim_id=indexing_claim_id,
                            )
                        except SummaryVectorGenerationLostError:
                            session.rollback()
                            logger.info(
                                "Stopped summary batch after losing dataset generation, dataset_id=%s",
                                dataset_id,
                            )
                            return summary_records
                        session.rollback()
                except Exception:
                    logger.exception("Failed to generate summary for segment %s", segment.id)
                    # The inner use case persists an error only for its exact
                    # summary id and generation version.
                    continue

            logger.info(
                "Completed summary generation for document %s: %s summaries generated and vectorized",
                document.id,
                len(summary_records),
            )
            return summary_records

    @staticmethod
    def disable_summaries_for_segments(
        dataset: Dataset,
        segment_ids: list[str] | None = None,
        disabled_by: str | None = None,
    ) -> None:
        """
        Disable summary records and durably schedule their exact vectors for removal.
        Unlike delete, this preserves the summary records but marks them as disabled.
        Every explicit request advances the summary version, including an
        already-disabled row, so a pending document-enable intent cannot revive
        the superseded revision.

        Args:
            dataset: Dataset containing the segments
            segment_ids: List of segment IDs to disable summaries for. If None, disable all.
            disabled_by: User ID who disabled the summaries
        """
        from libs.datetime_utils import naive_utc_now

        with session_factory.create_session() as session:
            stmt = select(DocumentSegmentSummary).where(
                DocumentSegmentSummary.dataset_id == dataset.id,
            )

            if segment_ids:
                stmt = stmt.where(DocumentSegmentSummary.chunk_id.in_(segment_ids))

            summaries = session.scalars(
                stmt.order_by(DocumentSegmentSummary.id).with_for_update().execution_options(populate_existing=True)
            ).all()

            if not summaries:
                return

            logger.info(
                "Disabling %s summary records for dataset %s, segment_ids: %s",
                len(summaries),
                dataset.id,
                len(segment_ids) if segment_ids else "all",
            )

            SQLAlchemySummaryVectorCleanupRepository.add_terminal_cleanups(
                session,
                tenant_id=dataset.tenant_id,
                summaries=summaries,
            )

            # Disable summary records (don't delete)
            now = naive_utc_now()
            session.execute(
                update(DocumentSegmentSummary)
                .where(DocumentSegmentSummary.id.in_(s.id for s in summaries))
                .values(
                    version=DocumentSegmentSummary.version + 1,
                    enabled=False,
                    disabled_at=now,
                    disabled_by=disabled_by,
                )
            )
            session.commit()
            logger.info("Disabled %s summary records for dataset %s", len(summaries), dataset.id)

    @staticmethod
    def delete_summaries_for_segments(
        dataset: Dataset,
        segment_ids: list[str] | None = None,
        *,
        session: Session,
    ) -> None:
        """
        Delete summary records and durably hand their exact vectors to cleanup.
        For disable operations, use disable_summaries_for_segments.

        Args:
            dataset: Dataset containing the segments
            segment_ids: List of segment IDs to delete summaries for. If None, delete all.
        """
        stmt = select(DocumentSegmentSummary).where(DocumentSegmentSummary.dataset_id == dataset.id)

        if segment_ids:
            stmt = stmt.where(DocumentSegmentSummary.chunk_id.in_(segment_ids))

        summaries = session.scalars(
            stmt.order_by(DocumentSegmentSummary.id).with_for_update().execution_options(populate_existing=True)
        ).all()

        if not summaries:
            return

        SQLAlchemySummaryVectorCleanupRepository.add_terminal_cleanups(
            session,
            tenant_id=dataset.tenant_id,
            summaries=summaries,
        )

        # Delete summary records in the caller's existing transaction.
        for summary in summaries:
            session.delete(summary)

        session.flush()
        logger.info("Deleted %s summary records for dataset %s", len(summaries), dataset.id)

    @staticmethod
    def get_segment_summary(
        segment_id: str,
        dataset_id: str,
        *,
        session: Session,
    ) -> DocumentSegmentSummary | None:
        """
        Get summary for a single segment.

        Args:
            segment_id: Segment ID (chunk_id)
            dataset_id: Dataset ID

        Keyword Args:
            session: SQLAlchemy session used to read summary records.

        Returns:
            DocumentSegmentSummary instance if found, None otherwise
        """
        return session.scalar(
            select(DocumentSegmentSummary)
            .where(
                DocumentSegmentSummary.chunk_id == segment_id,
                DocumentSegmentSummary.dataset_id == dataset_id,
                DocumentSegmentSummary.enabled.is_(True),
            )
            .limit(1)
        )

    @staticmethod
    def get_segments_summaries(
        segment_ids: list[str],
        dataset_id: str,
        *,
        session: Session,
    ) -> dict[str, DocumentSegmentSummary]:
        """
        Get summaries for multiple segments.

        Args:
            segment_ids: List of segment IDs (chunk_ids)
            dataset_id: Dataset ID

        Keyword Args:
            session: SQLAlchemy session used to read summary records.

        Returns:
            Dictionary mapping segment_id to DocumentSegmentSummary (only enabled summaries)
        """
        if not segment_ids:
            return {}

        summaries = session.scalars(
            select(DocumentSegmentSummary).where(
                DocumentSegmentSummary.chunk_id.in_(segment_ids),
                DocumentSegmentSummary.dataset_id == dataset_id,
                DocumentSegmentSummary.enabled.is_(True),
            )
        ).all()
        return {summary.chunk_id: summary for summary in summaries}

    @staticmethod
    def get_document_summaries(
        document_id: str,
        dataset_id: str,
        segment_ids: list[str] | None = None,
        *,
        session: Session,
    ) -> list[DocumentSegmentSummary]:
        """
        Get all summary records for a document.

        Args:
            document_id: Document ID
            dataset_id: Dataset ID
            segment_ids: Optional list of segment IDs to filter by

        Keyword Args:
            session: SQLAlchemy session used to read summary records.

        Returns:
            List of DocumentSegmentSummary instances (only enabled summaries)
        """
        stmt = select(DocumentSegmentSummary).where(
            DocumentSegmentSummary.document_id == document_id,
            DocumentSegmentSummary.dataset_id == dataset_id,
            DocumentSegmentSummary.enabled.is_(True),
        )

        if segment_ids:
            stmt = stmt.where(DocumentSegmentSummary.chunk_id.in_(segment_ids))

        return list(session.scalars(stmt).all())

    @staticmethod
    def get_document_summary_index_status(
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        *,
        session: Session,
    ) -> str | None:
        """
        Get summary_index_status for a single document.

        Args:
            document_id: Document ID
            dataset_id: Dataset ID
            tenant_id: Tenant ID

        Keyword Args:
            session: SQLAlchemy session used to read summary status.

        Returns:
            "SUMMARIZING" if there are pending summaries, None otherwise
        """
        # Get all segments for this document (excluding qa_model and re_segment)
        segment_ids = list(
            session.scalars(
                select(DocumentSegment.id).where(
                    DocumentSegment.document_id == document_id,
                    DocumentSegment.status != "re_segment",
                    DocumentSegment.tenant_id == tenant_id,
                )
            ).all()
        )

        if not segment_ids:
            return None

        # Get all summary records for these segments
        summaries = SummaryIndexAdapter.get_segments_summaries(segment_ids, dataset_id, session=session)
        summary_status_map = {chunk_id: summary.status for chunk_id, summary in summaries.items()}

        # Check if there are any "not_started" or "generating" status summaries
        has_pending_summaries = any(
            summary_status_map.get(segment_id) is not None  # Ensure summary exists (enabled=True)
            and summary_status_map[segment_id] in (SummaryStatus.NOT_STARTED, SummaryStatus.GENERATING)
            for segment_id in segment_ids
        )

        return "SUMMARIZING" if has_pending_summaries else None

    @staticmethod
    def get_documents_summary_index_status(
        document_ids: list[str],
        dataset_id: str,
        tenant_id: str,
        *,
        session: Session,
    ) -> dict[str, str | None]:
        """
        Get summary_index_status for multiple documents.

        Args:
            document_ids: List of document IDs
            dataset_id: Dataset ID
            tenant_id: Tenant ID

        Keyword Args:
            session: SQLAlchemy session used to read summary status.

        Returns:
            Dictionary mapping document_id to summary_index_status ("SUMMARIZING" or None)
        """
        if not document_ids:
            return {}

        # Get all segments for these documents (excluding qa_model and re_segment)
        segments = session.execute(
            select(DocumentSegment.id, DocumentSegment.document_id).where(
                DocumentSegment.document_id.in_(document_ids),
                DocumentSegment.status != "re_segment",
                DocumentSegment.tenant_id == tenant_id,
            )
        ).all()

        # Group segments by document_id
        document_segments_map: dict[str, list[str]] = {}
        for segment in segments:
            doc_id = str(segment.document_id)
            if doc_id not in document_segments_map:
                document_segments_map[doc_id] = []
            document_segments_map[doc_id].append(segment.id)

        # Get all summary records for these segments
        all_segment_ids = [seg.id for seg in segments]
        summaries = SummaryIndexAdapter.get_segments_summaries(all_segment_ids, dataset_id, session=session)
        summary_status_map = {chunk_id: summary.status for chunk_id, summary in summaries.items()}

        # Calculate summary_index_status for each document
        result: dict[str, str | None] = {}
        for doc_id in document_ids:
            segment_ids = document_segments_map.get(doc_id, [])
            if not segment_ids:
                # No segments, status is None (not started)
                result[doc_id] = None
                continue

            # Check if there are any "not_started" or "generating" status summaries
            # Only check enabled=True summaries (already filtered in query)
            # If segment has no summary record (summary_status_map.get returns None),
            # it means the summary is disabled (enabled=False) or not created yet, ignore it
            has_pending_summaries = any(
                summary_status_map.get(segment_id) is not None  # Ensure summary exists (enabled=True)
                and summary_status_map[segment_id] in (SummaryStatus.NOT_STARTED, SummaryStatus.GENERATING)
                for segment_id in segment_ids
            )

            if has_pending_summaries:
                # Task is still running (not started or generating)
                result[doc_id] = "SUMMARIZING"
            else:
                # All enabled=True summaries are "completed" or "error", task finished
                # Or no enabled=True summaries exist (all disabled)
                result[doc_id] = None

        return result

    @staticmethod
    def get_document_summary_status_detail(
        document_id: str,
        dataset_id: str,
        session: Session,
        *,
        tenant_id: str,
    ) -> DocumentSummaryStatusDetailDict:
        """
        Get detailed summary status for a document.

        Args:
            document_id: Document ID
            dataset_id: Dataset ID
            tenant_id: Owning tenant ID
            session: SQLAlchemy session used for segment lookup

        Returns:
            Dictionary containing:
            - total_segments: Total number of segments in the document
            - summary_status: Dictionary with status counts
              - completed: Number of summaries completed
              - generating: Number of summaries being generated
              - error: Number of summaries with errors
              - not_started: Number of segments without summary records
              - timeout: Number of summaries that timed out
            - summaries: List of summary records with status and content preview
        """
        from services.knowledge.dataset_service import SegmentService

        # Get all segments for this document
        segments = SegmentService.get_segments_by_document_and_dataset(
            tenant_id=tenant_id,
            document_id=document_id,
            dataset_id=dataset_id,
            session=session,
            status="completed",
            enabled=True,
        )

        total_segments = len(segments)

        # Get all summary records for these segments
        segment_ids = [segment.id for segment in segments]
        summaries = []
        if segment_ids:
            summaries = SummaryIndexAdapter.get_document_summaries(
                document_id=document_id,
                dataset_id=dataset_id,
                segment_ids=segment_ids,
                session=session,
            )

        # Create a mapping of chunk_id to summary
        summary_map = {summary.chunk_id: summary for summary in summaries}

        # Count statuses
        status_counts = {
            SummaryStatus.COMPLETED: 0,
            SummaryStatus.GENERATING: 0,
            SummaryStatus.ERROR: 0,
            SummaryStatus.NOT_STARTED: 0,
        }

        summary_list: list[SummaryEntryDict] = []
        for segment in segments:
            summary = summary_map.get(segment.id)
            if summary:
                status = SummaryStatus(summary.status)
                status_counts[status] = status_counts.get(status, 0) + 1
                summary_list.append(
                    {
                        "segment_id": segment.id,
                        "segment_position": segment.position,
                        "status": summary.status,
                        "summary_preview": (
                            summary.summary_content[:100] + "..."
                            if summary.summary_content and len(summary.summary_content) > 100
                            else summary.summary_content
                        ),
                        "error": summary.error,
                        "created_at": int(summary.created_at.timestamp()) if summary.created_at else None,
                        "updated_at": int(summary.updated_at.timestamp()) if summary.updated_at else None,
                    }
                )
            else:
                status_counts[SummaryStatus.NOT_STARTED] += 1
                summary_list.append(
                    {
                        "segment_id": segment.id,
                        "segment_position": segment.position,
                        "status": SummaryStatus.NOT_STARTED,
                        "summary_preview": None,
                        "error": None,
                        "created_at": None,
                        "updated_at": None,
                    }
                )

        return DocumentSummaryStatusDetailDict(
            total_segments=total_segments,
            summary_status=cast(dict[str, int], status_counts),
            summaries=summary_list,
        )


class _PreparedSummaryVectorWrite:
    def __init__(self, vector: Vector, document: Document, route: VectorRoute) -> None:
        self._vector = vector
        self._document = document
        self._route = route

    @property
    def route(self) -> VectorRoute:
        return self._route

    def submit(self) -> None:
        write_started = False

        def mark_write_started() -> None:
            nonlocal write_started
            write_started = True

        try:
            self._vector.add_texts(
                [self._document],
                duplicate_check=False,
                before_write=mark_write_started,
            )
        except Exception as error:
            if not write_started:
                raise SummaryVectorWriteNotSubmittedError(str(error)) from error
            raise


class SummaryVectorGateway:
    def __init__(self, dataset: Dataset, factory: Callable[[], Vector]) -> None:
        self._dataset, self._factory = dataset, factory
        self._model: ModelInstance | None = None

    def delete(self, claim: SummaryVectorCleanupClaim) -> None:
        if claim.route is None:
            raise ValueError("Summary vector cleanup has no trustworthy persisted route")
        vector_for_route(claim.route).delete_by_ids([claim.node_id])

    def count_tokens(self, content: str) -> int:
        model = self._resolve_model()
        tokens = model.get_text_embedding_num_tokens([content])
        return tokens[0] if tokens and isinstance(tokens[0], int) else 0

    def prepare(self, command: SummaryVectorInput, result: SummaryVectorResult) -> _PreparedSummaryVectorWrite:
        """Build a write that marks the exact vector-store mutation boundary.

        Embedding happens before that boundary. Failures there are reported as
        definitely not submitted; processor failures remain uncertain.
        """
        dataset_ref = command.ref.document.dataset
        if dataset_ref.tenant_id != self._dataset.tenant_id or dataset_ref.dataset_id != self._dataset.id:
            raise ValueError("Summary vector command does not belong to the dataset")
        if not command.content.strip():
            raise ValueError("Summary vector content is empty")
        vector = self._factory()
        route = capture_vector_route(self._dataset, vector, doc_form="summary")
        document = Document(
            page_content=command.content,
            metadata={
                "doc_id": result.node_id,
                "doc_hash": result.content_hash,
                "dataset_id": dataset_ref.dataset_id,
                "document_id": command.ref.document.document_id,
                "original_chunk_id": command.ref.segment_id,
                "doc_type": DocType.TEXT,
                "is_summary": True,
            },
        )
        return _PreparedSummaryVectorWrite(vector, document, route)

    def _resolve_model(self) -> ModelInstance:
        if self._model is not None:
            return self._model
        dataset = self._dataset
        model = ModelManager.for_tenant(tenant_id=dataset.tenant_id).get_model_instance(
            tenant_id=dataset.tenant_id,
            provider=dataset.embedding_model_provider or "",
            model_type=ModelType.TEXT_EMBEDDING,
            model=dataset.embedding_model or "",
        )
        self._model = model
        return model

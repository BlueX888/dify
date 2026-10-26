"""Durable, generation-fenced segment enable/disable reconciliation."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from celery import shared_task
from sqlalchemy import case, delete, select, update
from sqlalchemy.orm import Session, sessionmaker

from core.db.session_factory import session_factory
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.constant.doc_type import DocType
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.index_processor.index_processor import IndexProcessorFactory
from core.rag.models.document import AttachmentDocument, ChildDocument
from core.rag.models.document import Document as IndexDocument
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import IndexingStatus, SegmentStatus
from models.model import SegmentStateOutbox
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.child_vector_route_repository import SQLAlchemyChildVectorRouteStore
from repositories.knowledge.dataset_indexing_outbox_repository import SQLAlchemyDatasetIndexingOutboxRepository
from repositories.knowledge.dataset_read_repository import get_segment_child_chunks
from repositories.knowledge.segment_read_adapter import get_segment_attachments
from repositories.knowledge.summary_vector_cleanup_repository import SQLAlchemySummaryVectorCleanupRepository
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.summaries import SummaryVectorInput
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.indexing.vector_route import capture_vector_route
from services.knowledge.summaries.adapters import SummaryIndexAdapter
from tasks.delete_segment_from_index_task import (
    cleanup_segment_delete_blobs,
    cleanup_segment_index_in_session,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


@dataclass(frozen=True, slots=True)
class _IntentOwner:
    tenant_id: str
    dataset_id: str
    document_id: str
    segment_id: str


@dataclass(frozen=True, slots=True)
class _IntentBatchKey:
    tenant_id: str
    dataset_id: str
    document_id: str
    action: str


@dataclass(frozen=True, slots=True)
class _EnableBatchPublication:
    """Exact Dataset generation and physical target recorded before I/O."""

    generation_id: str | None
    route: VectorRoute | None
    summary_plans: tuple[_SummaryEnablePlan, ...]
    summary_intent_snapshots: tuple[_SummaryIntentSnapshot, ...]


@dataclass(frozen=True, slots=True)
class _SummaryEnablePlan:
    intent_id: str
    command: SummaryVectorInput


@dataclass(frozen=True, slots=True)
class _SummaryIntentSnapshot:
    intent_id: str
    summary_id: str | None
    summary_version: int | None


def _build_index_documents(
    segment: DocumentSegment,
    dataset: Dataset,
    document: Document,
    *,
    session: Session,
) -> tuple[IndexDocument, list[AttachmentDocument]]:
    indexed = IndexDocument(
        page_content=segment.content,
        metadata={
            "doc_id": segment.index_node_id,
            "doc_hash": segment.index_node_hash,
            "document_id": document.id,
            "dataset_id": dataset.id,
        },
    )
    if document.doc_form == IndexStructureType.PARENT_CHILD_INDEX:
        indexed.children = [
            ChildDocument(
                page_content=child.content,
                metadata={
                    "doc_id": child.index_node_id,
                    "doc_hash": child.index_node_hash,
                    "document_id": document.id,
                    "dataset_id": dataset.id,
                },
            )
            for child in get_segment_child_chunks(segment, session=session)
        ]
    attachments: list[AttachmentDocument] = []
    if dataset.is_multimodal:
        attachments = [
            AttachmentDocument(
                page_content=attachment["name"],
                metadata={
                    "doc_id": attachment["id"],
                    "doc_hash": "",
                    "document_id": document.id,
                    "dataset_id": dataset.id,
                    "doc_type": DocType.IMAGE,
                },
            )
            for attachment in get_segment_attachments(segment, session=session)
        ]
    return indexed, attachments


def _reconcile_summaries(
    *,
    action: Literal["enable", "disable"],
    segments: Sequence[DocumentSegment],
    intents: Sequence[SegmentStateOutbox],
    dataset: Dataset,
    document: Document,
    session: Session,
    published_summary_ids: frozenset[str] = frozenset(),
) -> None:
    """Keep summary rows/vectors fenced by the same segment generation locks."""

    if not segments:
        return
    # This matches the established summary-enable contract. Economy datasets
    # have no embedding configuration after a technique switch, so retained
    # summaries stay disabled instead of attempting an impossible vector write.
    if action == "enable" and dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
        return

    summaries = session.scalars(
        select(DocumentSegmentSummary)
        .where(
            DocumentSegmentSummary.dataset_id == dataset.id,
            DocumentSegmentSummary.document_id == document.id,
            DocumentSegmentSummary.chunk_id.in_([segment.id for segment in segments]),
        )
        .order_by(DocumentSegmentSummary.chunk_id, DocumentSegmentSummary.id)
        .with_for_update()
    ).all()
    if not summaries:
        return
    if action == "disable":
        SQLAlchemySummaryVectorCleanupRepository.add_terminal_cleanups(
            session,
            tenant_id=dataset.tenant_id,
            summaries=summaries,
        )
        now = naive_utc_now()
        disabled_by = {segment.id: segment.disabled_by for segment in segments}
        for summary in summaries:
            # A preceding enable phase may already have published a new physical
            # route while the final enabled flag is still false. Every newer
            # disable therefore transfers the current node and advances the
            # revision, even when the logical flag was already disabled.
            summary.version += 1
            summary.enabled = False
            summary.disabled_at = now
            summary.disabled_by = disabled_by.get(summary.chunk_id)
        return

    intents_by_segment = {intent.segment_id: intent for intent in intents}

    def owns_summary(summary: DocumentSegmentSummary) -> bool:
        intent = intents_by_segment.get(summary.chunk_id)
        if intent is None or intent.summary_version is None:
            # Missing evidence must never authorize an old enable delivery to
            # revive the current summary revision.
            return False
        if intent.summary_version == 0:
            return False
        return intent.summary_id == summary.id and intent.summary_version == summary.version

    for summary in summaries:
        # This is the freshly reloaded publication identity, not the plan's
        # optional previous node. Successful publication must have persisted it.
        if (
            not owns_summary(summary)
            or summary.id not in published_summary_ids
            or summary.enabled
            or not summary.summary_content
            or not summary.summary_index_node_id
        ):
            continue
        summary.version += 1
        summary.enabled = True
        summary.disabled_at = None
        summary.disabled_by = None


def _chunks(values: Sequence[str], size: int = _BATCH_SIZE) -> list[list[str]]:
    return [list(values[offset : offset + size]) for offset in range(0, len(values), size)]


def _process_delete_intent(
    intent_id: str,
    owner: _IntentOwner,
    *,
    new_session: sessionmaker[Session],
) -> bool:
    result = None
    with new_session.begin() as session:
        intent = session.scalar(
            select(SegmentStateOutbox)
            .where(
                SegmentStateOutbox.id == intent_id,
                SegmentStateOutbox.tenant_id == owner.tenant_id,
                SegmentStateOutbox.dataset_id == owner.dataset_id,
                SegmentStateOutbox.document_id == owner.document_id,
                SegmentStateOutbox.segment_id == owner.segment_id,
                SegmentStateOutbox.action == "delete",
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if intent is None:
            return False
        decoded = json.loads(intent.child_node_ids or "[]")
        if not isinstance(decoded, list) or not all(isinstance(node_id, str) for node_id in decoded):
            raise ValueError("Invalid child node snapshot on segment deletion intent")
        result = cleanup_segment_index_in_session(
            session,
            tenant_id=owner.tenant_id,
            dataset_id=owner.dataset_id,
            document_id=owner.document_id,
            segment_ids=(owner.segment_id,),
            index_node_ids=(intent.index_node_id,) if intent.index_node_id else (),
            child_node_ids=decoded,
            doc_form=intent.doc_form,
            upload_session_factory=new_session,
        )
        session.delete(intent)
    assert result is not None
    cleanup_segment_delete_blobs(result)
    return True


def _delete_owned_intents(session: Session, key: _IntentBatchKey, intent_ids: Sequence[str]) -> None:
    session.execute(
        delete(SegmentStateOutbox).where(
            SegmentStateOutbox.id.in_(intent_ids),
            SegmentStateOutbox.tenant_id == key.tenant_id,
            SegmentStateOutbox.dataset_id == key.dataset_id,
            SegmentStateOutbox.document_id == key.document_id,
            SegmentStateOutbox.action == key.action,
        )
    )


def _record_enable_batch_route(
    key: _IntentBatchKey,
    intent_ids: Sequence[str],
    *,
    new_session: sessionmaker[Session],
) -> _EnableBatchPublication | None:
    """Commit route evidence and return the immutable publication identity."""
    if key.action != "enable":
        return None
    with new_session.begin() as session:
        dataset = session.scalar(
            select(Dataset)
            .where(Dataset.id == key.dataset_id, Dataset.tenant_id == key.tenant_id)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        if dataset is None:
            return None
        document = session.scalar(
            select(Document)
            .where(
                Document.id == key.document_id,
                Document.tenant_id == key.tenant_id,
                Document.dataset_id == key.dataset_id,
            )
            .with_for_update()
        )
        if (
            document is None
            or not document.enabled
            or document.archived
            or document.indexing_status != IndexingStatus.COMPLETED
        ):
            return None
        candidates = session.execute(
            select(SegmentStateOutbox.id, SegmentStateOutbox.segment_id).where(
                SegmentStateOutbox.id.in_(intent_ids),
                SegmentStateOutbox.tenant_id == key.tenant_id,
                SegmentStateOutbox.dataset_id == key.dataset_id,
                SegmentStateOutbox.document_id == key.document_id,
                SegmentStateOutbox.action == "enable",
            )
        ).all()
        segment_ids = sorted({segment_id for _, segment_id in candidates})
        segments = session.scalars(
            select(DocumentSegment)
            .where(
                DocumentSegment.id.in_(segment_ids),
                DocumentSegment.tenant_id == key.tenant_id,
                DocumentSegment.dataset_id == key.dataset_id,
                DocumentSegment.document_id == key.document_id,
            )
            .order_by(DocumentSegment.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        intents = session.scalars(
            select(SegmentStateOutbox)
            .where(
                SegmentStateOutbox.id.in_(intent_ids),
                SegmentStateOutbox.tenant_id == key.tenant_id,
                SegmentStateOutbox.dataset_id == key.dataset_id,
                SegmentStateOutbox.document_id == key.document_id,
                SegmentStateOutbox.action == "enable",
            )
            .order_by(SegmentStateOutbox.segment_id, SegmentStateOutbox.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        segments_by_id = {segment.id: segment for segment in segments}
        current = [
            intent
            for intent in intents
            if (segment := segments_by_id.get(intent.segment_id)) is not None
            and _validated_action(segment, intent) == "enable"
            and segment.status == SegmentStatus.COMPLETED
            and segment.index_node_id is not None
        ]
        if not current:
            return None
        route = None
        summary_plans: list[_SummaryEnablePlan] = []
        if dataset.indexing_technique == IndexTechniqueType.HIGH_QUALITY:
            vector = Vector(dataset, session=session)
            route = capture_vector_route(dataset, vector, doc_form=str(document.doc_form))
            SQLAlchemyDatasetIndexingOutboxRepository.record_generation_writer_route_locked(
                session,
                dataset=dataset,
                generation_id=dataset.indexing_request_id,
                route=route,
            )
            summaries = session.scalars(
                select(DocumentSegmentSummary)
                .where(
                    DocumentSegmentSummary.dataset_id == dataset.id,
                    DocumentSegmentSummary.document_id == document.id,
                    DocumentSegmentSummary.chunk_id.in_([intent.segment_id for intent in current]),
                )
                .order_by(DocumentSegmentSummary.chunk_id, DocumentSegmentSummary.id)
                .with_for_update()
                .execution_options(populate_existing=True)
            ).all()
            current_by_segment = {intent.segment_id: intent for intent in current}
            for summary in summaries:
                owning_intent = current_by_segment.get(summary.chunk_id)
                if (
                    owning_intent is None
                    or owning_intent.summary_id != summary.id
                    or owning_intent.summary_version != summary.version
                    or not summary.summary_content
                ):
                    continue
                # A retained summary may have content but no node after an earlier
                # vectorization failure. None is an exact previous identity; the
                # durable summary writer will allocate and publish a fresh node.
                command = SummaryVectorInput(
                    ref=DatasetRef(dataset.tenant_id, dataset.id).document(document.id).segment(summary.chunk_id),
                    summary_id=summary.id,
                    version=summary.version,
                    content=summary.summary_content,
                    previous_node_id=summary.summary_index_node_id,
                    dataset_generation_id=dataset.indexing_request_id,
                )
                summary_plans.append(
                    _SummaryEnablePlan(
                        intent_id=owning_intent.id,
                        command=command,
                    )
                )
        return _EnableBatchPublication(
            generation_id=dataset.indexing_request_id,
            route=route,
            summary_plans=tuple(summary_plans),
            summary_intent_snapshots=tuple(
                _SummaryIntentSnapshot(
                    intent_id=intent.id,
                    summary_id=intent.summary_id,
                    summary_version=intent.summary_version,
                )
                for intent in current
            ),
        )


def _enable_publication_is_current(
    publication: _EnableBatchPublication | None,
    *,
    dataset: Dataset,
    document: Document,
    session: Session,
) -> bool:
    """Validate the committed receipt under the actual mutation's Dataset lock."""

    if publication is None or dataset.indexing_request_id != publication.generation_id:
        return False
    if publication.route is None:
        return dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY
    if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
        return False
    vector = Vector(dataset, session=session)
    current_route = capture_vector_route(dataset, vector, doc_form=str(document.doc_form))
    return current_route.key == publication.route.key


def _publish_enable_summaries(
    publication: _EnableBatchPublication,
    *,
    active_intent_ids: frozenset[str],
    new_session: sessionmaker[Session],
) -> frozenset[str]:
    """Publish exact summary snapshots through the durable writer protocol."""

    published: set[str] = set()
    for plan in publication.summary_plans:
        if plan.intent_id not in active_intent_ids:
            continue
        try:
            command = plan.command
            ref = command.ref
            with new_session() as session:
                dataset = session.scalar(
                    select(Dataset).where(
                        Dataset.id == ref.document.dataset.dataset_id,
                        Dataset.tenant_id == ref.document.dataset.tenant_id,
                        Dataset.indexing_request_id == publication.generation_id,
                    )
                )
                segment = session.scalar(
                    select(DocumentSegment).where(
                        DocumentSegment.id == ref.segment_id,
                        DocumentSegment.tenant_id == ref.document.dataset.tenant_id,
                        DocumentSegment.dataset_id == ref.document.dataset.dataset_id,
                        DocumentSegment.document_id == ref.document.document_id,
                        DocumentSegment.indexing_request_id == plan.intent_id,
                        DocumentSegment.enabled.is_(True),
                        DocumentSegment.status == SegmentStatus.COMPLETED,
                    )
                )
                summary = session.scalar(
                    select(DocumentSegmentSummary).where(
                        DocumentSegmentSummary.id == command.summary_id,
                        DocumentSegmentSummary.version == command.version,
                        DocumentSegmentSummary.dataset_id == ref.document.dataset.dataset_id,
                        DocumentSegmentSummary.document_id == ref.document.document_id,
                        DocumentSegmentSummary.chunk_id == ref.segment_id,
                    )
                )
                if dataset is None or segment is None or summary is None:
                    continue
                session.expunge_all()

            if SummaryIndexAdapter.vectorize_summary(
                summary,
                segment,
                dataset,
                new_session=new_session,
                vector_factory=lambda dataset=dataset: Vector(dataset, session=None),
                indexing_request_id=publication.generation_id,
                reuse_existing_node=True,
                expected_segment_state_intent_id=plan.intent_id,
            ):
                published.add(summary.id)
        except Exception:
            # Every summary has its own durable writer. Continue so phase three
            # can commit successful siblings while retaining this exact intent
            # for recovery, then report the remaining publication as pending.
            logger.exception(
                "Failed to publish summary during segment enable, summary_id=%s intent_id=%s",
                plan.command.summary_id,
                plan.intent_id,
            )
    return frozenset(published)


def _process_enable_intent_batch(
    key: _IntentBatchKey,
    intent_ids: Sequence[str],
    *,
    new_session: sessionmaker[Session],
) -> dict[str, bool]:
    """Commit body I/O, durable summary publication, then exact intent ACK."""

    results = dict.fromkeys(intent_ids, False)
    publication = _record_enable_batch_route(key, intent_ids, new_session=new_session)
    deferred_errors: list[str] = []
    active_intent_ids: frozenset[str] = frozenset()

    # Phase one commits body publication while retaining every durable intent.
    with new_session.begin() as session:
        dataset = session.scalar(
            select(Dataset)
            .where(Dataset.id == key.dataset_id, Dataset.tenant_id == key.tenant_id)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        if dataset is None:
            _delete_owned_intents(session, key, intent_ids)
            return results
        document = session.scalar(
            select(Document)
            .where(
                Document.id == key.document_id,
                Document.tenant_id == key.tenant_id,
                Document.dataset_id == key.dataset_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if document is None:
            _delete_owned_intents(session, key, intent_ids)
            return results
        candidates = session.execute(
            select(SegmentStateOutbox.id, SegmentStateOutbox.segment_id).where(
                SegmentStateOutbox.id.in_(intent_ids),
                SegmentStateOutbox.tenant_id == key.tenant_id,
                SegmentStateOutbox.dataset_id == key.dataset_id,
                SegmentStateOutbox.document_id == key.document_id,
                SegmentStateOutbox.action == "enable",
            )
        ).all()
        segment_ids = sorted({segment_id for _, segment_id in candidates})
        segments = session.scalars(
            select(DocumentSegment)
            .where(
                DocumentSegment.id.in_(segment_ids),
                DocumentSegment.tenant_id == key.tenant_id,
                DocumentSegment.dataset_id == key.dataset_id,
                DocumentSegment.document_id == key.document_id,
            )
            .order_by(DocumentSegment.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        segments_by_id = {segment.id: segment for segment in segments}
        intents = session.scalars(
            select(SegmentStateOutbox)
            .where(
                SegmentStateOutbox.id.in_(intent_ids),
                SegmentStateOutbox.tenant_id == key.tenant_id,
                SegmentStateOutbox.dataset_id == key.dataset_id,
                SegmentStateOutbox.document_id == key.document_id,
                SegmentStateOutbox.action == "enable",
            )
            .order_by(SegmentStateOutbox.segment_id, SegmentStateOutbox.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        valid: list[tuple[DocumentSegment, SegmentStateOutbox]] = []
        for intent in intents:
            segment = segments_by_id.get(intent.segment_id)
            if segment is None:
                session.delete(intent)
                continue
            if _validated_action(segment, intent) != "enable":
                if segment.indexing_request_id == intent.id:
                    segment.indexing_request_id = None
                session.delete(intent)
                continue
            valid.append((segment, intent))
        if valid and (
            not document.enabled or document.archived or document.indexing_status != IndexingStatus.COMPLETED
        ):
            deferred_errors.append(f"Document {document.id} is not ready for segment state reconciliation")
            valid = []
        if valid and not _enable_publication_is_current(
            publication,
            dataset=dataset,
            document=document,
            session=session,
        ):
            deferred_errors.append(
                f"Dataset {dataset.id} generation or vector route changed after segment publication preflight"
            )
            valid = []
        if valid:
            processor = IndexProcessorFactory(
                document.doc_form,
                attachment_vectors=SQLAlchemyAttachmentVectorStore(),
                child_vector_routes=SQLAlchemyChildVectorRouteStore(),
            ).init_index_processor()
            indexed_documents: list[IndexDocument] = []
            attachment_documents: dict[tuple[str, str], AttachmentDocument] = {}
            for segment, _intent in valid:
                indexed, attachments = _build_index_documents(segment, dataset, document, session=session)
                indexed_documents.append(indexed)
                for attachment in attachments:
                    attachment_documents[(document.id, str(attachment.metadata["doc_id"]))] = attachment
            processor.load(
                dataset,
                indexed_documents,
                multimodal_documents=list(attachment_documents.values()),
                session=session,
            )
            active_intent_ids = frozenset(intent.id for _segment, intent in valid)

    if deferred_errors:
        raise RuntimeError("; ".join(deferred_errors))
    if not active_intent_ids:
        return results
    assert publication is not None

    # Phase two owns a per-node cleanup writer before each external summary I/O.
    published_summary_ids = _publish_enable_summaries(
        publication,
        active_intent_ids=active_intent_ids,
        new_session=new_session,
    )

    # Phase three acknowledges only the exact token and parent generation that
    # survived both publications. A newer summary revision wins independently.
    with new_session.begin() as session:
        dataset = session.scalar(
            select(Dataset)
            .where(Dataset.id == key.dataset_id, Dataset.tenant_id == key.tenant_id)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        if dataset is None:
            return results
        document = session.scalar(
            select(Document)
            .where(
                Document.id == key.document_id,
                Document.tenant_id == key.tenant_id,
                Document.dataset_id == key.dataset_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if (
            document is None
            or not document.enabled
            or document.archived
            or document.indexing_status != IndexingStatus.COMPLETED
            or not _enable_publication_is_current(
                publication,
                dataset=dataset,
                document=document,
                session=session,
            )
        ):
            raise RuntimeError(f"Dataset {key.dataset_id} generation changed before segment enable acknowledgement")
        candidates = session.execute(
            select(SegmentStateOutbox.id, SegmentStateOutbox.segment_id).where(
                SegmentStateOutbox.id.in_(active_intent_ids),
                SegmentStateOutbox.tenant_id == key.tenant_id,
                SegmentStateOutbox.dataset_id == key.dataset_id,
                SegmentStateOutbox.document_id == key.document_id,
                SegmentStateOutbox.action == "enable",
            )
        ).all()
        segment_ids = sorted({segment_id for _, segment_id in candidates})
        segments = session.scalars(
            select(DocumentSegment)
            .where(
                DocumentSegment.id.in_(segment_ids),
                DocumentSegment.tenant_id == key.tenant_id,
                DocumentSegment.dataset_id == key.dataset_id,
                DocumentSegment.document_id == key.document_id,
            )
            .order_by(DocumentSegment.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        segments_by_id = {segment.id: segment for segment in segments}
        intents = session.scalars(
            select(SegmentStateOutbox)
            .where(
                SegmentStateOutbox.id.in_(active_intent_ids),
                SegmentStateOutbox.tenant_id == key.tenant_id,
                SegmentStateOutbox.dataset_id == key.dataset_id,
                SegmentStateOutbox.document_id == key.document_id,
                SegmentStateOutbox.action == "enable",
            )
            .order_by(SegmentStateOutbox.segment_id, SegmentStateOutbox.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        valid = [
            (segment, intent)
            for intent in intents
            if (segment := segments_by_id.get(intent.segment_id)) is not None
            and _validated_action(segment, intent) == "enable"
            and segment.status == SegmentStatus.COMPLETED
            and segment.index_node_id is not None
        ]
        summaries = session.scalars(
            select(DocumentSegmentSummary)
            .where(
                DocumentSegmentSummary.dataset_id == dataset.id,
                DocumentSegmentSummary.document_id == document.id,
                DocumentSegmentSummary.chunk_id.in_([segment.id for segment, _intent in valid]),
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        summaries_by_id = {summary.id: summary for summary in summaries}
        plans_by_intent = {plan.intent_id: plan for plan in publication.summary_plans}
        preflight_summary_snapshots = {
            snapshot.intent_id: (snapshot.summary_id, snapshot.summary_version)
            for snapshot in publication.summary_intent_snapshots
        }
        ackable: list[tuple[DocumentSegment, SegmentStateOutbox]] = []
        for segment, intent in valid:
            plan = plans_by_intent.get(intent.id)
            intent_summary_snapshot = (intent.summary_id, intent.summary_version)
            if preflight_summary_snapshots.get(intent.id) != intent_summary_snapshot:
                deferred_errors.append(f"Summary intent {intent.id} changed after publication preflight")
                continue
            if plan is not None:
                plan_summary_snapshot = (plan.command.summary_id, plan.command.version)
                if intent_summary_snapshot != plan_summary_snapshot:
                    deferred_errors.append(f"Summary intent {intent.id} changed after publication preflight")
                    continue
                current_summary = summaries_by_id.get(plan.command.summary_id)
                if (
                    current_summary is not None
                    and current_summary.version == plan.command.version
                    and current_summary.id not in published_summary_ids
                ):
                    deferred_errors.append(f"Summary {current_summary.id} publication remains pending")
                    continue
            ackable.append((segment, intent))
        _reconcile_summaries(
            action="enable",
            segments=[segment for segment, _intent in ackable],
            intents=[intent for _segment, intent in ackable],
            dataset=dataset,
            document=document,
            session=session,
            published_summary_ids=published_summary_ids,
        )
        for segment, intent in ackable:
            segment.indexing_request_id = None
            session.delete(intent)
            results[intent.id] = True

    if deferred_errors:
        raise RuntimeError("; ".join(deferred_errors))
    return results


def _process_state_intent_batch(
    key: _IntentBatchKey,
    intent_ids: Sequence[str],
    *,
    new_session: sessionmaker[Session],
) -> dict[str, bool]:
    """Apply one compatible bounded batch while fencing every segment independently."""

    results = dict.fromkeys(intent_ids, False)
    deferred_errors: list[str] = []
    if key.action == "enable":
        return _process_enable_intent_batch(key, intent_ids, new_session=new_session)
    if key.action != "disable":
        raise ValueError(f"Unsupported segment state action {key.action!r}")
    action: Literal["disable"] = "disable"

    with new_session.begin() as session:
        # A shared dataset lock freezes vector configuration without forcing
        # unrelated documents in this dataset to serialize their body writes.
        dataset = session.scalar(
            select(Dataset)
            .where(Dataset.id == key.dataset_id, Dataset.tenant_id == key.tenant_id)
            .with_for_update(read=True)
        )
        if dataset is None:
            _delete_owned_intents(session, key, intent_ids)
            return results
        document = session.scalar(
            select(Document)
            .where(
                Document.id == key.document_id,
                Document.tenant_id == key.tenant_id,
                Document.dataset_id == key.dataset_id,
            )
            .with_for_update()
        )
        if document is None:
            _delete_owned_intents(session, key, intent_ids)
            return results

        # Read identifiers only after locking the owner chain, then acquire
        # segment locks before intent locks to preserve the mutation lock order.
        candidates = session.execute(
            select(SegmentStateOutbox.id, SegmentStateOutbox.segment_id).where(
                SegmentStateOutbox.id.in_(intent_ids),
                SegmentStateOutbox.tenant_id == key.tenant_id,
                SegmentStateOutbox.dataset_id == key.dataset_id,
                SegmentStateOutbox.document_id == key.document_id,
                SegmentStateOutbox.action == action,
            )
        ).all()
        candidate_segment_ids = sorted({segment_id for _, segment_id in candidates})
        segments = session.scalars(
            select(DocumentSegment)
            .where(
                DocumentSegment.id.in_(candidate_segment_ids),
                DocumentSegment.tenant_id == key.tenant_id,
                DocumentSegment.dataset_id == key.dataset_id,
                DocumentSegment.document_id == key.document_id,
            )
            .order_by(DocumentSegment.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        segments_by_id = {segment.id: segment for segment in segments}
        intents = session.scalars(
            select(SegmentStateOutbox)
            .where(
                SegmentStateOutbox.id.in_(intent_ids),
                SegmentStateOutbox.tenant_id == key.tenant_id,
                SegmentStateOutbox.dataset_id == key.dataset_id,
                SegmentStateOutbox.document_id == key.document_id,
                SegmentStateOutbox.action == action,
            )
            .order_by(SegmentStateOutbox.segment_id, SegmentStateOutbox.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()

        valid: list[tuple[DocumentSegment, SegmentStateOutbox]] = []
        for intent in intents:
            segment = segments_by_id.get(intent.segment_id)
            if segment is None:
                session.delete(intent)
                continue
            if _validated_action(segment, intent) != action:
                if segment.indexing_request_id == intent.id:
                    segment.indexing_request_id = None
                session.delete(intent)
                continue
            valid.append((segment, intent))

        # Segment-level disables run while a document remains enabled; durable
        # document disable/archive commands run after the parent desired state
        # is committed. Exact segment intent ownership fences both forms, while
        # a later document enable replaces these intent IDs before publication.
        document_ready = document.indexing_status == IndexingStatus.COMPLETED or document.archived
        if valid and not document_ready:
            deferred_errors.append(f"Document {document.id} is not ready for segment state reconciliation")
            valid = []

        if valid:
            valid_segments = [segment for segment, _ in valid]
            # Bindings and attachment vectors survive disable; only the
            # generation-fenced body/child/keyword nodes are removed here.
            index_node_ids = [segment.index_node_id for segment in valid_segments if segment.index_node_id]
            if index_node_ids:
                processor = IndexProcessorFactory(
                    document.doc_form,
                    attachment_vectors=SQLAlchemyAttachmentVectorStore(),
                    child_vector_routes=SQLAlchemyChildVectorRouteStore(),
                ).init_index_processor()
                processor.clean(
                    dataset,
                    index_node_ids,
                    with_keywords=True,
                    delete_child_chunks=False,
                    session=session,
                )
            _reconcile_summaries(
                action=action,
                segments=valid_segments,
                intents=[intent for _segment, intent in valid],
                dataset=dataset,
                document=document,
                session=session,
            )
            for segment, intent in valid:
                if _validated_action(segment, intent) != action:
                    raise RuntimeError(f"Segment state generation changed during reconciliation: {segment.id}")
                segment.indexing_request_id = None
                session.delete(intent)
                results[intent.id] = True

    # Stale rows are committed independently of an incompatible not-ready row,
    # so one bad member cannot keep the rest of the batch from converging.
    if deferred_errors:
        raise RuntimeError("; ".join(deferred_errors))
    return results


def _group_intents(
    intent_ids: Sequence[str], *, new_session: sessionmaker[Session]
) -> dict[_IntentBatchKey, list[tuple[str, str]]]:
    with new_session() as session:
        rows = session.execute(
            select(
                SegmentStateOutbox.id,
                SegmentStateOutbox.tenant_id,
                SegmentStateOutbox.dataset_id,
                SegmentStateOutbox.document_id,
                SegmentStateOutbox.segment_id,
                SegmentStateOutbox.action,
            ).where(SegmentStateOutbox.id.in_(intent_ids))
        ).all()
    grouped: dict[_IntentBatchKey, list[tuple[str, str]]] = defaultdict(list)
    for intent_id, tenant_id, dataset_id, document_id, segment_id, action in rows:
        grouped[_IntentBatchKey(tenant_id, dataset_id, document_id, action)].append((intent_id, segment_id))
    return grouped


def _mark_attempted(intent_ids: Sequence[str], *, new_session: sessionmaker[Session]) -> None:
    with new_session.begin() as session:
        session.execute(
            update(SegmentStateOutbox)
            .where(SegmentStateOutbox.id.in_(intent_ids))
            .values(last_attempted_at=naive_utc_now())
        )


def _process_bounded_intents(intent_ids: Sequence[str], *, new_session: sessionmaker[Session]) -> dict[str, bool]:
    results = dict.fromkeys(intent_ids, False)
    first_error: Exception | None = None
    for key, grouped_rows in _group_intents(intent_ids, new_session=new_session).items():
        grouped_ids = [intent_id for intent_id, _ in grouped_rows]
        try:
            if key.action == "delete":
                for intent_id, segment_id in grouped_rows:
                    results[intent_id] = _process_delete_intent(
                        intent_id,
                        _IntentOwner(key.tenant_id, key.dataset_id, key.document_id, segment_id),
                        new_session=new_session,
                    )
            else:
                results.update(_process_state_intent_batch(key, grouped_ids, new_session=new_session))
        except Exception as error:
            logger.exception("Failed to reconcile segment state intents %s", grouped_ids)
            _mark_attempted(grouped_ids, new_session=new_session)
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error
    return results


def process_segment_state_intents(
    intent_ids: Sequence[str],
    *,
    new_session: sessionmaker[Session] | None = None,
) -> dict[str, bool]:
    """Apply durable intents in compatible, generation-fenced bounded batches."""

    session_maker = new_session or session_factory.get_session_maker()
    requested_ids = tuple(dict.fromkeys(intent_ids))
    results = dict.fromkeys(requested_ids, False)
    first_error: Exception | None = None
    for intent_batch in _chunks(requested_ids):
        try:
            results.update(_process_bounded_intents(intent_batch, new_session=session_maker))
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error
    return results


def process_segment_state_intent(
    intent_id: str,
    *,
    new_session: sessionmaker[Session] | None = None,
) -> bool:
    """Rolling-deploy bridge for a task that carries one durable intent."""

    return process_segment_state_intents((intent_id,), new_session=new_session)[intent_id]


def _validated_action(segment: DocumentSegment, intent: SegmentStateOutbox) -> Literal["enable", "disable"] | None:
    if segment.indexing_request_id != intent.id:
        return None
    if intent.action == "enable" and segment.enabled:
        return "enable"
    if intent.action == "disable" and not segment.enabled:
        return "disable"
    return None


@shared_task(queue="dataset")
def reconcile_segment_state_task(intent_id: str) -> None:
    """Consume the pre-batch task signature during rolling deployments."""

    process_segment_state_intent(intent_id)


@shared_task(queue="dataset")
def reconcile_segment_state_batch_task(intent_ids: list[str]) -> None:
    process_segment_state_intents(intent_ids)


def publish_segment_state_intents(intent_ids: Sequence[str]) -> None:
    for intent_batch in _chunks(tuple(dict.fromkeys(intent_ids))):
        reconcile_segment_state_batch_task.delay(intent_batch)


def drain_segment_state_outbox(
    *,
    new_session: sessionmaker[Session] | None = None,
    publish: Callable[[Sequence[str]], object] = publish_segment_state_intents,
) -> None:
    session_maker = new_session or session_factory.get_session_maker()
    # Broker I/O occurs only after the read session has closed.
    with session_maker() as session:
        rows = session.execute(
            select(
                SegmentStateOutbox.id,
                SegmentStateOutbox.tenant_id,
                SegmentStateOutbox.dataset_id,
                SegmentStateOutbox.document_id,
                SegmentStateOutbox.action,
            )
            .order_by(
                case((SegmentStateOutbox.last_attempted_at.is_(None), 0), else_=1),
                SegmentStateOutbox.last_attempted_at,
                SegmentStateOutbox.created_at,
                SegmentStateOutbox.id,
            )
            .limit(_BATCH_SIZE)
        ).all()
    grouped: dict[_IntentBatchKey, list[str]] = defaultdict(list)
    for intent_id, tenant_id, dataset_id, document_id, action in rows:
        grouped[_IntentBatchKey(tenant_id, dataset_id, document_id, action)].append(intent_id)
    for intent_ids in grouped.values():
        try:
            publish(intent_ids)
        except Exception:
            logger.exception("Failed to publish segment state intents %s", intent_ids)
            continue
        with session_maker.begin() as session:
            session.execute(
                update(SegmentStateOutbox)
                .where(SegmentStateOutbox.id.in_(intent_ids))
                .values(last_attempted_at=naive_utc_now())
            )


@shared_task(queue="dataset")
def drain_segment_state_outbox_task() -> None:
    drain_segment_state_outbox()

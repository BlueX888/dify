"""Infrastructure adapters for dataset segment use cases."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, cast

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from core.errors.error import LLMBadRequestError, ProviderTokenNotInitError
from core.model_manager import ModelManager
from graphon.model_runtime.entities.model_entities import ModelType
from libs.datetime_utils import naive_utc_now
from models.dataset import Dataset, Document, DocumentSegment
from models.enums import SegmentStatus
from models.model import SegmentStateOutbox
from repositories.knowledge.segment_read_adapter import (
    format_segment_attachments,
    sign_segment_text,
)
from repositories.knowledge.segment_repository import lock_segment_summary_snapshot
from services.knowledge.entities.resource_scope import DocumentRef
from services.knowledge.entities.segments import SegmentReadSnapshot, SegmentRecord, SegmentStateIntent
from services.knowledge.segments.application import (
    SegmentDatasetRecord,
    SegmentModelProviderError,
)

logger = logging.getLogger(__name__)

_SEGMENT_STATE_BATCH_SIZE = 100


class RedisSegmentClient(Protocol):
    def get(self, name: str | bytes) -> Any: ...

    def setnx(self, name: str | bytes, value: Any) -> Any: ...
    def lock(self, name: str, timeout: int) -> Any: ...


class SegmentPresentationAdapter:
    def present(self, snapshots: Sequence[SegmentReadSnapshot]) -> tuple[SegmentRecord, ...]:
        records = []
        for snapshot in snapshots:
            records.append(
                SegmentRecord.model_validate(
                    {
                        **snapshot.values,
                        "sign_content": sign_segment_text(
                            cast(str, snapshot.values["content"]),
                            allowed_uploads=snapshot.file_access.uploads,
                            allowed_tools=snapshot.file_access.tools,
                        ),
                        "child_chunks": snapshot.child_chunks,
                        "attachments": format_segment_attachments(snapshot.attachments),
                        "summary": snapshot.summary,
                    }
                )
            )
        return tuple(records)


class ModelManagerSegmentGuard:
    def check(self, dataset: SegmentDatasetRecord) -> None:
        try:
            ModelManager.for_tenant(tenant_id=dataset.workspace_id).get_model_instance(
                tenant_id=dataset.workspace_id,
                provider=dataset.embedding_model_provider or "",
                model_type=ModelType.TEXT_EMBEDDING,
                model=dataset.embedding_model or "",
            )
        except LLMBadRequestError as error:
            raise SegmentModelProviderError(kind="bad_request", description=str(error)) from error
        except ProviderTokenNotInitError as error:
            raise SegmentModelProviderError(kind="token", description=error.description) from error


class RedisSegmentIndexingState:
    def __init__(self, redis: RedisSegmentClient) -> None:
        self._redis = redis

    def lock(self, name: str, *, timeout: int):
        return self._redis.lock(name, timeout=timeout)

    def is_segment_indexing(self, segment_id: str, *, deleting: bool = False) -> bool:
        suffix = "delete_indexing" if deleting else "indexing"
        return self._redis.get(f"segment_{segment_id}_{suffix}") is not None

    def is_document_indexing(self, document_id: str) -> bool:
        return self._redis.get(f"document_{document_id}_indexing") is not None

    def set_batch_waiting(self, job_id: str) -> None:
        self._redis.setnx(f"segment_batch_import_{job_id}", "waiting")

    def get_batch_status(self, job_id: str) -> str | None:
        value = self._redis.get(f"segment_batch_import_{job_id}")
        if value is None:
            return None
        return value.decode() if isinstance(value, bytes) else value


class CelerySegmentBatchImportDispatcher:
    def __init__(self, *, delay: Callable[..., object]) -> None:
        self._delay = delay

    def dispatch(
        self,
        *,
        job_id: str,
        upload_file_id: str,
        dataset_id: str,
        document_id: str,
        workspace_id: str,
        actor_id: str,
    ) -> None:
        self._delay(job_id, upload_file_id, dataset_id, document_id, workspace_id, actor_id)


class SQLAlchemySegmentStateTransitions:
    """Persist segment state and its exact indexing generation in one transaction."""

    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def request(
        self,
        *,
        document_ref: DocumentRef,
        segment_ids: Sequence[str],
        action: Literal["enable", "disable"],
        actor_id: str,
        reconcile_settled: bool = False,
    ) -> tuple[SegmentStateIntent, ...]:
        if not segment_ids:
            return ()
        target_enabled = action == "enable"
        requested_ids = tuple(dict.fromkeys(segment_ids))
        with self._session_factory.begin() as session:
            dataset = session.scalar(
                select(Dataset)
                .where(
                    Dataset.id == document_ref.dataset.dataset_id,
                    Dataset.tenant_id == document_ref.dataset.tenant_id,
                )
                .with_for_update()
            )
            if dataset is None:
                return ()
            document = session.scalar(
                select(Document)
                .where(
                    Document.id == document_ref.document_id,
                    Document.dataset_id == dataset.id,
                    Document.tenant_id == dataset.tenant_id,
                )
                .with_for_update()
            )
            if document is None:
                return ()
            segments = session.scalars(
                select(DocumentSegment)
                .where(
                    DocumentSegment.id.in_(requested_ids),
                    DocumentSegment.tenant_id == dataset.tenant_id,
                    DocumentSegment.dataset_id == dataset.id,
                    DocumentSegment.document_id == document.id,
                )
                .order_by(DocumentSegment.id)
                .with_for_update()
            ).all()
            intents: list[SegmentStateIntent] = []
            now = naive_utc_now()
            for segment in segments:
                if segment.status != SegmentStatus.COMPLETED:
                    continue
                if reconcile_settled:
                    # A compatibility delivery may only repair the exact state
                    # that remains current under this lock. It must never
                    # reverse or replace a newer opposite transition.
                    if segment.enabled != target_enabled:
                        continue
                    if segment.indexing_request_id:
                        existing = session.get(SegmentStateOutbox, segment.indexing_request_id)
                        if existing is not None:
                            if (
                                existing.tenant_id == dataset.tenant_id
                                and existing.dataset_id == dataset.id
                                and existing.document_id == document.id
                                and existing.segment_id == segment.id
                                and existing.action == action
                            ):
                                intents.append(SegmentStateIntent(existing.id))
                            continue
                        segment.indexing_request_id = None
                    summary_id, summary_version = (
                        lock_segment_summary_snapshot(session, dataset_id=dataset.id, segment_id=segment.id)
                        if action == "enable"
                        else (None, None)
                    )
                    intent = SegmentStateOutbox(
                        tenant_id=dataset.tenant_id,
                        dataset_id=dataset.id,
                        document_id=document.id,
                        segment_id=segment.id,
                        action=action,
                        actor_id=actor_id,
                        summary_id=summary_id,
                        summary_version=summary_version,
                    )
                    session.add(intent)
                    segment.indexing_request_id = intent.id
                    intents.append(SegmentStateIntent(intent.id))
                    continue
                invalid_current_intent = False
                if segment.enabled == target_enabled and segment.indexing_request_id:
                    existing = session.scalar(
                        select(SegmentStateOutbox).where(
                            SegmentStateOutbox.id == segment.indexing_request_id,
                            SegmentStateOutbox.tenant_id == dataset.tenant_id,
                            SegmentStateOutbox.dataset_id == dataset.id,
                            SegmentStateOutbox.document_id == document.id,
                            SegmentStateOutbox.segment_id == segment.id,
                            SegmentStateOutbox.action == action,
                        )
                    )
                    if existing is not None:
                        intents.append(SegmentStateIntent(existing.id))
                        continue
                    invalid_current_intent = True
                if segment.enabled == target_enabled and not invalid_current_intent:
                    # A settled same-state request has no reconciliation work.
                    continue
                if segment.indexing_request_id:
                    session.execute(
                        delete(SegmentStateOutbox).where(SegmentStateOutbox.id == segment.indexing_request_id)
                    )
                summary_id, summary_version = (
                    lock_segment_summary_snapshot(session, dataset_id=dataset.id, segment_id=segment.id)
                    if action == "enable"
                    else (None, None)
                )
                intent = SegmentStateOutbox(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_id=document.id,
                    segment_id=segment.id,
                    action=action,
                    actor_id=actor_id,
                    summary_id=summary_id,
                    summary_version=summary_version,
                )
                session.add(intent)
                segment.indexing_request_id = intent.id
                segment.enabled = target_enabled
                segment.disabled_at = None if target_enabled else now
                segment.disabled_by = None if target_enabled else actor_id
                intents.append(SegmentStateIntent(intent.id))
            return tuple(intents)


def request_document_enable_intents_locked(
    session: Session,
    *,
    dataset: Dataset,
    document: Document,
    segments: Sequence[DocumentSegment],
    actor_id: str,
) -> tuple[SegmentStateIntent, ...]:
    """Persist document-enable reconciliation in the caller's owner transaction.

    The caller holds Dataset, Document, and ordered Segment locks. A document
    false-to-true transition is a new user command, so it always replaces an
    earlier segment intent and snapshots the summary revision at this command's
    serialization point. A later explicit summary disable then invalidates only
    this publication.
    """

    intents: list[SegmentStateIntent] = []
    for segment in segments:
        if segment.status != SegmentStatus.COMPLETED:
            continue
        if segment.indexing_request_id:
            session.execute(delete(SegmentStateOutbox).where(SegmentStateOutbox.id == segment.indexing_request_id))
        summary_id, summary_version = lock_segment_summary_snapshot(
            session,
            dataset_id=dataset.id,
            segment_id=segment.id,
        )
        intent = SegmentStateOutbox(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            segment_id=segment.id,
            action="enable",
            actor_id=actor_id,
            summary_id=summary_id,
            summary_version=summary_version,
        )
        session.add(intent)
        segment.indexing_request_id = intent.id
        segment.enabled = True
        segment.disabled_at = None
        segment.disabled_by = None
        intents.append(SegmentStateIntent(intent.id))
    return tuple(intents)


def request_document_disable_intents_locked(
    session: Session,
    *,
    dataset: Dataset,
    document: Document,
    segments: Sequence[DocumentSegment],
    actor_id: str,
) -> tuple[SegmentStateIntent, ...]:
    """Persist document-disable reconciliation in the caller's owner transaction.

    The caller holds Dataset, Document, and ordered Segment locks. Every
    segment receives a new exact disable generation, replacing a pending enable
    or older disable. ERROR and body-less segments still need an intent: they
    may own an earlier body node or summary vector that must be handed to
    durable cleanup.
    """

    now = naive_utc_now()
    intents: list[SegmentStateIntent] = []
    for segment in segments:
        if segment.indexing_request_id:
            session.execute(delete(SegmentStateOutbox).where(SegmentStateOutbox.id == segment.indexing_request_id))
            segment.indexing_request_id = None
        segment.enabled = False
        segment.disabled_at = now
        segment.disabled_by = actor_id
        segment.updated_at = now
        intent = SegmentStateOutbox(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            segment_id=segment.id,
            action="disable",
            actor_id=actor_id,
        )
        session.add(intent)
        segment.indexing_request_id = intent.id
        intents.append(SegmentStateIntent(intent.id))
    return tuple(intents)


class CelerySegmentStateDispatcher:
    def __init__(self, *, delay: Callable[[list[str]], object]) -> None:
        self._delay = delay

    def dispatch(self, intent_id: str) -> None:
        self.dispatch_many((intent_id,))

    def dispatch_many(self, intent_ids: Sequence[str]) -> None:
        requested_ids = list(dict.fromkeys(intent_ids))
        for offset in range(0, len(requested_ids), _SEGMENT_STATE_BATCH_SIZE):
            self._delay(requested_ids[offset : offset + _SEGMENT_STATE_BATCH_SIZE])

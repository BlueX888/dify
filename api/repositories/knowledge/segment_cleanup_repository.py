"""Transactional cleanup operations for segment-owned persistence."""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from models.dataset import (
    ChildChunk,
    Dataset,
    Document,
    DocumentSegment,
    DocumentSegmentSummary,
    SegmentAttachmentBinding,
)
from models.model import UploadFile
from repositories.knowledge.upload_file_repository import (
    UploadFileCleanupArtifact,
    claim_unreferenced_files_for_cleanup_in_session,
)
from repositories.knowledge.vector_cleanup_handoff_repository import (
    add_terminal_child_row_cleanups,
    add_terminal_summary_vector_cleanups,
)


@dataclass(frozen=True, slots=True)
class SegmentAttachmentReplacementResult:
    removed_ids: tuple[str, ...]
    changed: bool


def require_locked_segment_attachment_document_owner(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
) -> None:
    """Lock and validate the dataset/document owner chain before a writer can flush changes."""
    lock_segment_attachment_publication_dataset(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
    )
    with session.no_autoflush:
        owned_document_id = session.scalar(
            select(Document.id)
            .where(
                Document.id == document_id,
                Document.dataset_id == dataset_id,
                Document.tenant_id == tenant_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if owned_document_id is None:
            raise LookupError("Document no longer exists")


def lock_segment_attachment_publication_dataset(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
) -> None:
    """Take the writer-side dataset lock shared by binding and vector publication."""
    with session.no_autoflush:
        owned_dataset_id = session.scalar(
            select(Dataset.id)
            .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
    if owned_dataset_id is None:
        raise LookupError("Dataset no longer exists")


def require_locked_segment_attachment_owner(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    segment_id: str,
) -> None:
    """Lock and validate the complete owner chain before mutating bindings."""
    require_locked_segment_attachment_document_owner(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
    )
    with session.no_autoflush:
        owned_segment_id = session.scalar(
            select(DocumentSegment.id)
            .where(
                DocumentSegment.id == segment_id,
                DocumentSegment.document_id == document_id,
                DocumentSegment.dataset_id == dataset_id,
                DocumentSegment.tenant_id == tenant_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    if owned_segment_id is None:
        raise LookupError("Segment no longer exists")


def lock_segment_attachment_cleanup_dataset(session: Session, *, tenant_id: str, dataset_id: str) -> None:
    """Take the cleanup-side dataset lock without requiring a deleted owner to remain."""
    session.scalar(
        select(Dataset.id)
        .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )


def lock_owned_upload_files(
    session: Session,
    *,
    tenant_id: str,
    file_ids: Sequence[str],
) -> frozenset[str]:
    """Lock tenant-owned files by id after the caller has locked knowledge owners."""
    normalized_file_ids = sorted({file_id for file_id in file_ids if file_id})
    if not normalized_file_ids:
        return frozenset()
    return frozenset(
        session.scalars(
            select(UploadFile.id)
            .where(UploadFile.tenant_id == tenant_id, UploadFile.id.in_(normalized_file_ids))
            .order_by(UploadFile.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    )


def add_segment_attachment_bindings(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    segment_id: str,
    attachment_ids: Sequence[str],
) -> None:
    """Add bindings after locking the complete knowledge owner chain."""
    requested_ids = tuple(sorted(set(attachment_ids)))
    require_locked_segment_attachment_owner(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        segment_id=segment_id,
    )
    existing_file_ids = lock_owned_upload_files(session, tenant_id=tenant_id, file_ids=requested_ids)
    missing_ids = set(requested_ids) - existing_file_ids
    if missing_ids:
        raise LookupError(f"Attachment files no longer exist: {sorted(missing_ids)}")
    bound_ids = set(
        session.scalars(
            select(SegmentAttachmentBinding.attachment_id)
            .where(
                SegmentAttachmentBinding.tenant_id == tenant_id,
                SegmentAttachmentBinding.dataset_id == dataset_id,
                SegmentAttachmentBinding.document_id == document_id,
                SegmentAttachmentBinding.segment_id == segment_id,
                SegmentAttachmentBinding.attachment_id.in_(requested_ids),
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    )
    session.add_all(
        [
            SegmentAttachmentBinding(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                segment_id=segment_id,
                attachment_id=file_id,
            )
            for file_id in requested_ids
            if file_id not in bound_ids
        ]
    )


def replace_segment_attachment_bindings(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    segment_id: str,
    attachment_ids: Sequence[str],
) -> SegmentAttachmentReplacementResult:
    """Replace the current binding set under its complete knowledge owner."""
    requested_ids = tuple(sorted(set(attachment_ids)))
    require_locked_segment_attachment_owner(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        segment_id=segment_id,
    )
    current_ids = tuple(
        session.scalars(
            select(SegmentAttachmentBinding.attachment_id)
            .where(
                SegmentAttachmentBinding.tenant_id == tenant_id,
                SegmentAttachmentBinding.dataset_id == dataset_id,
                SegmentAttachmentBinding.document_id == document_id,
                SegmentAttachmentBinding.segment_id == segment_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    )
    existing_file_ids = lock_owned_upload_files(
        session,
        tenant_id=tenant_id,
        file_ids=(*current_ids, *requested_ids),
    )
    missing_ids = set(requested_ids) - existing_file_ids
    if missing_ids:
        raise LookupError(f"Attachment files no longer exist: {sorted(missing_ids)}")
    if set(current_ids) == set(requested_ids):
        return SegmentAttachmentReplacementResult(removed_ids=(), changed=False)
    session.execute(
        delete(SegmentAttachmentBinding).where(
            SegmentAttachmentBinding.tenant_id == tenant_id,
            SegmentAttachmentBinding.dataset_id == dataset_id,
            SegmentAttachmentBinding.document_id == document_id,
            SegmentAttachmentBinding.segment_id == segment_id,
        )
    )
    session.add_all(
        [
            SegmentAttachmentBinding(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                segment_id=segment_id,
                attachment_id=file_id,
            )
            for file_id in requested_ids
        ]
    )
    removed_file_ids = tuple(file_id for file_id in dict.fromkeys(current_ids) if file_id not in requested_ids)
    claim_unreferenced_files_for_cleanup_in_session(
        session,
        workspace_id=tenant_id,
        file_ids=removed_file_ids,
    )
    return SegmentAttachmentReplacementResult(removed_ids=removed_file_ids, changed=True)


def _lock_cleanup_owner_scope(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_ids: tuple[str, ...] | None,
    segment_ids: tuple[str, ...] | None,
) -> None:
    """Serialize cleanup with live writers without requiring owners to remain present."""
    lock_segment_attachment_cleanup_dataset(session, tenant_id=tenant_id, dataset_id=dataset_id)
    document_scope = [Document.tenant_id == tenant_id, Document.dataset_id == dataset_id]
    if document_ids is not None:
        document_scope.append(Document.id.in_(document_ids))
    session.scalars(
        select(Document.id)
        .where(*document_scope)
        .order_by(Document.id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).all()

    segment_scope = [DocumentSegment.tenant_id == tenant_id, DocumentSegment.dataset_id == dataset_id]
    if document_ids is not None:
        segment_scope.append(DocumentSegment.document_id.in_(document_ids))
    if segment_ids is not None:
        segment_scope.append(DocumentSegment.id.in_(segment_ids))
    session.scalars(
        select(DocumentSegment.id)
        .where(*segment_scope)
        .order_by(DocumentSegment.id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).all()


def detach_owned_segment_attachments(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str] | None = None,
    segment_ids: Sequence[str] | None = None,
    additional_file_ids_to_lock: Sequence[str] = (),
) -> tuple[str, ...]:
    """Delete scoped attachment ownership rows in the caller's transaction.

    The owner and binding rows are locked before UploadFile rows. Global file
    cleanup probes the reverse owner edge with NOWAIT and defers on conflict,
    so the current binding set can be consumed without a stale pre-read.
    """
    normalized_document_ids = tuple(dict.fromkeys(document_ids)) if document_ids is not None else None
    if normalized_document_ids is not None and not normalized_document_ids:
        return ()
    normalized_segment_ids = tuple(dict.fromkeys(segment_ids)) if segment_ids is not None else None
    if normalized_segment_ids is not None and not normalized_segment_ids:
        return ()

    binding_scope = [
        SegmentAttachmentBinding.tenant_id == tenant_id,
        SegmentAttachmentBinding.dataset_id == dataset_id,
    ]
    if normalized_document_ids is not None:
        binding_scope.append(SegmentAttachmentBinding.document_id.in_(normalized_document_ids))
    if normalized_segment_ids is not None:
        binding_scope.append(SegmentAttachmentBinding.segment_id.in_(normalized_segment_ids))

    _lock_cleanup_owner_scope(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_ids=normalized_document_ids,
        segment_ids=normalized_segment_ids,
    )
    current_file_ids = tuple(
        session.scalars(
            select(SegmentAttachmentBinding.attachment_id)
            .where(*binding_scope)
            .order_by(SegmentAttachmentBinding.segment_id, SegmentAttachmentBinding.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    )
    lock_owned_upload_files(
        session,
        tenant_id=tenant_id,
        file_ids=(*current_file_ids, *additional_file_ids_to_lock),
    )
    session.execute(delete(SegmentAttachmentBinding).where(*binding_scope))
    return tuple(dict.fromkeys(current_file_ids))


def delete_document_segments_and_claim_uploads(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    document_ids: Sequence[str],
    segment_ids: Sequence[str] | None = None,
    candidate_upload_file_ids: Sequence[str] = (),
) -> tuple[UploadFileCleanupArtifact, ...]:
    """Delete segment-owned rows and persist file cleanup in one transaction.

    The caller must finish external index deletion before invoking this helper.
    Keeping these writes in the repository gives segment ownership one SQL
    mutation boundary and preserves every file id needed after a crash.
    """
    normalized_document_ids = tuple(dict.fromkeys(document_ids))
    if not normalized_document_ids:
        return ()
    normalized_segment_ids = tuple(dict.fromkeys(segment_ids)) if segment_ids is not None else None
    if normalized_segment_ids is not None and not normalized_segment_ids:
        return ()

    attachment_ids = detach_owned_segment_attachments(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_ids=normalized_document_ids,
        segment_ids=normalized_segment_ids,
        additional_file_ids_to_lock=candidate_upload_file_ids,
    )
    child_scope = [
        ChildChunk.tenant_id == tenant_id,
        ChildChunk.dataset_id == dataset_id,
        ChildChunk.document_id.in_(normalized_document_ids),
    ]
    summary_scope = [
        DocumentSegmentSummary.dataset_id == dataset_id,
        DocumentSegmentSummary.document_id.in_(normalized_document_ids),
    ]
    segment_scope = [
        DocumentSegment.tenant_id == tenant_id,
        DocumentSegment.dataset_id == dataset_id,
        DocumentSegment.document_id.in_(normalized_document_ids),
    ]
    if normalized_segment_ids is not None:
        child_scope.append(ChildChunk.segment_id.in_(normalized_segment_ids))
        summary_scope.append(DocumentSegmentSummary.chunk_id.in_(normalized_segment_ids))
        segment_scope.append(DocumentSegment.id.in_(normalized_segment_ids))
    children = session.scalars(select(ChildChunk).where(*child_scope).with_for_update()).all()
    add_terminal_child_row_cleanups(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        children=children,
    )
    summaries = session.scalars(select(DocumentSegmentSummary).where(*summary_scope).with_for_update()).all()
    add_terminal_summary_vector_cleanups(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        summaries=summaries,
    )
    session.execute(delete(ChildChunk).where(*child_scope))
    session.execute(delete(DocumentSegmentSummary).where(*summary_scope))
    session.execute(delete(DocumentSegment).where(*segment_scope))
    return claim_unreferenced_files_for_cleanup_in_session(
        session,
        workspace_id=tenant_id,
        file_ids=(*attachment_ids, *candidate_upload_file_ids),
    )


def delete_orphan_segments_and_claim_uploads(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
) -> tuple[UploadFileCleanupArtifact, ...]:
    """Delete rows left after their documents while preserving physical vector routes."""

    attachment_ids = detach_owned_segment_attachments(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
    )
    children = session.scalars(
        select(ChildChunk)
        .where(ChildChunk.tenant_id == tenant_id, ChildChunk.dataset_id == dataset_id)
        .order_by(ChildChunk.document_id, ChildChunk.segment_id, ChildChunk.id)
        .with_for_update()
    ).all()
    add_terminal_child_row_cleanups(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        children=children,
    )
    summaries = session.scalars(
        select(DocumentSegmentSummary)
        .where(DocumentSegmentSummary.dataset_id == dataset_id)
        .order_by(DocumentSegmentSummary.document_id, DocumentSegmentSummary.chunk_id, DocumentSegmentSummary.id)
        .with_for_update()
    ).all()
    add_terminal_summary_vector_cleanups(
        session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        summaries=summaries,
    )
    session.execute(delete(ChildChunk).where(ChildChunk.tenant_id == tenant_id, ChildChunk.dataset_id == dataset_id))
    session.execute(delete(DocumentSegmentSummary).where(DocumentSegmentSummary.dataset_id == dataset_id))
    session.execute(
        delete(DocumentSegment).where(
            DocumentSegment.tenant_id == tenant_id,
            DocumentSegment.dataset_id == dataset_id,
        )
    )
    return claim_unreferenced_files_for_cleanup_in_session(
        session,
        workspace_id=tenant_id,
        file_ids=attachment_ids,
    )

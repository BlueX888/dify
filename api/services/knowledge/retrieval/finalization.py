"""Canonical final-result selection for every knowledge retrieval entrypoint."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace

from sqlalchemy.orm import Session

from core.app.file_access import (
    bind_file_access_scope,
    get_current_file_access_scope,
    grant_retriever_segment_access,
    grant_upload_file_access,
)
from core.rag.datasource.retrieval_service import RetrievalService
from core.rag.embedding.retrieval import (
    AttachmentInfoDict,
    FinalizedInternalRetrieval,
    FinalizedRetrievalItem,
    FinalizedRetrievalResult,
    RetrievalSegments,
)
from core.rag.models.document import Document
from core.tools.signature import sign_upload_file_preview_url
from models.model import UploadFile
from repositories.knowledge.segment_read_adapter import sign_segment_content
from services.knowledge.retrieval.attachments import (
    authorize_retrieval_records,
    expand_attachment_document_filter,
    expand_retrieval_attachment_candidates,
)


def _signed_attachment_info(attachment: UploadFile) -> AttachmentInfoDict:
    """Map one revalidated attachment without consulting formatter-era metadata."""
    extension = "." + attachment.extension.lstrip(".")
    return {
        "id": attachment.id,
        "name": attachment.name,
        "extension": extension,
        "mime_type": attachment.mime_type,
        "source_url": sign_upload_file_preview_url(attachment.id, extension.lstrip(".")),
        "size": attachment.size,
    }


def _finalize_retrieval_candidates(
    *,
    format_session: Session,
    authorization_session: Session,
    candidates: Sequence[Document],
    tenant_id: str,
    dataset_ids: Sequence[str],
    top_k: int,
) -> FinalizedRetrievalResult:
    """Validate, rank, publish, and sign a retrieval result exactly once.

    Vector, keyword, summary, child, and attachment hits are all treated as
    untrusted candidates until the canonical formatter resolves their current
    parent segment and the complete persisted owner chain is revalidated.
    """
    external_documents = [document for document in candidates if document.provider == "external"]
    dify_documents = [document for document in candidates if document.provider == "dify"]
    expanded = expand_retrieval_attachment_candidates(
        dify_documents,
        tenant_id=tenant_id,
        dataset_ids=dataset_ids,
        session=format_session,
    )
    records = RetrievalService.format_retrieval_documents(format_session, list(expanded.candidates))
    eligible_internal = authorize_retrieval_records(
        records,
        matched_attachment_ids=expanded.matched_attachment_ids,
        tenant_id=tenant_id,
        dataset_ids=dataset_ids,
        session=authorization_session,
    )

    eligible: list[Document | tuple[RetrievalSegments, tuple[UploadFile, ...]]] = [
        *external_documents,
        *eligible_internal,
    ]

    def score(item: Document | tuple[RetrievalSegments, tuple[UploadFile, ...]]) -> float:
        value = item.metadata.get("score") if isinstance(item, Document) else item[0].score
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    eligible.sort(key=score, reverse=True)
    selected = eligible[:top_k] if top_k else eligible

    selected_internal = [item for item in selected if not isinstance(item, Document)]
    selected_segment_ids = [record.segment.id for record, _ in selected_internal]
    selected_file_ids = [file.id for _, attachments in selected_internal for file in attachments]
    scope = get_current_file_access_scope()
    staged_scope = (
        replace(
            scope,
            granted_retriever_segment_ids=scope.granted_retriever_segment_ids | frozenset(selected_segment_ids),
            granted_upload_file_ids=scope.granted_upload_file_ids | frozenset(selected_file_ids),
        )
        if scope is not None
        else None
    )
    staging_context = bind_file_access_scope(staged_scope) if staged_scope is not None else nullcontext()

    finalized: list[FinalizedRetrievalItem] = []
    with staging_context:
        for item in selected:
            if isinstance(item, Document):
                finalized.append(item)
                continue

            record, attachments = item
            segment = record.segment
            signed_files = tuple(_signed_attachment_info(attachment) for attachment in attachments)
            published_record = record.model_copy(update={"files": list(signed_files) or None})
            finalized.append(
                FinalizedInternalRetrieval(
                    record=published_record,
                    attachments=attachments,
                    content=sign_segment_content(
                        segment,
                        session=authorization_session,
                        allowed_upload_file_ids=frozenset(attachment.id for attachment in attachments),
                    ),
                    files=signed_files,
                )
            )

    grant_retriever_segment_access(selected_segment_ids)
    grant_upload_file_access(selected_file_ids)

    return FinalizedRetrievalResult(items=tuple(finalized))


class KnowledgeRetrievalFinalizer:
    """Persistence-aware implementation of the core retrieval finalization port."""

    def expand_document_ids_filter(
        self,
        *,
        session: Session,
        tenant_id: str,
        dataset_id: str,
        document_ids: Sequence[str],
    ) -> tuple[str, ...]:
        return expand_attachment_document_filter(
            session=session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_ids=document_ids,
        )

    def finalize(
        self,
        *,
        format_session: Session,
        authorization_session: Session,
        candidates: Sequence[Document],
        tenant_id: str,
        dataset_ids: Sequence[str],
        top_k: int,
    ) -> FinalizedRetrievalResult:
        return _finalize_retrieval_candidates(
            format_session=format_session,
            authorization_session=authorization_session,
            candidates=candidates,
            tenant_id=tenant_id,
            dataset_ids=dataset_ids,
            top_k=top_k,
        )

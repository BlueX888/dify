from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.rag.models.document import Document
from models.dataset import DocumentSegment
from models.model import UploadFile


class AttachmentInfoDict(TypedDict):
    id: str
    name: str
    extension: str
    mime_type: str
    source_url: str
    size: int


class RetrievalChildChunk(BaseModel):
    """Retrieval segments."""

    id: str
    content: str
    score: float
    position: int


class RetrievalSegments(BaseModel):
    """Retrieval segments."""

    model_config = {"arbitrary_types_allowed": True}
    segment: DocumentSegment
    child_chunks: list[RetrievalChildChunk] | None = None
    score: float | None = None
    files: list[AttachmentInfoDict] | None = None
    summary: str | None = None  # Summary content if retrieved via summary index
    matched_attachment_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FinalizedInternalRetrieval:
    """A persisted retrieval record whose access and presentation are published."""

    record: RetrievalSegments
    attachments: tuple[UploadFile, ...]
    content: str
    files: tuple[AttachmentInfoDict, ...]

    @property
    def score(self) -> float:
        return float(self.record.score or 0.0)

    def trace_document(self) -> Document:
        segment = self.record.segment
        return Document(
            page_content=segment.content,
            provider="dify",
            metadata={
                "dataset_id": segment.dataset_id,
                "document_id": segment.document_id,
                "doc_id": segment.index_node_id or segment.id,
                "segment_id": segment.id,
                "score": self.record.score,
            },
        )


type FinalizedRetrievalItem = Document | FinalizedInternalRetrieval


@dataclass(frozen=True, slots=True)
class FinalizedRetrievalResult:
    """The only records allowed to affect output, access scope, or statistics."""

    items: tuple[FinalizedRetrievalItem, ...]

    @property
    def trace_documents(self) -> list[Document]:
        return [item if isinstance(item, Document) else item.trace_document() for item in self.items]

    @property
    def segment_ids(self) -> list[str]:
        return [item.record.segment.id for item in self.items if isinstance(item, FinalizedInternalRetrieval)]

    @property
    def upload_file_ids(self) -> list[str]:
        return [
            attachment.id
            for item in self.items
            if isinstance(item, FinalizedInternalRetrieval)
            for attachment in item.attachments
        ]


@runtime_checkable
class RetrievalFinalizer(Protocol):
    """Resolve raw retrieval candidates into the sole publishable result set."""

    def expand_document_ids_filter(
        self,
        *,
        session: Session,
        tenant_id: str,
        dataset_id: str,
        document_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Expand logical document filters to provider-side physical identities."""

    def finalize(
        self,
        *,
        format_session: Session,
        authorization_session: Session,
        candidates: Sequence[Document],
        tenant_id: str,
        dataset_ids: Sequence[str],
        top_k: int,
    ) -> FinalizedRetrievalResult: ...

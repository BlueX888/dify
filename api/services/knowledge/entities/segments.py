"""Detached knowledge segment values shared by application ports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from services.knowledge.entities.vector_route import VectorRoute

ChildVectorRoute = VectorRoute


class ChildChunkUpdateArgs(BaseModel):
    id: str | None = Field(default=None, description="Existing child chunk ID. Omit to create a new child chunk.")
    content: str = Field(description="Child chunk text content.")


class SegmentAttachmentRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: str
    name: str
    size: int
    extension: str
    mime_type: str | None
    source_url: str


@dataclass(frozen=True, slots=True)
class SegmentAttachmentFile:
    id: str
    name: str
    size: int
    extension: str
    mime_type: str


@dataclass(frozen=True, slots=True)
class SegmentFileAccess:
    uploads: frozenset[str]
    tools: frozenset[str]


@dataclass(frozen=True, slots=True)
class SegmentStateIntent:
    id: str


@dataclass(frozen=True, slots=True)
class ChildVectorWriteIntent:
    """Stable ownership of one child-vector write before external I/O."""

    id: str
    tenant_id: str
    dataset_id: str
    node_id: str
    writer_id: str
    claim_id: str
    route: ChildVectorRoute


@dataclass(frozen=True, slots=True)
class ChildVectorCleanupClaim:
    id: str
    tenant_id: str
    dataset_id: str
    node_id: str
    claim_id: str
    writer_finished: bool
    route: ChildVectorRoute | None


@dataclass(frozen=True, slots=True)
class ChildVectorPublication:
    """Terminal writer claims to consume with the child-row transaction."""

    claims: tuple[ChildVectorCleanupClaim, ...] = ()


@dataclass(frozen=True, slots=True)
class ChildVectorGuard:
    """Dataset generation and vector configuration captured under its shared lock."""

    required: bool
    generation_id: str | None
    indexing_technique: str
    embedding_model_provider: str | None
    embedding_model: str | None
    index_struct: str | None
    collection_binding_id: str | None


@dataclass(frozen=True, slots=True)
class SegmentCreateData:
    """Complete persistence input for one newly allocated segment."""

    id: str
    content: str
    word_count: int
    tokens: int
    created_by: str
    status: str
    answer: str | None = None
    keywords: tuple[str, ...] | None = ()
    index_node_id: str | None = None
    index_node_hash: str | None = None
    indexing_at: datetime | None = None
    completed_at: datetime | None = None
    attachment_ids: tuple[str, ...] = ()


class ChildChunkRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: str
    segment_id: str
    content: str
    position: int
    word_count: int
    type: str
    created_at: datetime
    updated_at: datetime


class SegmentRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: str
    position: int
    document_id: str
    content: str
    sign_content: str
    answer: str | None
    word_count: int
    tokens: int
    keywords: tuple[str, ...] | None
    index_node_id: str | None
    index_node_hash: str | None
    hit_count: int
    enabled: bool
    disabled_at: datetime | None
    disabled_by: str | None
    status: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    updated_by: str | None
    indexing_at: datetime | None
    completed_at: datetime | None
    error: str | None
    stopped_at: datetime | None
    child_chunks: tuple[ChildChunkRecord, ...]
    attachments: tuple[SegmentAttachmentRecord, ...]
    summary: str | None


@dataclass(frozen=True, slots=True)
class SegmentReadSnapshot:
    values: Mapping[str, object]
    child_chunks: tuple[ChildChunkRecord, ...]
    attachments: tuple[SegmentAttachmentFile, ...]
    summary: str | None
    file_access: SegmentFileAccess


class SegmentUpdateArgs(BaseModel):
    content: str | None = Field(default=None, description="Updated chunk text content.")
    answer: str | None = Field(default=None, description="Updated answer content for QA mode.")
    keywords: list[str] | None = Field(default=None, description="Updated keywords for the chunk.")
    regenerate_child_chunks: bool = Field(
        default=False,
        description="Whether to regenerate child chunks after updating a parent chunk.",
    )
    enabled: bool | None = Field(default=None, description="Whether the chunk is enabled.")
    attachment_ids: list[str] | None = Field(default=None, description="Attachment file IDs.")
    summary: str | None = Field(default=None, description="Summary content for summary index.")

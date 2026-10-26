"""Materialized dataset read contracts, independent of persistence and HTTP.

Timestamps remain datetimes until transport serialization. JSON configuration
objects retain their stored keys; this boundary does not apply response defaults.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import NamedTuple, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, JsonValue

from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.vector_route import VectorRoute


class DatasetIndexingGenerationLostError(RuntimeError):
    """An index writer no longer owns the dataset representation it captured."""


class DatasetIndexingRepresentation(BaseModel):
    """Immutable logical configuration plus durable physical-route evidence.

    ``Dataset`` configuration columns are the desired representation and do
    not prove that vectors were published there. The Dataset
    ``indexing_applied_representation`` receipt is advanced only when a vector
    phase completes its final database transaction; its ``vector_route`` is
    therefore the last fully applied physical identity.

    A new outbox target starts with ``vector_route=None``. Immediately before
    its first external mutation, the worker resolves the actual backend and
    collection and persists that route into the outbox target. If the worker
    loses ownership or its commit outcome is uncertain, this started-target
    receipt remains available for a superseding generation to clean. A target
    that never started has no physical identity to inherit.
    """

    model_config = ConfigDict(frozen=True)

    indexing_technique: str
    index_struct: str | None
    collection_binding_id: str | None
    embedding_model: str | None
    embedding_model_provider: str | None
    keyword_number: int | None
    is_multimodal: bool
    doc_form: str | None
    summary_index_setting: Mapping[str, JsonValue] | None
    # Exact applied/source or started-target destination; None is also valid
    # for economy indexing, which has no vector destination.
    vector_route: VectorRoute | None = None
    # Physical destinations inherited from superseded or partial generations.
    cleanup_routes: tuple[VectorRoute, ...] = ()


class DatasetIndexingJob(NamedTuple):
    """One source-to-target transition committed with a dataset configuration."""

    dataset: DatasetRef
    source: DatasetIndexingRepresentation
    target: DatasetIndexingRepresentation


class DatasetTagRecord(TypedDict):
    id: str
    name: str
    type: str


class DatasetDetailRecord(TypedDict):
    id: str
    name: str
    description: str | None
    provider: str
    permission: str
    data_source_type: str | None
    indexing_technique: str | None
    keyword_number: int | None
    app_count: int
    document_count: int
    word_count: int
    created_by: str
    author_name: str | None
    created_at: datetime
    updated_by: str | None
    updated_at: datetime
    embedding_model: str | None
    embedding_model_provider: str | None
    # None means availability has not been evaluated by the application service.
    embedding_available: bool | None
    retrieval_model_dict: Mapping[str, JsonValue]
    summary_index_setting: Mapping[str, JsonValue] | None
    tags: list[DatasetTagRecord]
    doc_form: str | None
    external_knowledge_info: Mapping[str, JsonValue] | None
    external_retrieval_model: Mapping[str, JsonValue] | None
    doc_metadata: Sequence[Mapping[str, str]]
    built_in_field_enabled: bool
    pipeline_id: str | None
    runtime_mode: str | None
    chunk_structure: str | None
    icon_info: Mapping[str, JsonValue] | None
    is_published: bool
    total_documents: int
    total_available_documents: int
    enable_api: bool
    is_multimodal: bool
    permission_keys: list[str]
    maintainer: str | None
    # Detail/create responses may omit membership or explicitly return null;
    # listing and updates materialize a list, including an empty one.
    partial_member_list: NotRequired[list[str] | None]


class DatasetPage(TypedDict):
    data: list[DatasetDetailRecord]
    has_more: bool
    total: int
    page: int
    limit: int

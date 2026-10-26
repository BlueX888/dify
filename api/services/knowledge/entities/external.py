"""Shared external knowledge contracts, independent of service implementations."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import AnyHttpUrl, BaseModel, Field, TypeAdapter, ValidationError

_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


def validate_external_knowledge_api_settings(settings: Mapping[str, Any]) -> None:
    """Validate the shared template settings before persistence or network I/O."""
    if not settings:
        raise ValueError("api list is empty")

    endpoint = settings.get("endpoint")
    if not endpoint:
        raise ValueError("endpoint is required")
    if not isinstance(endpoint, str):
        raise ValueError("endpoint must be a string")

    if endpoint.partition(":")[0].lower() not in {"http", "https"}:
        raise ValueError(f"invalid endpoint: {endpoint} must start with http:// or https://")
    try:
        parsed_url = _HTTP_URL_ADAPTER.validate_python(endpoint, strict=True)
    except ValidationError as error:
        raise ValueError(f"invalid endpoint: {endpoint}") from error
    if parsed_url.port == 0:
        raise ValueError(f"invalid endpoint: {endpoint}")

    api_key = settings.get("api_key")
    if not api_key:
        raise ValueError("api_key is required")
    if not isinstance(api_key, str):
        raise ValueError("api_key must be a string")


class ExternalKnowledgeApiSetting(BaseModel):
    url: str
    headers: dict[str, str]
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExternalKnowledgeRetrievalRequest:
    """Owned provider configuration and retrieval input, independent of HTTP."""

    endpoint: str
    api_key: str | None
    knowledge_id: str
    query: str
    retrieval_model: dict[str, Any]
    metadata_filters: dict[str, Any] | None


class ExternalDatasetCreatePayload(BaseModel):
    """Validated fields required to create an external dataset binding.

    The console controller owns HTTP concerns, but the service also needs this
    contract when creating the tenant-scoped dataset and external knowledge
    binding. Keep it outside controllers so service imports do not depend on
    Flask blueprint initialization.
    """

    external_knowledge_api_id: str
    external_knowledge_id: str
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=400)
    external_retrieval_model: dict[str, object] | None = Field(default=None)

"""Shared HTTP translation for knowledge hit-testing failures."""

from dataclasses import dataclass
from typing import Never

from core.errors.error import (
    LLMBadRequestError,
    ModelCurrentlyNotSupportError,
    ProviderTokenNotInitError,
    QuotaExceededError,
)
from libs.exception import BaseHTTPException
from services.errors.dataset import DatasetAccessDeniedError, DatasetNotFoundError
from services.errors.index import IndexNotInitializedError
from services.errors.knowledge_retrieval import HitTestingFailedError, HitTestingInvocationError


@dataclass(frozen=True, slots=True)
class HitTestingHttpErrors:
    """Surface-owned HTTP errors used by the shared domain-error mapping."""

    not_found: type[BaseHTTPException]
    access_denied: type[BaseHTTPException]
    not_initialized: type[BaseHTTPException]
    provider_not_initialized: type[BaseHTTPException]
    provider_quota_exceeded: type[BaseHTTPException]
    model_not_supported: type[BaseHTTPException]
    completion_request: type[BaseHTTPException]
    internal_server: type[BaseHTTPException]


def raise_hit_testing_http_error(error: Exception, errors: HitTestingHttpErrors) -> Never:
    """Translate one hit-testing failure without changing its public status or code."""

    if isinstance(error, DatasetNotFoundError):
        raise errors.not_found(description="Dataset not found.") from error
    if isinstance(error, DatasetAccessDeniedError):
        raise errors.access_denied(description=str(error)) from error
    if isinstance(error, IndexNotInitializedError):
        raise errors.not_initialized() from error
    if isinstance(error, ProviderTokenNotInitError):
        raise errors.provider_not_initialized(error.description) from error
    if isinstance(error, QuotaExceededError):
        raise errors.provider_quota_exceeded() from error
    if isinstance(error, ModelCurrentlyNotSupportError):
        raise errors.model_not_supported() from error
    if isinstance(error, LLMBadRequestError):
        raise errors.provider_not_initialized(
            "No Embedding Model or Reranking Model available. Please configure a valid provider "
            "in the Settings -> Model Provider."
        ) from error
    if isinstance(error, HitTestingInvocationError):
        raise errors.completion_request(str(error)) from error
    if isinstance(error, HitTestingFailedError):
        raise errors.internal_server() from error
    raise error

"""Tenant- and provider-scoped cache keys for website crawl jobs."""

from hashlib import sha256
from urllib.parse import SplitResult, urlsplit, urlunsplit

_SUPPORTED_PROVIDERS = frozenset({"firecrawl", "watercrawl", "jinareader"})
_DEFAULT_BASE_URLS = {
    "firecrawl": "https://api.firecrawl.dev",
    "watercrawl": "https://app.watercrawl.dev",
    "jinareader": "https://adaptivecrawl-kir3wx7b3a-uc.a.run.app",
}


def website_crawl_provider_instance_id(*, provider: str, base_url: str | None = None) -> str:
    """Return a non-secret stable identity for one provider endpoint."""
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError("Invalid provider")
    effective_url = (
        _DEFAULT_BASE_URLS[provider] if provider == "jinareader" else base_url or _DEFAULT_BASE_URLS[provider]
    )
    canonical_url = _normalize_provider_base_url(effective_url, origin_only=provider == "watercrawl")
    return sha256(f"{provider}\0{canonical_url}".encode()).hexdigest()


def website_crawl_legacy_watercrawl_instance_id(*, base_url: str | None = None) -> str:
    """Reproduce the former path-sensitive Watercrawl identity for upgrade checks only."""
    effective_url = base_url or _DEFAULT_BASE_URLS["watercrawl"]
    canonical_url = _normalize_provider_base_url(effective_url, origin_only=False)
    return sha256(f"watercrawl\0{canonical_url}".encode()).hexdigest()


def website_crawl_cache_key(*, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> str:
    """Build an object-storage key without embedding the untrusted remote job ID."""
    _validate_scope(
        tenant_id=tenant_id,
        provider=provider,
        provider_instance_id=provider_instance_id,
        job_id=job_id,
    )
    job_digest = sha256(job_id.encode("utf-8")).hexdigest()
    return f"website_files/{tenant_id}/{provider}/{provider_instance_id}/{job_digest}.json"


def website_crawl_timer_key(*, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> str:
    """Build the matching Redis key used to report crawl duration."""
    _validate_scope(
        tenant_id=tenant_id,
        provider=provider,
        provider_instance_id=provider_instance_id,
        job_id=job_id,
    )
    job_digest = sha256(job_id.encode("utf-8")).hexdigest()
    return f"website_crawl:{tenant_id}:{provider}:{provider_instance_id}:{job_digest}"


def _normalize_provider_base_url(value: str, *, origin_only: bool) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Invalid provider base URL")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("Invalid provider base URL")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Invalid provider base URL") from error
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    authority = host if port is None or port == default_port else f"{host}:{port}"
    path = "" if origin_only else parsed.path.rstrip("/")
    return urlunsplit(SplitResult(parsed.scheme.lower(), authority, path, "", ""))


def _validate_scope(*, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> None:
    if not tenant_id or any(char in tenant_id for char in "/\\\x00"):
        raise ValueError("Invalid tenant ID")
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError("Invalid provider")
    if len(provider_instance_id) != 64 or any(char not in "0123456789abcdef" for char in provider_instance_id):
        raise ValueError("Invalid provider instance ID")
    if not job_id:
        raise ValueError("Invalid job ID")

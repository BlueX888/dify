from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from typing import Any, NotRequired, Protocol, TypedDict, cast

import httpx
from flask_login import current_user

from core.helper import encrypter
from core.helper.http_client_pooling import get_pooled_http_client
from core.rag.extractor.entity.extract_setting import WebsiteJobRef
from core.rag.extractor.firecrawl.firecrawl_app import CrawlStatusResponse, FirecrawlApp, FirecrawlDocumentData
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from services.data_source.provider_service import DatasourceProviderService
from services.data_source.watercrawl_provider import WaterCrawlProvider
from services.data_source.website_crawl_cache import (
    website_crawl_cache_key,
    website_crawl_legacy_watercrawl_instance_id,
    website_crawl_provider_instance_id,
    website_crawl_timer_key,
)

# Reuse pooled HTTP clients to avoid creating new connections per request and ease testing.
# Both clients carry a bounded read/connect timeout so a stalled Jina or
# adaptive-crawl endpoint fails fast instead of pinning a worker. The values
# match the floor used in the WaterCrawl PR (#37512). See #39859.
_jina_http_client: httpx.Client = get_pooled_http_client(
    "website:jinareader",
    lambda: httpx.Client(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
    ),
)
_adaptive_http_client: httpx.Client = get_pooled_http_client(
    "website:adaptivecrawl",
    lambda: httpx.Client(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
    ),
)


@dataclass
class CrawlOptions:
    """Options for crawling operations."""

    limit: int = 1
    crawl_sub_pages: bool = False
    only_main_content: bool = False
    includes: str | None = None
    excludes: str | None = None
    prompt: str | None = None
    max_depth: int | None = None
    use_sitemap: bool = True

    def get_include_paths(self) -> list[str]:
        """Get list of include paths from comma-separated string."""
        return self.includes.split(",") if self.includes else []

    def get_exclude_paths(self) -> list[str]:
        """Get list of exclude paths from comma-separated string."""
        return self.excludes.split(",") if self.excludes else []


@dataclass
class CrawlRequest:
    """Request container for crawling operations."""

    url: str
    provider: str
    options: CrawlOptions


@dataclass
class ScrapeRequest:
    """Request container for scraping operations."""

    provider: str
    url: str
    tenant_id: str
    only_main_content: bool


@dataclass
class WebsiteCrawlApiRequest:
    """Request container for website crawl API arguments."""

    provider: str
    url: str
    options: dict[str, Any]

    def to_crawl_request(self) -> CrawlRequest:
        """Convert API request to internal CrawlRequest."""
        options = CrawlOptions(
            limit=self.options.get("limit", 1),
            crawl_sub_pages=self.options.get("crawl_sub_pages", False),
            only_main_content=self.options.get("only_main_content", False),
            includes=self.options.get("includes"),
            excludes=self.options.get("excludes"),
            prompt=self.options.get("prompt"),
            max_depth=self.options.get("max_depth"),
            use_sitemap=self.options.get("use_sitemap", True),
        )
        return CrawlRequest(url=self.url, provider=self.provider, options=options)

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> WebsiteCrawlApiRequest:
        """Create from Flask-RESTful parsed arguments."""
        provider = args.get("provider")
        url = args.get("url")
        options = args.get("options", {})

        if not provider:
            raise ValueError("Provider is required")
        if not url:
            raise ValueError("URL is required")
        if not options:
            raise ValueError("Options are required")

        return cls(provider=provider, url=url, options=options)


@dataclass
class WebsiteCrawlStatusApiRequest:
    """Request container for website crawl status API arguments."""

    provider: str
    job_id: str
    source_handle: str | None = None

    @classmethod
    def from_args(cls, args: dict[str, Any], job_id: str) -> WebsiteCrawlStatusApiRequest:
        """Create from Flask-RESTful parsed arguments."""
        provider = args.get("provider")
        source_handle = args.get("source_handle")
        if not provider:
            raise ValueError("Provider is required")
        if not job_id:
            raise ValueError("Job ID is required")
        if not source_handle or not isinstance(source_handle, str):
            raise ValueError("Source handle is required")

        return cls(provider=provider, job_id=job_id, source_handle=source_handle)


class CrawlStatusDict(TypedDict):
    status: str
    job_id: str
    total: int
    current: int
    data: list[Any]
    source_handle: NotRequired[str]
    time_consuming: NotRequired[str | float]


class WebsiteCrawlJobStore(Protocol):
    """Persist and verify the tenant/provider boundary of remote crawl jobs."""

    def register_created(
        self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str
    ) -> str | None: ...

    def claim_verified(
        self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str
    ) -> str | None: ...

    def get_owned_identity(self, *, tenant_id: str, source_handle: str) -> tuple[str, str, str] | None: ...

    def is_owned_by(self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> bool: ...

    def has_owner(self, *, provider: str, provider_instance_id: str, job_id: str) -> bool: ...

    def owned_provider_instance_ids(self, *, tenant_id: str, provider: str, job_id: str) -> tuple[str, ...]: ...


class WebsiteService:
    """Service class for website crawling operations using different providers."""

    def __init__(self, *, providers: DatasourceProviderService, crawl_jobs: WebsiteCrawlJobStore) -> None:
        self._providers = providers
        self._crawl_jobs = crawl_jobs

    def _get_credentials_and_config(self, tenant_id: str, provider: str) -> tuple[Any, Any]:
        """Get and validate credentials for a provider."""
        if provider == "firecrawl":
            plugin_id = "langgenius/firecrawl_datasource"
        elif provider == "watercrawl":
            plugin_id = "watercrawl/watercrawl_datasource"
        elif provider == "jinareader":
            plugin_id = "langgenius/jina_datasource"
        else:
            raise ValueError("Invalid provider")
        credential = self._providers.get_datasource_credentials(
            tenant_id=tenant_id,
            provider=provider,
            plugin_id=plugin_id,
        )
        if provider == "firecrawl":
            return credential.get("firecrawl_api_key"), credential
        elif provider in {"watercrawl", "jinareader"}:
            return credential.get("api_key"), credential
        else:
            raise ValueError("Invalid provider")

    @classmethod
    def _get_decrypted_api_key(cls, tenant_id: str, config: dict[str, Any]) -> str:
        """Decrypt and return the API key from config."""
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("API key not found in configuration")
        return encrypter.decrypt_token(tenant_id=tenant_id, token=api_key)

    @classmethod
    def document_create_args_validate(cls, args: dict[str, Any]):
        """Validate arguments for document creation."""
        try:
            WebsiteCrawlApiRequest.from_args(args)
        except ValueError as e:
            raise ValueError(f"Invalid arguments: {e}") from e

    def crawl_url(self, api_request: WebsiteCrawlApiRequest) -> dict[str, Any]:
        """Crawl a URL using the specified provider with typed request."""
        request = api_request.to_crawl_request()
        tenant_id = current_user.current_tenant_id

        api_key, config = self._get_credentials_and_config(tenant_id, request.provider)
        provider_instance_id = website_crawl_provider_instance_id(
            provider=request.provider,
            base_url=config.get("base_url"),
        )

        if request.provider == "firecrawl":
            result = self._crawl_with_firecrawl(request=request, api_key=api_key, config=config)
        elif request.provider == "watercrawl":
            result = self._crawl_with_watercrawl(request=request, api_key=api_key, config=config)
        elif request.provider == "jinareader":
            result = self._crawl_with_jinareader(request=request, api_key=api_key)
        else:
            raise ValueError("Invalid provider")

        job_id = result.get("job_id")
        if job_id is not None:
            if not isinstance(job_id, str) or not job_id:
                raise ValueError("Invalid crawl job ID")
            self._validate_provider_job_id(job_id)
            if request.provider == "watercrawl":
                self._reject_foreign_legacy_owner(
                    tenant_id=tenant_id,
                    provider=request.provider,
                    provider_instance_id=provider_instance_id,
                    legacy_provider_instance_id=website_crawl_legacy_watercrawl_instance_id(
                        base_url=config.get("base_url")
                    ),
                    job_id=job_id,
                )
            source_handle = self._register_created_job(
                tenant_id=tenant_id,
                provider=request.provider,
                provider_instance_id=provider_instance_id,
                job_id=job_id,
            )
            result["source_handle"] = source_handle
            if request.provider == "firecrawl":
                redis_client.setex(
                    website_crawl_timer_key(
                        tenant_id=tenant_id,
                        provider=request.provider,
                        provider_instance_id=provider_instance_id,
                        job_id=job_id,
                    ),
                    3600,
                    str(datetime.datetime.now().timestamp()),
                )
        return result

    def resolve_job_ref(
        self,
        *,
        tenant_id: str,
        provider: str,
        job_id: str,
        source_handle: str | None = None,
    ) -> WebsiteJobRef:
        """Resolve a caller-selected job only through its server-issued handle."""
        job_ref = self._resolve_job_ref_from_handle(
            tenant_id=tenant_id,
            provider=provider,
            job_id=job_id,
            source_handle=source_handle,
        )
        # The handle determines identity. Current credentials may authorize that
        # exact provider instance, but must never reinterpret it as another one.
        self._get_credentials_for_job_ref(job_ref)
        return job_ref

    def _resolve_job_ref_from_handle(
        self,
        *,
        tenant_id: str,
        provider: str,
        job_id: str,
        source_handle: str | None,
    ) -> WebsiteJobRef:
        if provider == "jinareader" and not job_id:
            if source_handle is not None:
                raise ValueError("Source handle does not match crawl job")
            return WebsiteJobRef(
                tenant_id=tenant_id,
                provider=provider,
                provider_instance_id=website_crawl_provider_instance_id(provider=provider),
                job_id=job_id,
            )
        self._validate_job_lookup_inputs(provider=provider, job_id=job_id)
        if source_handle is None:
            raise ValueError("Website source handle is required")
        identity = self._crawl_jobs.get_owned_identity(tenant_id=tenant_id, source_handle=source_handle)
        if identity is None:
            raise ValueError("Crawl job not found")
        owned_provider, provider_instance_id, owned_job_id = identity
        if owned_provider != provider or owned_job_id != job_id:
            raise ValueError("Source handle does not match crawl job")
        return WebsiteJobRef(
            tenant_id=tenant_id,
            provider=provider,
            provider_instance_id=provider_instance_id,
            job_id=job_id,
        )

    def resolve_stored_job_ref(self, *, tenant_id: str, provider: str, job_id: str) -> WebsiteJobRef:
        """Recover a legacy source only from unambiguous durable ownership evidence."""
        if provider == "jinareader" and not job_id:
            return WebsiteJobRef(
                tenant_id=tenant_id,
                provider=provider,
                provider_instance_id=website_crawl_provider_instance_id(provider=provider),
                job_id=job_id,
            )
        self._validate_job_lookup_inputs(provider=provider, job_id=job_id)
        owned_instance_ids = self._crawl_jobs.owned_provider_instance_ids(
            tenant_id=tenant_id,
            provider=provider,
            job_id=job_id,
        )
        if not owned_instance_ids:
            raise ValueError("Crawl job provider instance is not recorded")
        owned_instances = set(owned_instance_ids)
        if provider == "watercrawl":
            api_key, config = self._get_credentials_and_config(tenant_id, provider)
            provider_instance_id = website_crawl_provider_instance_id(
                provider=provider,
                base_url=config.get("base_url"),
            )
            legacy_provider_instance_id = website_crawl_legacy_watercrawl_instance_id(base_url=config.get("base_url"))
            if owned_instances <= {provider_instance_id, legacy_provider_instance_id}:
                if provider_instance_id not in owned_instances:
                    self._verify_remote_job_access(job_id=job_id, provider=provider, api_key=api_key, config=config)
                    self._claim_verified_job(
                        tenant_id=tenant_id,
                        provider=provider,
                        provider_instance_id=provider_instance_id,
                        job_id=job_id,
                    )
                owned_instance_id = provider_instance_id
            elif len(owned_instances) != 1:
                raise ValueError("Crawl job provider instance is ambiguous")
            else:
                owned_instance_id = next(iter(owned_instances))
        else:
            if len(owned_instances) != 1:
                raise ValueError("Crawl job provider instance is ambiguous")
            owned_instance_id = next(iter(owned_instances))
        return WebsiteJobRef(
            tenant_id=tenant_id,
            provider=provider,
            provider_instance_id=owned_instance_id,
            job_id=job_id,
        )

    def get_crawl_url_data_by_ref(
        self, job_ref: WebsiteJobRef, url: str, *, use_job: bool = True
    ) -> dict[str, Any] | None:
        """Read a crawl result without reinterpreting its persisted provider instance."""
        api_key, config = self._get_credentials_for_job_ref(job_ref)
        if job_ref.provider == "firecrawl":
            return self._get_firecrawl_url_data(
                job_ref.job_id,
                job_ref.tenant_id,
                job_ref.provider_instance_id,
                url,
                api_key,
                config,
            )
        if job_ref.provider == "watercrawl":
            return self._get_watercrawl_url_data(job_ref.job_id, url, api_key, config)
        if job_ref.provider == "jinareader":
            return self._get_jinareader_url_data(job_ref.job_id if use_job else "", url, api_key)
        raise ValueError("Invalid provider")

    def get_scrape_url_data_by_ref(self, job_ref: WebsiteJobRef, url: str, only_main_content: bool) -> dict[str, Any]:
        """Scrape through the same provider instance that owns the persisted source."""
        api_key, config = self._get_credentials_for_job_ref(job_ref)
        request = ScrapeRequest(
            provider=job_ref.provider,
            url=url,
            tenant_id=job_ref.tenant_id,
            only_main_content=only_main_content,
        )
        if job_ref.provider == "firecrawl":
            return self._scrape_with_firecrawl(request=request, api_key=api_key, config=config)
        if job_ref.provider == "watercrawl":
            return self._scrape_with_watercrawl(request=request, api_key=api_key, config=config)
        raise ValueError("Invalid provider")

    def _get_credentials_for_job_ref(self, job_ref: WebsiteJobRef) -> tuple[Any, dict[str, Any]]:
        if job_ref.provider not in {"firecrawl", "watercrawl", "jinareader"}:
            raise ValueError("Invalid provider")
        api_key, config = self._get_credentials_and_config(job_ref.tenant_id, job_ref.provider)
        current_instance_id = website_crawl_provider_instance_id(
            provider=job_ref.provider,
            base_url=config.get("base_url"),
        )
        if current_instance_id != job_ref.provider_instance_id:
            raise ValueError("Website provider endpoint has changed")
        if not job_ref.job_id:
            if job_ref.provider != "jinareader":
                raise ValueError("Crawl job not found")
            return api_key, config
        if not self._crawl_jobs.is_owned_by(
            tenant_id=job_ref.tenant_id,
            provider=job_ref.provider,
            provider_instance_id=job_ref.provider_instance_id,
            job_id=job_ref.job_id,
        ):
            raise ValueError("Crawl job not found")
        return api_key, config

    @classmethod
    def _crawl_with_firecrawl(cls, request: CrawlRequest, api_key: str, config: dict[str, Any]) -> dict[str, Any]:
        firecrawl_app = FirecrawlApp(api_key=api_key, base_url=config.get("base_url"))

        params: dict[str, Any]
        if not request.options.crawl_sub_pages:
            params = {
                "includePaths": [],
                "excludePaths": [],
                "limit": 1,
                "scrapeOptions": {"onlyMainContent": request.options.only_main_content},
            }
        else:
            params = {
                "includePaths": request.options.get_include_paths(),
                "excludePaths": request.options.get_exclude_paths(),
                "limit": request.options.limit,
                "scrapeOptions": {"onlyMainContent": request.options.only_main_content},
            }

        # Add optional prompt for Firecrawl v2 crawl-params compatibility
        if request.options.prompt:
            params["prompt"] = request.options.prompt

        job_id = firecrawl_app.crawl_url(request.url, params)
        return {"status": "active", "job_id": job_id}

    @classmethod
    def _crawl_with_watercrawl(cls, request: CrawlRequest, api_key: str, config: dict[str, Any]) -> dict[str, Any]:
        # Convert CrawlOptions back to dict format for WaterCrawlProvider
        options = {
            "limit": request.options.limit,
            "crawl_sub_pages": request.options.crawl_sub_pages,
            "only_main_content": request.options.only_main_content,
            "includes": request.options.includes,
            "excludes": request.options.excludes,
            "max_depth": request.options.max_depth,
            "use_sitemap": request.options.use_sitemap,
        }
        return dict(
            WaterCrawlProvider(api_key=api_key, base_url=config.get("base_url")).crawl_url(
                url=request.url, options=options
            )
        )

    @classmethod
    def _crawl_with_jinareader(cls, request: CrawlRequest, api_key: str) -> dict[str, Any]:
        if not request.options.crawl_sub_pages:
            response = _jina_http_client.get(
                f"https://r.jina.ai/{request.url}",
                headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            if response.json().get("code") != 200:
                raise ValueError("Failed to crawl:")
            return {"status": "active", "data": response.json().get("data")}
        else:
            response = _adaptive_http_client.post(
                "https://adaptivecrawl-kir3wx7b3a-uc.a.run.app",
                json={
                    "url": request.url,
                    "maxPages": request.options.limit,
                    "useSitemap": request.options.use_sitemap,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            if response.json().get("code") != 200:
                raise ValueError("Failed to crawl")
            return {"status": "active", "job_id": response.json().get("data", {}).get("taskId")}

    def get_crawl_status_typed(self, api_request: WebsiteCrawlStatusApiRequest) -> CrawlStatusDict:
        """Get crawl status through the stable identity behind a server-issued handle."""
        tenant_id = current_user.current_tenant_id
        job_ref = self._resolve_job_ref_from_handle(
            tenant_id=tenant_id,
            provider=api_request.provider,
            job_id=api_request.job_id,
            source_handle=api_request.source_handle,
        )
        api_key, config = self._get_credentials_for_job_ref(job_ref)

        if api_request.provider == "firecrawl":
            result = self._get_firecrawl_status(
                api_request.job_id,
                tenant_id,
                job_ref.provider_instance_id,
                api_key,
                config,
            )
        elif api_request.provider == "watercrawl":
            result = self._get_watercrawl_status(api_request.job_id, api_key, config)
        elif api_request.provider == "jinareader":
            result = self._get_jinareader_status(api_request.job_id, api_key)
        else:
            raise ValueError("Invalid provider")
        assert api_request.source_handle is not None
        result["source_handle"] = api_request.source_handle
        return result

    @classmethod
    def _get_firecrawl_status(
        cls,
        job_id: str,
        tenant_id: str,
        provider_instance_id: str,
        api_key: str,
        config: dict[str, Any],
    ) -> CrawlStatusDict:
        result = cls._fetch_firecrawl_status(job_id, api_key, config)
        return cls._build_firecrawl_status(job_id, tenant_id, provider_instance_id, result)

    @staticmethod
    def _fetch_firecrawl_status(job_id: str, api_key: str, config: dict[str, Any]) -> CrawlStatusResponse:
        firecrawl_app = FirecrawlApp(api_key=api_key, base_url=config.get("base_url"))
        return firecrawl_app.check_crawl_status(job_id)

    @classmethod
    def _build_firecrawl_status(
        cls,
        job_id: str,
        tenant_id: str,
        provider_instance_id: str,
        result: CrawlStatusResponse,
    ) -> CrawlStatusDict:
        crawl_status_data: CrawlStatusDict = {
            "status": result["status"],
            "job_id": job_id,
            "total": result["total"] or 0,
            "current": result["current"] or 0,
            "data": result["data"],
        }
        if crawl_status_data["status"] == "completed":
            cls._save_firecrawl_crawl_data(
                website_crawl_cache_key(
                    tenant_id=tenant_id,
                    provider="firecrawl",
                    provider_instance_id=provider_instance_id,
                    job_id=job_id,
                ),
                result["data"],
            )
            website_crawl_time_cache_key = website_crawl_timer_key(
                tenant_id=tenant_id,
                provider="firecrawl",
                provider_instance_id=provider_instance_id,
                job_id=job_id,
            )
            start_time = redis_client.get(website_crawl_time_cache_key)
            if start_time:
                end_time = datetime.datetime.now().timestamp()
                time_consuming = abs(end_time - float(start_time))
                crawl_status_data["time_consuming"] = f"{time_consuming:.2f}"
                redis_client.delete(website_crawl_time_cache_key)
        return crawl_status_data

    @classmethod
    def _get_watercrawl_status(cls, job_id: str, api_key: str, config: dict[str, Any]) -> CrawlStatusDict:
        return cast(CrawlStatusDict, dict(WaterCrawlProvider(api_key, config.get("base_url")).get_crawl_status(job_id)))

    @classmethod
    def _get_jinareader_status(cls, job_id: str, api_key: str) -> CrawlStatusDict:
        response = _adaptive_http_client.post(
            "https://adaptivecrawlstatus-kir3wx7b3a-uc.a.run.app",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"taskId": job_id},
        )
        data = cls._require_jina_response_data(response, expected_job_id=job_id)
        status = data.get("status")
        if status not in {"active", "pending", "processing", "running", "completed", "failed", "cancelled"}:
            raise ValueError("Invalid Jina Reader crawl status response")
        urls = data.get("urls", [])
        processed = data.get("processed", {})
        failed = data.get("failed", {})
        if not isinstance(urls, list) or not isinstance(processed, dict) or not isinstance(failed, dict):
            raise ValueError("Invalid Jina Reader crawl status response")
        crawl_status_data: CrawlStatusDict = {
            "status": status,
            "job_id": job_id,
            "total": len(urls),
            "current": len(processed) + len(failed),
            "data": [],
            "time_consuming": data.get("duration", 0) / 1000,
        }

        if crawl_status_data["status"] == "completed":
            response = _adaptive_http_client.post(
                "https://adaptivecrawlstatus-kir3wx7b3a-uc.a.run.app",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={"taskId": job_id, "urls": list(processed.keys())},
            )
            data = cls._require_jina_response_data(response, expected_job_id=job_id)
            processed_data = data.get("processed", {})
            if not isinstance(processed_data, dict):
                raise ValueError("Invalid Jina Reader crawl data response")
            formatted_data = [
                {
                    "title": item.get("data", {}).get("title"),
                    "source_url": item.get("data", {}).get("url"),
                    "description": item.get("data", {}).get("description"),
                    "markdown": item.get("data", {}).get("content"),
                }
                for item in processed_data.values()
            ]
            crawl_status_data["data"] = formatted_data
        return crawl_status_data

    @classmethod
    def _get_firecrawl_url_data(
        cls,
        job_id: str,
        tenant_id: str,
        provider_instance_id: str,
        url: str,
        api_key: str,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        crawl_data: list[FirecrawlDocumentData] | None = None
        file_key = website_crawl_cache_key(
            tenant_id=tenant_id,
            provider="firecrawl",
            provider_instance_id=provider_instance_id,
            job_id=job_id,
        )
        if storage.exists(file_key):
            stored_data = storage.load_once(file_key)
            if stored_data:
                decoded = json.loads(stored_data.decode("utf-8"))
                if not isinstance(decoded, list):
                    raise ValueError("Invalid crawl cache data")
                crawl_data = cast(list[FirecrawlDocumentData], decoded)
        else:
            firecrawl_app = FirecrawlApp(api_key=api_key, base_url=config.get("base_url"))
            result = firecrawl_app.check_crawl_status(job_id)
            if result["status"] != "completed":
                raise ValueError("Crawl job is not completed")
            crawl_data = result["data"]
            cls._save_firecrawl_crawl_data(file_key, crawl_data)

        if crawl_data:
            for item in crawl_data:
                if item["source_url"] == url:
                    return dict(item)
        return None

    def _reject_foreign_legacy_owner(
        self,
        *,
        tenant_id: str,
        provider: str,
        provider_instance_id: str,
        legacy_provider_instance_id: str | None,
        job_id: str,
    ) -> None:
        if (
            legacy_provider_instance_id is not None
            and legacy_provider_instance_id != provider_instance_id
            and self._crawl_jobs.has_owner(
                provider=provider,
                provider_instance_id=legacy_provider_instance_id,
                job_id=job_id,
            )
            and not self._crawl_jobs.is_owned_by(
                tenant_id=tenant_id,
                provider=provider,
                provider_instance_id=legacy_provider_instance_id,
                job_id=job_id,
            )
        ):
            raise ValueError("Crawl job not found")

    @classmethod
    def _validate_job_lookup_inputs(cls, *, provider: str, job_id: str) -> None:
        if provider not in {"firecrawl", "watercrawl", "jinareader"}:
            raise ValueError("Invalid provider")
        if not job_id:
            raise ValueError("Crawl job not found")

    @staticmethod
    def _validate_provider_job_id(job_id: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9_-]{1,255}", job_id) is None:
            raise ValueError("Invalid crawl job ID")

    def _register_created_job(
        self,
        *,
        tenant_id: str,
        provider: str,
        provider_instance_id: str,
        job_id: str,
    ) -> str:
        source_handle = self._crawl_jobs.register_created(
            tenant_id=tenant_id,
            provider=provider,
            provider_instance_id=provider_instance_id,
            job_id=job_id,
        )
        if source_handle is None:
            raise ValueError("Crawl job not found")
        return source_handle

    def _claim_verified_job(
        self,
        *,
        tenant_id: str,
        provider: str,
        provider_instance_id: str,
        job_id: str,
    ) -> None:
        if not self._crawl_jobs.claim_verified(
            tenant_id=tenant_id,
            provider=provider,
            provider_instance_id=provider_instance_id,
            job_id=job_id,
        ):
            raise ValueError("Crawl job not found")

    @classmethod
    def _verify_remote_job_access(cls, *, job_id: str, provider: str, api_key: str, config: dict[str, Any]) -> None:
        """Validate a pre-migration job through this tenant's provider-scoped credential."""
        cls._require_remote_verification_api_key(api_key)
        if provider == "firecrawl":
            result: CrawlStatusResponse | CrawlStatusDict = cls._fetch_firecrawl_status(job_id, api_key, config)
        elif provider == "watercrawl":
            result = cls._get_watercrawl_status(job_id, api_key, config)
        else:
            raise ValueError("Invalid provider")
        cls._validate_remote_status_for_claim(provider=provider, job_id=job_id, result=result)

    @staticmethod
    def _require_remote_verification_api_key(api_key: object) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("Provider credential cannot verify crawl job ownership")

    @staticmethod
    def _validate_remote_status_for_claim(
        *, provider: str, job_id: str, result: CrawlStatusResponse | CrawlStatusDict
    ) -> None:
        status = result.get("status")
        valid_statuses = {
            "active",
            "pending",
            "processing",
            "running",
            "scraping",
            "completed",
            "error",
            "failed",
            "cancelled",
            "canceled",
        }
        if status not in valid_statuses:
            raise ValueError("Invalid crawl status response")
        response_job_id = result.get("job_id")
        if provider == "watercrawl" and response_job_id != job_id:
            raise ValueError("Crawl job response does not match requested job")

    @staticmethod
    def _require_jina_response_data(response: httpx.Response, *, expected_job_id: str) -> dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("code") != 200:
            raise ValueError("Invalid Jina Reader response")
        data = payload.get("data")
        if not isinstance(data, dict) or not data:
            raise ValueError("Invalid Jina Reader response")
        response_job_id = data.get("_id", data.get("taskId"))
        # Adaptive status returns the task identity as `_id`; older test and
        # proxy fixtures may use `taskId`. Bind either spelling exactly.
        if expected_job_id and response_job_id is not None and response_job_id != expected_job_id:
            raise ValueError("Jina Reader response does not match requested job")
        return data

    @staticmethod
    def _save_firecrawl_crawl_data(file_key: str, crawl_data: list[FirecrawlDocumentData]) -> None:
        if not crawl_data:
            return
        try:
            if storage.exists(file_key):
                storage.delete(file_key)
            storage.save(file_key, json.dumps(crawl_data).encode("utf-8"))
        except Exception as error:
            raise Exception(f"Error saving crawl data: {error}") from error

    @classmethod
    def _get_watercrawl_url_data(
        cls, job_id: str, url: str, api_key: str, config: dict[str, Any]
    ) -> dict[str, Any] | None:
        result = WaterCrawlProvider(api_key, config.get("base_url")).get_crawl_url_data(job_id, url)
        return dict(result) if result is not None else None

    @classmethod
    def _get_jinareader_url_data(cls, job_id: str, url: str, api_key: str) -> dict[str, Any] | None:
        if not job_id:
            response = _jina_http_client.get(
                f"https://r.jina.ai/{url}",
                headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            return cls._require_jina_response_data(response, expected_job_id="")
        else:
            # Get crawl status first
            status_response = _adaptive_http_client.post(
                "https://adaptivecrawlstatus-kir3wx7b3a-uc.a.run.app",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={"taskId": job_id},
            )
            status_data = cls._require_jina_response_data(status_response, expected_job_id=job_id)
            if status_data.get("status") != "completed":
                raise ValueError("Crawl job is not completed")

            # Get processed data
            data_response = _adaptive_http_client.post(
                "https://adaptivecrawlstatus-kir3wx7b3a-uc.a.run.app",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={"taskId": job_id, "urls": list(status_data.get("processed", {}).keys())},
            )
            processed_data = cls._require_jina_response_data(data_response, expected_job_id=job_id)
            processed = processed_data.get("processed", {})
            if not isinstance(processed, dict):
                raise ValueError("Invalid Jina Reader crawl data response")
            for item in processed.values():
                if item.get("data", {}).get("url") == url:
                    return dict(item.get("data", {}))
        return None

    @classmethod
    def _scrape_with_firecrawl(cls, request: ScrapeRequest, api_key: str, config: dict[str, Any]) -> dict[str, Any]:
        firecrawl_app = FirecrawlApp(api_key=api_key, base_url=config.get("base_url"))
        params = {"onlyMainContent": request.only_main_content}
        return dict(firecrawl_app.scrape_url(url=request.url, params=params))

    @classmethod
    def _scrape_with_watercrawl(cls, request: ScrapeRequest, api_key: str, config: dict[str, Any]) -> dict[str, Any]:
        return dict(WaterCrawlProvider(api_key=api_key, base_url=config.get("base_url")).scrape_url(request.url))

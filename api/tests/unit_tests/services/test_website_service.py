"""Unit tests for services.data_source.website_service.

Focuses on provider dispatching, argument validation, and provider-specific branches
without making any real network/storage/redis calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, create_autospec, patch

import pytest

import services.data_source.website_service as website_service_module
from services.data_source.credential_gateway import DatasourceProviderCredentialStore
from services.data_source.website_crawl_cache import (
    website_crawl_cache_key,
    website_crawl_provider_instance_id,
    website_crawl_timer_key,
)
from services.data_source.website_service import (
    CrawlOptions,
    WebsiteCrawlApiRequest,
    WebsiteCrawlJobStore,
    WebsiteCrawlStatusApiRequest,
    WebsiteService,
)

_FIRECRAWL_INSTANCE = website_crawl_provider_instance_id(provider="firecrawl")


@dataclass(frozen=True)
class _DummyHttpxResponse:
    payload: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


@dataclass
class _HttpPostSequence:
    responses: list[_DummyHttpxResponse]
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def __call__(self, url: str, **kwargs: object) -> _DummyHttpxResponse:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


@dataclass
class _ScrapeClient:
    result: dict[str, Any]
    calls: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)

    def scrape_url(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((url, params))
        return self.result


def _website_service(
    *, providers: website_service_module.DatasourceProviderService, crawl_jobs: WebsiteCrawlJobStore | None = None
) -> WebsiteService:
    if crawl_jobs is None:
        crawl_jobs = create_autospec(WebsiteCrawlJobStore, instance=True, spec_set=True)
        crawl_jobs.is_owned_by.return_value = True
        crawl_jobs.has_owner.return_value = True
        crawl_jobs.register_created.return_value = "source-handle"
        crawl_jobs.claim_verified.return_value = "source-handle"
    return WebsiteService(
        providers=providers,
        crawl_jobs=crawl_jobs,
    )


@pytest.fixture(autouse=True)
def stub_current_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        website_service_module,
        "current_user",
        type("User", (), {"current_tenant_id": "tenant-1"})(),
    )


def test_crawl_options_include_exclude_paths() -> None:
    options = CrawlOptions(includes="a,b", excludes="x,y")
    assert options.get_include_paths() == ["a", "b"]
    assert options.get_exclude_paths() == ["x", "y"]

    empty = CrawlOptions(includes=None, excludes=None)
    assert empty.get_include_paths() == []
    assert empty.get_exclude_paths() == []


def test_website_crawl_api_request_from_args_valid_and_to_crawl_request() -> None:
    args = {
        "provider": "firecrawl",
        "url": "https://example.com",
        "options": {
            "limit": 2,
            "crawl_sub_pages": True,
            "only_main_content": True,
            "includes": "a,b",
            "excludes": "x",
            "prompt": "hi",
            "max_depth": 3,
            "use_sitemap": False,
        },
    }

    api_req = WebsiteCrawlApiRequest.from_args(args)
    crawl_req = api_req.to_crawl_request()

    assert crawl_req.provider == "firecrawl"
    assert crawl_req.url == "https://example.com"
    assert crawl_req.options.limit == 2
    assert crawl_req.options.crawl_sub_pages is True
    assert crawl_req.options.only_main_content is True
    assert crawl_req.options.get_include_paths() == ["a", "b"]
    assert crawl_req.options.get_exclude_paths() == ["x"]
    assert crawl_req.options.prompt == "hi"
    assert crawl_req.options.max_depth == 3
    assert crawl_req.options.use_sitemap is False


@pytest.mark.parametrize(
    ("args", "missing_msg"),
    [
        ({}, "Provider is required"),
        ({"provider": "firecrawl"}, "URL is required"),
        ({"provider": "firecrawl", "url": "https://example.com"}, "Options are required"),
    ],
)
def test_website_crawl_api_request_from_args_requires_fields(args: dict[str, Any], missing_msg: str) -> None:
    with pytest.raises(ValueError, match=missing_msg):
        WebsiteCrawlApiRequest.from_args(args)


def test_website_crawl_status_api_request_from_args_requires_fields() -> None:
    with pytest.raises(ValueError, match="Provider is required"):
        WebsiteCrawlStatusApiRequest.from_args({}, job_id="job-1")

    with pytest.raises(ValueError, match="Job ID is required"):
        WebsiteCrawlStatusApiRequest.from_args({"provider": "firecrawl"}, job_id="")

    with pytest.raises(ValueError, match="Source handle is required"):
        WebsiteCrawlStatusApiRequest.from_args({"provider": "firecrawl"}, job_id="job-1")

    req = WebsiteCrawlStatusApiRequest.from_args(
        {"provider": "firecrawl", "source_handle": "source-handle"}, job_id="job-1"
    )
    assert req.provider == "firecrawl"
    assert req.job_id == "job-1"
    assert req.source_handle == "source-handle"


def test_get_credentials_and_config_selects_plugin_id_and_key_firecrawl(monkeypatch: pytest.MonkeyPatch) -> None:
    service_instance = MagicMock(name="DatasourceProviderService-instance")
    service_instance.get_datasource_credentials.return_value = {"firecrawl_api_key": "k", "base_url": "b"}
    monkeypatch.setattr(website_service_module, "DatasourceProviderService", MagicMock(return_value=service_instance))

    api_key, config = _website_service(
        providers=website_service_module.DatasourceProviderService(
            credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
        )
    )._get_credentials_and_config("tenant-1", "firecrawl")
    assert api_key == "k"
    assert config["base_url"] == "b"

    service_instance.get_datasource_credentials.assert_called_once_with(
        tenant_id="tenant-1",
        provider="firecrawl",
        plugin_id="langgenius/firecrawl_datasource",
    )


@pytest.mark.parametrize(
    ("provider", "plugin_id"),
    [
        ("watercrawl", "watercrawl/watercrawl_datasource"),
        ("jinareader", "langgenius/jina_datasource"),
    ],
)
def test_get_credentials_and_config_selects_plugin_id_and_key_api_key(
    monkeypatch: pytest.MonkeyPatch, provider: str, plugin_id: str
) -> None:
    service_instance = MagicMock(name="DatasourceProviderService-instance")
    service_instance.get_datasource_credentials.return_value = {"api_key": "enc-key", "base_url": "b"}
    monkeypatch.setattr(website_service_module, "DatasourceProviderService", MagicMock(return_value=service_instance))

    api_key, config = _website_service(
        providers=website_service_module.DatasourceProviderService(
            credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
        )
    )._get_credentials_and_config("tenant-1", provider)
    assert api_key == "enc-key"
    assert config["base_url"] == "b"

    service_instance.get_datasource_credentials.assert_called_once_with(
        tenant_id="tenant-1",
        provider=provider,
        plugin_id=plugin_id,
    )


def test_get_credentials_and_config_rejects_invalid_provider() -> None:
    with pytest.raises(ValueError, match="Invalid provider"):
        _website_service(
            providers=website_service_module.DatasourceProviderService(
                credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
            )
        )._get_credentials_and_config("tenant-1", "unknown")


def test_get_credentials_and_config_hits_unreachable_guard_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakyProvider:
        def __init__(self) -> None:
            self._eq_calls = 0

        def __hash__(self) -> int:
            return 1

        def __eq__(self, other: object) -> bool:
            if other == "firecrawl":
                self._eq_calls += 1
                return self._eq_calls == 1
            return False

        def __repr__(self) -> str:
            return "FlakyProvider()"

    service_instance = MagicMock(name="DatasourceProviderService-instance")
    service_instance.get_datasource_credentials.return_value = {"firecrawl_api_key": "k"}
    monkeypatch.setattr(website_service_module, "DatasourceProviderService", MagicMock(return_value=service_instance))

    with pytest.raises(ValueError, match="Invalid provider"):
        _website_service(
            providers=website_service_module.DatasourceProviderService(
                credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
            )
        )._get_credentials_and_config("tenant-1", FlakyProvider())  # type: ignore[arg-type]


def test_get_decrypted_api_key_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(website_service_module.encrypter, "decrypt_token", MagicMock())
    with pytest.raises(ValueError, match="API key not found in configuration"):
        WebsiteService._get_decrypted_api_key("tenant-1", {})


def test_get_decrypted_api_key_decrypts(monkeypatch: pytest.MonkeyPatch) -> None:
    decrypt_mock = MagicMock(return_value="plain")
    monkeypatch.setattr(website_service_module.encrypter, "decrypt_token", decrypt_mock)

    assert WebsiteService._get_decrypted_api_key("tenant-1", {"api_key": "enc"}) == "plain"
    decrypt_mock.assert_called_once_with(tenant_id="tenant-1", token="enc")


def test_document_create_args_validate_wraps_error_message() -> None:
    with pytest.raises(ValueError, match=r"^Invalid arguments: Provider is required$"):
        WebsiteService.document_create_args_validate({})


def test_crawl_url_dispatches_by_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    api_request = WebsiteCrawlApiRequest(provider="firecrawl", url="https://example.com", options={"limit": 1})
    crawl_request = api_request.to_crawl_request()

    monkeypatch.setattr(
        WebsiteService,
        "_get_credentials_and_config",
        lambda _self, _tenant_id, _provider: ("k", {"base_url": "https://api.firecrawl.dev/"}),
    )
    firecrawl_mock = MagicMock(return_value={"status": "active", "job_id": "j1"})
    monkeypatch.setattr(WebsiteService, "_crawl_with_firecrawl", firecrawl_mock)

    crawl_jobs = create_autospec(WebsiteCrawlJobStore, instance=True, spec_set=True)
    crawl_jobs.register_created.return_value = "source-handle"
    redis_mock = MagicMock()
    monkeypatch.setattr(website_service_module, "redis_client", redis_mock)
    result = _website_service(
        providers=website_service_module.DatasourceProviderService(
            credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
        ),
        crawl_jobs=crawl_jobs,
    ).crawl_url(api_request)

    assert result == {"status": "active", "job_id": "j1", "source_handle": "source-handle"}
    firecrawl_mock.assert_called_once()
    assert firecrawl_mock.call_args.kwargs["request"] == crawl_request
    crawl_jobs.register_created.assert_called_once_with(
        tenant_id="tenant-1",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="j1",
    )
    redis_mock.setex.assert_called_once_with(
        website_crawl_timer_key(
            tenant_id="tenant-1",
            provider="firecrawl",
            provider_instance_id=_FIRECRAWL_INSTANCE,
            job_id="j1",
        ),
        3600,
        redis_mock.setex.call_args.args[2],
    )


@pytest.mark.parametrize(
    ("provider", "method_name"),
    [
        ("watercrawl", "_crawl_with_watercrawl"),
        ("jinareader", "_crawl_with_jinareader"),
    ],
)
def test_crawl_url_dispatches_other_providers(monkeypatch: pytest.MonkeyPatch, provider: str, method_name: str) -> None:
    api_request = WebsiteCrawlApiRequest(provider=provider, url="https://example.com", options={"limit": 1})
    monkeypatch.setattr(
        WebsiteService,
        "_get_credentials_and_config",
        lambda _self, _tenant_id, _provider: ("k", {"base_url": "https://api.firecrawl.dev/"}),
    )

    impl_mock = MagicMock(return_value={"status": "active"})
    monkeypatch.setattr(WebsiteService, method_name, impl_mock)

    assert _website_service(
        providers=website_service_module.DatasourceProviderService(
            credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
        )
    ).crawl_url(api_request) == {"status": "active"}
    impl_mock.assert_called_once()


def test_crawl_url_rejects_invalid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    api_request = WebsiteCrawlApiRequest(provider="bad", url="https://example.com", options={"limit": 1})
    monkeypatch.setattr(WebsiteService, "_get_credentials_and_config", MagicMock(return_value=("k", {})))

    with pytest.raises(ValueError, match="Invalid provider"):
        _website_service(
            providers=website_service_module.DatasourceProviderService(
                credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
            )
        ).crawl_url(api_request)


def test_crawl_url_rejects_unsafe_provider_job_id_before_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    api_request = WebsiteCrawlApiRequest(provider="firecrawl", url="https://example.com", options={"limit": 1})
    monkeypatch.setattr(WebsiteService, "_get_credentials_and_config", MagicMock(return_value=("k", {})))
    monkeypatch.setattr(
        WebsiteService,
        "_crawl_with_firecrawl",
        lambda _self, **_kwargs: {"status": "active", "job_id": "../../other/job"},
    )
    crawl_jobs = create_autospec(WebsiteCrawlJobStore, instance=True, spec_set=True)

    with pytest.raises(ValueError, match="Invalid crawl job ID"):
        _website_service(
            providers=website_service_module.DatasourceProviderService(
                credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
            ),
            crawl_jobs=crawl_jobs,
        ).crawl_url(api_request)

    crawl_jobs.register_created.assert_not_called()


def test_crawl_with_firecrawl_builds_params_single_page(monkeypatch: pytest.MonkeyPatch) -> None:
    firecrawl_instance = MagicMock(name="FirecrawlApp-instance")
    firecrawl_instance.crawl_url.return_value = "job-1"
    firecrawl_cls = MagicMock(return_value=firecrawl_instance)
    monkeypatch.setattr(website_service_module, "FirecrawlApp", firecrawl_cls)

    req = WebsiteCrawlApiRequest(
        provider="firecrawl", url="https://example.com", options={"limit": 5}
    ).to_crawl_request()
    req.options.crawl_sub_pages = False
    req.options.only_main_content = True

    result = WebsiteService._crawl_with_firecrawl(request=req, api_key="k", config={"base_url": "b"})

    assert result == {"status": "active", "job_id": "job-1"}

    firecrawl_cls.assert_called_once_with(api_key="k", base_url="b")
    firecrawl_instance.crawl_url.assert_called_once()
    _, params = firecrawl_instance.crawl_url.call_args.args
    assert params["limit"] == 1
    assert params["includePaths"] == []
    assert params["excludePaths"] == []
    assert params["scrapeOptions"] == {"onlyMainContent": True}


def test_crawl_with_firecrawl_builds_params_multi_page_including_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    firecrawl_instance = MagicMock(name="FirecrawlApp-instance")
    firecrawl_instance.crawl_url.return_value = "job-2"
    monkeypatch.setattr(website_service_module, "FirecrawlApp", MagicMock(return_value=firecrawl_instance))
    monkeypatch.setattr(website_service_module, "redis_client", MagicMock())

    req = WebsiteCrawlApiRequest(
        provider="firecrawl",
        url="https://example.com",
        options={
            "crawl_sub_pages": True,
            "limit": 3,
            "only_main_content": False,
            "includes": "a,b",
            "excludes": "x",
            "prompt": "use this",
        },
    ).to_crawl_request()

    WebsiteService._crawl_with_firecrawl(request=req, api_key="k", config={"base_url": None})
    _, params = firecrawl_instance.crawl_url.call_args.args
    assert params["includePaths"] == ["a", "b"]
    assert params["excludePaths"] == ["x"]
    assert params["limit"] == 3
    assert params["scrapeOptions"] == {"onlyMainContent": False}
    assert params["prompt"] == "use this"


def test_crawl_with_watercrawl_passes_options_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_instance = MagicMock()
    provider_instance.crawl_url.return_value = {"status": "active", "job_id": "w1"}
    provider_cls = MagicMock(return_value=provider_instance)
    monkeypatch.setattr(website_service_module, "WaterCrawlProvider", provider_cls)

    req = WebsiteCrawlApiRequest(
        provider="watercrawl",
        url="https://example.com",
        options={
            "limit": 2,
            "crawl_sub_pages": True,
            "only_main_content": True,
            "includes": "a",
            "excludes": None,
            "max_depth": 5,
            "use_sitemap": False,
        },
    ).to_crawl_request()

    result = WebsiteService._crawl_with_watercrawl(request=req, api_key="k", config={"base_url": "b"})
    assert result == {"status": "active", "job_id": "w1"}

    provider_cls.assert_called_once_with(api_key="k", base_url="b")
    provider_instance.crawl_url.assert_called_once_with(
        url="https://example.com",
        options={
            "limit": 2,
            "crawl_sub_pages": True,
            "only_main_content": True,
            "includes": "a",
            "excludes": None,
            "max_depth": 5,
            "use_sitemap": False,
        },
    )


def test_crawl_with_jinareader_single_page_success(monkeypatch: pytest.MonkeyPatch) -> None:
    get_mock = MagicMock(return_value=_DummyHttpxResponse({"code": 200, "data": {"title": "t"}}))
    monkeypatch.setattr(website_service_module._jina_http_client, "get", get_mock)

    req = WebsiteCrawlApiRequest(
        provider="jinareader", url="https://example.com", options={"crawl_sub_pages": False}
    ).to_crawl_request()
    req.options.crawl_sub_pages = False

    result = WebsiteService._crawl_with_jinareader(request=req, api_key="k")
    assert result == {"status": "active", "data": {"title": "t"}}
    get_mock.assert_called_once()


def test_crawl_with_jinareader_single_page_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        website_service_module._jina_http_client,
        "get",
        MagicMock(return_value=_DummyHttpxResponse({"code": 500})),
    )
    req = WebsiteCrawlApiRequest(
        provider="jinareader", url="https://example.com", options={"crawl_sub_pages": False}
    ).to_crawl_request()
    req.options.crawl_sub_pages = False

    with pytest.raises(ValueError, match="Failed to crawl:"):
        WebsiteService._crawl_with_jinareader(request=req, api_key="k")


def test_crawl_with_jinareader_multi_page_success(monkeypatch: pytest.MonkeyPatch) -> None:
    post_mock = MagicMock(return_value=_DummyHttpxResponse({"code": 200, "data": {"taskId": "t1"}}))
    monkeypatch.setattr(website_service_module._adaptive_http_client, "post", post_mock)

    req = WebsiteCrawlApiRequest(
        provider="jinareader",
        url="https://example.com",
        options={"crawl_sub_pages": True, "limit": 5, "use_sitemap": True},
    ).to_crawl_request()
    req.options.crawl_sub_pages = True

    result = WebsiteService._crawl_with_jinareader(request=req, api_key="k")
    assert result == {"status": "active", "job_id": "t1"}
    post_mock.assert_called_once()


def test_crawl_with_jinareader_multi_page_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        website_service_module._adaptive_http_client, "post", MagicMock(return_value=_DummyHttpxResponse({"code": 400}))
    )
    req = WebsiteCrawlApiRequest(
        provider="jinareader",
        url="https://example.com",
        options={"crawl_sub_pages": True, "limit": 2, "use_sitemap": False},
    ).to_crawl_request()
    req.options.crawl_sub_pages = True

    with pytest.raises(ValueError, match="Failed to crawl$"):
        WebsiteService._crawl_with_jinareader(request=req, api_key="k")


def test_get_crawl_status_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        WebsiteService,
        "_get_credentials_and_config",
        lambda _self, _tenant_id, _provider: ("k", {"base_url": "https://api.firecrawl.dev/"}),
    )
    firecrawl_status = MagicMock(return_value={"status": "active"})
    monkeypatch.setattr(WebsiteService, "_get_firecrawl_status", firecrawl_status)
    firecrawl_jobs = create_autospec(WebsiteCrawlJobStore, instance=True, spec_set=True)
    firecrawl_jobs.get_owned_identity.return_value = ("firecrawl", _FIRECRAWL_INSTANCE, "job-1")
    firecrawl_jobs.is_owned_by.return_value = True

    result = _website_service(
        providers=website_service_module.DatasourceProviderService(
            credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
        ),
        crawl_jobs=firecrawl_jobs,
    ).get_crawl_status_typed(
        WebsiteCrawlStatusApiRequest(provider="firecrawl", job_id="job-1", source_handle="firecrawl-handle")
    )
    assert result == {"status": "active", "source_handle": "firecrawl-handle"}
    firecrawl_status.assert_called_once_with(
        "job-1",
        "tenant-1",
        _FIRECRAWL_INSTANCE,
        "k",
        {"base_url": "https://api.firecrawl.dev/"},
    )

    watercrawl_status = MagicMock(return_value={"status": "active", "job_id": "w"})
    monkeypatch.setattr(WebsiteService, "_get_watercrawl_status", watercrawl_status)
    watercrawl_jobs = create_autospec(WebsiteCrawlJobStore, instance=True, spec_set=True)
    watercrawl_jobs.get_owned_identity.return_value = (
        "watercrawl",
        website_crawl_provider_instance_id(provider="watercrawl", base_url="https://api.firecrawl.dev/"),
        "job-2",
    )
    watercrawl_jobs.is_owned_by.return_value = True
    assert _website_service(
        providers=website_service_module.DatasourceProviderService(
            credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
        ),
        crawl_jobs=watercrawl_jobs,
    ).get_crawl_status_typed(
        WebsiteCrawlStatusApiRequest(provider="watercrawl", job_id="job-2", source_handle="watercrawl-handle")
    ) == {
        "status": "active",
        "job_id": "w",
        "source_handle": "watercrawl-handle",
    }
    watercrawl_status.assert_called_once_with("job-2", "k", {"base_url": "https://api.firecrawl.dev/"})

    jinareader_status = MagicMock(return_value={"status": "active", "job_id": "j"})
    monkeypatch.setattr(WebsiteService, "_get_jinareader_status", jinareader_status)
    jina_jobs = create_autospec(WebsiteCrawlJobStore, instance=True, spec_set=True)
    jina_jobs.get_owned_identity.return_value = (
        "jinareader",
        website_crawl_provider_instance_id(provider="jinareader", base_url="https://api.firecrawl.dev/"),
        "job-3",
    )
    jina_jobs.is_owned_by.return_value = True
    assert _website_service(
        providers=website_service_module.DatasourceProviderService(
            credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
        ),
        crawl_jobs=jina_jobs,
    ).get_crawl_status_typed(
        WebsiteCrawlStatusApiRequest(provider="jinareader", job_id="job-3", source_handle="jina-handle")
    ) == {
        "status": "active",
        "job_id": "j",
        "source_handle": "jina-handle",
    }
    jinareader_status.assert_called_once_with("job-3", "k")


def test_get_crawl_status_typed_rejects_invalid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(WebsiteService, "_get_credentials_and_config", MagicMock(return_value=("k", {})))
    with pytest.raises(ValueError, match="Invalid provider"):
        _website_service(
            providers=website_service_module.DatasourceProviderService(
                credentials=create_autospec(DatasourceProviderCredentialStore, instance=True)
            )
        ).get_crawl_status_typed(WebsiteCrawlStatusApiRequest(provider="bad", job_id="j"))


def test_get_firecrawl_status_adds_time_consuming_when_completed_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    firecrawl_instance = MagicMock()
    firecrawl_instance.check_crawl_status.return_value = {"status": "completed", "total": 2, "current": 2, "data": []}
    monkeypatch.setattr(website_service_module, "FirecrawlApp", MagicMock(return_value=firecrawl_instance))

    redis_mock = MagicMock()
    redis_mock.get.return_value = b"100.0"
    monkeypatch.setattr(website_service_module, "redis_client", redis_mock)

    with patch.object(website_service_module.datetime, "datetime") as datetime_mock:
        datetime_mock.now.return_value = datetime.fromtimestamp(105.0, tz=UTC)
        result = WebsiteService._get_firecrawl_status(
            job_id="job-1",
            tenant_id="tenant-1",
            provider_instance_id=_FIRECRAWL_INSTANCE,
            api_key="k",
            config={"base_url": "b"},
        )

    assert result["status"] == "completed"
    assert result["time_consuming"] == "5.00"
    redis_mock.delete.assert_called_once_with(
        website_crawl_timer_key(
            tenant_id="tenant-1",
            provider="firecrawl",
            provider_instance_id=_FIRECRAWL_INSTANCE,
            job_id="job-1",
        )
    )


def test_get_firecrawl_status_completed_without_cache_does_not_add_time(monkeypatch: pytest.MonkeyPatch) -> None:
    firecrawl_instance = MagicMock()
    firecrawl_instance.check_crawl_status.return_value = {"status": "completed", "total": 1, "current": 1, "data": []}
    monkeypatch.setattr(website_service_module, "FirecrawlApp", MagicMock(return_value=firecrawl_instance))

    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    monkeypatch.setattr(website_service_module, "redis_client", redis_mock)

    result = WebsiteService._get_firecrawl_status(
        job_id="job-1",
        tenant_id="tenant-1",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        api_key="k",
        config={"base_url": None},
    )
    assert result["status"] == "completed"
    assert "time_consuming" not in result
    redis_mock.delete.assert_not_called()


def test_get_firecrawl_status_scopes_cache_and_propagates_save_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    crawl_data = [{"source_url": "https://example.com", "markdown": "private"}]
    firecrawl_instance = MagicMock()
    firecrawl_instance.check_crawl_status.return_value = {
        "status": "completed",
        "total": 1,
        "current": 1,
        "data": crawl_data,
    }
    monkeypatch.setattr(website_service_module, "FirecrawlApp", MagicMock(return_value=firecrawl_instance))
    storage_mock = MagicMock()
    storage_mock.exists.return_value = False
    storage_mock.save.side_effect = RuntimeError("save failed")
    monkeypatch.setattr(website_service_module, "storage", storage_mock)

    with pytest.raises(Exception, match="Error saving crawl data"):
        WebsiteService._get_firecrawl_status(
            job_id="job-1",
            tenant_id="tenant-1",
            provider_instance_id=_FIRECRAWL_INSTANCE,
            api_key="k",
            config={"base_url": None},
        )

    storage_mock.save.assert_called_once_with(
        website_crawl_cache_key(
            tenant_id="tenant-1",
            provider="firecrawl",
            provider_instance_id=_FIRECRAWL_INSTANCE,
            job_id="job-1",
        ),
        json.dumps(crawl_data).encode("utf-8"),
    )


def test_get_watercrawl_status_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_instance = MagicMock()
    provider_instance.get_crawl_status.return_value = {"status": "active", "job_id": "w1"}
    monkeypatch.setattr(website_service_module, "WaterCrawlProvider", MagicMock(return_value=provider_instance))

    assert WebsiteService._get_watercrawl_status("job-1", "k", {"base_url": "b"}) == {
        "status": "active",
        "job_id": "w1",
    }
    provider_instance.get_crawl_status.assert_called_once_with("job-1")


def test_get_jinareader_status_active(monkeypatch: pytest.MonkeyPatch) -> None:
    post_mock = _HttpPostSequence(
        [
            _DummyHttpxResponse(
                {
                    "code": 200,
                    "data": {
                        "taskId": "job-1",
                        "status": "active",
                        "urls": ["a", "b"],
                        "processed": {"a": {}},
                        "failed": {"b": {}},
                        "duration": 3000,
                    },
                }
            )
        ]
    )
    monkeypatch.setattr(website_service_module._adaptive_http_client, "post", post_mock)

    result = WebsiteService._get_jinareader_status("job-1", "k")
    assert result["status"] == "active"
    assert result["total"] == 2
    assert result["current"] == 2
    assert result["time_consuming"] == 3.0
    assert result["data"] == []
    assert len(post_mock.calls) == 1


def test_get_jinareader_status_completed_formats_processed_items(monkeypatch: pytest.MonkeyPatch) -> None:
    status_payload = {
        "code": 200,
        "data": {
            "taskId": "job-1",
            "status": "completed",
            "urls": ["u1"],
            "processed": {"u1": {}},
            "failed": {},
            "duration": 1000,
        },
    }
    processed_payload = {
        "code": 200,
        "data": {
            "taskId": "job-1",
            "processed": {
                "u1": {
                    "data": {
                        "title": "t",
                        "url": "u1",
                        "description": "d",
                        "content": "md",
                    }
                }
            },
        },
    }
    post_mock = MagicMock(side_effect=[_DummyHttpxResponse(status_payload), _DummyHttpxResponse(processed_payload)])
    monkeypatch.setattr(website_service_module._adaptive_http_client, "post", post_mock)

    result = WebsiteService._get_jinareader_status("job-1", "k")
    assert result["status"] == "completed"
    assert result["data"] == [{"title": "t", "source_url": "u1", "description": "d", "markdown": "md"}]
    assert post_mock.call_count == 2


def test_get_firecrawl_url_data_reads_from_storage_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    stored_list = [{"source_url": "https://example.com", "title": "t"}]
    stored = json.dumps(stored_list).encode("utf-8")

    storage_mock = MagicMock()
    storage_mock.exists.return_value = True
    storage_mock.load_once.return_value = stored
    monkeypatch.setattr(website_service_module, "storage", storage_mock)

    monkeypatch.setattr(website_service_module, "FirecrawlApp", MagicMock())

    result = WebsiteService._get_firecrawl_url_data(
        "job-1",
        "tenant-1",
        _FIRECRAWL_INSTANCE,
        "https://example.com",
        "k",
        {"base_url": "b"},
    )
    assert result == {"source_url": "https://example.com", "title": "t"}
    assert result is not stored_list[0]


def test_get_firecrawl_url_data_returns_none_when_storage_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    storage_mock = MagicMock()
    storage_mock.exists.return_value = True
    storage_mock.load_once.return_value = b""
    monkeypatch.setattr(website_service_module, "storage", storage_mock)

    assert (
        WebsiteService._get_firecrawl_url_data("job-1", "tenant-1", _FIRECRAWL_INSTANCE, "https://example.com", "k", {})
        is None
    )


def test_get_firecrawl_url_data_raises_when_job_not_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    storage_mock = MagicMock()
    storage_mock.exists.return_value = False
    monkeypatch.setattr(website_service_module, "storage", storage_mock)

    firecrawl_instance = MagicMock()
    firecrawl_instance.check_crawl_status.return_value = {"status": "active"}
    monkeypatch.setattr(website_service_module, "FirecrawlApp", MagicMock(return_value=firecrawl_instance))

    with pytest.raises(ValueError, match="Crawl job is not completed"):
        WebsiteService._get_firecrawl_url_data(
            "job-1",
            "tenant-1",
            _FIRECRAWL_INSTANCE,
            "https://example.com",
            "k",
            {"base_url": None},
        )


def test_get_firecrawl_url_data_returns_none_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    storage_mock = MagicMock()
    storage_mock.exists.return_value = False
    monkeypatch.setattr(website_service_module, "storage", storage_mock)

    firecrawl_instance = MagicMock()
    firecrawl_instance.check_crawl_status.return_value = {"status": "completed", "data": [{"source_url": "x"}]}
    monkeypatch.setattr(website_service_module, "FirecrawlApp", MagicMock(return_value=firecrawl_instance))

    assert (
        WebsiteService._get_firecrawl_url_data(
            "job-1",
            "tenant-1",
            _FIRECRAWL_INSTANCE,
            "https://example.com",
            "k",
            {"base_url": "b"},
        )
        is None
    )


def test_get_watercrawl_url_data_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_instance = MagicMock()
    provider_instance.get_crawl_url_data.return_value = {"source_url": "u"}
    monkeypatch.setattr(website_service_module, "WaterCrawlProvider", MagicMock(return_value=provider_instance))

    result = WebsiteService._get_watercrawl_url_data("job-1", "u", "k", {"base_url": "b"})
    assert result == {"source_url": "u"}
    provider_instance.get_crawl_url_data.assert_called_once_with("job-1", "u")


def test_get_jinareader_url_data_without_job_id_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        website_service_module._jina_http_client,
        "get",
        MagicMock(return_value=_DummyHttpxResponse({"code": 200, "data": {"url": "u"}})),
    )
    assert WebsiteService._get_jinareader_url_data("", "u", "k") == {"url": "u"}


def test_get_jinareader_url_data_without_job_id_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        website_service_module._jina_http_client,
        "get",
        MagicMock(return_value=_DummyHttpxResponse({"code": 500})),
    )
    with pytest.raises(ValueError, match="Invalid Jina Reader response"):
        WebsiteService._get_jinareader_url_data("", "u", "k")


def test_get_jinareader_url_data_with_job_id_completed_returns_matching_item(monkeypatch: pytest.MonkeyPatch) -> None:
    status_payload = {
        "code": 200,
        "data": {"taskId": "job-1", "status": "completed", "processed": {"u1": {}}},
    }
    processed_payload = {
        "code": 200,
        "data": {"taskId": "job-1", "processed": {"u1": {"data": {"url": "u", "title": "t"}}}},
    }

    post_mock = MagicMock(side_effect=[_DummyHttpxResponse(status_payload), _DummyHttpxResponse(processed_payload)])
    monkeypatch.setattr(website_service_module._adaptive_http_client, "post", post_mock)

    assert WebsiteService._get_jinareader_url_data("job-1", "u", "k") == {"url": "u", "title": "t"}
    assert post_mock.call_count == 2


def test_get_jinareader_url_data_with_job_id_not_completed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    post_mock = _HttpPostSequence([_DummyHttpxResponse({"code": 200, "data": {"taskId": "job-1", "status": "active"}})])
    monkeypatch.setattr(website_service_module._adaptive_http_client, "post", post_mock)

    with pytest.raises(ValueError, match=r"Crawl job is no\s*t completed"):
        WebsiteService._get_jinareader_url_data("job-1", "u", "k")


def test_get_jinareader_url_data_with_job_id_completed_but_not_found_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_payload = {
        "code": 200,
        "data": {"taskId": "job-1", "status": "completed", "processed": {"u1": {}}},
    }
    processed_payload = {
        "code": 200,
        "data": {"taskId": "job-1", "processed": {"u1": {"data": {"url": "other"}}}},
    }

    post_mock = MagicMock(side_effect=[_DummyHttpxResponse(status_payload), _DummyHttpxResponse(processed_payload)])
    monkeypatch.setattr(website_service_module._adaptive_http_client, "post", post_mock)

    assert WebsiteService._get_jinareader_url_data("job-1", "u", "k") is None


def test_scrape_with_firecrawl_calls_app(monkeypatch: pytest.MonkeyPatch) -> None:
    firecrawl_instance = _ScrapeClient({"markdown": "m"})
    monkeypatch.setattr(website_service_module, "FirecrawlApp", lambda **_kwargs: firecrawl_instance)

    result = WebsiteService._scrape_with_firecrawl(
        request=website_service_module.ScrapeRequest(
            provider="firecrawl",
            url="u",
            tenant_id="tenant-1",
            only_main_content=True,
        ),
        api_key="k",
        config={"base_url": "b"},
    )
    assert result == {"markdown": "m"}
    assert firecrawl_instance.calls == [("u", {"onlyMainContent": True})]


def test_scrape_with_watercrawl_calls_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_instance = _ScrapeClient({"markdown": "m"})
    monkeypatch.setattr(website_service_module, "WaterCrawlProvider", lambda **_kwargs: provider_instance)

    result = WebsiteService._scrape_with_watercrawl(
        request=website_service_module.ScrapeRequest(
            provider="watercrawl",
            url="u",
            tenant_id="tenant-1",
            only_main_content=False,
        ),
        api_key="k",
        config={"base_url": "b"},
    )
    assert result == {"markdown": "m"}
    assert provider_instance.calls == [("u", None)]


def test_pooled_clients_carry_bounded_timeouts() -> None:
    """Regression for #39859: the Jina and adaptive-crawl pooled clients
    must carry a read/connect timeout so a stalled endpoint fails fast
    instead of pinning a worker. Same shape as the WaterCrawl hardening
    that landed in PR #37512.
    """
    jina = website_service_module._jina_http_client
    adaptive = website_service_module._adaptive_http_client

    # Read and connect bounds are set.
    assert jina.timeout is not None
    assert adaptive.timeout is not None

    # The values match the documented floor (read 30.0 / connect 5.0).
    assert jina.timeout.read == 30.0
    assert jina.timeout.connect == 5.0
    assert adaptive.timeout.read == 30.0
    assert adaptive.timeout.connect == 5.0

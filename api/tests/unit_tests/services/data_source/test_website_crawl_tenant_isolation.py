"""Regression tests for tenant-owned asynchronous website crawl jobs."""

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import cast
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, event
from sqlalchemy.orm import Session, sessionmaker

import services.data_source.website_service as website_service_module
from core.rag.extractor.entity.datasource_type import DatasourceType
from core.rag.extractor.entity.extract_setting import ExtractSetting, WebsiteInfo, WebsiteJobRef
from core.rag.extractor.firecrawl.firecrawl_app import CrawlStatusResponse, FirecrawlDocumentData
from core.rag.extractor.watercrawl.client import WaterCrawlAPIClient
from models.dataset import Document
from models.enums import DataSourceType as DocumentDataSourceType
from models.source import WebsiteCrawlJob
from repositories.data_source.website_crawl_job_repository import SQLAlchemyWebsiteCrawlJobRepository
from repositories.data_source.website_source_repository import SQLAlchemyWebsiteSourceRepository
from services.data_source.provider_service import DatasourceProviderService
from services.data_source.website_crawl_cache import (
    website_crawl_cache_key,
    website_crawl_provider_instance_id,
)
from services.data_source.website_service import WebsiteCrawlApiRequest, WebsiteCrawlStatusApiRequest, WebsiteService
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.adapters.estimate import IndexingEstimateAdapter
from services.knowledge.indexing.adapters.sources import WebsiteSourceAdapter
from services.knowledge.indexing.errors import IndexingInputSourceError
from services.knowledge.indexing.estimate import IndexingEstimateExecutionError, StoredSource
from tests.unit_tests.model_factories import make_dataset, make_document, make_tenant

_FIRECRAWL_INSTANCE = website_crawl_provider_instance_id(provider="firecrawl")
_WATERCRAWL_INSTANCE = website_crawl_provider_instance_id(provider="watercrawl")
_JINA_INSTANCE = website_crawl_provider_instance_id(provider="jinareader")


@dataclass
class _CredentialProvider:
    credentials: dict[str, object]
    calls: list[dict[str, str]] = field(default_factory=list)

    def get_datasource_credentials(self, *, tenant_id: str, provider: str, plugin_id: str) -> dict[str, object]:
        self.calls.append({"tenant_id": tenant_id, "provider": provider, "plugin_id": plugin_id})
        return dict(self.credentials)


@dataclass
class _FirecrawlStatusClient:
    result: CrawlStatusResponse | Exception
    calls: list[str] = field(default_factory=list)

    def check_crawl_status(self, job_id: str) -> CrawlStatusResponse:
        self.calls.append(job_id)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@dataclass
class _WatercrawlResultClient:
    result: dict[str, object]

    def get_crawl_url_data(self, _job_id: str, _url: str) -> dict[str, object]:
        return self.result


@dataclass
class _HttpResponse:
    payload: dict[str, object]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class _TransportResponse:
    pass


def _instance_for(provider: str) -> str:
    return website_crawl_provider_instance_id(provider=provider)


def _service(
    session_factory: sessionmaker[Session], *, credentials: dict[str, object] | None = None
) -> tuple[WebsiteService, _CredentialProvider, SQLAlchemyWebsiteCrawlJobRepository]:
    providers = _CredentialProvider(
        credentials
        if credentials is not None
        else {
            "firecrawl_api_key": "firecrawl-key",
            "api_key": "provider-key",
        }
    )
    jobs = SQLAlchemyWebsiteCrawlJobRepository(session_factory=session_factory)
    return WebsiteService(providers=cast(DatasourceProviderService, providers), crawl_jobs=jobs), providers, jobs


def test_estimate_chain_rejects_foreign_firecrawl_job_before_reading_legacy_cache(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, providers, jobs = _service(sqlite_session_factory)
    assert jobs.register_created(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="shared-job",
    )
    setting = ExtractSetting(
        datasource_type=DatasourceType.WEBSITE,
        website_info=WebsiteInfo(
            job_ref=WebsiteJobRef(
                tenant_id="tenant-b",
                provider="firecrawl",
                provider_instance_id=_FIRECRAWL_INSTANCE,
                job_id="shared-job",
            ),
            url="https://private.example/page",
            mode="crawl",
        ),
        document_model="text_model",
    )

    with (
        patch("extensions.application_services.data_sources.build_website_service", return_value=service),
        patch.object(website_service_module, "storage", autospec=True) as storage,
    ):
        # This is the legacy unscoped object used by the vulnerable estimate path.
        storage.exists.return_value = True
        storage.load_once.return_value = json.dumps(
            [{"source_url": "https://private.example/page", "markdown": "tenant-a secret"}]
        ).encode()

        with pytest.raises(IndexingEstimateExecutionError, match="Crawl job not found"):
            IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
                "tenant-b",
                [setting],
                {"mode": "automatic"},
                doc_form="text_model",
            )

    assert len(providers.calls) == 1
    storage.exists.assert_not_called()
    storage.load_once.assert_not_called()


def test_same_tenant_provider_reads_only_its_scoped_firecrawl_cache(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, providers, jobs = _service(sqlite_session_factory)
    assert jobs.register_created(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="shared-job",
    )
    scoped_key = website_crawl_cache_key(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="shared-job",
    )

    with patch.object(website_service_module, "storage", autospec=True) as storage:
        storage.exists.return_value = True
        storage.load_once.return_value = json.dumps(
            [{"source_url": "https://private.example/page", "markdown": "tenant-a secret"}]
        ).encode()

        job_ref = service.resolve_stored_job_ref(tenant_id="tenant-a", provider="firecrawl", job_id="shared-job")
        result = service.get_crawl_url_data_by_ref(job_ref, "https://private.example/page")

    assert result == {"source_url": "https://private.example/page", "markdown": "tenant-a secret"}
    storage.exists.assert_called_once_with(scoped_key)
    storage.load_once.assert_called_once_with(scoped_key)
    assert len(providers.calls) == 1


def test_same_job_id_isolated_across_providers(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, _, jobs = _service(sqlite_session_factory)
    assert jobs.register_created(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="shared-job",
    )
    assert jobs.register_created(
        tenant_id="tenant-a",
        provider="watercrawl",
        provider_instance_id=_WATERCRAWL_INSTANCE,
        job_id="shared-job",
    )
    assert jobs.is_owned_by(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="shared-job",
    )
    assert jobs.is_owned_by(
        tenant_id="tenant-a",
        provider="watercrawl",
        provider_instance_id=_WATERCRAWL_INSTANCE,
        job_id="shared-job",
    )
    assert not jobs.has_owner(provider="jinareader", provider_instance_id=_JINA_INSTANCE, job_id="shared-job")
    assert website_crawl_cache_key(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="shared-job",
    ) != website_crawl_cache_key(
        tenant_id="tenant-a",
        provider="watercrawl",
        provider_instance_id=_WATERCRAWL_INSTANCE,
        job_id="shared-job",
    )

    watercrawl = _WatercrawlResultClient({"source_url": "https://water.example", "markdown": "water"})
    with (
        patch.object(website_service_module, "WaterCrawlProvider", return_value=watercrawl),
        patch.object(website_service_module, "storage", autospec=True) as storage,
    ):
        job_ref = service.resolve_stored_job_ref(tenant_id="tenant-a", provider="watercrawl", job_id="shared-job")
        result = service.get_crawl_url_data_by_ref(job_ref, "https://water.example")

    assert result == {"source_url": "https://water.example", "markdown": "water"}
    storage.exists.assert_not_called()
    storage.load_once.assert_not_called()


def test_jina_create_result_can_be_registered_by_multiple_tenants(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, _, jobs = _service(sqlite_session_factory)
    request = WebsiteCrawlApiRequest(
        provider="jinareader",
        url="https://shared.example",
        options={"crawl_sub_pages": True},
    )

    with patch.object(
        WebsiteService,
        "_crawl_with_jinareader",
        side_effect=[
            {"status": "active", "job_id": "deterministic-job"},
            {"status": "active", "job_id": "deterministic-job"},
        ],
    ):
        with patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"):
            result_a = service.crawl_url(request)
        with patch.object(website_service_module, "current_user", current_tenant_id="tenant-b"):
            result_b = service.crawl_url(request)

    handle_a = result_a["source_handle"]
    handle_b = result_b["source_handle"]
    assert isinstance(handle_a, str)
    assert isinstance(handle_b, str)
    assert handle_a != handle_b
    assert (
        service.resolve_job_ref(
            tenant_id="tenant-a",
            provider="jinareader",
            job_id="deterministic-job",
            source_handle=handle_a,
        ).provider_instance_id
        == _JINA_INSTANCE
    )
    assert (
        service.resolve_job_ref(
            tenant_id="tenant-b",
            provider="jinareader",
            job_id="deterministic-job",
            source_handle=handle_b,
        ).provider_instance_id
        == _JINA_INSTANCE
    )

    assert jobs.is_owned_by(
        tenant_id="tenant-a",
        provider="jinareader",
        provider_instance_id=_JINA_INSTANCE,
        job_id="deterministic-job",
    )
    assert jobs.is_owned_by(
        tenant_id="tenant-b",
        provider="jinareader",
        provider_instance_id=_JINA_INSTANCE,
        job_id="deterministic-job",
    )
    assert not jobs.claim_verified(
        tenant_id="tenant-c",
        provider="jinareader",
        provider_instance_id=_JINA_INSTANCE,
        job_id="deterministic-job",
    )


def test_registered_jina_status_accepts_historical_response_without_identity_echo(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, _, jobs = _service(sqlite_session_factory)
    source_handle = jobs.register_created(
        tenant_id="tenant-a",
        provider="jinareader",
        provider_instance_id=_JINA_INSTANCE,
        job_id="known-job",
    )
    assert source_handle is not None
    response = _HttpResponse(
        {
            "code": 200,
            "data": {"status": "active", "urls": [], "processed": {}, "failed": {}},
        }
    )

    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(website_service_module._adaptive_http_client, "post", return_value=response),
    ):
        result = service.get_crawl_status_typed(
            WebsiteCrawlStatusApiRequest(
                provider="jinareader",
                job_id="known-job",
                source_handle=source_handle,
            )
        )

    assert result["status"] == "active"
    assert result["source_handle"] == source_handle


def test_status_rejects_unknown_owner_before_credentials_or_provider_call(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, providers, jobs = _service(sqlite_session_factory)
    source_handle = jobs.register_created(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="shared-job",
    )
    assert source_handle is not None

    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-b"),
        patch.object(website_service_module, "FirecrawlApp") as firecrawl,
    ):
        with pytest.raises(ValueError, match="Crawl job not found"):
            service.get_crawl_status_typed(
                WebsiteCrawlStatusApiRequest(
                    provider="firecrawl",
                    job_id="shared-job",
                    source_handle=source_handle,
                )
            )

    assert providers.calls == []
    firecrawl.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "job_id"),
    [("watercrawl", "shared-job"), ("firecrawl", "other-job")],
)
def test_source_handle_rejects_tampered_provider_or_job_before_credentials(
    sqlite_session_factory: sessionmaker[Session],
    provider: str,
    job_id: str,
) -> None:
    service, providers, jobs = _service(sqlite_session_factory)
    source_handle = jobs.register_created(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="shared-job",
    )
    assert source_handle is not None

    with pytest.raises(ValueError, match="Source handle does not match crawl job"):
        service.resolve_job_ref(
            tenant_id="tenant-a",
            provider=provider,
            job_id=job_id,
            source_handle=source_handle,
        )

    assert providers.calls == []


def test_untrusted_job_id_is_hashed_in_scoped_cache_key(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    weird_job_id = "../../foreign/cache-object"
    service, _, jobs = _service(sqlite_session_factory)
    assert jobs.register_created(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id=weird_job_id,
    )
    scoped_key = website_crawl_cache_key(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id=weird_job_id,
    )
    assert weird_job_id not in scoped_key
    assert ".." not in scoped_key

    with patch.object(website_service_module, "storage", autospec=True) as storage:
        storage.exists.return_value = True
        storage.load_once.return_value = json.dumps(
            [{"source_url": "https://private.example/page", "markdown": "safe"}]
        ).encode()
        job_ref = service.resolve_stored_job_ref(tenant_id="tenant-a", provider="firecrawl", job_id=weird_job_id)
        result = service.get_crawl_url_data_by_ref(job_ref, "https://private.example/page")

    assert result is not None
    assert result["markdown"] == "safe"
    storage.load_once.assert_called_once_with(scoped_key)


@pytest.mark.parametrize("provider", ["firecrawl", "watercrawl"])
def test_empty_async_job_id_fails_before_credentials_cache_or_provider(
    sqlite_session_factory: sessionmaker[Session], provider: str
) -> None:
    service, providers, _ = _service(sqlite_session_factory)

    with (
        patch.object(website_service_module, "storage", autospec=True) as storage,
        patch.object(website_service_module, "FirecrawlApp") as firecrawl,
        patch.object(website_service_module, "WaterCrawlProvider") as watercrawl,
    ):
        with pytest.raises(ValueError, match="Crawl job not found"):
            service.resolve_stored_job_ref(tenant_id="tenant-a", provider=provider, job_id="")

    assert providers.calls == []
    storage.exists.assert_not_called()
    storage.load_once.assert_not_called()
    firecrawl.assert_not_called()
    watercrawl.assert_not_called()


def test_estimate_resolves_created_source_handle_and_uses_scoped_cache(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, providers, jobs = _service(sqlite_session_factory)
    crawl_data: list[FirecrawlDocumentData] = [
        {
            "source_url": "https://legacy.example/page",
            "markdown": "legacy tenant content",
            "title": "Legacy",
            "description": None,
        }
    ]
    firecrawl = _FirecrawlStatusClient(
        CrawlStatusResponse(
            status="completed",
            total=1,
            current=1,
            data=crawl_data,
        )
    )
    scoped_key = website_crawl_cache_key(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="legacy-job",
    )

    with (
        patch("extensions.application_services.data_sources.build_website_service", return_value=service),
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(
            WebsiteService,
            "_crawl_with_firecrawl",
            return_value={"status": "active", "job_id": "legacy-job"},
        ),
        patch.object(website_service_module, "FirecrawlApp", return_value=firecrawl),
        patch.object(website_service_module, "storage", autospec=True) as storage,
    ):
        storage.exists.return_value = False
        crawl_result = service.crawl_url(
            WebsiteCrawlApiRequest(
                provider="firecrawl",
                url="https://legacy.example/page",
                options={"crawl_sub_pages": True},
            )
        )
        source_handle = crawl_result["source_handle"]
        assert isinstance(source_handle, str)
        setting = WebsiteSourceAdapter(jobs=service).resolve_selection(
            workspace_id="tenant-a",
            provider="firecrawl",
            job_id="legacy-job",
            source_handle=source_handle,
            url="https://legacy.example/page",
            mode="crawl",
            only_main_content=False,
            document_model="text_model",
        )
        result = IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
            "tenant-a",
            [setting],
            {"mode": "automatic", "rules": {}},
            doc_form="text_model",
        )

    assert result.total_segments == 1
    assert result.preview[0].content == "legacy tenant content"
    assert jobs.is_owned_by(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=_FIRECRAWL_INSTANCE,
        job_id="legacy-job",
    )
    assert len(providers.calls) == 3
    assert firecrawl.calls == ["legacy-job"]
    storage.save.assert_called_once_with(scoped_key, json.dumps(crawl_data).encode("utf-8"))
    assert all(call.args[0] != "website_files/legacy-job.txt" for call in storage.exists.call_args_list)


def test_async_jina_source_without_handle_fails_closed(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, providers, jobs = _service(sqlite_session_factory)
    with (
        patch.object(website_service_module._jina_http_client, "get") as get,
        patch.object(website_service_module._adaptive_http_client, "post") as status,
    ):
        with pytest.raises(ValueError, match="source handle is required"):
            service.resolve_job_ref(
                tenant_id="tenant-a",
                provider="jinareader",
                job_id="legacy-job",
            )

    assert not jobs.has_owner(provider="jinareader", provider_instance_id=_JINA_INSTANCE, job_id="legacy-job")
    assert providers.calls == []
    get.assert_not_called()
    status.assert_not_called()


def test_status_without_source_handle_fails_closed_before_remote_probe(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, providers, jobs = _service(sqlite_session_factory)
    firecrawl = _FirecrawlStatusClient({"status": "active", "total": 1, "current": 0, "data": []})

    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(website_service_module, "FirecrawlApp", return_value=firecrawl),
    ):
        with pytest.raises(ValueError, match="source handle is required"):
            service.get_crawl_status_typed(WebsiteCrawlStatusApiRequest(provider="firecrawl", job_id="legacy-job"))

    assert not jobs.has_owner(provider="firecrawl", provider_instance_id=_FIRECRAWL_INSTANCE, job_id="legacy-job")
    assert providers.calls == []
    assert firecrawl.calls == []


def test_exclusive_registration_is_idempotent_for_owner_and_rejects_other_tenant(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    _, _, jobs = _service(sqlite_session_factory)

    first_handle = jobs.register_created(
        tenant_id="tenant-a", provider="firecrawl", provider_instance_id=_FIRECRAWL_INSTANCE, job_id="job-1"
    )
    second_handle = jobs.register_created(
        tenant_id="tenant-a", provider="firecrawl", provider_instance_id=_FIRECRAWL_INSTANCE, job_id="job-1"
    )
    assert first_handle is not None
    assert second_handle == first_handle
    assert not jobs.register_created(
        tenant_id="tenant-b", provider="firecrawl", provider_instance_id=_FIRECRAWL_INSTANCE, job_id="job-1"
    )
    assert jobs.is_owned_by(
        tenant_id="tenant-a", provider="firecrawl", provider_instance_id=_FIRECRAWL_INSTANCE, job_id="job-1"
    )


def test_same_firecrawl_job_isolated_across_self_hosted_instances(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service_a, providers_a, jobs = _service(
        sqlite_session_factory,
        credentials={"firecrawl_api_key": "key-a", "base_url": "https://crawl-a.example/api/"},
    )
    service_b, providers_b, _ = _service(
        sqlite_session_factory,
        credentials={"firecrawl_api_key": "key-b", "base_url": "https://crawl-b.example/api"},
    )
    request = WebsiteCrawlApiRequest(
        provider="firecrawl",
        url="https://source.example",
        options={"crawl_sub_pages": True},
    )
    instance_a = website_crawl_provider_instance_id(provider="firecrawl", base_url="https://crawl-a.example/api")
    instance_b = website_crawl_provider_instance_id(provider="firecrawl", base_url="https://crawl-b.example/api")

    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(
            WebsiteService,
            "_crawl_with_firecrawl",
            side_effect=[
                {"status": "active", "job_id": "same-job"},
                {"status": "active", "job_id": "same-job"},
            ],
        ),
    ):
        result_a = service_a.crawl_url(request)
        result_b = service_b.crawl_url(request)

    handle_a = result_a["source_handle"]
    handle_b = result_b["source_handle"]
    assert isinstance(handle_a, str)
    assert isinstance(handle_b, str)
    assert handle_a != handle_b
    providers_a.credentials = {"firecrawl_api_key": "key-a", "base_url": "https://crawl-a.example/api/"}
    providers_b.credentials = {"firecrawl_api_key": "key-b", "base_url": "https://crawl-b.example/api"}
    assert (
        service_a.resolve_job_ref(
            tenant_id="tenant-a", provider="firecrawl", job_id="same-job", source_handle=handle_a
        ).provider_instance_id
        == instance_a
    )
    assert (
        service_b.resolve_job_ref(
            tenant_id="tenant-a", provider="firecrawl", job_id="same-job", source_handle=handle_b
        ).provider_instance_id
        == instance_b
    )
    with pytest.raises(ValueError, match="endpoint has changed"):
        service_b.resolve_job_ref(tenant_id="tenant-a", provider="firecrawl", job_id="same-job", source_handle=handle_a)

    assert jobs.is_owned_by(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=instance_a,
        job_id="same-job",
    )
    assert jobs.is_owned_by(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=instance_b,
        job_id="same-job",
    )
    key_a = website_crawl_cache_key(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=instance_a,
        job_id="same-job",
    )
    key_b = website_crawl_cache_key(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=instance_b,
        job_id="same-job",
    )
    payloads = {
        key_a: json.dumps([{"source_url": "https://source.example", "markdown": "instance-a"}]).encode(),
        key_b: json.dumps([{"source_url": "https://source.example", "markdown": "instance-b"}]).encode(),
    }
    with patch.object(website_service_module, "storage", autospec=True) as storage:
        storage.exists.return_value = True
        storage.load_once.side_effect = payloads.__getitem__
        result_a = service_a.get_crawl_url_data_by_ref(
            WebsiteJobRef(
                tenant_id="tenant-a",
                provider="firecrawl",
                provider_instance_id=instance_a,
                job_id="same-job",
            ),
            "https://source.example",
        )
        result_b = service_b.get_crawl_url_data_by_ref(
            WebsiteJobRef(
                tenant_id="tenant-a",
                provider="firecrawl",
                provider_instance_id=instance_b,
                job_id="same-job",
            ),
            "https://source.example",
        )
        setting_a = WebsiteSourceAdapter(jobs=service_a).resolve_selection(
            workspace_id="tenant-a",
            provider="firecrawl",
            job_id="same-job",
            source_handle=handle_a,
            url="https://source.example",
            mode="crawl",
            only_main_content=False,
            document_model="text_model",
        )
        setting_b = WebsiteSourceAdapter(jobs=service_b).resolve_selection(
            workspace_id="tenant-a",
            provider="firecrawl",
            job_id="same-job",
            source_handle=handle_b,
            url="https://source.example",
            mode="crawl",
            only_main_content=False,
            document_model="text_model",
        )
        with patch(
            "extensions.application_services.data_sources.build_website_service",
            side_effect=[service_a, service_b],
        ):
            estimate_a = IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
                "tenant-a", [setting_a], {"mode": "automatic", "rules": {}}, doc_form="text_model"
            )
            estimate_b = IndexingEstimateAdapter(session_factory=sqlite_session_factory).run(
                "tenant-a", [setting_b], {"mode": "automatic", "rules": {}}, doc_form="text_model"
            )

    assert key_a != key_b
    assert result_a is not None
    assert result_a["markdown"] == "instance-a"
    assert result_b is not None
    assert result_b["markdown"] == "instance-b"
    assert estimate_a.preview[0].content == "instance-a"
    assert estimate_b.preview[0].content == "instance-b"


def test_persisted_job_ref_is_not_reinterpreted_after_endpoint_switch(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    base_url_a = "https://crawl-a.example/api"
    base_url_b = "https://crawl-b.example/api"
    instance_a = website_crawl_provider_instance_id(provider="firecrawl", base_url=base_url_a)
    instance_b = website_crawl_provider_instance_id(provider="firecrawl", base_url=base_url_b)
    service, providers, jobs = _service(
        sqlite_session_factory,
        credentials={"firecrawl_api_key": "rotated-key", "base_url": base_url_a},
    )
    assert jobs.register_created(
        tenant_id="tenant-a", provider="firecrawl", provider_instance_id=instance_a, job_id="same-job"
    )
    assert jobs.register_created(
        tenant_id="tenant-a", provider="firecrawl", provider_instance_id=instance_b, job_id="same-job"
    )
    job_ref = WebsiteJobRef(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=instance_a,
        job_id="same-job",
    )
    key_a = website_crawl_cache_key(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=instance_a,
        job_id="same-job",
    )

    with patch.object(website_service_module, "storage", autospec=True) as storage:
        storage.exists.return_value = True
        storage.load_once.return_value = json.dumps(
            [{"source_url": "https://source.example", "markdown": "instance-a"}]
        ).encode()
        assert service.get_crawl_url_data_by_ref(job_ref, "https://source.example") == {
            "source_url": "https://source.example",
            "markdown": "instance-a",
        }

        providers.credentials = {
            "firecrawl_api_key": "key-b",
            "base_url": base_url_b,
        }
        with pytest.raises(ValueError, match="endpoint has changed"):
            service.get_crawl_url_data_by_ref(job_ref, "https://source.example")

    storage.exists.assert_called_once_with(key_a)
    storage.load_once.assert_called_once_with(key_a)

    providers.credentials = {
        "firecrawl_api_key": "key-a",
        "base_url": base_url_a,
    }
    with pytest.raises(ValueError, match="source handle is required"):
        service.resolve_job_ref(tenant_id="tenant-a", provider="firecrawl", job_id="same-job")


def test_legacy_document_migrates_from_unique_owner_and_rejects_switched_endpoint(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    base_url_a = "https://crawl-a.example/api"
    base_url_b = "https://crawl-b.example/api"
    instance_a = website_crawl_provider_instance_id(provider="firecrawl", base_url=base_url_a)
    service, providers, jobs = _service(
        sqlite_session_factory,
        credentials={"firecrawl_api_key": "key-b", "base_url": base_url_b},
    )
    assert jobs.register_created(
        tenant_id="tenant-a", provider="firecrawl", provider_instance_id=instance_a, job_id="legacy-job"
    )
    source_info: dict[str, object] = {
        "provider": "firecrawl",
        "job_id": "legacy-job",
        "url": "https://source.example",
        "mode": "crawl",
        "only_main_content": True,
    }
    tenant = make_tenant(tenant_id="tenant-a")
    dataset = make_dataset(dataset_id="dataset-a", tenant_id=tenant.id)
    document = make_document(
        document_id="document-a",
        dataset_id=dataset.id,
        tenant_id=tenant.id,
        data_source_type=DocumentDataSourceType.WEBSITE_CRAWL,
    )
    document.data_source_info = json.dumps(source_info)
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, dataset, document])

    source = StoredSource(
        document_ref=DatasetRef("tenant-a", "dataset-a").document("document-a"),
        source_type="website_crawl",
        source_info=source_info,
        document_model="text_model",
    )
    adapter = WebsiteSourceAdapter(
        jobs=service,
        migrations=SQLAlchemyWebsiteSourceRepository(session_factory=sqlite_session_factory),
    )
    setting = adapter.resolve(source)

    assert setting.website_info is not None
    assert setting.website_info.job_ref.provider_instance_id == instance_a
    assert providers.calls == []
    with sqlite_session_factory() as session:
        stored = session.get_one(Document, "document-a")
        assert stored.data_source_info_dict["provider_instance_id"] == instance_a

    with patch.object(website_service_module, "storage", autospec=True) as storage:
        with pytest.raises(ValueError, match="endpoint has changed"):
            service.get_crawl_url_data_by_ref(setting.website_info.job_ref, "https://source.example")
    storage.exists.assert_not_called()


def test_legacy_document_without_durable_owner_fails_closed(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, providers, _ = _service(sqlite_session_factory)
    source_info: dict[str, object] = {
        "provider": "firecrawl",
        "job_id": "legacy-job",
        "url": "https://source.example",
        "mode": "crawl",
    }
    adapter = WebsiteSourceAdapter(jobs=service)

    with pytest.raises(IndexingInputSourceError, match="provider instance is not recorded"):
        adapter.resolve(
            StoredSource(
                document_ref=DatasetRef("tenant-a", "dataset-a").document("document-a"),
                source_type="website_crawl",
                source_info=source_info,
                document_model="text_model",
            )
        )

    assert providers.calls == []


@pytest.mark.parametrize("provider", ["firecrawl", "watercrawl"])
def test_exclusive_owner_lookup_uses_the_complete_unique_index(
    sqlite_engine: Engine,
    sqlite_session_factory: sessionmaker[Session],
    provider: str,
) -> None:
    with sqlite_session_factory.begin() as session:
        session.add_all([make_tenant(tenant_id="tenant-a"), make_tenant(tenant_id="tenant-b")])
        session.add_all(
            WebsiteCrawlJob(
                tenant_id="tenant-a" if index % 2 else "tenant-b",
                provider=provider,
                provider_instance_id="shared-default-endpoint",
                job_id=f"history-job-{index}",
                exclusive_job_id=f"history-job-{index}",
            )
            for index in range(100)
        )
    jobs = SQLAlchemyWebsiteCrawlJobRepository(session_factory=sqlite_session_factory)
    statements: list[tuple[str, tuple[str | int, ...]]] = []

    def capture_lookup(
        _connection: object,
        _cursor: object,
        statement: str,
        parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT") and "website_crawl_jobs" in statement:
            assert isinstance(parameters, tuple)
            bound_parameters: list[str | int] = []
            for value in parameters:
                assert isinstance(value, str | int)
                bound_parameters.append(value)
            statements.append((statement, tuple(bound_parameters)))

    event.listen(sqlite_engine, "before_cursor_execute", capture_lookup)
    try:
        assert jobs.has_owner(
            provider=provider, provider_instance_id="shared-default-endpoint", job_id="history-job-63"
        )
        assert not jobs.has_owner(
            provider=provider, provider_instance_id="shared-default-endpoint", job_id="unknown-job"
        )
        assert not jobs.has_owner(provider=provider, provider_instance_id="another-endpoint", job_id="history-job-63")
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", capture_lookup)

    assert len(statements) == 3
    with sqlite_engine.connect() as connection:
        for statement, parameters in statements:
            plan = connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters).all()
            detail = " ".join(str(row[-1]) for row in plan)
            assert "SEARCH website_crawl_jobs USING INDEX" in detail
            assert "provider=? AND provider_instance_id=? AND exclusive_job_id=?" in detail


@pytest.mark.parametrize("provider", ["firecrawl", "watercrawl"])
def test_exclusive_job_id_is_scoped_to_provider_instance(
    sqlite_session_factory: sessionmaker[Session], provider: str
) -> None:
    _, _, jobs = _service(sqlite_session_factory)
    if provider == "watercrawl":
        base_url_a = "https://crawl-a.example/instance"
        base_url_b = "https://crawl-b.example/instance"
    else:
        base_url_a = "https://crawl.example/instance-a"
        base_url_b = "https://crawl.example/instance-b"
    instance_a = website_crawl_provider_instance_id(provider=provider, base_url=base_url_a)
    instance_b = website_crawl_provider_instance_id(provider=provider, base_url=base_url_b)

    assert jobs.register_created(
        tenant_id="tenant-a",
        provider=provider,
        provider_instance_id=instance_a,
        job_id="same-job",
    )
    assert jobs.register_created(
        tenant_id="tenant-b",
        provider=provider,
        provider_instance_id=instance_b,
        job_id="same-job",
    )


def test_watercrawl_create_registers_same_job_on_distinct_self_hosted_instances(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service_a, _, jobs = _service(
        sqlite_session_factory,
        credentials={"api_key": "key-a", "base_url": "https://crawl-a.example/instance"},
    )
    service_b, _, _ = _service(
        sqlite_session_factory,
        credentials={"api_key": "key-b", "base_url": "https://crawl-b.example/instance/"},
    )
    request = WebsiteCrawlApiRequest(
        provider="watercrawl",
        url="https://source.example",
        options={"crawl_sub_pages": True},
    )
    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(
            WebsiteService,
            "_crawl_with_watercrawl",
            return_value={"status": "active", "job_id": "same-job"},
        ),
    ):
        service_a.crawl_url(request)
        service_b.crawl_url(request)

    for base_url in (
        "https://crawl-a.example/instance",
        "https://crawl-b.example/instance",
    ):
        assert jobs.is_owned_by(
            tenant_id="tenant-a",
            provider="watercrawl",
            provider_instance_id=website_crawl_provider_instance_id(provider="watercrawl", base_url=base_url),
            job_id="same-job",
        )


def test_same_self_hosted_instance_rejects_foreign_owner_even_with_different_api_key(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    base_url = "HTTPS://CRAWL.EXAMPLE:443/api/"
    instance_id = website_crawl_provider_instance_id(provider="firecrawl", base_url=base_url)
    service, _, jobs = _service(
        sqlite_session_factory,
        credentials={"firecrawl_api_key": "tenant-b-key", "base_url": "https://crawl.example/api"},
    )
    assert jobs.register_created(
        tenant_id="tenant-a",
        provider="firecrawl",
        provider_instance_id=instance_id,
        job_id="owned-job",
    )

    with patch.object(website_service_module, "FirecrawlApp") as firecrawl:
        with pytest.raises(ValueError, match="Crawl job not found"):
            service.get_crawl_url_data_by_ref(
                WebsiteJobRef(
                    tenant_id="tenant-b",
                    provider="firecrawl",
                    provider_instance_id=instance_id,
                    job_id="owned-job",
                ),
                "https://source.example",
            )

    firecrawl.assert_not_called()


def test_watercrawl_paths_on_same_origin_share_owner_boundary(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    origin_instance = website_crawl_provider_instance_id(
        provider="watercrawl", base_url="https://crawl.example/tenant-a"
    )
    service, _, jobs = _service(
        sqlite_session_factory,
        credentials={"api_key": "tenant-b-key", "base_url": "https://crawl.example/tenant-b/"},
    )
    assert jobs.register_created(
        tenant_id="tenant-a",
        provider="watercrawl",
        provider_instance_id=origin_instance,
        job_id="owned-job",
    )

    with patch.object(website_service_module, "WaterCrawlProvider") as watercrawl:
        with pytest.raises(ValueError, match="Crawl job not found"):
            service.get_crawl_url_data_by_ref(
                WebsiteJobRef(
                    tenant_id="tenant-b",
                    provider="watercrawl",
                    provider_instance_id=origin_instance,
                    job_id="owned-job",
                ),
                "https://source.example",
            )

    watercrawl.assert_not_called()


def test_legacy_path_scoped_watercrawl_owner_requires_origin_verification(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    base_url = "https://crawl.example/legacy-path"
    legacy_path_instance = sha256(f"watercrawl\0{base_url}".encode()).hexdigest()
    origin_instance = website_crawl_provider_instance_id(provider="watercrawl", base_url=base_url)
    service, _, jobs = _service(
        sqlite_session_factory,
        credentials={"api_key": "tenant-a-key", "base_url": base_url},
    )
    assert legacy_path_instance != origin_instance
    assert jobs.register_created(
        tenant_id="tenant-a",
        provider="watercrawl",
        provider_instance_id=legacy_path_instance,
        job_id="legacy-job",
    )

    with patch.object(
        WebsiteService,
        "_get_watercrawl_status",
        return_value={
            "status": "active",
            "job_id": "legacy-job",
            "total": 1,
            "current": 0,
            "data": [],
        },
    ) as status:
        resolved = service.resolve_stored_job_ref(
            tenant_id="tenant-a",
            provider="watercrawl",
            job_id="legacy-job",
        )

    status.assert_called_once()
    assert jobs.is_owned_by(
        tenant_id="tenant-a",
        provider="watercrawl",
        provider_instance_id=origin_instance,
        job_id="legacy-job",
    )
    assert resolved.provider_instance_id == origin_instance


def test_legacy_path_scoped_watercrawl_foreign_owner_is_rejected(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    base_url = "https://crawl.example/legacy-path"
    legacy_path_instance = sha256(f"watercrawl\0{base_url}".encode()).hexdigest()
    service, _, jobs = _service(
        sqlite_session_factory,
        credentials={"api_key": "tenant-b-key", "base_url": f"{base_url}/"},
    )
    assert jobs.register_created(
        tenant_id="tenant-a",
        provider="watercrawl",
        provider_instance_id=legacy_path_instance,
        job_id="legacy-job",
    )

    with patch.object(website_service_module, "WaterCrawlProvider") as watercrawl:
        with pytest.raises(ValueError, match="provider instance is not recorded"):
            service.resolve_stored_job_ref(tenant_id="tenant-b", provider="watercrawl", job_id="legacy-job")

    watercrawl.assert_not_called()
    request = WebsiteCrawlApiRequest(
        provider="watercrawl",
        url="https://source.example",
        options={"crawl_sub_pages": True},
    )
    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-b"),
        patch.object(
            WebsiteService,
            "_crawl_with_watercrawl",
            return_value={"status": "active", "job_id": "legacy-job"},
        ),
    ):
        with pytest.raises(ValueError, match="Crawl job not found"):
            service.crawl_url(request)

    assert not jobs.has_owner(
        provider="watercrawl",
        provider_instance_id=website_crawl_provider_instance_id(provider="watercrawl", base_url=base_url),
        job_id="legacy-job",
    )


def test_watercrawl_transport_url_and_instance_ignore_configured_base_path() -> None:
    client = WaterCrawlAPIClient("key", "HTTPS://CRAWL.EXAMPLE:443/configured/path/")
    response = _TransportResponse()
    with patch("core.rag.extractor.watercrawl.client.ssrf_proxy.make_request", return_value=response) as request:
        assert client._request("GET", "/api/v1/core/crawl-requests/job/") is response

    assert request.call_args.args[1] == "https://CRAWL.EXAMPLE:443/api/v1/core/crawl-requests/job/"
    assert website_crawl_provider_instance_id(
        provider="watercrawl", base_url="HTTPS://CRAWL.EXAMPLE:443/configured/path/"
    ) == website_crawl_provider_instance_id(provider="watercrawl", base_url="https://crawl.example/another/path")


def test_provider_instance_normalizes_defaults_and_never_depends_on_credentials() -> None:
    assert website_crawl_provider_instance_id(provider="firecrawl") == website_crawl_provider_instance_id(
        provider="firecrawl", base_url="HTTPS://API.FIRECRAWL.DEV:443/"
    )
    assert website_crawl_provider_instance_id(provider="watercrawl") == website_crawl_provider_instance_id(
        provider="watercrawl", base_url="https://APP.WATERCRAWL.DEV:443/"
    )
    assert website_crawl_provider_instance_id(
        provider="firecrawl",
        base_url="https://self-hosted.example",
    ) == website_crawl_provider_instance_id(provider="firecrawl", base_url="https://self-hosted.example/")
    assert website_crawl_provider_instance_id(
        provider="watercrawl", base_url="https://self-hosted.example/one"
    ) == website_crawl_provider_instance_id(provider="watercrawl", base_url="https://self-hosted.example/two")


@pytest.mark.parametrize("provider", ["firecrawl", "watercrawl", "jinareader"])
def test_status_rejects_unknown_unsafe_job_id_before_remote_probe(
    sqlite_session_factory: sessionmaker[Session], provider: str
) -> None:
    service, providers, jobs = _service(sqlite_session_factory)

    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(website_service_module, "FirecrawlApp") as firecrawl,
        patch.object(website_service_module, "WaterCrawlProvider") as watercrawl,
        patch.object(website_service_module._adaptive_http_client, "post") as jina,
    ):
        with pytest.raises(ValueError, match="Crawl job not found"):
            service.get_crawl_status_typed(
                WebsiteCrawlStatusApiRequest(
                    provider=provider,
                    job_id="../../other/job",
                    source_handle="untrusted-handle",
                )
            )

    assert providers.calls == []
    firecrawl.assert_not_called()
    watercrawl.assert_not_called()
    jina.assert_not_called()
    assert not jobs.has_owner(
        provider=provider,
        provider_instance_id=_instance_for(provider),
        job_id="../../other/job",
    )


def test_watercrawl_tampered_source_handle_is_not_probed_or_claimed(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, _, jobs = _service(sqlite_session_factory)
    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(
            WebsiteService,
            "_get_watercrawl_status",
            return_value={"status": "active", "job_id": "other-job", "total": 1, "current": 0, "data": []},
        ) as status,
    ):
        with pytest.raises(ValueError, match="Crawl job not found"):
            service.get_crawl_status_typed(
                WebsiteCrawlStatusApiRequest(
                    provider="watercrawl",
                    job_id="legacy-job",
                    source_handle="tampered-handle",
                )
            )

    status.assert_not_called()
    assert not jobs.has_owner(provider="watercrawl", provider_instance_id=_WATERCRAWL_INSTANCE, job_id="legacy-job")


@pytest.mark.parametrize("payload", [{"code": 401}, {}, {"code": 200, "data": {}}])
def test_jina_error_response_is_not_claimed(
    sqlite_session_factory: sessionmaker[Session], payload: dict[str, object]
) -> None:
    service, _, jobs = _service(sqlite_session_factory)
    source_handle = jobs.register_created(
        tenant_id="tenant-a",
        provider="jinareader",
        provider_instance_id=_JINA_INSTANCE,
        job_id="legacy-job",
    )
    assert source_handle is not None
    response = _HttpResponse(payload)
    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(website_service_module._adaptive_http_client, "post", return_value=response),
    ):
        with pytest.raises(ValueError, match="Invalid Jina Reader response"):
            service.get_crawl_status_typed(
                WebsiteCrawlStatusApiRequest(
                    provider="jinareader",
                    job_id="legacy-job",
                    source_handle=source_handle,
                )
            )

    assert jobs.is_owned_by(
        tenant_id="tenant-a",
        provider="jinareader",
        provider_instance_id=_JINA_INSTANCE,
        job_id="legacy-job",
    )


def test_jina_response_with_mismatched_task_identity_is_not_claimed(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, _, jobs = _service(sqlite_session_factory)
    source_handle = jobs.register_created(
        tenant_id="tenant-a",
        provider="jinareader",
        provider_instance_id=_JINA_INSTANCE,
        job_id="legacy-job",
    )
    assert source_handle is not None
    data: dict[str, object] = {
        "status": "active",
        "urls": [],
        "processed": {},
        "failed": {},
    }
    data["taskId"] = "other-job"
    response = _HttpResponse({"code": 200, "data": data})
    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(website_service_module._adaptive_http_client, "post", return_value=response),
    ):
        with pytest.raises(ValueError, match="does not match requested job"):
            service.get_crawl_status_typed(
                WebsiteCrawlStatusApiRequest(
                    provider="jinareader",
                    job_id="legacy-job",
                    source_handle=source_handle,
                )
            )

    assert jobs.is_owned_by(
        tenant_id="tenant-a",
        provider="jinareader",
        provider_instance_id=_JINA_INSTANCE,
        job_id="legacy-job",
    )


def test_unknown_jina_status_fails_closed_before_remote_probe(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, providers, jobs = _service(sqlite_session_factory)
    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(website_service_module._adaptive_http_client, "post") as post,
    ):
        with pytest.raises(ValueError, match="Crawl job not found"):
            service.get_crawl_status_typed(
                WebsiteCrawlStatusApiRequest(
                    provider="jinareader",
                    job_id="legacy-job",
                    source_handle="unknown-handle",
                )
            )

    assert providers.calls == []
    post.assert_not_called()
    assert not jobs.has_owner(provider="jinareader", provider_instance_id=_JINA_INSTANCE, job_id="legacy-job")


def test_firecrawl_tampered_source_handle_is_not_probed_or_claimed(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    service, _, jobs = _service(sqlite_session_factory)
    firecrawl = _FirecrawlStatusClient({"status": "mystery", "total": 0, "current": 0, "data": []})
    with (
        patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"),
        patch.object(website_service_module, "FirecrawlApp", return_value=firecrawl),
    ):
        with pytest.raises(ValueError, match="Crawl job not found"):
            service.get_crawl_status_typed(
                WebsiteCrawlStatusApiRequest(
                    provider="firecrawl",
                    job_id="legacy-job",
                    source_handle="tampered-handle",
                )
            )

    assert firecrawl.calls == []
    assert not jobs.has_owner(provider="firecrawl", provider_instance_id=_FIRECRAWL_INSTANCE, job_id="legacy-job")


@pytest.mark.parametrize("provider", ["firecrawl", "watercrawl"])
def test_unknown_job_requires_server_issued_source_handle(
    sqlite_session_factory: sessionmaker[Session], provider: str
) -> None:
    service, _, jobs = _service(sqlite_session_factory, credentials={})

    with patch.object(website_service_module, "current_user", current_tenant_id="tenant-a"):
        with pytest.raises(ValueError, match="source handle is required"):
            service.get_crawl_status_typed(WebsiteCrawlStatusApiRequest(provider=provider, job_id="legacy-job"))

    assert not jobs.has_owner(
        provider=provider,
        provider_instance_id=_instance_for(provider),
        job_id="legacy-job",
    )

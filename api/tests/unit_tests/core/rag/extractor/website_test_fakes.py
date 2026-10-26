"""Typed collaborators for extractor tests that patch WebsiteService methods."""

from typing import Never, cast

from services.data_source.provider_service import DatasourceProviderService
from services.data_source.website_service import WebsiteService


class FailingDatasourceProvider:
    def get_datasource_credentials(self, *, tenant_id: str, provider: str, plugin_id: str) -> Never:
        raise AssertionError(f"unexpected credential lookup tenant={tenant_id} provider={provider} plugin={plugin_id}")


class FailingWebsiteCrawlJobStore:
    def register_created(self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> Never:
        scope = tenant_id, provider, provider_instance_id, job_id
        raise AssertionError(f"unexpected crawl job registration: {scope}")

    def claim_verified(self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> Never:
        raise AssertionError(f"unexpected crawl job claim: {tenant_id}/{provider}/{provider_instance_id}/{job_id}")

    def get_owned_identity(self, *, tenant_id: str, source_handle: str) -> Never:
        raise AssertionError(f"unexpected source handle lookup: {tenant_id}/{source_handle}")

    def is_owned_by(self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> Never:
        scope = tenant_id, provider, provider_instance_id, job_id
        raise AssertionError(f"unexpected crawl job ownership lookup: {scope}")

    def has_owner(self, *, provider: str, provider_instance_id: str, job_id: str) -> Never:
        raise AssertionError(f"unexpected crawl job owner lookup: {provider}/{provider_instance_id}/{job_id}")

    def owned_provider_instance_ids(self, *, tenant_id: str, provider: str, job_id: str) -> Never:
        raise AssertionError(f"unexpected crawl job instance lookup: {tenant_id}/{provider}/{job_id}")


def unused_website_service() -> WebsiteService:
    """Build the real service while making accidental collaborator access fail."""
    return WebsiteService(
        providers=cast(DatasourceProviderService, FailingDatasourceProvider()),
        crawl_jobs=FailingWebsiteCrawlJobStore(),
    )

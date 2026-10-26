"""Remove crawl caches whose remaining document references belong to a deleted dataset."""

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select

from core.db.session_factory import session_factory
from extensions.ext_storage import storage
from models.dataset import Document
from models.source import WebsiteCrawlJob
from services.data_source.website_crawl_cache import website_crawl_cache_key

logger = logging.getLogger(__name__)


class FirecrawlDataSourceInfo(BaseModel):
    provider: Literal["firecrawl"]
    job_id: str
    provider_instance_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        if not v or any(char in v for char in "/\\\x00"):
            raise ValueError("Invalid job_id")
        return v


def _firecrawl_job_ref(data_source_info: str | None) -> tuple[str, str | None] | None:
    if not data_source_info:
        return None
    try:
        info = FirecrawlDataSourceInfo.model_validate_json(data_source_info)
    except (ValueError, TypeError, ValidationError):
        return None
    return info.job_id, info.provider_instance_id


def cleanup_dataset_crawl_cache(tenant_id: str, dataset_id: str) -> None:
    """Delete only unshared Firecrawl caches, before deleting their document references.

    Crawl keys are tenant/provider scoped; another dataset in the same tenant may still share one.
    Storage failures propagate, preserving the database references for a later retry.
    """
    with session_factory.create_session() as session:
        candidate_refs = {
            job_ref
            for info in session.scalars(
                select(Document.data_source_info).where(
                    Document.tenant_id == tenant_id,
                    Document.dataset_id == dataset_id,
                    Document.data_source_type == "website_crawl",
                )
            )
            if (job_ref := _firecrawl_job_ref(info)) is not None
        }
        if not candidate_refs:
            return
        candidate_job_ids = {job_id for job_id, _ in candidate_refs}
        provider_instances_by_job: dict[str, set[str]] = {}
        for job_id, provider_instance_id in session.execute(
            select(WebsiteCrawlJob.job_id, WebsiteCrawlJob.provider_instance_id).where(
                WebsiteCrawlJob.tenant_id == tenant_id,
                WebsiteCrawlJob.provider == "firecrawl",
                WebsiteCrawlJob.job_id.in_(candidate_job_ids),
            )
        ):
            provider_instances_by_job.setdefault(job_id, set()).add(provider_instance_id)
        resolved_candidates: set[tuple[str, str]] = set()
        for job_id, persisted_instance_id in candidate_refs:
            if persisted_instance_id is not None:
                if persisted_instance_id in provider_instances_by_job.get(job_id, set()):
                    resolved_candidates.add((job_id, persisted_instance_id))
                continue
            known_instances = provider_instances_by_job.get(job_id, set())
            # A legacy source did not record its provider instance. Delete a
            # cache only when durable ownership makes that identity unique;
            # ambiguity must retain all possible instance caches.
            if len(known_instances) == 1:
                resolved_candidates.add((job_id, next(iter(known_instances))))
        if not resolved_candidates:
            return
        references = session.scalars(
            select(Document.data_source_info)
            .where(
                Document.tenant_id == tenant_id,
                Document.dataset_id != dataset_id,
                Document.data_source_type == "website_crawl",
            )
            .execution_options(yield_per=500)
        )
        for info in references:
            job_ref = _firecrawl_job_ref(info)
            if job_ref is None:
                continue
            job_id, persisted_instance_id = job_ref
            if persisted_instance_id is None:
                resolved_candidates = {candidate for candidate in resolved_candidates if candidate[0] != job_id}
            else:
                resolved_candidates.discard((job_id, persisted_instance_id))
        candidates = {
            website_crawl_cache_key(
                tenant_id=tenant_id,
                provider="firecrawl",
                provider_instance_id=provider_instance_id,
                job_id=job_id,
            )
            for job_id, provider_instance_id in resolved_candidates
        }
    # Finish the read transaction before accessing object storage.
    for key in sorted(candidates):
        try:
            storage.delete(key)
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception(
                "Failed to delete dataset crawl cache tenant=%s dataset=%s key=%s", tenant_id, dataset_id, key
            )
            raise

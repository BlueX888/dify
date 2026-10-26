import json
from datetime import datetime
from typing import Any, TypedDict
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import TypeBase
from .types import AdjustedJSON, LongText, StringUUID, adjusted_json_index


class DataSourceOauthBinding(TypeBase):
    __tablename__ = "data_source_oauth_bindings"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="source_binding_pkey"),
        sa.Index("source_binding_tenant_id_idx", "tenant_id"),
        adjusted_json_index("source_info_idx", "source_info"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID, insert_default=lambda: str(uuid4()), default_factory=lambda: str(uuid4()), init=False
    )
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    access_token: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(255), nullable=False)
    source_info: Mapped[dict[str, Any]] = mapped_column(AdjustedJSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), init=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        init=False,
    )
    disabled: Mapped[bool] = mapped_column(sa.Boolean, nullable=True, default=False)


class DataSourceApiKeyAuthBindingDict(TypedDict):
    id: str
    tenant_id: str
    category: str
    provider: str
    credentials: Any
    created_at: float
    updated_at: float
    disabled: bool


class DataSourceApiKeyAuthBinding(TypeBase):
    __tablename__ = "data_source_api_key_auth_bindings"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="data_source_api_key_auth_binding_pkey"),
        sa.Index("data_source_api_key_auth_binding_tenant_id_idx", "tenant_id"),
        sa.Index("data_source_api_key_auth_binding_provider_idx", "provider"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID, insert_default=lambda: str(uuid4()), default_factory=lambda: str(uuid4()), init=False
    )
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    category: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(255), nullable=False)
    credentials: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)  # JSON
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), init=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        init=False,
    )
    disabled: Mapped[bool] = mapped_column(sa.Boolean, nullable=True, default=False)

    def to_dict(self) -> DataSourceApiKeyAuthBindingDict:
        result: DataSourceApiKeyAuthBindingDict = {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "category": self.category,
            "provider": self.provider,
            "credentials": json.loads(self.credentials) if self.credentials else None,
            "created_at": self.created_at.timestamp(),
            "updated_at": self.updated_at.timestamp(),
            "disabled": self.disabled,
        }
        return result


class WebsiteCrawlJob(TypeBase):
    """Durable ownership record for an asynchronous website crawl job."""

    __tablename__ = "website_crawl_jobs"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="website_crawl_job_pkey"),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_instance_id",
            "job_id",
            name="website_crawl_job_owner_job_key",
        ),
        sa.UniqueConstraint(
            "provider",
            "provider_instance_id",
            "exclusive_job_id",
            name="website_crawl_job_exclusive_job_key",
        ),
    )

    id: Mapped[str] = mapped_column(
        StringUUID, insert_default=lambda: str(uuid4()), default_factory=lambda: str(uuid4()), init=False
    )
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_instance_id: Mapped[str] = mapped_column(String(64), nullable=False)
    job_id: Mapped[str] = mapped_column(String(255), nullable=False)
    # Firecrawl and Watercrawl jobs are credential-owned and therefore have one
    # claimant per provider instance. Jina task IDs are deterministic from
    # crawl input, so several tenants may legitimately receive the same ID.
    exclusive_job_id: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), init=False
    )

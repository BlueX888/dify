"""SQLAlchemy persistence for tenant-owned website crawl jobs."""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from models.source import WebsiteCrawlJob


class SQLAlchemyWebsiteCrawlJobRepository:
    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def register_created(self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> str | None:
        """Register a job returned by a provider create call.

        Jina task IDs are deterministic from crawl inputs, so the same ID may
        legitimately be registered by several tenants. Other providers use an
        exclusive database key per provider instance to make first ownership
        atomic.
        """
        return self._register(
            tenant_id=tenant_id,
            provider=provider,
            provider_instance_id=provider_instance_id,
            job_id=job_id,
            exclusive=provider != "jinareader",
        )

    def claim_verified(self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> str | None:
        """Claim a legacy job after credential-scoped provider verification."""
        if provider == "jinareader":
            return None
        return self._register(
            tenant_id=tenant_id,
            provider=provider,
            provider_instance_id=provider_instance_id,
            job_id=job_id,
            exclusive=True,
        )

    def _register(
        self,
        *,
        tenant_id: str,
        provider: str,
        provider_instance_id: str,
        job_id: str,
        exclusive: bool,
    ) -> str | None:
        try:
            with self._session_factory.begin() as session:
                owned_handle = self._owned_handle(
                    session,
                    tenant_id=tenant_id,
                    provider=provider,
                    provider_instance_id=provider_instance_id,
                    job_id=job_id,
                )
                if owned_handle is not None:
                    return owned_handle
                if exclusive and self._has_owner(
                    session,
                    provider=provider,
                    provider_instance_id=provider_instance_id,
                    job_id=job_id,
                ):
                    return None
                job = WebsiteCrawlJob(
                    tenant_id=tenant_id,
                    provider=provider,
                    provider_instance_id=provider_instance_id,
                    job_id=job_id,
                    exclusive_job_id=job_id if exclusive else None,
                )
                session.add(job)
                handle = job.id
            return handle
        except IntegrityError:
            # Another request may have registered the same provider result or
            # won the exclusive claim while this transaction was in flight.
            return self.owned_handle(
                tenant_id=tenant_id,
                provider=provider,
                provider_instance_id=provider_instance_id,
                job_id=job_id,
            )

    def owned_handle(self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> str | None:
        with self._session_factory() as session:
            return self._owned_handle(
                session,
                tenant_id=tenant_id,
                provider=provider,
                provider_instance_id=provider_instance_id,
                job_id=job_id,
            )

    def get_owned_identity(self, *, tenant_id: str, source_handle: str) -> tuple[str, str, str] | None:
        """Resolve an opaque server-issued handle within its tenant boundary."""
        with self._session_factory() as session:
            row = session.execute(
                select(
                    WebsiteCrawlJob.provider,
                    WebsiteCrawlJob.provider_instance_id,
                    WebsiteCrawlJob.job_id,
                ).where(
                    WebsiteCrawlJob.id == source_handle,
                    WebsiteCrawlJob.tenant_id == tenant_id,
                )
            ).one_or_none()
            return tuple(row) if row is not None else None

    def is_owned_by(self, *, tenant_id: str, provider: str, provider_instance_id: str, job_id: str) -> bool:
        return (
            self.owned_handle(
                tenant_id=tenant_id,
                provider=provider,
                provider_instance_id=provider_instance_id,
                job_id=job_id,
            )
            is not None
        )

    def owned_provider_instance_ids(self, *, tenant_id: str, provider: str, job_id: str) -> tuple[str, ...]:
        """Return the durable provider instances this tenant registered for a job ID."""
        with self._session_factory() as session:
            return tuple(
                session.scalars(
                    select(WebsiteCrawlJob.provider_instance_id)
                    .where(
                        WebsiteCrawlJob.tenant_id == tenant_id,
                        WebsiteCrawlJob.provider == provider,
                        WebsiteCrawlJob.job_id == job_id,
                    )
                    .distinct()
                    .order_by(WebsiteCrawlJob.provider_instance_id)
                ).all()
            )

    @staticmethod
    def _owned_handle(
        session: Session,
        *,
        tenant_id: str,
        provider: str,
        provider_instance_id: str,
        job_id: str,
    ) -> str | None:
        return session.scalar(
            select(WebsiteCrawlJob.id)
            .where(
                WebsiteCrawlJob.tenant_id == tenant_id,
                WebsiteCrawlJob.provider == provider,
                WebsiteCrawlJob.provider_instance_id == provider_instance_id,
                WebsiteCrawlJob.job_id == job_id,
            )
            .limit(1)
        )

    def has_owner(self, *, provider: str, provider_instance_id: str, job_id: str) -> bool:
        with self._session_factory() as session:
            return self._has_owner(
                session,
                provider=provider,
                provider_instance_id=provider_instance_id,
                job_id=job_id,
            )

    @staticmethod
    def _has_owner(session: Session, *, provider: str, provider_instance_id: str, job_id: str) -> bool:
        """Probe an exclusive provider's complete unique ownership key."""
        return (
            session.scalar(
                select(WebsiteCrawlJob.id)
                .where(
                    WebsiteCrawlJob.provider == provider,
                    WebsiteCrawlJob.provider_instance_id == provider_instance_id,
                    WebsiteCrawlJob.exclusive_job_id == job_id,
                )
                .limit(1)
            )
            is not None
        )

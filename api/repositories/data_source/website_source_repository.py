"""Persistence for stable provider-instance references on website documents."""

import json
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.extractor.entity.extract_setting import WebsiteJobRef
from models.dataset import Document


class SQLAlchemyWebsiteSourceRepository:
    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def persist_legacy_job_ref(
        self,
        *,
        workspace_id: str,
        dataset_id: str,
        document_id: str,
        expected_source_info: Mapping[str, object],
        job_ref: WebsiteJobRef,
    ) -> None:
        """Attach a verified instance only if the legacy source is still unchanged."""
        if job_ref.tenant_id != workspace_id:
            raise ValueError("Website source tenant does not match the document")
        with self._session_factory.begin() as session:
            document = session.scalar(
                select(Document)
                .where(
                    Document.tenant_id == workspace_id,
                    Document.dataset_id == dataset_id,
                    Document.id == document_id,
                    Document.data_source_type == "website_crawl",
                )
                .with_for_update()
            )
            if document is None:
                raise ValueError("Website source document no longer exists")
            current = document.data_source_info_dict
            if current is None:
                raise ValueError("Website source document has no source information")
            current_instance = current.get("provider_instance_id")
            if current_instance is not None:
                if current_instance != job_ref.provider_instance_id:
                    raise ValueError("Website source provider instance changed concurrently")
                return
            if current != dict(expected_source_info):
                raise ValueError("Website source changed concurrently")
            if current.get("provider") != job_ref.provider or current.get("job_id", "") != job_ref.job_id:
                raise ValueError("Website source job reference does not match the document")
            current["provider_instance_id"] = job_ref.provider_instance_id
            document.data_source_info = json.dumps(current)

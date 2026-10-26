from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from core.rag.models.document import Document


@dataclass(frozen=True)
class CreatedUploadArtifact:
    """A storage object created while constructing its UploadFile record."""

    id: str
    key: str


@dataclass(frozen=True)
class ExtractionResult:
    documents: list[Document]
    created_upload_artifacts: tuple[CreatedUploadArtifact, ...] = ()
    source_version: str | None = None

    @property
    def created_upload_file_ids(self) -> tuple[str, ...]:
        """Compatibility view for owners that only need committed record ids."""

        return tuple(artifact.id for artifact in self.created_upload_artifacts)


class UploadFileExtractionInput(BaseModel):
    """Detached values required to extract one tenant-owned upload."""

    id: str
    tenant_id: str
    key: str
    created_by: str
    model_config = ConfigDict(from_attributes=True)


class StoredDocumentExtractionInput(BaseModel):
    """Detached document values needed to record a source's last edited time."""

    id: str
    tenant_id: str
    dataset_id: str
    data_source_info_dict: dict[str, object] | None
    model_config = ConfigDict(from_attributes=True)


class NotionInfo(BaseModel):
    """
    Notion import info.
    """

    credential_id: str | None = None
    notion_workspace_id: str | None = ""
    notion_obj_id: str
    notion_page_type: str
    document: StoredDocumentExtractionInput | None = None
    tenant_id: str
    notion_access_token: str | None = Field(default=None, exclude=True, repr=False)


class WebsiteJobRef(BaseModel):
    """Stable ownership and provider-instance identity for one website crawl job."""

    tenant_id: str
    provider: str
    provider_instance_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    job_id: str


class WebsiteInfo(BaseModel):
    """
    website import info.
    """

    job_ref: WebsiteJobRef
    url: str
    mode: str
    only_main_content: bool = False

    @property
    def provider(self) -> str:
        return self.job_ref.provider

    @property
    def job_id(self) -> str:
        return self.job_ref.job_id

    @property
    def tenant_id(self) -> str:
        return self.job_ref.tenant_id


class ExtractSetting(BaseModel):
    """
    Model class for provider response.
    """

    datasource_type: str
    upload_file: UploadFileExtractionInput | None = None
    notion_info: NotionInfo | None = None
    website_info: WebsiteInfo | None = None
    document_model: str | None = None

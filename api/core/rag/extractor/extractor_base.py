"""Abstract interface for document loader implementations."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from core.rag.extractor.entity.extract_setting import CreatedUploadArtifact


class BaseExtractor(ABC):
    """Interface for extract files."""

    on_upload_artifact_created: Callable[[CreatedUploadArtifact], None] | None = None

    def register_created_upload_artifact(self, artifact: CreatedUploadArtifact) -> None:
        """Persist caller-owned cleanup evidence before writing the object."""
        if self.on_upload_artifact_created is not None:
            self.on_upload_artifact_created(artifact)

    @property
    def created_upload_artifacts(self) -> tuple[CreatedUploadArtifact, ...]:
        """Storage objects and UploadFile identities created by the latest extraction."""
        return ()

    @property
    def created_upload_file_ids(self) -> tuple[str, ...]:
        """Compatibility view of created artifact ids."""

        return tuple(artifact.id for artifact in self.created_upload_artifacts)

    @abstractmethod
    def extract(self):
        raise NotImplementedError

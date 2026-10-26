"""SQLAlchemy repository for upload files used by knowledge extraction."""

from collections.abc import Sequence
from typing import override

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.extractor.entity.extract_setting import UploadFileExtractionInput
from models.model import UploadFile
from repositories.upload_file_repository import (
    SQLAlchemyUploadFileCleanupRepository,
    UploadFileCleanupArtifact,
    claim_upload_files_for_cleanup_in_session,
    get_document_source_upload_file_id,
    register_upload_file_cleanup_candidates_in_session,
)

__all__ = [
    "SQLAlchemyKnowledgeUploadRepository",
    "get_document_source_upload_file_id",
    "register_upload_file_cleanup_candidates_in_session",
]


def _query_files(session: Session, *, workspace_id: str, file_ids: Sequence[str]) -> dict[str, UploadFile]:
    if not file_ids:
        return {}
    uploads = session.scalars(
        select(UploadFile).where(UploadFile.id.in_(file_ids), UploadFile.tenant_id == workspace_id)
    )
    return {upload.id: upload for upload in uploads}


def query_upload_extraction_inputs(
    session: Session, *, workspace_id: str, file_ids: Sequence[str]
) -> dict[str, UploadFileExtractionInput]:
    """Read detached extraction inputs without changing the caller's session lifecycle."""
    return {
        upload.id: UploadFileExtractionInput.model_validate(upload)
        for upload in _query_files(session, workspace_id=workspace_id, file_ids=file_ids).values()
    }


def claim_unreferenced_files_for_cleanup_in_session(
    session: Session,
    *,
    workspace_id: str,
    file_ids: Sequence[str],
) -> tuple[UploadFileCleanupArtifact, ...]:
    """Submit knowledge candidates to the authoritative file-domain claim."""
    return claim_upload_files_for_cleanup_in_session(
        session,
        workspace_id=workspace_id,
        file_ids=file_ids,
    )


class SQLAlchemyKnowledgeUploadRepository(SQLAlchemyUploadFileCleanupRepository):
    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_by_id(self, *, workspace_id: str, file_id: str) -> UploadFileExtractionInput | None:
        return self.get_by_ids(workspace_id=workspace_id, file_ids=(file_id,)).get(file_id)

    def get_by_ids(self, *, workspace_id: str, file_ids: Sequence[str]) -> dict[str, UploadFileExtractionInput]:
        with self._session_factory() as session:
            return query_upload_extraction_inputs(session, workspace_id=workspace_id, file_ids=file_ids)

    def get_files(self, *, workspace_id: str, file_ids: Sequence[str]) -> dict[str, UploadFile]:
        with self._session_factory() as session:
            return _query_files(session, workspace_id=workspace_id, file_ids=file_ids)

    def get_existing_file_ids(self, *, workspace_id: str, file_ids: Sequence[str]) -> frozenset[str]:
        with self._session_factory() as session:
            return frozenset(_query_files(session, workspace_id=workspace_id, file_ids=file_ids))

    def get_file_name(self, *, workspace_id: str, upload_file_id: str) -> str | None:
        upload = self.get_files(workspace_id=workspace_id, file_ids=(upload_file_id,)).get(upload_file_id)
        return upload.name if upload is not None else None

    @override
    def claim_unreferenced_files_for_cleanup(
        self, *, workspace_id: str, file_ids: Sequence[str]
    ) -> tuple[UploadFileCleanupArtifact, ...]:
        """Persist deletion intent before removing owned, unreferenced file rows."""
        with self._session_factory.begin() as session:
            return self.claim_unreferenced_files_for_cleanup_in_session(
                session,
                workspace_id=workspace_id,
                file_ids=file_ids,
            )

    @override
    def claim_unreferenced_files_for_cleanup_in_session(
        self,
        session: Session,
        *,
        workspace_id: str,
        file_ids: Sequence[str],
    ) -> tuple[UploadFileCleanupArtifact, ...]:
        return claim_unreferenced_files_for_cleanup_in_session(
            session,
            workspace_id=workspace_id,
            file_ids=file_ids,
        )

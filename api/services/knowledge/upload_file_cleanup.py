"""Conservative liveness checks for tenant-owned knowledge uploads.

Segment attachment bindings are the durable ownership record for extracted
images. Text references can only keep a file alive; they never authorize its
deletion because segment text is editable by workspace users.
"""

import logging
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from datetime import datetime
from threading import Event, Thread
from typing import Protocol

from core.rag.extractor.entity.extract_setting import CreatedUploadArtifact

UPLOAD_ARTIFACT_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


class UploadFileCleanupArtifact(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def cleanup_id(self) -> str: ...

    @property
    def tenant_id(self) -> str: ...

    @property
    def key(self) -> str: ...

    @property
    def writer_id(self) -> str | None: ...

    @property
    def writer_finished_at(self) -> datetime | None: ...

    @property
    def cleanup_started_at(self) -> datetime | None: ...


class UploadFileCleanupClaim(Protocol):
    def register_created_upload_artifacts_for_cleanup(
        self, *, workspace_id: str, artifacts: Sequence[CreatedUploadArtifact]
    ) -> None: ...

    def renew_created_upload_artifact_writes(self, *, workspace_id: str, upload_file_ids: Sequence[str]) -> None: ...

    def abandon_created_upload_artifact_writes(self, *, workspace_id: str, upload_file_ids: Sequence[str]) -> None: ...

    def claim_unreferenced_files_for_cleanup(
        self, *, workspace_id: str, file_ids: Sequence[str]
    ) -> Sequence[UploadFileCleanupArtifact]: ...

    def list_pending_cleanup_artifacts(self, *, limit: int) -> Sequence[UploadFileCleanupArtifact]: ...

    def cleanup_artifact_is_referenced(self, *, tenant_id: str, upload_file_id: str) -> bool: ...

    def mark_cleanup_attempted(self, *, tenant_id: str, cleanup_id: str) -> None: ...

    def complete_cleanup(self, *, tenant_id: str, cleanup_id: str) -> None: ...


@contextmanager
def maintain_created_upload_artifact_writers(
    *,
    workspace_id: str,
    upload_file_ids: Callable[[], Sequence[str]],
    files: UploadFileCleanupClaim,
    logger: logging.Logger,
    interval_seconds: float = UPLOAD_ARTIFACT_HEARTBEAT_SECONDS,
) -> Generator[None]:
    """Renew dynamically-discovered artifact writers until durable handoff."""
    if interval_seconds <= 0:
        raise ValueError("upload artifact heartbeat interval must be positive")
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            current_ids = tuple(dict.fromkeys(upload_file_ids()))
            if not current_ids:
                continue
            try:
                files.renew_created_upload_artifact_writes(
                    workspace_id=workspace_id,
                    upload_file_ids=current_ids,
                )
            except Exception:
                logger.exception("Upload artifact writer heartbeat failed, workspace_id=%s", workspace_id)

    thread = Thread(target=heartbeat, name=f"upload-artifact-heartbeat-{workspace_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning("Upload artifact writer heartbeat did not stop promptly, workspace_id=%s", workspace_id)


def cleanup_claimed_upload_files(
    *,
    artifacts: Sequence[UploadFileCleanupArtifact],
    files: UploadFileCleanupClaim,
    delete_blob: Callable[[str], None],
    logger: logging.Logger,
) -> tuple[str, ...]:
    failed_cleanup_ids: list[str] = []
    for artifact in artifacts:
        if files.cleanup_artifact_is_referenced(
            tenant_id=artifact.tenant_id,
            upload_file_id=artifact.id,
        ):
            logger.debug(
                "Upload blob cleanup deferred because the file is referenced again, upload_file_id=%s, cleanup_id=%s",
                artifact.id,
                artifact.cleanup_id,
            )
            continue
        retry = False
        try:
            delete_blob(artifact.key)
        except FileNotFoundError:
            logger.debug("Upload blob already absent, upload_file_id=%s, key=%s", artifact.id, artifact.key)
        except Exception:
            logger.exception(
                "Upload blob cleanup failed; the durable cleanup record will be retried, "
                "upload_file_id=%s, cleanup_id=%s, key=%s",
                artifact.id,
                artifact.cleanup_id,
                artifact.key,
            )
            retry = True
        if not retry:
            if artifact.writer_id is not None and artifact.writer_finished_at is None:
                # A timed-out or disconnected PUT can materialize after this
                # delete. Its tombstone must remain so later drains reconcile
                # that uncertain remote outcome.
                try:
                    files.mark_cleanup_attempted(tenant_id=artifact.tenant_id, cleanup_id=artifact.cleanup_id)
                except Exception:
                    logger.exception("Failed to update upload cleanup retry order, cleanup_id=%s", artifact.cleanup_id)
                continue
            try:
                files.complete_cleanup(tenant_id=artifact.tenant_id, cleanup_id=artifact.cleanup_id)
            except Exception:
                logger.exception(
                    "Upload blob was deleted but cleanup acknowledgement failed; the durable record will be retried, "
                    "upload_file_id=%s, cleanup_id=%s, key=%s",
                    artifact.id,
                    artifact.cleanup_id,
                    artifact.key,
                )
                retry = True

        if retry:
            try:
                files.mark_cleanup_attempted(tenant_id=artifact.tenant_id, cleanup_id=artifact.cleanup_id)
            except Exception:
                logger.exception("Failed to update upload cleanup retry order, cleanup_id=%s", artifact.cleanup_id)
            failed_cleanup_ids.append(artifact.cleanup_id)
    return tuple(failed_cleanup_ids)


def cleanup_unreferenced_upload_files(
    *,
    workspace_id: str,
    file_ids: Sequence[str],
    files: UploadFileCleanupClaim,
    delete_blob: Callable[[str], None],
    logger: logging.Logger,
) -> tuple[str, ...]:
    """Persist deletion intent transactionally, then delete blobs outside the transaction."""
    claimed = files.claim_unreferenced_files_for_cleanup(workspace_id=workspace_id, file_ids=file_ids)
    cleanup_claimed_upload_files(artifacts=claimed, files=files, delete_blob=delete_blob, logger=logger)
    return tuple(artifact.id for artifact in claimed)


def cleanup_created_upload_artifacts(
    *,
    workspace_id: str,
    artifacts: Sequence[CreatedUploadArtifact],
    files: UploadFileCleanupClaim,
    delete_blob: Callable[[str], None],
    logger: logging.Logger,
) -> tuple[str, ...]:
    """Persist exact cleanup evidence before checking whether created objects were published."""
    unique_artifacts = tuple({artifact.id: artifact for artifact in artifacts}.values())
    if not unique_artifacts:
        return ()
    files.abandon_created_upload_artifact_writes(
        workspace_id=workspace_id,
        upload_file_ids=tuple(artifact.id for artifact in unique_artifacts),
    )
    files.register_created_upload_artifacts_for_cleanup(
        workspace_id=workspace_id,
        artifacts=unique_artifacts,
    )
    return cleanup_unreferenced_upload_files(
        workspace_id=workspace_id,
        file_ids=tuple(artifact.id for artifact in unique_artifacts),
        files=files,
        delete_blob=delete_blob,
        logger=logger,
    )


def drain_pending_upload_file_cleanups(
    *,
    files: UploadFileCleanupClaim,
    delete_blob: Callable[[str], None],
    logger: logging.Logger,
    limit: int,
) -> tuple[str, ...]:
    """Attempt one durable cleanup batch and return records that still need retry."""
    pending = files.list_pending_cleanup_artifacts(limit=limit)
    return cleanup_claimed_upload_files(artifacts=pending, files=files, delete_blob=delete_blob, logger=logger)

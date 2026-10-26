import logging
from datetime import UTC, datetime
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.engine import ExecutionContext
from sqlalchemy.orm import Session, sessionmaker

from core.rag.extractor.entity.extract_setting import CreatedUploadArtifact
from extensions.storage.storage_type import StorageType
from graphon.file import FileTransferMethod, FileType
from models.enums import CreatorUserRole
from models.model import MessageFile, UploadFile, UploadFileCleanupOutbox
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository
from repositories.upload_file_repository import UploadFilePublicationLostError
from services.knowledge.upload_file_cleanup import (
    cleanup_claimed_upload_files,
    cleanup_created_upload_artifacts,
    cleanup_unreferenced_upload_files,
    drain_pending_upload_file_cleanups,
    maintain_created_upload_artifact_writers,
)

logger = logging.getLogger(__name__)


def _persist_upload(session_factory: sessionmaker[Session], *, tenant_id: str, key: str) -> UploadFile:
    upload = UploadFile(
        tenant_id=tenant_id,
        storage_type=StorageType.LOCAL,
        key=key,
        name="image.png",
        size=7,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="account-1",
        created_at=datetime.now(UTC),
        used=False,
    )
    with session_factory.begin() as session:
        session.add(upload)
    return upload


def test_temporary_storage_failure_is_retried_from_persistent_outbox(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    upload = _persist_upload(sqlite_session_factory, tenant_id="tenant-1", key="tenant-1/image.png")
    first_attempts: list[str] = []

    def fail_delete(key: str) -> None:
        first_attempts.append(key)
        raise OSError("storage temporarily unavailable")

    first_repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    cleaned_ids = cleanup_unreferenced_upload_files(
        workspace_id="tenant-1",
        file_ids=(upload.id,),
        files=first_repository,
        delete_blob=fail_delete,
        logger=logger,
    )

    assert cleaned_ids == (upload.id,)
    assert first_attempts == [upload.key]
    with sqlite_session_factory() as session:
        assert session.get(UploadFile, upload.id) is None
        pending = session.scalar(
            select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == upload.id)
        )
        assert pending is not None
        cleanup_id = pending.id

    retried_keys: list[str] = []
    second_repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    failures = drain_pending_upload_file_cleanups(
        files=second_repository,
        delete_blob=retried_keys.append,
        logger=logger,
        limit=100,
    )

    assert failures == ()
    assert retried_keys == [upload.key]
    with sqlite_session_factory() as session:
        assert session.get(UploadFileCleanupOutbox, cleanup_id) is None


def test_uncommitted_artifact_key_is_durable_when_blob_cleanup_fails(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    artifact = CreatedUploadArtifact(id=str(uuid4()), key="tenant-1/uncommitted.png")
    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)

    def fail_delete(_key: str) -> None:
        raise OSError("storage unavailable")

    cleaned_ids = cleanup_created_upload_artifacts(
        workspace_id="tenant-1",
        artifacts=(artifact,),
        files=repository,
        delete_blob=fail_delete,
        logger=logger,
    )

    assert cleaned_ids == (artifact.id,)
    with sqlite_session_factory() as session:
        pending = session.scalar(
            select(UploadFileCleanupOutbox).where(
                UploadFileCleanupOutbox.tenant_id == "tenant-1",
                UploadFileCleanupOutbox.upload_file_id == artifact.id,
            )
        )
        assert pending is not None
        assert pending.key == artifact.key


def test_cleanup_fences_late_writer_and_retains_uncertain_put_tombstone(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    artifact = CreatedUploadArtifact(id=str(uuid4()), key="tenant-1/uncertain-put.png")
    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    writer_id = repository.begin_created_upload_artifact_write(
        workspace_id="tenant-1",
        artifact=artifact,
    )
    deleted_keys: list[str] = []

    assert cleanup_created_upload_artifacts(
        workspace_id="tenant-1",
        artifacts=(artifact,),
        files=repository,
        delete_blob=deleted_keys.append,
        logger=logger,
    ) == (artifact.id,)

    with pytest.raises(UploadFilePublicationLostError):
        repository.finish_created_upload_artifact_write(
            workspace_id="tenant-1",
            upload_file_id=artifact.id,
            writer_id=writer_id,
        )

    assert deleted_keys == [artifact.key]
    with sqlite_session_factory() as session:
        marker = session.scalar(select(UploadFileCleanupOutbox))
        assert marker is not None
        assert marker.writer_id == writer_id
        assert marker.writer_finished_at is None
        assert marker.cleanup_started_at is not None

    assert (
        drain_pending_upload_file_cleanups(
            files=repository,
            delete_blob=deleted_keys.append,
            logger=logger,
            limit=10,
        )
        == ()
    )
    assert deleted_keys == [artifact.key, artifact.key]
    with sqlite_session_factory() as session:
        assert session.scalar(select(UploadFileCleanupOutbox)) is not None


def test_drainer_skips_active_writer_lease_then_reconciles_after_expiry(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    artifact = CreatedUploadArtifact(id=str(uuid4()), key="tenant-1/leased-put.png")
    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    repository.begin_created_upload_artifact_write(workspace_id="tenant-1", artifact=artifact)
    deleted_keys: list[str] = []

    assert (
        drain_pending_upload_file_cleanups(
            files=repository,
            delete_blob=deleted_keys.append,
            logger=logger,
            limit=10,
        )
        == ()
    )
    assert deleted_keys == []
    with sqlite_session_factory.begin() as session:
        marker = session.scalar(select(UploadFileCleanupOutbox))
        assert marker is not None
        assert marker.cleanup_started_at is None
        marker.writer_expires_at = datetime(2000, 1, 1)

    assert (
        drain_pending_upload_file_cleanups(
            files=repository,
            delete_blob=deleted_keys.append,
            logger=logger,
            limit=10,
        )
        == ()
    )
    assert deleted_keys == [artifact.key]
    with sqlite_session_factory() as session:
        marker = session.scalar(select(UploadFileCleanupOutbox))
        assert marker is not None
        assert marker.cleanup_started_at is not None
        assert marker.writer_finished_at is None


def test_writer_heartbeat_renews_dynamically_registered_artifact(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = CreatedUploadArtifact(id=str(uuid4()), key="tenant-1/heartbeat.png")
    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    repository.begin_created_upload_artifact_write(workspace_id="tenant-1", artifact=artifact)
    renewed = Event()
    original_renew = repository.renew_created_upload_artifact_writes

    def renew(*, workspace_id: str, upload_file_ids: tuple[str, ...]) -> None:
        original_renew(workspace_id=workspace_id, upload_file_ids=upload_file_ids)
        renewed.set()

    monkeypatch.setattr(repository, "renew_created_upload_artifact_writes", renew)
    with maintain_created_upload_artifact_writers(
        workspace_id="tenant-1",
        upload_file_ids=lambda: (artifact.id,),
        files=repository,
        logger=logger,
        interval_seconds=0.01,
    ):
        assert renewed.wait(timeout=1)

    with sqlite_session_factory() as session:
        marker = session.scalar(select(UploadFileCleanupOutbox))
        assert marker is not None
        assert marker.writer_expires_at is not None
        assert marker.writer_expires_at > datetime.now(UTC).replace(tzinfo=None)


def test_created_artifact_cleanup_preserves_a_formally_referenced_upload(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = str(uuid4())
    upload = _persist_upload(sqlite_session_factory, tenant_id=tenant_id, key=f"{tenant_id}/published.png")
    with sqlite_session_factory.begin() as session:
        session.add(
            MessageFile(
                message_id=str(uuid4()),
                type=FileType.IMAGE,
                transfer_method=FileTransferMethod.LOCAL_FILE,
                created_by_role=CreatorUserRole.ACCOUNT,
                created_by=str(uuid4()),
                upload_file_id=upload.id,
            )
        )
    deleted_keys: list[str] = []

    cleaned_ids = cleanup_created_upload_artifacts(
        workspace_id=tenant_id,
        artifacts=(CreatedUploadArtifact(upload.id, upload.key),),
        files=SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory),
        delete_blob=deleted_keys.append,
        logger=logger,
    )

    assert cleaned_ids == ()
    assert deleted_keys == []
    with sqlite_session_factory() as session:
        assert session.get(UploadFile, upload.id) is not None
        pending = session.scalar(
            select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == upload.id)
        )
        assert pending is not None
        assert pending.key == upload.key


def test_success_removes_cleanup_record(sqlite_session_factory: sessionmaker[Session]) -> None:
    upload = _persist_upload(sqlite_session_factory, tenant_id="tenant-1", key="tenant-1/success.png")
    deleted_keys: list[str] = []

    def delete_after_commit(key: str) -> None:
        with sqlite_session_factory() as session:
            assert session.get(UploadFile, upload.id) is None
            assert (
                session.scalar(
                    select(UploadFileCleanupOutbox).where(UploadFileCleanupOutbox.upload_file_id == upload.id)
                )
                is not None
            )
        deleted_keys.append(key)

    cleanup_unreferenced_upload_files(
        workspace_id="tenant-1",
        file_ids=(upload.id,),
        files=SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory),
        delete_blob=delete_after_commit,
        logger=logger,
    )

    assert deleted_keys == [upload.key]
    with sqlite_session_factory() as session:
        assert session.scalar(select(UploadFileCleanupOutbox)) is None


def test_missing_blob_completes_cleanup(sqlite_session_factory: sessionmaker[Session]) -> None:
    upload = _persist_upload(sqlite_session_factory, tenant_id="tenant-1", key="tenant-1/missing.png")

    def absent_blob(_key: str) -> None:
        raise FileNotFoundError

    cleanup_unreferenced_upload_files(
        workspace_id="tenant-1",
        file_ids=(upload.id,),
        files=SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory),
        delete_blob=absent_blob,
        logger=logger,
    )

    with sqlite_session_factory() as session:
        assert session.scalar(select(UploadFileCleanupOutbox)) is None


def test_redelivery_is_idempotent_and_cleanup_acknowledgement_is_tenant_scoped(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    upload = _persist_upload(sqlite_session_factory, tenant_id="tenant-1", key="tenant-1/retry.png")
    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    first_claim = repository.claim_unreferenced_files_for_cleanup(
        workspace_id="tenant-1",
        file_ids=(upload.id,),
    )
    second_claim = repository.claim_unreferenced_files_for_cleanup(
        workspace_id="tenant-1",
        file_ids=(upload.id,),
    )

    assert len(first_claim) == 1
    assert second_claim == first_claim
    repository.complete_cleanup(tenant_id="tenant-2", cleanup_id=first_claim[0].cleanup_id)
    assert repository.list_pending_cleanup_artifacts(limit=100) == first_claim

    repository.complete_cleanup(tenant_id="tenant-1", cleanup_id=first_claim[0].cleanup_id)
    deleted_keys: list[str] = []
    failures = drain_pending_upload_file_cleanups(
        files=repository,
        delete_blob=deleted_keys.append,
        logger=logger,
        limit=100,
    )
    assert failures == ()
    assert deleted_keys == []


def test_failed_batch_rotates_so_later_cleanup_is_not_starved(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    pending: list[UploadFileCleanupOutbox] = []
    for index in range(101):
        artifact = UploadFileCleanupOutbox(
            tenant_id="tenant-1",
            upload_file_id=f"10000000-0000-0000-0000-{index:012x}",
            key="deletable" if index == 100 else f"blocked-{index}",
        )
        artifact.id = f"00000000-0000-0000-0000-{index:012x}"
        pending.append(artifact)
    with sqlite_session_factory.begin() as session:
        session.add_all(pending)

    successful_keys: list[str] = []

    def delete_blob(key: str) -> None:
        if key != "deletable":
            raise OSError("storage rejected key")
        successful_keys.append(key)

    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    first_failures = drain_pending_upload_file_cleanups(
        files=repository,
        delete_blob=delete_blob,
        logger=logger,
        limit=100,
    )
    assert len(first_failures) == 100
    assert successful_keys == []

    second_failures = drain_pending_upload_file_cleanups(
        files=repository,
        delete_blob=delete_blob,
        logger=logger,
        limit=100,
    )
    assert len(second_failures) == 99
    assert successful_keys == ["deletable"]
    with sqlite_session_factory() as session:
        assert session.scalar(select(func.count()).select_from(UploadFileCleanupOutbox)) == 100


def test_drainer_discovers_without_locks_then_claims_upload_before_outbox(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    upload = _persist_upload(sqlite_session_factory, tenant_id="tenant-1", key="tenant-1/ordered.png")
    with sqlite_session_factory.begin() as session:
        session.add(
            UploadFileCleanupOutbox(
                tenant_id=upload.tenant_id,
                upload_file_id=upload.id,
                key=upload.key,
            )
        )

    statements: list[tuple[str, str]] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        context: ExecutionContext,
        _executemany: bool,
    ) -> None:
        compiled = context.compiled
        clause = str(compiled.statement) if compiled is not None else statement
        statements.append((" ".join(statement.lower().split()), " ".join(clause.lower().split())))

    with sqlite_session_factory() as session:
        engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", capture_statement)
    try:
        repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
        claimed = repository.list_pending_cleanup_artifacts(limit=1)
    finally:
        event.remove(engine, "before_cursor_execute", capture_statement)

    assert tuple(artifact.id for artifact in claimed) == (upload.id,)
    discovery_index = next(
        index
        for index, (executed, _clause) in enumerate(statements)
        if "from upload_file_cleanup_outbox left outer join upload_files" in executed
    )
    upload_lock_index = next(
        index
        for index, (executed, clause) in enumerate(statements)
        if index > discovery_index
        and "from upload_files" in executed
        and "upload_file_cleanup_outbox" not in executed
        and "for update" in clause
    )
    outbox_lock_index = next(
        index
        for index, (executed, clause) in enumerate(statements)
        if index > upload_lock_index
        and "from upload_file_cleanup_outbox" in executed
        and "left outer join" not in executed
        and "for update" in clause
    )
    assert discovery_index < upload_lock_index < outbox_lock_index
    assert "for update" not in statements[discovery_index][1]


def test_drainer_preserves_cleanup_key_when_upload_file_row_is_already_missing(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    cleanup = UploadFileCleanupOutbox(
        tenant_id="tenant-1",
        upload_file_id=str(uuid4()),
        key="tenant-1/already-removed.png",
    )
    with sqlite_session_factory.begin() as session:
        session.add(cleanup)

    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    claimed = repository.list_pending_cleanup_artifacts(limit=1)

    assert len(claimed) == 1
    assert claimed[0].id == cleanup.upload_file_id
    assert claimed[0].cleanup_id == cleanup.id
    assert claimed[0].key == cleanup.key
    deleted_keys: list[str] = []
    assert (
        cleanup_claimed_upload_files(
            artifacts=claimed,
            files=repository,
            delete_blob=deleted_keys.append,
            logger=logger,
        )
        == ()
    )
    assert deleted_keys == [cleanup.key]
    with sqlite_session_factory() as session:
        assert session.get(UploadFileCleanupOutbox, cleanup.id) is None


def test_replay_and_blob_delete_recheck_preserve_a_file_referenced_after_claim(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = str(uuid4())
    upload = _persist_upload(sqlite_session_factory, tenant_id=tenant_id, key=f"{tenant_id}/shared.png")
    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    claimed = repository.claim_unreferenced_files_for_cleanup(
        workspace_id=tenant_id,
        file_ids=(upload.id,),
    )
    assert len(claimed) == 1

    with sqlite_session_factory.begin() as session:
        session.add(
            MessageFile(
                message_id=str(uuid4()),
                type=FileType.IMAGE,
                transfer_method=FileTransferMethod.LOCAL_FILE,
                created_by_role=CreatorUserRole.ACCOUNT,
                created_by=str(uuid4()),
                upload_file_id=upload.id,
            )
        )

    assert repository.list_pending_cleanup_artifacts(limit=100) == ()
    deleted_keys: list[str] = []
    assert (
        cleanup_claimed_upload_files(
            artifacts=claimed,
            files=repository,
            delete_blob=deleted_keys.append,
            logger=logger,
        )
        == ()
    )
    assert deleted_keys == []
    with sqlite_session_factory() as session:
        assert session.get(UploadFileCleanupOutbox, claimed[0].cleanup_id) is not None

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import tasks.upload_file_cleanup_task as task_module
from extensions.storage.storage_type import StorageType
from models.enums import CreatorUserRole
from models.model import UploadFile, UploadFileCleanupOutbox
from repositories.knowledge.upload_file_repository import SQLAlchemyKnowledgeUploadRepository


def test_periodic_drain_retries_persistent_cleanup_record(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = UploadFile(
        tenant_id="tenant-1",
        storage_type=StorageType.LOCAL,
        key="tenant-1/pending.png",
        name="pending.png",
        size=7,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="account-1",
        created_at=datetime.now(UTC),
        used=False,
    )
    with sqlite_session_factory.begin() as session:
        session.add(upload)
    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    claimed = repository.claim_unreferenced_files_for_cleanup(
        workspace_id=upload.tenant_id,
        file_ids=(upload.id,),
    )
    assert len(claimed) == 1

    failed_keys: list[str] = []

    def fail_delete(key: str) -> None:
        failed_keys.append(key)
        raise OSError("storage unavailable")

    monkeypatch.setattr(task_module.storage, "delete", fail_delete)
    with pytest.raises(RuntimeError, match="upload blob cleanup batch did not complete"):
        task_module.drain_upload_file_cleanup_outbox_task.run()

    assert failed_keys == [upload.key]
    with sqlite_session_factory() as session:
        assert session.scalar(select(UploadFileCleanupOutbox)) is not None

    retried_keys: list[str] = []
    monkeypatch.setattr(task_module.storage, "delete", retried_keys.append)
    task_module.drain_upload_file_cleanup_outbox_task.run()

    assert retried_keys == [upload.key]
    with sqlite_session_factory() as session:
        assert session.scalar(select(UploadFileCleanupOutbox)) is None

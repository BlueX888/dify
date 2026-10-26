from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Event, local
from uuid import uuid4

from sqlalchemy import delete, event, select
from sqlalchemy.orm import Session, sessionmaker

from extensions.storage.storage_type import StorageType
from graphon.file import FileTransferMethod, FileType
from models.enums import CreatorUserRole, MessageFileBelongsTo
from models.model import MessageFile, UploadFile, UploadFileCleanupOutbox
from repositories.upload_file_repository import (
    SQLAlchemyUploadFileCleanupRepository,
    UploadFileCleanupArtifact,
    lock_and_validate_owned_upload_files_in_session,
)


class _WorkerState(local):
    def __init__(self) -> None:
        self.role: str | None = None


def test_message_owner_committed_after_drainer_snapshot_keeps_upload_file(mysql_session: Session) -> None:
    engine = mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ")
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id = str(uuid4())
    actor_id = str(uuid4())
    message_id = str(uuid4())
    upload = UploadFile(
        tenant_id=tenant_id,
        storage_type=StorageType.LOCAL,
        key=f"message-cleanup-race/{uuid4()}.txt",
        name="message.txt",
        size=1,
        extension="txt",
        mime_type="text/plain",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=actor_id,
        created_at=datetime.now(UTC),
        used=True,
    )
    with sessions.begin() as seed:
        seed.add(upload)
        seed.flush()
        seed.add(
            UploadFileCleanupOutbox(
                tenant_id=tenant_id,
                upload_file_id=upload.id,
                key=upload.key,
            )
        )
    upload_id = upload.id

    repository = SQLAlchemyUploadFileCleanupRepository(session_factory=sessions)
    producer_locked = Event()
    cleanup_snapshot_established = Event()
    cleanup_upload_lock_attempted = Event()
    allow_producer_commit = Event()
    worker = _WorkerState()

    def normalized(statement: str) -> str:
        return " ".join(statement.lower().split())

    def is_pending_read(statement: str) -> bool:
        sql = normalized(statement)
        return sql.startswith("select") and " from upload_file_cleanup_outbox " in sql

    def is_upload_exclusive_lock(statement: str) -> bool:
        sql = normalized(statement)
        return sql.startswith("select") and " from upload_files " in sql and " for update" in sql

    def observe_cleanup_attempt(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if worker.role != "cleanup":
            return
        if is_upload_exclusive_lock(statement):
            assert cleanup_snapshot_established.is_set()
            cleanup_upload_lock_attempted.set()

    def observe_cleanup_result(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if worker.role == "cleanup" and is_pending_read(statement):
            sql = normalized(statement)
            assert " for update" not in sql
            assert " lock in share mode" not in sql
            cleanup_snapshot_established.set()

    def create_message_owner() -> None:
        worker.role = "producer"
        with sessions.begin() as producer:
            locked = lock_and_validate_owned_upload_files_in_session(
                producer,
                tenant_id=tenant_id,
                upload_file_ids=(upload_id,),
            )
            assert tuple(locked) == (upload_id,)
            producer_locked.set()
            assert allow_producer_commit.wait(timeout=10)
            producer.add(
                MessageFile(
                    message_id=message_id,
                    type=FileType.DOCUMENT,
                    transfer_method=FileTransferMethod.LOCAL_FILE,
                    created_by_role=CreatorUserRole.ACCOUNT,
                    created_by=actor_id,
                    belongs_to=MessageFileBelongsTo.USER,
                    upload_file_id=locked[upload_id].id,
                )
            )

    def drain_pending_cleanup() -> tuple[UploadFileCleanupArtifact, ...]:
        worker.role = "cleanup"
        return repository.list_pending_cleanup_artifacts(limit=1)

    event.listen(engine, "before_cursor_execute", observe_cleanup_attempt)
    event.listen(engine, "after_cursor_execute", observe_cleanup_result)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            producer = executor.submit(create_message_owner)
            try:
                assert producer_locked.wait(timeout=10)
                drainer = executor.submit(drain_pending_cleanup)
                assert cleanup_snapshot_established.wait(timeout=10)
                assert cleanup_upload_lock_attempted.wait(timeout=10)
                assert not drainer.done()
                allow_producer_commit.set()
                producer.result(timeout=10)
                assert drainer.result(timeout=10) == ()
            finally:
                allow_producer_commit.set()

        with sessions() as verify:
            assert verify.get(UploadFile, upload_id) is not None
            assert (
                verify.scalar(
                    select(MessageFile.id).where(
                        MessageFile.message_id == message_id,
                        MessageFile.upload_file_id == upload_id,
                    )
                )
                is not None
            )
            assert (
                verify.scalar(
                    select(UploadFileCleanupOutbox.id).where(
                        UploadFileCleanupOutbox.tenant_id == tenant_id,
                        UploadFileCleanupOutbox.upload_file_id == upload_id,
                    )
                )
                is not None
            )
    finally:
        allow_producer_commit.set()
        event.remove(engine, "before_cursor_execute", observe_cleanup_attempt)
        event.remove(engine, "after_cursor_execute", observe_cleanup_result)
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(MessageFile).where(
                    MessageFile.message_id == message_id,
                    MessageFile.upload_file_id == upload_id,
                )
            )
            cleanup.execute(
                delete(UploadFileCleanupOutbox).where(
                    UploadFileCleanupOutbox.tenant_id == tenant_id,
                    UploadFileCleanupOutbox.upload_file_id == upload_id,
                )
            )
            cleanup.execute(delete(UploadFile).where(UploadFile.tenant_id == tenant_id, UploadFile.id == upload_id))

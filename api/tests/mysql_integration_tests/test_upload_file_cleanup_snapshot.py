from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event, Thread, local
from uuid import uuid4

import pytest
from sqlalchemy import delete, event, select
from sqlalchemy.orm import Session, sessionmaker

import tasks.delete_conversation_task as delete_conversation_task_module
from core.app.entities.app_invoke_entities import InvokeFrom, UserFrom
from core.app.file_access import FileAccessScope, bind_file_access_scope, get_current_file_access_scope
from core.rag.datasource.retrieval_service import RetrievalService
from core.rag.embedding.retrieval import FinalizedInternalRetrieval, RetrievalSegments
from core.rag.models.document import AttachmentDocument
from core.rag.models.document import Document as IndexDocument
from core.workflow.nodes.human_input.enums import HumanInputFormKind, HumanInputFormStatus
from extensions.storage.storage_type import StorageType
from models.dataset import Dataset, Document, DocumentSegment, SegmentAttachmentBinding
from models.enums import (
    ConversationFromSource,
    CreatorUserRole,
    DataSourceType,
    DocumentCreatedFrom,
    SegmentStatus,
)
from models.human_input import (
    HumanInputForm,
    HumanInputFormRecipient,
    HumanInputFormUploadFile,
    HumanInputFormUploadToken,
    RecipientType,
)
from models.model import App, AppMode, Conversation, IconType, UploadFile, UploadFileCleanupOutbox
from repositories.human_input_file_upload_repository import SQLAlchemyHumanInputFileUploadRepository
from repositories.knowledge.segment_cleanup_repository import (
    add_segment_attachment_bindings,
    delete_document_segments_and_claim_uploads,
    detach_owned_segment_attachments,
    replace_segment_attachment_bindings,
    require_locked_segment_attachment_owner,
)
from repositories.knowledge.segment_repository import SQLAlchemySegmentRepository
from repositories.knowledge.upload_file_repository import claim_unreferenced_files_for_cleanup_in_session
from repositories.upload_file_repository import SQLAlchemyUploadFileCleanupRepository, UploadFileCleanupArtifact
from services.human_input_file_upload_service import InvalidUploadTokenError
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.estimate import StoredSource
from services.knowledge.indexing.execution import IndexingDocument
from services.knowledge.retrieval import finalization as retrieval_finalization
from services.knowledge.retrieval.finalization import KnowledgeRetrievalFinalizer


def _repeatable_read_sessions(mysql_session: Session) -> sessionmaker[Session]:
    engine = mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ")
    return sessionmaker(bind=engine, expire_on_commit=False)


def _upload(*, tenant_id: str, key: str, created_by: str) -> UploadFile:
    return UploadFile(
        tenant_id=tenant_id,
        storage_type=StorageType.LOCAL,
        key=key,
        name="image.png",
        size=1,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=created_by,
        created_at=datetime.now(UTC),
        used=True,
    )


class _WorkerState(local):
    def __init__(self) -> None:
        self.role: str | None = None


class _SessionFactoryBoundary:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def create_session(self) -> Session:
        return self._sessions()


@dataclass(frozen=True, slots=True)
class _HumanInputUploadFixture:
    tenant_id: str
    app_id: str
    conversation_id: str
    form_id: str
    recipient_id: str
    upload_token_id: str
    upload_id: str


def _seed_human_input_upload_publication(sessions: sessionmaker[Session]) -> _HumanInputUploadFixture:
    tenant_id = str(uuid4())
    app_id = str(uuid4())
    conversation_id = str(uuid4())
    form_id = str(uuid4())
    recipient_id = str(uuid4())
    upload_token_id = str(uuid4())
    created_by = str(uuid4())
    upload = _upload(
        tenant_id=tenant_id,
        key=f"human-input-owner/{uuid4()}.png",
        created_by=created_by,
    )
    with sessions.begin() as seed:
        app = App(
            id=app_id,
            tenant_id=tenant_id,
            name="HITL owner-chain app",
            description="",
            mode=AppMode.WORKFLOW,
            icon_type=IconType.EMOJI,
            icon="app",
            icon_background="#FFFFFF",
            enable_site=True,
            enable_api=True,
            created_by=created_by,
            updated_by=created_by,
        )
        conversation = Conversation(
            id=conversation_id,
            app_id=app_id,
            mode=AppMode.WORKFLOW,
            name="deleted HITL conversation",
            inputs={},
            from_source=ConversationFromSource.CONSOLE,
            from_account_id=created_by,
            is_deleted=True,
        )
        form = HumanInputForm(
            id=form_id,
            tenant_id=tenant_id,
            app_id=app_id,
            workflow_run_id=None,
            conversation_id=conversation_id,
            form_kind=HumanInputFormKind.RUNTIME,
            node_id="human-input",
            form_definition="{}",
            rendered_content="content",
            status=HumanInputFormStatus.WAITING,
            expiration_time=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
        )
        recipient = HumanInputFormRecipient(
            id=recipient_id,
            form_id=form_id,
            delivery_id=str(uuid4()),
            recipient_type=RecipientType.STANDALONE_WEB_APP,
            recipient_payload="{}",
            access_token=f"form-{uuid4()}",
        )
        upload_token = HumanInputFormUploadToken(
            id=upload_token_id,
            tenant_id=tenant_id,
            app_id=app_id,
            form_id=form_id,
            recipient_id=recipient_id,
            token=f"hitl-upload-{uuid4()}",
        )
        marker = UploadFileCleanupOutbox(
            tenant_id=tenant_id,
            upload_file_id=upload.id,
            key=upload.key,
            writer_id=str(uuid4()),
            writer_finished_at=datetime.now(UTC).replace(tzinfo=None),
        )
        seed.add_all([app, conversation, form, recipient, upload_token, upload, marker])
    return _HumanInputUploadFixture(
        tenant_id=tenant_id,
        app_id=app_id,
        conversation_id=conversation_id,
        form_id=form_id,
        recipient_id=recipient_id,
        upload_token_id=upload_token_id,
        upload_id=upload.id,
    )


def _delete_human_input_upload_fixture(sessions: sessionmaker[Session], fixture: _HumanInputUploadFixture) -> None:
    with sessions.begin() as cleanup:
        cleanup.execute(
            delete(HumanInputFormUploadFile).where(HumanInputFormUploadFile.upload_file_id == fixture.upload_id)
        )
        cleanup.execute(
            delete(HumanInputFormUploadToken).where(HumanInputFormUploadToken.id == fixture.upload_token_id)
        )
        cleanup.execute(delete(HumanInputFormRecipient).where(HumanInputFormRecipient.id == fixture.recipient_id))
        cleanup.execute(delete(HumanInputForm).where(HumanInputForm.id == fixture.form_id))
        cleanup.execute(delete(Conversation).where(Conversation.id == fixture.conversation_id))
        cleanup.execute(
            delete(UploadFileCleanupOutbox).where(
                UploadFileCleanupOutbox.tenant_id == fixture.tenant_id,
                UploadFileCleanupOutbox.upload_file_id == fixture.upload_id,
            )
        )
        cleanup.execute(delete(UploadFile).where(UploadFile.id == fixture.upload_id))
        cleanup.execute(delete(App).where(App.id == fixture.app_id))


def _seed_attachment_replacement(
    sessions: sessionmaker[Session],
) -> tuple[str, str, str, str, str, str]:
    tenant_id = str(uuid4())
    created_by = str(uuid4())
    dataset = Dataset(tenant_id=tenant_id, name=f"cleanup-race-{uuid4()}", created_by=created_by)
    with sessions.begin() as seed:
        seed.add(dataset)
        seed.flush()
        document = Document(
            tenant_id=tenant_id,
            dataset_id=dataset.id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch=f"batch-{uuid4()}",
            name="race.txt",
            created_from=DocumentCreatedFrom.API,
            created_by=created_by,
        )
        segment = DocumentSegment(
            tenant_id=tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=1,
            content="attachment race",
            word_count=2,
            tokens=2,
            created_by=created_by,
        )
        segment.index_node_id = segment.id
        segment.index_node_hash = f"hash-{segment.id}"
        first = _upload(tenant_id=tenant_id, key=f"cleanup-race/{uuid4()}-a.png", created_by=created_by)
        replacement = _upload(tenant_id=tenant_id, key=f"cleanup-race/{uuid4()}-b.png", created_by=created_by)
        seed.add_all([document, segment, first, replacement])
        seed.flush()
        seed.add(
            SegmentAttachmentBinding(
                tenant_id=tenant_id,
                dataset_id=dataset.id,
                document_id=document.id,
                segment_id=segment.id,
                attachment_id=first.id,
            )
        )
    return tenant_id, dataset.id, document.id, segment.id, first.id, replacement.id


def _delete_attachment_replacement_rows(
    sessions: sessionmaker[Session],
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    segment_id: str,
    file_ids: tuple[str, ...],
) -> None:
    with sessions.begin() as cleanup:
        cleanup.execute(
            delete(SegmentAttachmentBinding).where(
                SegmentAttachmentBinding.tenant_id == tenant_id,
                SegmentAttachmentBinding.dataset_id == dataset_id,
                SegmentAttachmentBinding.document_id == document_id,
                SegmentAttachmentBinding.segment_id == segment_id,
            )
        )
        cleanup.execute(
            delete(UploadFileCleanupOutbox).where(
                UploadFileCleanupOutbox.tenant_id == tenant_id,
                UploadFileCleanupOutbox.upload_file_id.in_(file_ids),
            )
        )
        cleanup.execute(
            delete(DocumentSegment).where(
                DocumentSegment.tenant_id == tenant_id,
                DocumentSegment.dataset_id == dataset_id,
                DocumentSegment.document_id == document_id,
                DocumentSegment.id == segment_id,
            )
        )
        cleanup.execute(
            delete(Document).where(
                Document.tenant_id == tenant_id,
                Document.dataset_id == dataset_id,
                Document.id == document_id,
            )
        )
        cleanup.execute(delete(UploadFile).where(UploadFile.tenant_id == tenant_id, UploadFile.id.in_(file_ids)))
        cleanup.execute(delete(Dataset).where(Dataset.tenant_id == tenant_id, Dataset.id == dataset_id))


def _seed_attachment_cleanup_lock_race(
    sessions: sessionmaker[Session],
) -> tuple[str, str, str, str, str, str]:
    tenant_id, dataset_id, document_id, segment_id, first_id, replacement_id = _seed_attachment_replacement(sessions)
    with sessions.begin() as seed:
        replacement = seed.get_one(UploadFile, replacement_id)
        segment = seed.get_one(DocumentSegment, segment_id)
        segment.content = f"replacement ![image](/files/{replacement_id}/image-preview)"
        cleanup_row = UploadFileCleanupOutbox(
            tenant_id=tenant_id,
            upload_file_id=replacement_id,
            key=replacement.key,
        )
        seed.add(cleanup_row)
        seed.flush()
        # Keep this row ahead of unrelated retries after a failed CI test.
        cleanup_row.created_at = datetime(2000, 1, 1)
    return tenant_id, dataset_id, document_id, segment_id, first_id, replacement_id


def test_initial_claim_and_drainer_share_upload_then_outbox_lock_order(mysql_session: Session) -> None:
    """The drainer's candidate scan must not recreate the Outbox-to-Upload deadlock cycle."""

    engine = mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ")
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id = str(uuid4())
    actor_id = str(uuid4())
    upload = _upload(
        tenant_id=tenant_id,
        key=f"cleanup-lock-order/{uuid4()}.txt",
        created_by=actor_id,
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
    worker = _WorkerState()
    claimant_upload_locked = Event()
    claimant_outbox_attempted = Event()
    allow_claimant_outbox = Event()
    drainer_candidate_read = Event()
    drainer_upload_attempted = Event()
    drainer_finished = Event()
    claimant_results: list[tuple[UploadFileCleanupArtifact, ...]] = []
    drainer_results: list[tuple[UploadFileCleanupArtifact, ...]] = []
    errors: list[BaseException] = []

    def normalized(statement: str) -> str:
        return " ".join(statement.lower().replace("`", "").split())

    def is_upload_lock(statement: str) -> bool:
        sql = normalized(statement)
        return sql.startswith("select") and " from upload_files " in sql and " for update" in sql

    def is_outbox_read(statement: str) -> bool:
        sql = normalized(statement)
        return sql.startswith("select") and " from upload_file_cleanup_outbox " in sql

    def before_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        sql = normalized(statement)
        if worker.role == "claimant" and is_outbox_read(statement) and " for update" in sql:
            assert claimant_upload_locked.is_set()
            claimant_outbox_attempted.set()
            assert allow_claimant_outbox.wait(timeout=10)
        elif worker.role == "drainer" and is_upload_lock(statement):
            assert drainer_candidate_read.is_set()
            drainer_upload_attempted.set()

    def after_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        sql = normalized(statement)
        if worker.role == "claimant" and is_upload_lock(statement):
            claimant_upload_locked.set()
        elif worker.role == "drainer" and is_outbox_read(statement) and not drainer_candidate_read.is_set():
            assert " for update" not in sql
            assert " lock in share mode" not in sql
            drainer_candidate_read.set()

    def claim_directly() -> None:
        try:
            worker.role = "claimant"
            claimant_results.append(
                repository.claim_unreferenced_files_for_cleanup(
                    workspace_id=tenant_id,
                    file_ids=(upload_id,),
                )
            )
        except BaseException as error:
            errors.append(error)

    def drain_pending() -> None:
        try:
            worker.role = "drainer"
            drainer_results.append(repository.list_pending_cleanup_artifacts(limit=1))
        except BaseException as error:
            errors.append(error)
        finally:
            drainer_finished.set()

    claimant = Thread(target=claim_directly)
    drainer = Thread(target=drain_pending)
    event.listen(engine, "before_cursor_execute", before_sql)
    event.listen(engine, "after_cursor_execute", after_sql)
    try:
        claimant.start()
        assert claimant_outbox_attempted.wait(timeout=10)
        drainer.start()
        assert drainer_candidate_read.wait(timeout=10)
        assert drainer_upload_attempted.wait(timeout=10)
        # No Python mutex blocks the drainer: it is queued on the claimant's
        # real InnoDB UploadFile X lock while the outbox remains unlocked.
        assert not drainer_finished.wait(timeout=0.25)
        allow_claimant_outbox.set()
        claimant.join(timeout=10)
        drainer.join(timeout=10)

        assert not claimant.is_alive()
        assert not drainer.is_alive()
        assert errors == []
        assert len(claimant_results) == 1
        assert len(claimant_results[0]) == 1
        assert len(drainer_results) == 1
        cleanup_id = claimant_results[0][0].cleanup_id
        assert len(drainer_results[0]) == 1
        drained = drainer_results[0][0]
        assert (drained.id, drained.cleanup_id, drained.tenant_id, drained.key) == (
            upload_id,
            cleanup_id,
            tenant_id,
            upload.key,
        )
        with sessions() as verify:
            assert verify.get(UploadFile, upload_id) is None
            current = verify.scalar(
                select(UploadFileCleanupOutbox).where(
                    UploadFileCleanupOutbox.tenant_id == tenant_id,
                    UploadFileCleanupOutbox.upload_file_id == upload_id,
                )
            )
            assert current is not None
            assert current.id == cleanup_id
            assert current.cleanup_started_at is not None
    finally:
        allow_claimant_outbox.set()
        if claimant.ident is not None:
            claimant.join(timeout=10)
        if drainer.ident is not None:
            drainer.join(timeout=10)
        event.remove(engine, "before_cursor_execute", before_sql)
        event.remove(engine, "after_cursor_execute", after_sql)
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(UploadFileCleanupOutbox).where(
                    UploadFileCleanupOutbox.tenant_id == tenant_id,
                    UploadFileCleanupOutbox.upload_file_id == upload_id,
                )
            )
            cleanup.execute(delete(UploadFile).where(UploadFile.tenant_id == tenant_id, UploadFile.id == upload_id))


@pytest.mark.parametrize("cleanup_entrypoint", ["direct", "drainer"])
def test_attachment_owner_nowait_defers_cleanup_without_upload_segment_deadlock(
    mysql_session: Session,
    cleanup_entrypoint: str,
) -> None:
    """An owner holding Segment X must make Upload-first cleanup defer without deleting the file."""

    engine = mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ")
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id, dataset_id, document_id, segment_id, first_id, replacement_id = _seed_attachment_cleanup_lock_race(
        sessions
    )
    repository = SQLAlchemyUploadFileCleanupRepository(session_factory=sessions)
    worker = _WorkerState()
    cleanup_upload_locked = Event()
    cleanup_savepoint_started = Event()
    cleanup_owner_attempted = Event()
    cleanup_savepoint_rolled_back = Event()
    allow_cleanup_owner_read = Event()
    writer_segment_locked = Event()
    writer_upload_attempted = Event()
    writer_finished = Event()
    cleanup_results: list[tuple[UploadFileCleanupArtifact, ...]] = []
    errors: list[BaseException] = []

    def normalized(statement: str) -> str:
        return " ".join(statement.lower().replace("`", "").split())

    def before_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        sql = normalized(statement)
        if worker.role == "cleanup" and sql.startswith("savepoint "):
            cleanup_savepoint_started.set()
        elif worker.role == "cleanup" and sql.startswith("rollback to savepoint "):
            cleanup_savepoint_rolled_back.set()
        elif worker.role == "cleanup" and " from document_segments " in sql and "nowait" in sql:
            assert cleanup_upload_locked.is_set()
            assert cleanup_savepoint_started.is_set()
            # Use an exclusive NOWAIT read: MySQL 8 does not accept the legacy
            # LOCK IN SHARE MODE NOWAIT rendering used by read=True.
            assert " for update nowait" in sql
            cleanup_owner_attempted.set()
            assert allow_cleanup_owner_read.wait(timeout=10)
        elif (
            worker.role == "writer"
            and sql.startswith("select")
            and " from upload_files " in sql
            and " for update" in sql
        ):
            writer_upload_attempted.set()

    def after_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        sql = normalized(statement)
        if (
            worker.role == "cleanup"
            and sql.startswith("select")
            and " from upload_files " in sql
            and " for update" in sql
        ):
            cleanup_upload_locked.set()

    def drain_once() -> None:
        try:
            worker.role = "cleanup"
            if cleanup_entrypoint == "direct":
                with sessions.begin() as cleanup:
                    cleanup_results.append(
                        claim_unreferenced_files_for_cleanup_in_session(
                            cleanup,
                            workspace_id=tenant_id,
                            file_ids=(replacement_id,),
                        )
                    )
            else:
                cleanup_results.append(repository.list_pending_cleanup_artifacts(limit=1))
        except BaseException as error:
            errors.append(error)

    def publish_owner() -> None:
        try:
            worker.role = "writer"
            with sessions.begin() as writer:
                # Existing indexing callers can already hold the owner chain
                # when attachment publication reaches the file-domain guard.
                require_locked_segment_attachment_owner(
                    writer,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    segment_id=segment_id,
                )
                writer_segment_locked.set()
                add_segment_attachment_bindings(
                    writer,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    segment_id=segment_id,
                    attachment_ids=(replacement_id,),
                )
        except BaseException as error:
            errors.append(error)
        finally:
            writer_finished.set()

    cleanup_thread = Thread(target=drain_once)
    writer_thread = Thread(target=publish_owner)
    event.listen(engine, "before_cursor_execute", before_sql)
    event.listen(engine, "after_cursor_execute", after_sql)
    try:
        cleanup_thread.start()
        assert cleanup_owner_attempted.wait(timeout=10)
        writer_thread.start()
        assert writer_segment_locked.wait(timeout=10)
        assert writer_upload_attempted.wait(timeout=10)
        # The writer is waiting on the real InnoDB UploadFile X lock while the
        # cleanup owner read is paused. NOWAIT must break this cycle promptly.
        assert not writer_finished.wait(timeout=0.25)
        allow_cleanup_owner_read.set()
        cleanup_thread.join(timeout=10)
        writer_thread.join(timeout=10)

        assert not cleanup_thread.is_alive()
        assert not writer_thread.is_alive()
        assert errors == []
        assert cleanup_savepoint_rolled_back.is_set()
        assert cleanup_results == [()]

        # A later real drainer pass observes the committed binding and keeps
        # both the file and its durable retry evidence.
        assert repository.list_pending_cleanup_artifacts(limit=1) == ()
        with sessions() as verify:
            assert verify.get(UploadFile, replacement_id) is not None
            assert set(
                verify.scalars(
                    select(SegmentAttachmentBinding.attachment_id).where(
                        SegmentAttachmentBinding.tenant_id == tenant_id,
                        SegmentAttachmentBinding.dataset_id == dataset_id,
                        SegmentAttachmentBinding.document_id == document_id,
                        SegmentAttachmentBinding.segment_id == segment_id,
                    )
                )
            ) == {first_id, replacement_id}
            retained = verify.scalar(
                select(UploadFileCleanupOutbox).where(
                    UploadFileCleanupOutbox.tenant_id == tenant_id,
                    UploadFileCleanupOutbox.upload_file_id == replacement_id,
                )
            )
            assert retained is not None
            assert retained.cleanup_started_at is None
            assert retained.last_attempted_at is not None
    finally:
        allow_cleanup_owner_read.set()
        if cleanup_thread.ident is not None:
            cleanup_thread.join(timeout=10)
        if writer_thread.ident is not None:
            writer_thread.join(timeout=10)
        event.remove(engine, "before_cursor_execute", before_sql)
        event.remove(engine, "after_cursor_execute", after_sql)
        _delete_attachment_replacement_rows(
            sessions,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            file_ids=(first_id, replacement_id),
        )


def test_finalizer_current_read_drops_attachment_detached_after_repeatable_read_snapshot(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, document_id, segment_id, attachment_id, replacement_id = _seed_attachment_replacement(
        sessions
    )
    old_signed_link = f"/files/{attachment_id}/image-preview?sign=old"
    with sessions.begin() as session:
        document = session.get_one(Document, document_id)
        document.indexing_status = "completed"
        document.enabled = True
        document.archived = False
        segment = session.get_one(DocumentSegment, segment_id)
        segment.content = f"attachment ![image]({old_signed_link})"
        segment.status = SegmentStatus.COMPLETED
        segment.enabled = True

    candidate = IndexDocument(
        page_content="attachment candidate",
        provider="dify",
        metadata={
            "doc_id": attachment_id,
            "doc_type": "image",
            "document_id": document_id,
            "dataset_id": dataset_id,
            "score": 0.9,
        },
    )
    format_documents = RetrievalService.format_retrieval_documents
    detached = Event()

    def format_then_detach(session: Session, candidates: list[IndexDocument]) -> list[RetrievalSegments]:
        records = format_documents(session, candidates)
        assert len(records) == 1
        assert records[0].files is not None
        assert [file["id"] for file in records[0].files] == [attachment_id]
        # Keep the formatter transaction and its REPEATABLE READ snapshot open.
        # A second real Session commits the production detach path before the
        # finalizer rechecks ownership in the original Session.
        with sessions.begin() as writer:
            removed = detach_owned_segment_attachments(
                writer,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_ids=(document_id,),
                segment_ids=(segment_id,),
            )
            assert removed == (attachment_id,)
        detached.set()
        return records

    monkeypatch.setattr(
        RetrievalService,
        "format_retrieval_documents",
        staticmethod(format_then_detach),
    )
    signed_file_ids: list[str] = []

    def sign_attachment(upload_file_id: str, _extension: str) -> str:
        signed_file_ids.append(upload_file_id)
        return f"signed:{upload_file_id}"

    monkeypatch.setattr(retrieval_finalization, "sign_upload_file_preview_url", sign_attachment)
    scope = FileAccessScope(
        tenant_id=tenant_id,
        user_id=str(uuid4()),
        user_from=UserFrom.END_USER,
        invoke_from=InvokeFrom.WEB_APP,
    )

    try:
        with bind_file_access_scope(scope), sessions() as finalizer_session:
            result = KnowledgeRetrievalFinalizer().finalize(
                format_session=finalizer_session,
                authorization_session=finalizer_session,
                candidates=[candidate],
                tenant_id=tenant_id,
                dataset_ids=[dataset_id],
                top_k=1,
            )
            current_scope = get_current_file_access_scope()

        assert detached.is_set()
        assert len(result.items) == 1
        finalized = result.items[0]
        assert isinstance(finalized, FinalizedInternalRetrieval)
        assert finalized.attachments == ()
        assert finalized.files == ()
        assert finalized.record.files is None
        assert signed_file_ids == []
        assert current_scope is not None
        assert current_scope.granted_upload_file_ids == frozenset()
        assert current_scope.granted_retriever_segment_ids == frozenset({segment_id})
        assert old_signed_link not in finalized.content
        assert "sign=" not in finalized.content
        assert "timestamp=" not in finalized.content
        with sessions() as verifier:
            assert verifier.get(UploadFile, attachment_id) is not None
            assert (
                verifier.scalar(
                    select(SegmentAttachmentBinding.id).where(
                        SegmentAttachmentBinding.tenant_id == tenant_id,
                        SegmentAttachmentBinding.dataset_id == dataset_id,
                        SegmentAttachmentBinding.document_id == document_id,
                        SegmentAttachmentBinding.segment_id == segment_id,
                        SegmentAttachmentBinding.attachment_id == attachment_id,
                    )
                )
                is None
            )
    finally:
        _delete_attachment_replacement_rows(
            sessions,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            file_ids=(attachment_id, replacement_id),
        )


def test_cleanup_current_read_keeps_file_bound_after_snapshot(mysql_session: Session) -> None:
    sessions = _repeatable_read_sessions(mysql_session)
    tenant_id = str(uuid4())
    dataset_id = str(uuid4())
    document_id = str(uuid4())
    segment_id = str(uuid4())
    upload = UploadFile(
        tenant_id=tenant_id,
        storage_type=StorageType.LOCAL,
        key=f"cleanup-snapshot/{uuid4()}.png",
        name="image.png",
        size=1,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=str(uuid4()),
        created_at=datetime.now(UTC),
        used=True,
    )
    with sessions.begin() as seed:
        seed.add(upload)
    upload_id = upload.id

    try:
        with sessions() as cleanup:
            assert (
                list(
                    cleanup.scalars(
                        select(SegmentAttachmentBinding.id).where(
                            SegmentAttachmentBinding.tenant_id == tenant_id,
                            SegmentAttachmentBinding.attachment_id == upload_id,
                        )
                    )
                )
                == []
            )

            with sessions.begin() as writer:
                locked_upload_id = writer.scalar(
                    select(UploadFile.id)
                    .where(UploadFile.tenant_id == tenant_id, UploadFile.id == upload_id)
                    .with_for_update()
                )
                assert locked_upload_id == upload_id
                writer.add(
                    SegmentAttachmentBinding(
                        tenant_id=tenant_id,
                        dataset_id=dataset_id,
                        document_id=document_id,
                        segment_id=segment_id,
                        attachment_id=upload_id,
                    )
                )

            artifacts = claim_unreferenced_files_for_cleanup_in_session(
                cleanup,
                workspace_id=tenant_id,
                file_ids=(upload_id,),
            )
            cleanup.commit()

        assert artifacts == ()
        with sessions() as verify:
            assert verify.get(UploadFile, upload_id) is not None
            assert (
                verify.scalar(
                    select(SegmentAttachmentBinding.id).where(
                        SegmentAttachmentBinding.tenant_id == tenant_id,
                        SegmentAttachmentBinding.attachment_id == upload_id,
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
                is None
            )
    finally:
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(SegmentAttachmentBinding).where(
                    SegmentAttachmentBinding.tenant_id == tenant_id,
                    SegmentAttachmentBinding.attachment_id == upload_id,
                )
            )
            cleanup.execute(
                delete(UploadFileCleanupOutbox).where(
                    UploadFileCleanupOutbox.tenant_id == tenant_id,
                    UploadFileCleanupOutbox.upload_file_id == upload_id,
                )
            )
            cleanup.execute(delete(UploadFile).where(UploadFile.tenant_id == tenant_id, UploadFile.id == upload_id))


def test_cleanup_current_read_keeps_file_added_to_human_input_after_snapshot(mysql_session: Session) -> None:
    """The Human Input owner query must escape an older MySQL REPEATABLE READ snapshot."""

    sessions = _repeatable_read_sessions(mysql_session)
    fixture = _seed_human_input_upload_publication(sessions)

    try:
        with sessions() as cleanup:
            # Establish the old snapshot using the exact owner table. The
            # production writer then commits through a separate real Session.
            assert (
                cleanup.scalar(
                    select(HumanInputFormUploadFile.id).where(
                        HumanInputFormUploadFile.tenant_id == fixture.tenant_id,
                        HumanInputFormUploadFile.upload_file_id == fixture.upload_id,
                    )
                )
                is None
            )
            SQLAlchemyHumanInputFileUploadRepository(session_factory=sessions).add_file(
                tenant_id=fixture.tenant_id,
                app_id=fixture.app_id,
                form_id=fixture.form_id,
                upload_token_id=fixture.upload_token_id,
                file_id=fixture.upload_id,
            )

            artifacts = claim_unreferenced_files_for_cleanup_in_session(
                cleanup,
                workspace_id=fixture.tenant_id,
                file_ids=(fixture.upload_id,),
            )
            cleanup.commit()

        assert artifacts == ()
        with sessions() as verify:
            assert verify.get(UploadFile, fixture.upload_id) is not None
            binding = verify.scalar(
                select(HumanInputFormUploadFile).where(
                    HumanInputFormUploadFile.tenant_id == fixture.tenant_id,
                    HumanInputFormUploadFile.upload_file_id == fixture.upload_id,
                )
            )
            assert binding is not None
            assert (binding.app_id, binding.form_id, binding.upload_token_id) == (
                fixture.app_id,
                fixture.form_id,
                fixture.upload_token_id,
            )
            assert (
                verify.scalar(
                    select(UploadFileCleanupOutbox.id).where(
                        UploadFileCleanupOutbox.tenant_id == fixture.tenant_id,
                        UploadFileCleanupOutbox.upload_file_id == fixture.upload_id,
                    )
                )
                is None
            )
    finally:
        _delete_human_input_upload_fixture(sessions, fixture)


def test_human_input_upload_publication_serializes_before_conversation_cleanup(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conversation cleanup must wait for the real owner-chain publication transaction."""

    engine = mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ")
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    fixture = _seed_human_input_upload_publication(sessions)
    monkeypatch.setattr(delete_conversation_task_module, "session_factory", _SessionFactoryBoundary(sessions))
    worker = _WorkerState()
    publication_reached_insert = Event()
    allow_publication_insert = Event()
    cleanup_requested_form_lock = Event()
    cleanup_finished = Event()
    publication_errors: list[BaseException] = []
    cleanup_errors: list[BaseException] = []
    cleanup_results: list[bool] = []

    def observe_owner_locks(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        normalized = " ".join(statement.lower().replace("`", "").split())
        if worker.role == "publisher" and normalized.startswith("insert into human_input_form_upload_files"):
            publication_reached_insert.set()
            assert allow_publication_insert.wait(timeout=10)
        if worker.role == "cleanup" and " from human_input_forms " in normalized and " for update" in normalized:
            cleanup_requested_form_lock.set()

    def publish_upload() -> None:
        try:
            worker.role = "publisher"
            SQLAlchemyHumanInputFileUploadRepository(session_factory=sessions).add_file(
                tenant_id=fixture.tenant_id,
                app_id=fixture.app_id,
                form_id=fixture.form_id,
                upload_token_id=fixture.upload_token_id,
                file_id=fixture.upload_id,
            )
        except BaseException as error:
            publication_errors.append(error)

    def clean_conversation() -> None:
        try:
            worker.role = "cleanup"
            cleanup_results.append(
                delete_conversation_task_module._cleanup_conversation_related_data(fixture.conversation_id)
            )
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            cleanup_finished.set()

    publisher = Thread(target=publish_upload)
    cleanup = Thread(target=clean_conversation)
    event.listen(engine, "before_cursor_execute", observe_owner_locks)
    try:
        publisher.start()
        assert publication_reached_insert.wait(timeout=10)
        cleanup.start()
        assert cleanup_requested_form_lock.wait(timeout=10)
        assert not cleanup_finished.wait(timeout=0.25)

        allow_publication_insert.set()
        publisher.join(timeout=10)
        cleanup.join(timeout=10)
        assert not publisher.is_alive()
        assert not cleanup.is_alive()
        assert publication_errors == []
        assert cleanup_results == []
        assert len(cleanup_errors) == 1
        assert str(cleanup_errors[0]) == "Human-input upload links changed while cleanup owners were being locked"

        # The losing cleanup transaction rolled back instead of deleting a
        # newly published link without durable blob-cleanup evidence. Its
        # normal retry sees the committed link, registers the candidate, and
        # atomically deletes the full owner chain.
        assert delete_conversation_task_module._cleanup_conversation_related_data(fixture.conversation_id) is True

        with sessions() as verify:
            assert verify.get(UploadFile, fixture.upload_id) is not None
            assert verify.get(Conversation, fixture.conversation_id) is None
            assert verify.get(HumanInputForm, fixture.form_id) is None
            assert verify.get(HumanInputFormRecipient, fixture.recipient_id) is None
            assert verify.get(HumanInputFormUploadToken, fixture.upload_token_id) is None
            assert (
                verify.scalar(
                    select(HumanInputFormUploadFile.id).where(
                        HumanInputFormUploadFile.upload_file_id == fixture.upload_id
                    )
                )
                is None
            )
            candidate = verify.scalar(
                select(UploadFileCleanupOutbox).where(
                    UploadFileCleanupOutbox.tenant_id == fixture.tenant_id,
                    UploadFileCleanupOutbox.upload_file_id == fixture.upload_id,
                )
            )
            assert candidate is not None
            assert candidate.writer_id is None
            assert candidate.writer_finished_at is None
    finally:
        allow_publication_insert.set()
        if publisher.is_alive():
            publisher.join(timeout=10)
        if cleanup.is_alive():
            cleanup.join(timeout=10)
        event.remove(engine, "before_cursor_execute", observe_owner_locks)
        _delete_human_input_upload_fixture(sessions, fixture)


def test_conversation_cleanup_winner_rejects_late_human_input_upload_publication(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A publisher waiting behind owner deletion must leave its finished marker retryable."""

    engine = mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ")
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    fixture = _seed_human_input_upload_publication(sessions)
    monkeypatch.setattr(delete_conversation_task_module, "session_factory", _SessionFactoryBoundary(sessions))
    worker = _WorkerState()
    cleanup_locked_form = Event()
    allow_cleanup_commit = Event()
    publication_requested_form_lock = Event()
    publication_finished = Event()
    cleanup_errors: list[BaseException] = []
    cleanup_results: list[bool] = []
    publication_errors: list[BaseException] = []

    def pause_after_cleanup_owner_lock(
        _connection, _cursor, statement: str, _parameters, _context, _executemany
    ) -> None:
        normalized = " ".join(statement.lower().replace("`", "").split())
        if worker.role == "cleanup" and " from human_input_forms " in normalized and " for update" in normalized:
            cleanup_locked_form.set()
            assert allow_cleanup_commit.wait(timeout=10)

    def observe_publication_owner_lock(
        _connection, _cursor, statement: str, _parameters, _context, _executemany
    ) -> None:
        normalized = " ".join(statement.lower().replace("`", "").split())
        if (
            worker.role == "publisher"
            and " from human_input_forms " in normalized
            and (" lock in share mode" in normalized or " for share" in normalized)
        ):
            publication_requested_form_lock.set()

    def clean_conversation() -> None:
        try:
            worker.role = "cleanup"
            cleanup_results.append(
                delete_conversation_task_module._cleanup_conversation_related_data(fixture.conversation_id)
            )
        except BaseException as error:
            cleanup_errors.append(error)

    def publish_upload() -> None:
        try:
            worker.role = "publisher"
            SQLAlchemyHumanInputFileUploadRepository(session_factory=sessions).add_file(
                tenant_id=fixture.tenant_id,
                app_id=fixture.app_id,
                form_id=fixture.form_id,
                upload_token_id=fixture.upload_token_id,
                file_id=fixture.upload_id,
            )
        except BaseException as error:
            publication_errors.append(error)
        finally:
            publication_finished.set()

    cleanup = Thread(target=clean_conversation)
    publisher = Thread(target=publish_upload)
    event.listen(engine, "after_cursor_execute", pause_after_cleanup_owner_lock)
    event.listen(engine, "before_cursor_execute", observe_publication_owner_lock)
    try:
        cleanup.start()
        assert cleanup_locked_form.wait(timeout=10)
        publisher.start()
        assert publication_requested_form_lock.wait(timeout=10)
        assert not publication_finished.wait(timeout=0.25)

        allow_cleanup_commit.set()
        cleanup.join(timeout=10)
        publisher.join(timeout=10)
        assert not cleanup.is_alive()
        assert not publisher.is_alive()
        assert cleanup_errors == []
        assert cleanup_results == [True]
        assert len(publication_errors) == 1
        assert isinstance(publication_errors[0], InvalidUploadTokenError)

        with sessions() as verify:
            assert verify.get(UploadFile, fixture.upload_id) is not None
            assert verify.get(HumanInputForm, fixture.form_id) is None
            assert (
                verify.scalar(
                    select(HumanInputFormUploadFile.id).where(
                        HumanInputFormUploadFile.upload_file_id == fixture.upload_id
                    )
                )
                is None
            )
            marker = verify.scalar(
                select(UploadFileCleanupOutbox).where(
                    UploadFileCleanupOutbox.tenant_id == fixture.tenant_id,
                    UploadFileCleanupOutbox.upload_file_id == fixture.upload_id,
                )
            )
            assert marker is not None
            assert marker.writer_id is not None
            assert marker.writer_finished_at is not None
    finally:
        allow_cleanup_commit.set()
        if cleanup.is_alive():
            cleanup.join(timeout=10)
        if publisher.is_alive():
            publisher.join(timeout=10)
        event.remove(engine, "after_cursor_execute", pause_after_cleanup_owner_lock)
        event.remove(engine, "before_cursor_execute", observe_publication_owner_lock)
        _delete_human_input_upload_fixture(sessions, fixture)


def test_cleanup_after_repeatable_read_snapshot_consumes_current_attachment_binding(
    mysql_session: Session,
) -> None:
    sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, document_id, segment_id, first_id, replacement_id = _seed_attachment_replacement(sessions)
    try:
        with sessions() as cleanup:
            assert list(
                cleanup.scalars(
                    select(SegmentAttachmentBinding.attachment_id).where(
                        SegmentAttachmentBinding.tenant_id == tenant_id,
                        SegmentAttachmentBinding.segment_id == segment_id,
                    )
                )
            ) == [first_id]
            SQLAlchemySegmentRepository(session_factory=sessions).replace_attachments(
                DatasetRef(tenant_id, dataset_id).document(document_id).segment(segment_id),
                (replacement_id,),
            )

            artifacts = delete_document_segments_and_claim_uploads(
                cleanup,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_ids=(document_id,),
                segment_ids=(segment_id,),
            )
            cleanup.commit()

        # The cleanup Session established a stale X snapshot, but the owner
        # and binding locking reads must consume Y after the concurrent commit.
        assert {artifact.id for artifact in artifacts} == {replacement_id}

        with sessions() as verify:
            assert verify.get(DocumentSegment, segment_id) is None
            assert (
                list(
                    verify.scalars(
                        select(DocumentSegment.id).where(
                            DocumentSegment.tenant_id == tenant_id,
                            DocumentSegment.dataset_id == dataset_id,
                            DocumentSegment.document_id == document_id,
                        )
                    )
                )
                == []
            )
            assert (
                list(
                    verify.scalars(
                        select(SegmentAttachmentBinding.id).where(
                            SegmentAttachmentBinding.tenant_id == tenant_id,
                            SegmentAttachmentBinding.segment_id == segment_id,
                        )
                    )
                )
                == []
            )
            assert verify.get(UploadFile, first_id) is None
            assert verify.get(UploadFile, replacement_id) is None
            assert set(
                verify.scalars(
                    select(UploadFileCleanupOutbox.upload_file_id).where(
                        UploadFileCleanupOutbox.tenant_id == tenant_id,
                        UploadFileCleanupOutbox.upload_file_id.in_((first_id, replacement_id)),
                    )
                )
            ) == {first_id, replacement_id}
    finally:
        _delete_attachment_replacement_rows(
            sessions,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            file_ids=(first_id, replacement_id),
        )


def test_cleanup_after_indexing_commit_consumes_all_current_attachment_bindings(mysql_session: Session) -> None:
    sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, document_id, segment_id, first_id, replacement_id = _seed_attachment_replacement(sessions)
    ref = DatasetRef(tenant_id, dataset_id).document(document_id)
    indexing_document = IndexingDocument(
        source=StoredSource(ref, "upload_file", {}, "text_model"),
        processing_rule={},
        doc_language="English",
        need_summary=False,
    )
    chunk = IndexDocument(
        page_content="attachment race updated",
        metadata={"doc_id": segment_id, "doc_hash": f"updated-{segment_id}"},
        attachments=[AttachmentDocument(page_content="replacement", metadata={"doc_id": replacement_id})],
    )
    try:
        with sessions() as cleanup:
            assert list(
                cleanup.scalars(
                    select(SegmentAttachmentBinding.attachment_id).where(
                        SegmentAttachmentBinding.tenant_id == tenant_id,
                        SegmentAttachmentBinding.segment_id == segment_id,
                    )
                )
            ) == [first_id]

            SQLAlchemySegmentRepository(session_factory=sessions).save_for_indexing(indexing_document, [chunk], [3])

            artifacts = delete_document_segments_and_claim_uploads(
                cleanup,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_ids=(document_id,),
                segment_ids=(segment_id,),
            )
            cleanup.commit()

        assert {artifact.id for artifact in artifacts} == {first_id, replacement_id}

        with sessions() as verify:
            assert verify.get(DocumentSegment, segment_id) is None
            assert (
                list(
                    verify.scalars(
                        select(DocumentSegment.id).where(
                            DocumentSegment.tenant_id == tenant_id,
                            DocumentSegment.dataset_id == dataset_id,
                            DocumentSegment.document_id == document_id,
                        )
                    )
                )
                == []
            )
            assert (
                list(
                    verify.scalars(
                        select(SegmentAttachmentBinding.id).where(
                            SegmentAttachmentBinding.tenant_id == tenant_id,
                            SegmentAttachmentBinding.segment_id == segment_id,
                        )
                    )
                )
                == []
            )
            assert verify.get(UploadFile, first_id) is None
            assert verify.get(UploadFile, replacement_id) is None
            assert set(
                verify.scalars(
                    select(UploadFileCleanupOutbox.upload_file_id).where(
                        UploadFileCleanupOutbox.tenant_id == tenant_id,
                        UploadFileCleanupOutbox.upload_file_id.in_((first_id, replacement_id)),
                    )
                )
            ) == {first_id, replacement_id}
    finally:
        _delete_attachment_replacement_rows(
            sessions,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            file_ids=(first_id, replacement_id),
        )


def test_attachment_replacement_uses_current_binding_after_repeatable_read_snapshot(
    mysql_session: Session,
) -> None:
    """A stale X snapshot must not stop the later writer from replacing current Y with Z."""

    sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, document_id, segment_id, first_id, replacement_id = _seed_attachment_replacement(sessions)
    third = _upload(
        tenant_id=tenant_id,
        key=f"cleanup-race/{uuid4()}-c.png",
        created_by=str(uuid4()),
    )
    with sessions.begin() as seed:
        seed.add(third)
    third_id = third.id

    try:
        with sessions() as later_writer:
            assert list(
                later_writer.scalars(
                    select(SegmentAttachmentBinding.attachment_id).where(
                        SegmentAttachmentBinding.tenant_id == tenant_id,
                        SegmentAttachmentBinding.segment_id == segment_id,
                    )
                )
            ) == [first_id]

            SQLAlchemySegmentRepository(session_factory=sessions).replace_attachments(
                DatasetRef(tenant_id, dataset_id).document(document_id).segment(segment_id),
                (replacement_id,),
            )
            result = replace_segment_attachment_bindings(
                later_writer,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                segment_id=segment_id,
                attachment_ids=(third_id,),
            )
            later_writer.commit()

        assert result.changed is True
        assert result.removed_ids == (replacement_id,)
        with sessions() as verify:
            assert set(
                verify.scalars(
                    select(SegmentAttachmentBinding.attachment_id).where(
                        SegmentAttachmentBinding.tenant_id == tenant_id,
                        SegmentAttachmentBinding.dataset_id == dataset_id,
                        SegmentAttachmentBinding.document_id == document_id,
                        SegmentAttachmentBinding.segment_id == segment_id,
                    )
                )
            ) == {third_id}
            assert verify.get(UploadFile, first_id) is None
            assert verify.get(UploadFile, replacement_id) is None
            assert verify.get(UploadFile, third_id) is not None
            assert set(
                verify.scalars(
                    select(UploadFileCleanupOutbox.upload_file_id).where(
                        UploadFileCleanupOutbox.tenant_id == tenant_id,
                        UploadFileCleanupOutbox.upload_file_id.in_((first_id, replacement_id, third_id)),
                    )
                )
            ) == {first_id, replacement_id}
    finally:
        _delete_attachment_replacement_rows(
            sessions,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            file_ids=(first_id, replacement_id, third_id),
        )


def test_cleanup_committing_first_rejects_waiting_attachment_replacement(mysql_session: Session) -> None:
    sessions = _repeatable_read_sessions(mysql_session)
    tenant_id, dataset_id, document_id, segment_id, first_id, replacement_id = _seed_attachment_replacement(sessions)
    cleanup_locked = Event()
    allow_cleanup = Event()
    writer_started = Event()
    cleanup_errors: list[BaseException] = []
    writer_errors: list[BaseException] = []

    def clean_segment() -> None:
        try:
            with sessions.begin() as cleanup:
                attachment_ids = detach_owned_segment_attachments(
                    cleanup,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    document_ids=(document_id,),
                    segment_ids=(segment_id,),
                )
                assert attachment_ids == (first_id,)
                cleanup_locked.set()
                assert allow_cleanup.wait(timeout=10)
                delete_document_segments_and_claim_uploads(
                    cleanup,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    document_ids=(document_id,),
                    segment_ids=(segment_id,),
                    candidate_upload_file_ids=attachment_ids,
                )
        except BaseException as error:
            cleanup_errors.append(error)

    def write_replacement() -> None:
        try:
            writer_started.set()
            with sessions.begin() as writer:
                replace_segment_attachment_bindings(
                    writer,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    segment_id=segment_id,
                    attachment_ids=(replacement_id,),
                )
        except BaseException as error:
            writer_errors.append(error)

    cleanup_thread = Thread(target=clean_segment)
    writer_thread = Thread(target=write_replacement)
    try:
        cleanup_thread.start()
        assert cleanup_locked.wait(timeout=10)
        writer_thread.start()
        assert writer_started.wait(timeout=10)
        allow_cleanup.set()
        cleanup_thread.join(timeout=10)
        writer_thread.join(timeout=10)

        assert not cleanup_thread.is_alive()
        assert not writer_thread.is_alive()
        assert cleanup_errors == []
        assert len(writer_errors) == 1
        assert isinstance(writer_errors[0], LookupError)
        assert str(writer_errors[0]) == "Segment no longer exists"
        with sessions() as verify:
            assert verify.get(DocumentSegment, segment_id) is None
            assert (
                list(
                    verify.scalars(
                        select(SegmentAttachmentBinding.id).where(
                            SegmentAttachmentBinding.tenant_id == tenant_id,
                            SegmentAttachmentBinding.segment_id == segment_id,
                        )
                    )
                )
                == []
            )
            assert verify.get(UploadFile, first_id) is None
            assert verify.get(UploadFile, replacement_id) is not None
            assert set(
                verify.scalars(
                    select(UploadFileCleanupOutbox.upload_file_id).where(
                        UploadFileCleanupOutbox.tenant_id == tenant_id,
                        UploadFileCleanupOutbox.upload_file_id.in_((first_id, replacement_id)),
                    )
                )
            ) == {first_id}
    finally:
        allow_cleanup.set()
        cleanup_thread.join(timeout=10)
        writer_thread.join(timeout=10)
        _delete_attachment_replacement_rows(
            sessions,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment_id,
            file_ids=(first_id, replacement_id),
        )

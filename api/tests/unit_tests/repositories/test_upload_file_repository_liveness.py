import base64
import json
import logging
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import ORMExecuteState, Session, sessionmaker
from sqlalchemy.sql import ClauseElement, Executable

from core.workflow.file_reference import build_file_reference
from graphon.file import FileTransferMethod, FileType
from graphon.variables import SegmentType
from models.enums import ConversationFromSource, CreatorUserRole, ExecutionOffLoadType, InvokeFrom, MessageStatus
from models.human_input import HumanInputFormUploadFile
from models.model import MessageFile, UploadFile, UploadFileCleanupOutbox
from models.workflow import (
    ConversationVariable,
    WorkflowDraftVariable,
    WorkflowDraftVariableFile,
    WorkflowNodeExecutionOffload,
)
from repositories.knowledge.upload_file_repository import (
    SQLAlchemyKnowledgeUploadRepository,
    register_upload_file_cleanup_candidates_in_session,
)
from repositories.upload_file_repository import (
    UploadFileReferenceNotFoundError,
    _get_normalized_reference_ids,
    lock_and_validate_owned_upload_files_in_session,
)
from services.knowledge.upload_file_cleanup import drain_pending_upload_file_cleanups
from tests.unit_tests.model_factories import make_app, make_conversation, make_message, make_upload_file


def _id() -> str:
    return str(uuid4())


def test_message_file_liveness_uses_mysql_locking_current_read(sqlite_session: Session) -> None:
    statements: list[Executable] = []

    def record_statement(state: ORMExecuteState) -> None:
        statements.append(state.statement)

    event.listen(sqlite_session, "do_orm_execute", record_statement)
    try:
        assert (
            _get_normalized_reference_ids(
                sqlite_session,
                tenant_id="tenant-id",
                candidates={"upload-file-id"},
            )
            == set()
        )
    finally:
        event.remove(sqlite_session, "do_orm_execute", record_statement)

    message_file_statement = statements[0]
    assert isinstance(message_file_statement, ClauseElement)
    sql = str(message_file_statement.compile(dialect=mysql.dialect())).upper()
    assert sql.endswith((" LOCK IN SHARE MODE", " FOR SHARE"))


def test_human_input_file_liveness_uses_mysql_locking_current_read(sqlite_session: Session) -> None:
    statements: list[Executable] = []

    def record_statement(state: ORMExecuteState) -> None:
        statements.append(state.statement)

    event.listen(sqlite_session, "do_orm_execute", record_statement)
    try:
        assert (
            _get_normalized_reference_ids(
                sqlite_session,
                tenant_id="tenant-id",
                candidates={"upload-file-id"},
            )
            == set()
        )
    finally:
        event.remove(sqlite_session, "do_orm_execute", record_statement)

    human_input_statement = next(
        statement for statement in statements if HumanInputFormUploadFile.__tablename__ in str(statement)
    )
    assert isinstance(human_input_statement, ClauseElement)
    sql = str(human_input_statement.compile(dialect=mysql.dialect())).upper()
    assert sql.endswith((" LOCK IN SHARE MODE", " FOR SHARE"))


def test_lock_and_validate_owned_upload_files_is_complete_and_tenant_scoped(sqlite_session: Session) -> None:
    tenant_id = _id()
    owned = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/owned")
    second_owned = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/second")
    foreign = make_upload_file(file_id=_id(), tenant_id=_id(), key="uploads/foreign")
    sqlite_session.add_all([owned, second_owned, foreign])
    sqlite_session.commit()

    locked = lock_and_validate_owned_upload_files_in_session(
        sqlite_session,
        tenant_id=tenant_id,
        upload_file_ids=(second_owned.id, owned.id, second_owned.id),
    )

    assert list(locked) == sorted((owned.id, second_owned.id))
    with pytest.raises(UploadFileReferenceNotFoundError) as exc_info:
        lock_and_validate_owned_upload_files_in_session(
            sqlite_session,
            tenant_id=tenant_id,
            upload_file_ids=(owned.id, foreign.id, "missing-file"),
        )
    assert exc_info.value.missing_file_ids == tuple(sorted((foreign.id, "missing-file")))


def test_candidate_registration_is_tenant_scoped_idempotent_and_rolls_back_with_caller(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = _id()
    owned = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/owned")
    foreign = make_upload_file(file_id=_id(), tenant_id=_id(), key="uploads/foreign")
    with sqlite_session_factory.begin() as session:
        session.add_all([owned, foreign])

    with sqlite_session_factory() as session:
        register_upload_file_cleanup_candidates_in_session(
            session,
            workspace_id=tenant_id,
            file_ids=(owned.id, foreign.id),
        )
        session.rollback()
    with sqlite_session_factory() as session:
        assert session.scalar(select(UploadFileCleanupOutbox)) is None

    with sqlite_session_factory.begin() as session:
        register_upload_file_cleanup_candidates_in_session(
            session,
            workspace_id=tenant_id,
            file_ids=(owned.id, foreign.id),
        )
        register_upload_file_cleanup_candidates_in_session(
            session,
            workspace_id=tenant_id,
            file_ids=(owned.id,),
        )

    with sqlite_session_factory() as session:
        pending = session.scalars(select(UploadFileCleanupOutbox)).all()
        assert [(artifact.tenant_id, artifact.upload_file_id, artifact.key) for artifact in pending] == [
            (tenant_id, owned.id, owned.key)
        ]
        assert session.get(UploadFile, owned.id) is not None
        assert session.get(UploadFile, foreign.id) is not None


@pytest.mark.parametrize("writer_finished_at", [None, datetime(2026, 1, 1)])
def test_candidate_registration_preserves_existing_writer_receipt(
    sqlite_session_factory: sessionmaker[Session],
    writer_finished_at: datetime | None,
) -> None:
    tenant_id = _id()
    upload = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/writer-owned")
    writer_id = _id()
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                upload,
                UploadFileCleanupOutbox(
                    tenant_id=tenant_id,
                    upload_file_id=upload.id,
                    key=upload.key,
                    writer_id=writer_id,
                    writer_finished_at=writer_finished_at,
                    writer_expires_at=datetime(2099, 1, 1),
                ),
            ]
        )

    with sqlite_session_factory.begin() as session:
        register_upload_file_cleanup_candidates_in_session(
            session,
            workspace_id=tenant_id,
            file_ids=(upload.id,),
        )

    with sqlite_session_factory() as session:
        receipts = session.scalars(select(UploadFileCleanupOutbox)).all()
        assert len(receipts) == 1
        assert receipts[0].writer_id == writer_id
        assert receipts[0].writer_finished_at == writer_finished_at
        assert receipts[0].writer_expires_at == datetime(2099, 1, 1)


def test_cleanup_claim_preserves_every_shared_upload_file_owner(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = _id()
    app_id = _id()
    uploads = {
        owner: make_upload_file(file_id=_id(), tenant_id=tenant_id, key=f"uploads/{owner}")
        for owner in ("message", "draft-offload", "execution-offload", "human-input", "workflow-value", "orphan")
    }
    app = make_app(app_id=app_id, tenant_id=tenant_id)
    message_file = MessageFile(
        message_id=_id(),
        type=FileType.DOCUMENT,
        transfer_method=FileTransferMethod.LOCAL_FILE,
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=_id(),
        upload_file_id=uploads["message"].id,
    )
    draft_offload = WorkflowDraftVariableFile(
        tenant_id=tenant_id,
        app_id=app_id,
        user_id=_id(),
        upload_file_id=uploads["draft-offload"].id,
        size=1,
        length=None,
        value_type=SegmentType.STRING,
    )
    execution_offload = WorkflowNodeExecutionOffload(
        tenant_id=tenant_id,
        app_id=app_id,
        node_execution_id=None,
        type_=ExecutionOffLoadType.OUTPUTS,
        file_id=uploads["execution-offload"].id,
    )
    human_input_file = HumanInputFormUploadFile(
        tenant_id=tenant_id,
        app_id=app_id,
        form_id=_id(),
        upload_file_id=uploads["human-input"].id,
        upload_token_id=_id(),
    )
    workflow_value = ConversationVariable(
        id=_id(),
        conversation_id=_id(),
        app_id=app_id,
        data=json.dumps(
            {
                "value": {
                    "transfer_method": "local_file",
                    "upload_file_id": uploads["workflow-value"].id,
                }
            }
        ),
    )
    sqlite_session.add_all(
        [
            app,
            *uploads.values(),
            message_file,
            draft_offload,
            execution_offload,
            human_input_file,
            workflow_value,
        ]
    )
    sqlite_session.commit()

    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    artifacts = repository.claim_unreferenced_files_for_cleanup(
        workspace_id=tenant_id,
        file_ids=[upload.id for upload in uploads.values()],
    )

    assert [artifact.id for artifact in artifacts] == [uploads["orphan"].id]
    with sqlite_session_factory() as session:
        surviving_ids = set(session.scalars(select(UploadFile.id)))
        assert surviving_ids == {upload.id for owner, upload in uploads.items() if owner != "orphan"}
        queued_ids = set(session.scalars(select(UploadFileCleanupOutbox.upload_file_id)))
        assert queued_ids == {uploads["orphan"].id}


def test_foreign_tenant_owner_does_not_keep_same_tenant_candidate_alive(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = _id()
    upload = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/candidate")
    sqlite_session.add_all(
        [
            upload,
            HumanInputFormUploadFile(
                tenant_id=_id(),
                app_id=_id(),
                form_id=_id(),
                upload_file_id=upload.id,
                upload_token_id=_id(),
            ),
        ]
    )
    sqlite_session.commit()

    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    artifacts = repository.claim_unreferenced_files_for_cleanup(workspace_id=tenant_id, file_ids=[upload.id])

    assert [artifact.id for artifact in artifacts] == [upload.id]


def test_serialized_workflow_values_preserve_canonical_and_real_file_ids(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = _id()
    app_id = _id()
    canonical_upload = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/canonical")
    real_id_upload = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/real-id")
    app = make_app(app_id=app_id, tenant_id=tenant_id)
    conversation_value = ConversationVariable(
        id=_id(),
        conversation_id=_id(),
        app_id=app_id,
        data=json.dumps(
            {
                "value": build_file_reference(
                    record_id=canonical_upload.id,
                    storage_key=canonical_upload.key,
                )
            }
        ),
    )
    draft_value = WorkflowDraftVariable(
        id=_id(),
        app_id=app_id,
        user_id=_id(),
        node_id="node",
        name="file",
        description="",
        selector=json.dumps(["node", "file"]),
        value_type=SegmentType.FILE,
        value=json.dumps({"value": {"real_file_id": real_id_upload.id}}),
        visible=True,
        editable=True,
        node_execution_id=None,
        file_id=None,
        is_default_value=False,
    )
    sqlite_session.add_all([app, canonical_upload, real_id_upload, conversation_value, draft_value])
    sqlite_session.commit()

    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    artifacts = repository.claim_unreferenced_files_for_cleanup(
        workspace_id=tenant_id,
        file_ids=(canonical_upload.id, real_id_upload.id),
    )

    assert artifacts == ()


def test_malformed_chat_reference_does_not_block_cleanup_batch(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = _id()
    app_id = _id()
    conversation_id = _id()
    orphan = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/orphan")
    canonical = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/canonical")
    historical = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/historical")
    canonical_reference = build_file_reference(record_id=canonical.id, storage_key=canonical.key)
    app = make_app(app_id=app_id, tenant_id=tenant_id)
    conversation = make_conversation(
        conversation_id=conversation_id,
        app_id=app_id,
        inputs={},
        from_source=ConversationFromSource.CONSOLE,
        from_account_id=_id(),
    )
    message = make_message(
        message_id=_id(),
        app_id=app_id,
        conversation_id=conversation_id,
        inputs={},
        query="files",
        message={
            "content": f"broken dify-file-ref:W10= beside {canonical_reference}",
            "legacy": {"transfer_method": "local_file", "reference": historical.id},
        },
        answer="done",
        status=MessageStatus.NORMAL,
        message_unit_price=Decimal(0),
        answer_unit_price=Decimal(0),
        total_price=Decimal(0),
        currency="USD",
        from_source=ConversationFromSource.CONSOLE,
        from_account_id=_id(),
        invoke_from=InvokeFrom.EXPLORE,
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([app, conversation, message, orphan, canonical, historical])
        register_upload_file_cleanup_candidates_in_session(
            session,
            workspace_id=tenant_id,
            file_ids=(orphan.id, canonical.id, historical.id),
        )

    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    deleted_keys: list[str] = []

    assert (
        drain_pending_upload_file_cleanups(
            files=repository,
            delete_blob=deleted_keys.append,
            logger=logging.getLogger(__name__),
            limit=100,
        )
        == ()
    )
    assert deleted_keys == [orphan.key]
    with sqlite_session_factory() as session:
        assert session.get(UploadFile, orphan.id) is None
        assert session.get(UploadFile, canonical.id) is not None
        assert session.get(UploadFile, historical.id) is not None
        assert set(session.scalars(select(UploadFileCleanupOutbox.upload_file_id))) == {
            canonical.id,
            historical.id,
        }


@pytest.mark.parametrize("storage_key", [None, 42])
def test_historical_invalid_canonical_storage_key_still_protects_upload(
    sqlite_session_factory: sessionmaker[Session],
    storage_key: object,
) -> None:
    tenant_id = _id()
    app_id = _id()
    conversation_id = _id()
    upload = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/historical-canonical")
    payload = json.dumps(
        {"record_id": upload.id, "storage_key": storage_key},
        separators=(",", ":"),
    ).encode()
    reference = f"dify-file-ref:{base64.urlsafe_b64encode(payload).decode()}"
    app = make_app(app_id=app_id, tenant_id=tenant_id)
    conversation = make_conversation(
        conversation_id=conversation_id,
        app_id=app_id,
        inputs={},
        from_source=ConversationFromSource.CONSOLE,
        from_account_id=_id(),
    )
    message = make_message(
        message_id=_id(),
        app_id=app_id,
        conversation_id=conversation_id,
        inputs={},
        query="historical file",
        message={"content": reference},
        answer="done",
        status=MessageStatus.NORMAL,
        message_unit_price=Decimal(0),
        answer_unit_price=Decimal(0),
        total_price=Decimal(0),
        currency="USD",
        from_source=ConversationFromSource.CONSOLE,
        from_account_id=_id(),
        invoke_from=InvokeFrom.EXPLORE,
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([app, conversation, message, upload])

    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    assert repository.claim_unreferenced_files_for_cleanup(workspace_id=tenant_id, file_ids=(upload.id,)) == ()
    with sqlite_session_factory() as session:
        assert session.get(UploadFile, upload.id) is not None
        assert session.scalar(select(UploadFileCleanupOutbox)) is None


def test_live_opaque_draft_offload_fails_closed_for_nested_file_references(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = _id()
    app_id = _id()
    app = make_app(app_id=app_id, tenant_id=tenant_id)
    candidate = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/nested-candidate")
    referenced_candidate = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/referenced")
    offload_upload = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/offloaded-value")
    message_file = MessageFile(
        message_id=_id(),
        type=FileType.DOCUMENT,
        transfer_method=FileTransferMethod.LOCAL_FILE,
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=_id(),
        upload_file_id=referenced_candidate.id,
    )
    offload = WorkflowDraftVariableFile(
        tenant_id=tenant_id,
        app_id=app_id,
        user_id=_id(),
        upload_file_id=offload_upload.id,
        size=100,
        length=1,
        value_type=SegmentType.OBJECT,
    )
    draft_value = WorkflowDraftVariable(
        id=_id(),
        app_id=app_id,
        user_id=_id(),
        node_id="node",
        name="offloaded",
        description="",
        selector=json.dumps(["node", "offloaded"]),
        value_type=SegmentType.OBJECT,
        value="{}",
        visible=True,
        editable=True,
        node_execution_id=None,
        file_id=offload.id,
        is_default_value=False,
    )
    sqlite_session.add_all([app, candidate, referenced_candidate, offload_upload, message_file, offload, draft_value])
    sqlite_session.commit()

    repository = SQLAlchemyKnowledgeUploadRepository(session_factory=sqlite_session_factory)
    assert (
        repository.claim_unreferenced_files_for_cleanup(
            workspace_id=tenant_id,
            file_ids=(candidate.id, referenced_candidate.id),
        )
        == ()
    )
    assert (
        repository.claim_unreferenced_files_for_cleanup(
            workspace_id=tenant_id,
            file_ids=(candidate.id, referenced_candidate.id),
        )
        == ()
    )
    with sqlite_session_factory() as session:
        assert session.get(UploadFile, candidate.id) is not None
        assert session.get(UploadFile, referenced_candidate.id) is not None
        assert (
            session.scalar(
                select(func.count())
                .select_from(UploadFileCleanupOutbox)
                .where(
                    UploadFileCleanupOutbox.tenant_id == tenant_id,
                    UploadFileCleanupOutbox.upload_file_id == candidate.id,
                )
            )
            == 1
        )

    sqlite_session.delete(draft_value)
    sqlite_session.delete(offload)
    sqlite_session.commit()
    deleted_keys: list[str] = []

    def delete_after_claim(key: str) -> None:
        with sqlite_session_factory() as session:
            assert session.get(UploadFile, candidate.id) is None
            assert (
                session.scalar(
                    select(UploadFileCleanupOutbox.id).where(
                        UploadFileCleanupOutbox.tenant_id == tenant_id,
                        UploadFileCleanupOutbox.upload_file_id == candidate.id,
                    )
                )
                is not None
            )
        deleted_keys.append(key)

    assert (
        drain_pending_upload_file_cleanups(
            files=repository,
            delete_blob=delete_after_claim,
            logger=logging.getLogger(__name__),
            limit=100,
        )
        == ()
    )
    assert deleted_keys == [candidate.key]
    with sqlite_session_factory() as session:
        assert session.get(UploadFile, candidate.id) is None
        assert session.get(UploadFile, referenced_candidate.id) is not None
        assert session.scalar(select(UploadFileCleanupOutbox)) is None


def test_foreign_tenant_opaque_offload_does_not_defer_candidate_cleanup(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = _id()
    foreign_tenant_id = _id()
    foreign_app_id = _id()
    candidate = make_upload_file(file_id=_id(), tenant_id=tenant_id, key="uploads/candidate")
    foreign_offload_upload = make_upload_file(file_id=_id(), tenant_id=foreign_tenant_id, key="uploads/foreign-offload")
    foreign_app = make_app(app_id=foreign_app_id, tenant_id=foreign_tenant_id)
    foreign_offload = WorkflowDraftVariableFile(
        tenant_id=foreign_tenant_id,
        app_id=foreign_app_id,
        user_id=_id(),
        upload_file_id=foreign_offload_upload.id,
        size=100,
        length=1,
        value_type=SegmentType.OBJECT,
    )
    foreign_value = WorkflowDraftVariable(
        id=_id(),
        app_id=foreign_app_id,
        user_id=_id(),
        node_id="node",
        name="offloaded",
        description="",
        selector=json.dumps(["node", "offloaded"]),
        value_type=SegmentType.OBJECT,
        value="{}",
        visible=True,
        editable=True,
        node_execution_id=None,
        file_id=foreign_offload.id,
        is_default_value=False,
    )
    sqlite_session.add_all([candidate, foreign_offload_upload, foreign_app, foreign_offload, foreign_value])
    sqlite_session.commit()

    artifacts = SQLAlchemyKnowledgeUploadRepository(
        session_factory=sqlite_session_factory
    ).claim_unreferenced_files_for_cleanup(
        workspace_id=tenant_id,
        file_ids=(candidate.id,),
    )
    assert [artifact.id for artifact in artifacts] == [candidate.id]

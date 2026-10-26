"""Shared UploadFile liveness and durable cleanup persistence.

Upload files can be referenced by messages, workflows, human-input forms, and
knowledge resources. Resource-specific callers provide candidate ids only;
this module owns the complete cross-domain liveness decision before an
UploadFile row or its blob is removed.
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from os.path import commonprefix
from typing import Any, cast
from uuid import uuid4

from psycopg2.errors import LockNotAvailable
from pydantic import BaseModel
from pymysql.err import OperationalError as PyMySQLOperationalError
from sqlalchemy import String, and_, case, delete, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from core.rag.extractor.entity.extract_setting import CreatedUploadArtifact
from core.workflow.file_reference import (
    build_file_reference,
    is_canonical_file_reference,
    resolve_file_record_id,
    resolve_local_file_record_id,
)
from libs.datetime_utils import naive_utc_now
from models import Account, App, HumanInputFormUploadFile, MessageFile
from models.account import Tenant
from models.agent import (
    Agent,
    AgentConfigDraft,
    AgentConfigSnapshot,
    AgentIconType,
    AgentWorkspaceBinding,
    WorkflowAgentNodeBinding,
)
from models.dataset import (
    ChildChunk,
    Dataset,
    Document,
    DocumentSegment,
    PipelineCustomizedTemplate,
    SegmentAttachmentBinding,
)
from models.enums import DataSourceType
from models.human_input import HumanInputForm
from models.model import Conversation, IconType, Message, Site, UploadFile, UploadFileCleanupOutbox
from models.snippet import CustomizedSnippet
from models.tools import MCPToolProvider
from models.workflow import (
    ConversationVariable,
    Workflow,
    WorkflowDraftVariable,
    WorkflowDraftVariableFile,
    WorkflowNodeExecutionModel,
    WorkflowNodeExecutionOffload,
    WorkflowPause,
    WorkflowRun,
)

_REFERENCE_QUERY_BATCH_SIZE = 100
_MARKDOWN_IMAGE_TARGET_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")
_UPLOAD_PREVIEW_ID_PATTERN = re.compile(r"/files/([a-f0-9-]+)/(?:file-preview|image-preview)(?:\?.*?)?")
_EMBEDDED_CANONICAL_REFERENCE_PATTERN = re.compile(r"dify-file-ref:[A-Za-z0-9_-]+={0,2}")
_CANONICAL_REFERENCE_PREFIX = "dify-file-ref:"
_UPLOAD_WRITER_LEASE_SECONDS = 300
_MYSQL_LOCK_NOWAIT_ERROR = 3572


@dataclass(frozen=True)
class UploadFileCleanupArtifact:
    id: str
    cleanup_id: str
    tenant_id: str
    key: str
    writer_id: str | None = None
    writer_finished_at: datetime | None = None
    writer_expires_at: datetime | None = None
    cleanup_started_at: datetime | None = None


class UploadFilePublicationLostError(RuntimeError):
    """A cleanup worker fenced a newly-created object before publication."""


@dataclass(frozen=True)
class _UploadFileLiveness:
    """Separate definite owners from candidates whose cleanup must be deferred."""

    referenced_ids: frozenset[str]
    deferred_ids: frozenset[str]

    @property
    def retained_ids(self) -> set[str]:
        return set(self.referenced_ids | self.deferred_ids)


class UploadFileReferenceNotFoundError(ValueError):
    """Raised when a tenant does not own every file a durable owner will reference."""

    def __init__(self, missing_file_ids: Iterable[str]) -> None:
        self.missing_file_ids = tuple(sorted(set(missing_file_ids)))
        super().__init__(f"Upload files no longer exist: {list(self.missing_file_ids)}")


def _is_owner_lock_not_available(error: OperationalError) -> bool:
    """Recognize only explicit NOWAIT conflicts used by owner current-reads."""

    original = error.orig
    if isinstance(original, LockNotAvailable):
        return True
    return bool(
        isinstance(original, PyMySQLOperationalError) and original.args and original.args[0] == _MYSQL_LOCK_NOWAIT_ERROR
    )


def _to_cleanup_artifact(artifact: UploadFileCleanupOutbox) -> UploadFileCleanupArtifact:
    return UploadFileCleanupArtifact(
        id=artifact.upload_file_id,
        cleanup_id=artifact.id,
        tenant_id=artifact.tenant_id,
        key=artifact.key,
        writer_id=artifact.writer_id,
        writer_finished_at=artifact.writer_finished_at,
        writer_expires_at=artifact.writer_expires_at,
        cleanup_started_at=artifact.cleanup_started_at,
    )


def _candidate_batches(candidate_ids: set[str]) -> list[tuple[str, ...]]:
    ordered_ids = sorted(candidate_ids)
    return [
        tuple(ordered_ids[offset : offset + _REFERENCE_QUERY_BATCH_SIZE])
        for offset in range(0, len(ordered_ids), _REFERENCE_QUERY_BATCH_SIZE)
    ]


def _source_upload_file_id(*, data_source_type: str, data_source_info: str | None) -> str | None:
    if not data_source_info:
        return None
    try:
        value = json.loads(data_source_info)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if data_source_type == DataSourceType.LOCAL_FILE:
        return resolve_local_file_record_id(value)
    if data_source_type != DataSourceType.UPLOAD_FILE:
        return None
    file_id = value.get("upload_file_id")
    return file_id if isinstance(file_id, str) and file_id else None


def get_document_source_upload_file_id(document: Document) -> str | None:
    """Return the UploadFile owned by a persisted document source."""
    return _source_upload_file_id(
        data_source_type=document.data_source_type,
        data_source_info=document.data_source_info,
    )


def _markdown_upload_file_ids(content: str) -> set[str]:
    """Match the same upload-preview markdown targets consumed by index processors."""
    return {
        match.group(1)
        for target in _MARKDOWN_IMAGE_TARGET_PATTERN.findall(content)
        if (match := _UPLOAD_PREVIEW_ID_PATTERN.search(target)) is not None
    }


def _canonical_reference_query_prefix(file_id: str) -> str:
    """Return a stable prefix shared by canonical references with or without a storage key."""
    return commonprefix(
        (
            build_file_reference(record_id=file_id),
            build_file_reference(record_id=file_id, storage_key="query-prefix"),
        )
    )


def _get_knowledge_binding_reference_ids(
    session: Session,
    *,
    tenant_id: str,
    candidates: set[str],
) -> set[str]:
    referenced: set[str] = set()
    for candidate_batch in _candidate_batches(candidates):
        referenced.update(
            session.scalars(
                select(SegmentAttachmentBinding.attachment_id)
                .where(
                    SegmentAttachmentBinding.tenant_id == tenant_id,
                    SegmentAttachmentBinding.attachment_id.in_(candidate_batch),
                )
                .with_for_update(nowait=True)
            )
        )
    return referenced


def _get_knowledge_markdown_reference_ids(
    session: Session,
    *,
    tenant_id: str,
    candidates: set[str],
) -> set[str]:
    referenced: set[str] = set()
    for candidate_batch in _candidate_batches(candidates):
        preview_tokens = {candidate_id: f"/files/{candidate_id}/" for candidate_id in candidate_batch}
        segment_predicates = [
            or_(
                DocumentSegment.content.contains(preview_token, autoescape=True),
                DocumentSegment.answer.contains(preview_token, autoescape=True),
            )
            for preview_token in preview_tokens.values()
        ]
        segment_texts = session.execute(
            select(DocumentSegment.content, DocumentSegment.answer)
            .where(
                DocumentSegment.tenant_id == tenant_id,
                or_(*segment_predicates),
            )
            .with_for_update(nowait=True)
        ).all()
        for content in (
            text for segment_content, answer in segment_texts for text in (segment_content, answer) if text
        ):
            referenced.update(set(candidate_batch).intersection(_markdown_upload_file_ids(content)))

        child_candidates = set(candidate_batch) - referenced
        if not child_candidates:
            continue
        child_predicates = [
            ChildChunk.content.contains(preview_tokens[candidate_id], autoescape=True)
            for candidate_id in child_candidates
        ]
        child_texts = session.scalars(
            select(ChildChunk.content)
            .where(
                ChildChunk.tenant_id == tenant_id,
                or_(*child_predicates),
            )
            .with_for_update(nowait=True)
        )
        for content in child_texts:
            referenced.update(child_candidates.intersection(_markdown_upload_file_ids(content)))
    return referenced


def _get_knowledge_source_reference_ids(
    session: Session,
    *,
    tenant_id: str,
    candidates: set[str],
    candidate_keys: Mapping[str, set[str]],
) -> set[str]:
    referenced: set[str] = set()
    for candidate_batch in _candidate_batches(candidates):
        source_predicates = [
            predicate
            for candidate_id in candidate_batch
            for predicate in (
                Document.data_source_info.contains(candidate_id, autoescape=True),
                Document.data_source_info.contains(
                    _canonical_reference_query_prefix(candidate_id),
                    autoescape=True,
                ),
            )
        ]
        source_predicates.extend(
            Document.data_source_info.contains(key, autoescape=True)
            for candidate_id in candidate_batch
            for key in candidate_keys.get(candidate_id, ())
        )
        source_rows = session.execute(
            select(Document.data_source_type, Document.data_source_info)
            .where(
                Document.tenant_id == tenant_id,
                Document.data_source_type.in_((DataSourceType.UPLOAD_FILE, DataSourceType.LOCAL_FILE)),
                Document.data_source_info.is_not(None),
                or_(*source_predicates),
            )
            .with_for_update(nowait=True)
        ).all()
        for data_source_type, data_source_info in source_rows:
            upload_file_id = _source_upload_file_id(
                data_source_type=data_source_type,
                data_source_info=data_source_info,
            )
            if upload_file_id in candidate_batch:
                referenced.add(upload_file_id)
                continue
            if data_source_info:
                try:
                    source_value: Any = json.loads(data_source_info)
                except json.JSONDecodeError:
                    source_value = None
                if isinstance(source_value, Mapping):
                    referenced.update(
                        set(candidate_batch).intersection(
                            _ids_for_storage_key(source_value.get("storage_key"), candidate_keys)
                        )
                    )
    return referenced


def _get_knowledge_reference_ids(
    session: Session,
    *,
    tenant_id: str,
    candidates: set[str],
    candidate_keys: Mapping[str, set[str]],
) -> set[str]:
    referenced = _get_knowledge_binding_reference_ids(session, tenant_id=tenant_id, candidates=candidates)
    remaining = candidates - referenced
    if remaining:
        referenced.update(_get_knowledge_markdown_reference_ids(session, tenant_id=tenant_id, candidates=remaining))
    remaining = candidates - referenced
    if remaining:
        referenced.update(
            _get_knowledge_source_reference_ids(
                session,
                tenant_id=tenant_id,
                candidates=remaining,
                candidate_keys=candidate_keys,
            )
        )
    return referenced


def _candidate_storage_keys(
    session: Session,
    *,
    tenant_id: str,
    candidates: set[str],
) -> dict[str, set[str]]:
    keys: dict[str, set[str]] = {}
    rows = list(
        session.execute(
            select(UploadFile.id, UploadFile.key).where(
                UploadFile.tenant_id == tenant_id,
                UploadFile.id.in_(candidates),
            )
        ).all()
    )
    rows.extend(
        session.execute(
            select(UploadFileCleanupOutbox.upload_file_id, UploadFileCleanupOutbox.key).where(
                UploadFileCleanupOutbox.tenant_id == tenant_id,
                UploadFileCleanupOutbox.upload_file_id.in_(candidates),
            )
        ).all()
    )
    for file_id, key in rows:
        if key:
            keys.setdefault(file_id, set()).add(key)
    return keys


def _ids_for_storage_key(storage_key: object, candidate_keys: Mapping[str, set[str]]) -> set[str]:
    if not isinstance(storage_key, str):
        return set()
    return {file_id for file_id, keys in candidate_keys.items() if storage_key in keys}


def _json_upload_file_ids(
    value: object,
    candidates: set[str],
    candidate_keys: Mapping[str, set[str]],
) -> set[str]:
    """Extract exact UploadFile references from persisted payloads."""
    referenced: set[str] = set()
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        upload_file_id = value.get("upload_file_id")
        if isinstance(upload_file_id, str) and upload_file_id in candidates:
            referenced.add(upload_file_id)
        local_file_id = resolve_local_file_record_id(value)
        if local_file_id in candidates:
            referenced.add(local_file_id)
        referenced.update(_ids_for_storage_key(value.get("storage_key"), candidate_keys))
        if value.get("file_kind") == "upload_file" or value.get("transfer_method") == "local_file":
            for field in ("file_id", "id"):
                encoded_id = value.get(field)
                if isinstance(encoded_id, str):
                    file_id = resolve_file_record_id(encoded_id)
                    if file_id in candidates:
                        referenced.add(file_id)
        for nested in value.values():
            referenced.update(_json_upload_file_ids(nested, candidates, candidate_keys))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            referenced.update(_json_upload_file_ids(nested, candidates, candidate_keys))
    elif isinstance(value, str):
        if is_canonical_file_reference(value):
            file_id = resolve_file_record_id(value)
            if file_id in candidates:
                referenced.add(file_id)
        for canonical_reference in _EMBEDDED_CANONICAL_REFERENCE_PATTERN.findall(value):
            file_id = resolve_file_record_id(canonical_reference)
            if file_id in candidates:
                referenced.add(file_id)
        referenced.update(candidates.intersection(_UPLOAD_PREVIEW_ID_PATTERN.findall(value)))
    return referenced


def _parse_json_reference_rows(
    rows: Iterable[object],
    candidates: set[str],
    candidate_keys: Mapping[str, set[str]],
) -> set[str]:
    referenced: set[str] = set()
    for payload in rows:
        if not payload:
            continue
        if isinstance(payload, str):
            try:
                value: Any = json.loads(payload)
            except json.JSONDecodeError:
                # A malformed historical JSON payload cannot be safely
                # interpreted. Preserve candidates with visible exact or
                # canonical tokens, and still recognize embedded URLs.
                referenced.update(
                    candidate
                    for candidate in candidates
                    if candidate in payload or _canonical_reference_query_prefix(candidate) in payload
                )
                referenced.update(_json_upload_file_ids(payload, candidates, candidate_keys))
                continue
        else:
            value = payload
        referenced.update(_json_upload_file_ids(value, candidates, candidate_keys))
    return referenced


def _parse_text_reference_rows(
    rows: Iterable[str | None],
    candidates: set[str],
    candidate_keys: Mapping[str, set[str]],
) -> set[str]:
    referenced: set[str] = set()
    for payload in rows:
        if not payload:
            continue
        referenced.update(_json_upload_file_ids(payload, candidates, candidate_keys))
    return referenced


def _payload_reference_predicates(
    column: Any,
    candidate_batch: tuple[str, ...],
    candidate_keys: Mapping[str, set[str]],
) -> list[Any]:
    tokens = [*candidate_batch, _CANONICAL_REFERENCE_PREFIX]
    tokens.extend(key for candidate_id in candidate_batch for key in candidate_keys.get(candidate_id, ()))
    searchable_column = column.cast(String)
    return [searchable_column.contains(token, autoescape=True) for token in dict.fromkeys(tokens)]


def _get_normalized_reference_ids(session: Session, *, tenant_id: str, candidates: set[str]) -> set[str]:
    referenced: set[str] = set()

    def add(values: Iterable[str | None]) -> None:
        referenced.update(value for value in values if value is not None)

    for candidate_batch in _candidate_batches(candidates):
        # Message writers take an UploadFile shared lock before inserting their
        # owner row. Cleanup already holds the conflicting UploadFile exclusive
        # lock here, so this must also be a locking current read: a plain SELECT
        # can keep using the transaction's older REPEATABLE READ snapshot after
        # a blocked writer commits.
        add(
            session.scalars(
                select(MessageFile.upload_file_id)
                .where(MessageFile.upload_file_id.in_(candidate_batch))
                .with_for_update(read=True)
            )
        )
        add(
            session.scalars(
                select(HumanInputFormUploadFile.upload_file_id)
                .where(
                    HumanInputFormUploadFile.tenant_id == tenant_id,
                    HumanInputFormUploadFile.upload_file_id.in_(candidate_batch),
                )
                .with_for_update(read=True)
            )
        )
        add(
            session.scalars(
                select(WorkflowNodeExecutionOffload.file_id).where(
                    WorkflowNodeExecutionOffload.tenant_id == tenant_id,
                    WorkflowNodeExecutionOffload.file_id.in_(candidate_batch),
                )
            )
        )
        add(
            session.scalars(
                select(WorkflowDraftVariableFile.upload_file_id).where(
                    WorkflowDraftVariableFile.tenant_id == tenant_id,
                    WorkflowDraftVariableFile.upload_file_id.in_(candidate_batch),
                )
            )
        )
        add(session.scalars(select(Account.avatar).where(Account.avatar.in_(candidate_batch))))
        add(
            session.scalars(
                select(App.icon).where(
                    App.tenant_id == tenant_id,
                    App.icon_type == IconType.IMAGE,
                    App.icon.in_(candidate_batch),
                )
            )
        )
        add(
            session.scalars(
                select(Site.icon)
                .join(App, App.id == Site.app_id)
                .where(
                    App.tenant_id == tenant_id,
                    Site.icon_type == IconType.IMAGE,
                    Site.icon.in_(candidate_batch),
                )
            )
        )
        add(
            session.scalars(
                select(Agent.icon).where(
                    Agent.tenant_id == tenant_id,
                    Agent.icon_type == AgentIconType.IMAGE,
                    Agent.icon.in_(candidate_batch),
                )
            )
        )
        add(
            session.scalars(
                select(MCPToolProvider.icon).where(
                    MCPToolProvider.tenant_id == tenant_id,
                    MCPToolProvider.icon.in_(candidate_batch),
                )
            )
        )
    return referenced


def _get_workflow_payload_reference_ids(
    session: Session,
    *,
    tenant_id: str,
    candidates: set[str],
    candidate_keys: Mapping[str, set[str]],
) -> set[str]:
    """Protect local files serialized inside live workflow values and traces."""
    referenced: set[str] = set()
    for candidate_batch in _candidate_batches(candidates):
        # Build each query against its own nullable columns. Pushing candidate
        # tokens into SQL avoids parsing every workflow row in the workspace.
        run_columns = (WorkflowRun.graph, WorkflowRun.inputs, WorkflowRun.outputs)
        run_rows = session.execute(
            select(*run_columns).where(
                WorkflowRun.tenant_id == tenant_id,
                or_(
                    *(
                        predicate
                        for column in run_columns
                        for predicate in _payload_reference_predicates(column, candidate_batch, candidate_keys)
                    )
                ),
            )
        ).all()
        referenced.update(
            _parse_json_reference_rows(
                (payload for row in run_rows for payload in row),
                set(candidate_batch),
                candidate_keys,
            )
        )

        execution_columns = (
            WorkflowNodeExecutionModel.inputs,
            WorkflowNodeExecutionModel.process_data,
            WorkflowNodeExecutionModel.outputs,
            WorkflowNodeExecutionModel.execution_metadata,
        )
        execution_rows = session.execute(
            select(*execution_columns).where(
                WorkflowNodeExecutionModel.tenant_id == tenant_id,
                or_(
                    *(
                        predicate
                        for column in execution_columns
                        for predicate in _payload_reference_predicates(column, candidate_batch, candidate_keys)
                    )
                ),
            )
        ).all()
        referenced.update(
            _parse_json_reference_rows(
                (payload for row in execution_rows for payload in row),
                set(candidate_batch),
                candidate_keys,
            )
        )

        draft_rows = session.scalars(
            select(WorkflowDraftVariable.value)
            .join(App, App.id == WorkflowDraftVariable.app_id)
            .where(
                App.tenant_id == tenant_id,
                or_(*_payload_reference_predicates(WorkflowDraftVariable.value, candidate_batch, candidate_keys)),
            )
        )
        referenced.update(_parse_json_reference_rows(draft_rows, set(candidate_batch), candidate_keys))

        conversation_rows = session.scalars(
            select(ConversationVariable.data)
            .join(App, App.id == ConversationVariable.app_id)
            .where(
                App.tenant_id == tenant_id,
                or_(*_payload_reference_predicates(ConversationVariable.data, candidate_batch, candidate_keys)),
            )
        )
        referenced.update(_parse_json_reference_rows(conversation_rows, set(candidate_batch), candidate_keys))
    return referenced


def _agent_job_config_file_ids(value: object, candidates: set[str]) -> set[str]:
    """Extract schema-defined upload refs from WorkflowAgentNodeBinding JSON."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {candidate for candidate in candidates if candidate in value}
    if not isinstance(value, Mapping):
        return set()

    references: list[object] = []
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping):
        file_refs = metadata.get("file_refs")
        if isinstance(file_refs, list):
            references.extend(file_refs)
    declared_outputs = value.get("declared_outputs")
    if isinstance(declared_outputs, list):
        for output in declared_outputs:
            if not isinstance(output, Mapping):
                continue
            check = output.get("check")
            if isinstance(check, Mapping):
                references.append(check.get("benchmark_file_ref"))

    referenced: set[str] = set()
    for reference in references:
        if isinstance(reference, BaseModel):
            reference = reference.model_dump(mode="json")
        if not isinstance(reference, Mapping):
            continue
        for field in ("upload_file_id", "file_id", "id", "reference"):
            encoded_id = reference.get(field)
            if not isinstance(encoded_id, str):
                continue
            file_id = resolve_file_record_id(encoded_id)
            if file_id in candidates:
                referenced.add(file_id)
                break
    return referenced


def _image_icon_upload_file_id(value: object, candidates: set[str]) -> str | None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping) or value.get("icon_type") != "image":
        return None
    icon = value.get("icon")
    return icon if isinstance(icon, str) and icon in candidates else None


def _get_persisted_payload_reference_ids(
    session: Session,
    *,
    tenant_id: str,
    candidates: set[str],
    candidate_keys: Mapping[str, set[str]],
) -> set[str]:
    """Protect file refs in non-execution persisted app, chat, form, and agent state."""
    referenced: set[str] = set()
    for candidate_batch in _candidate_batches(candidates):
        batch = set(candidate_batch)

        workflow_columns = (Workflow.graph, Workflow._conversation_variables)
        workflow_rows = session.execute(
            select(*workflow_columns).where(
                Workflow.tenant_id == tenant_id,
                or_(
                    *(
                        predicate
                        for column in workflow_columns
                        for predicate in _payload_reference_predicates(column, candidate_batch, candidate_keys)
                    )
                ),
            )
        ).all()
        referenced.update(
            _parse_json_reference_rows(
                (payload for row in workflow_rows for payload in row),
                batch,
                candidate_keys,
            )
        )

        conversation_json_rows = session.scalars(
            select(Conversation._inputs)
            .join(App, App.id == Conversation.app_id)
            .where(
                App.tenant_id == tenant_id,
                or_(*_payload_reference_predicates(Conversation._inputs, candidate_batch, candidate_keys)),
            )
        )
        referenced.update(_parse_json_reference_rows(conversation_json_rows, batch, candidate_keys))
        conversation_text_columns = (Conversation.introduction, Conversation.system_instruction)
        conversation_text_rows = session.execute(
            select(*conversation_text_columns)
            .join(App, App.id == Conversation.app_id)
            .where(
                App.tenant_id == tenant_id,
                or_(
                    *(
                        predicate
                        for column in conversation_text_columns
                        for predicate in _payload_reference_predicates(column, candidate_batch, candidate_keys)
                    )
                ),
            )
        ).all()
        referenced.update(
            _parse_text_reference_rows(
                (payload for row in conversation_text_rows for payload in row),
                batch,
                candidate_keys,
            )
        )

        message_json_columns = (Message._inputs, Message.message)
        message_json_rows = session.execute(
            select(*message_json_columns)
            .join(App, App.id == Message.app_id)
            .where(
                App.tenant_id == tenant_id,
                or_(
                    *(
                        predicate
                        for column in message_json_columns
                        for predicate in _payload_reference_predicates(column, candidate_batch, candidate_keys)
                    )
                ),
            )
        ).all()
        referenced.update(
            _parse_json_reference_rows(
                (payload for row in message_json_rows for payload in row),
                batch,
                candidate_keys,
            )
        )
        message_answers = session.scalars(
            select(Message.answer)
            .join(App, App.id == Message.app_id)
            .where(
                App.tenant_id == tenant_id,
                or_(*_payload_reference_predicates(Message.answer, candidate_batch, candidate_keys)),
            )
        )
        referenced.update(_parse_text_reference_rows(message_answers, batch, candidate_keys))

        form_json_columns = (HumanInputForm.form_definition, HumanInputForm.submitted_data)
        form_json_rows = session.execute(
            select(*form_json_columns).where(
                HumanInputForm.tenant_id == tenant_id,
                or_(
                    *(
                        predicate
                        for column in form_json_columns
                        for predicate in _payload_reference_predicates(column, candidate_batch, candidate_keys)
                    )
                ),
            )
        ).all()
        referenced.update(
            _parse_json_reference_rows(
                (payload for row in form_json_rows for payload in row),
                batch,
                candidate_keys,
            )
        )
        rendered_forms = session.scalars(
            select(HumanInputForm.rendered_content).where(
                HumanInputForm.tenant_id == tenant_id,
                or_(*_payload_reference_predicates(HumanInputForm.rendered_content, candidate_batch, candidate_keys)),
            )
        )
        referenced.update(_parse_text_reference_rows(rendered_forms, batch, candidate_keys))

        for model, column in (
            (Dataset, Dataset.icon_info),
            (PipelineCustomizedTemplate, PipelineCustomizedTemplate.icon),
            (CustomizedSnippet, CustomizedSnippet.icon_info),
        ):
            icon_rows = session.scalars(
                select(column).where(
                    model.tenant_id == tenant_id,
                    or_(*_payload_reference_predicates(column, candidate_batch, candidate_keys)),
                )
            )
            referenced.update(
                icon_id
                for icon_info in icon_rows
                if (icon_id := _image_icon_upload_file_id(icon_info, batch)) is not None
            )

        for model, column in (
            (AgentConfigDraft, AgentConfigDraft.config_snapshot),
            (AgentConfigSnapshot, AgentConfigSnapshot.config_snapshot),
            (AgentWorkspaceBinding, AgentWorkspaceBinding.session_snapshot),
        ):
            rows = session.scalars(
                select(column).where(
                    model.tenant_id == tenant_id,
                    or_(*_payload_reference_predicates(column, candidate_batch, candidate_keys)),
                )
            )
            referenced.update(_parse_json_reference_rows(rows, batch, candidate_keys))

        job_configs = session.scalars(
            select(WorkflowAgentNodeBinding.node_job_config).where(
                WorkflowAgentNodeBinding.tenant_id == tenant_id,
                or_(
                    *_payload_reference_predicates(
                        WorkflowAgentNodeBinding.node_job_config,
                        candidate_batch,
                        candidate_keys,
                    )
                ),
            )
        ).all()
        referenced.update(_parse_json_reference_rows(job_configs, batch, candidate_keys))
        referenced.update(file_id for config in job_configs for file_id in _agent_job_config_file_ids(config, batch))
    return referenced


def _get_workspace_config_reference_ids(
    session: Session,
    *,
    tenant_id: str,
    candidates: set[str],
) -> set[str]:
    payload = session.scalar(select(Tenant.custom_config).where(Tenant.id == tenant_id))
    if not payload:
        return set()
    try:
        value: Any = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return {
            candidate
            for candidate in candidates
            if candidate in payload or _canonical_reference_query_prefix(candidate) in payload
        }
    logo = value.get("replace_webapp_logo") if isinstance(value, Mapping) else None
    return {logo} if isinstance(logo, str) and logo in candidates else set()


def _has_live_opaque_workflow_payload(session: Session, *, tenant_id: str) -> bool:
    """Return whether the tenant has live offloaded payloads that cannot be inspected in SQL.

    Offloaded workflow values may contain nested local-file references that are
    absent from their truncated relational columns. Until those references have
    a durable index, retaining candidates is the only safe decision.
    """
    execution_offload_id = session.scalar(
        select(WorkflowNodeExecutionOffload.id)
        .join(
            WorkflowNodeExecutionModel,
            WorkflowNodeExecutionModel.id == WorkflowNodeExecutionOffload.node_execution_id,
        )
        .where(
            WorkflowNodeExecutionOffload.tenant_id == tenant_id,
            WorkflowNodeExecutionModel.tenant_id == tenant_id,
        )
        .limit(1)
    )
    if execution_offload_id is not None:
        return True
    draft_offload_id = session.scalar(
        select(WorkflowDraftVariableFile.id)
        .join(WorkflowDraftVariable, WorkflowDraftVariable.file_id == WorkflowDraftVariableFile.id)
        .join(App, App.id == WorkflowDraftVariable.app_id)
        .where(
            WorkflowDraftVariableFile.tenant_id == tenant_id,
            App.tenant_id == tenant_id,
        )
        .limit(1)
    )
    if draft_offload_id is not None:
        return True
    pause_id = session.scalar(
        select(WorkflowPause.id)
        .join(WorkflowRun, WorkflowRun.id == WorkflowPause.workflow_run_id)
        .where(
            WorkflowRun.tenant_id == tenant_id,
            WorkflowPause.resumed_at.is_(None),
        )
        .limit(1)
    )
    return pause_id is not None


def _get_upload_file_liveness(
    session: Session,
    *,
    tenant_id: str,
    candidate_ids: Sequence[str],
) -> _UploadFileLiveness:
    """Classify candidates as definitely referenced, opaque, or unreferenced."""
    candidates = {str(file_id) for file_id in candidate_ids if file_id}
    if not candidates:
        return _UploadFileLiveness(frozenset(), frozenset())
    candidate_keys = _candidate_storage_keys(session, tenant_id=tenant_id, candidates=candidates)
    referenced = _get_knowledge_reference_ids(
        session,
        tenant_id=tenant_id,
        candidates=candidates,
        candidate_keys=candidate_keys,
    )
    remaining = candidates - referenced
    if remaining:
        referenced.update(_get_normalized_reference_ids(session, tenant_id=tenant_id, candidates=remaining))
    remaining = candidates - referenced
    if remaining:
        referenced.update(
            _get_workflow_payload_reference_ids(
                session,
                tenant_id=tenant_id,
                candidates=remaining,
                candidate_keys=candidate_keys,
            )
        )
    remaining = candidates - referenced
    if remaining:
        referenced.update(
            _get_persisted_payload_reference_ids(
                session,
                tenant_id=tenant_id,
                candidates=remaining,
                candidate_keys=candidate_keys,
            )
        )
    remaining = candidates - referenced
    if remaining:
        referenced.update(_get_workspace_config_reference_ids(session, tenant_id=tenant_id, candidates=remaining))
    remaining = candidates - referenced
    deferred_ids = remaining if remaining and _has_live_opaque_workflow_payload(session, tenant_id=tenant_id) else set()
    return _UploadFileLiveness(frozenset(referenced), frozenset(deferred_ids))


def _get_upload_file_liveness_or_defer(
    session: Session,
    *,
    tenant_id: str,
    candidate_ids: Sequence[str],
) -> _UploadFileLiveness:
    """Run owner current-reads without letting a reverse-order owner lock block cleanup.

    Legacy knowledge writers may already hold an owner row while waiting for
    UploadFile. Cleanup owns UploadFile first, so its knowledge owner reads use
    NOWAIT. A lock conflict is not evidence that a file is unreferenced: roll
    back only the savepoint and conservatively defer every candidate. Other
    database failures remain fatal.
    """
    candidates = frozenset(str(file_id) for file_id in candidate_ids if file_id)
    if not candidates:
        return _UploadFileLiveness(frozenset(), frozenset())
    try:
        with session.begin_nested():
            return _get_upload_file_liveness(
                session,
                tenant_id=tenant_id,
                candidate_ids=tuple(candidates),
            )
    except OperationalError as error:
        if not _is_owner_lock_not_available(error):
            raise
        return _UploadFileLiveness(frozenset(), candidates)


def get_upload_file_reference_ids(
    session: Session,
    *,
    tenant_id: str,
    candidate_ids: Sequence[str],
) -> set[str]:
    """Return candidates that cannot currently be proven safe to delete."""
    return _get_upload_file_liveness_or_defer(
        session,
        tenant_id=tenant_id,
        candidate_ids=candidate_ids,
    ).retained_ids


def lock_and_validate_owned_upload_files_in_session(
    session: Session,
    *,
    tenant_id: str,
    upload_file_ids: Sequence[str],
) -> dict[str, UploadFile]:
    """Lock every tenant-owned file before adding a durable reference to it.

    Cleanup takes an exclusive UploadFile lock before inspecting owner rows.
    Writers take this shared lock first and retain it through their owner-row
    commit, so cleanup observes either the new owner or the file's removal.
    """
    requested_ids = tuple(sorted(set(upload_file_ids)))
    if not requested_ids:
        return {}
    uploads = session.scalars(
        select(UploadFile)
        .where(
            UploadFile.tenant_id == tenant_id,
            UploadFile.id.in_(requested_ids),
        )
        .order_by(UploadFile.id)
        .with_for_update(read=True)
        .execution_options(populate_existing=True)
    ).all()
    uploads_by_id = {upload.id: upload for upload in uploads}
    missing_ids = set(requested_ids) - uploads_by_id.keys()
    if missing_ids:
        raise UploadFileReferenceNotFoundError(missing_ids)
    return uploads_by_id


def register_upload_file_cleanup_candidates_in_session(
    session: Session,
    *,
    workspace_id: str,
    file_ids: Sequence[str],
) -> None:
    """Persist tenant-owned candidates without checking liveness or deleting rows.

    Resource owners that already hold their own row locks use this entry point
    to avoid reversing the file-domain lock order. The periodic drainer later
    locks UploadFile first, checks every owner, and promotes only definitely
    unreferenced candidates to blob cleanup.
    """
    candidate_ids = sorted({file_id for file_id in file_ids if file_id})
    if not candidate_ids:
        return
    uploads = session.scalars(
        select(UploadFile)
        .where(
            UploadFile.tenant_id == workspace_id,
            UploadFile.id.in_(candidate_ids),
        )
        .order_by(UploadFile.id)
    ).all()
    if not uploads:
        return
    register_created_upload_artifacts_for_cleanup_in_session(
        session,
        workspace_id=workspace_id,
        artifacts=tuple(CreatedUploadArtifact(id=upload.id, key=upload.key) for upload in uploads),
    )


def register_created_upload_artifacts_for_cleanup_in_session(
    session: Session,
    *,
    workspace_id: str,
    artifacts: Sequence[CreatedUploadArtifact],
) -> None:
    """Persist blob-cleanup evidence even when the UploadFile commit outcome is unknown."""
    artifacts_by_id = {artifact.id: artifact for artifact in artifacts if artifact.id and artifact.key}
    if not artifacts_by_id:
        return
    pending = {
        row.upload_file_id: row
        for row in session.scalars(
            select(UploadFileCleanupOutbox).where(
                UploadFileCleanupOutbox.tenant_id == workspace_id,
                UploadFileCleanupOutbox.upload_file_id.in_(artifacts_by_id),
            )
        )
    }
    for artifact in artifacts_by_id.values():
        existing = pending.get(artifact.id)
        if existing is not None:
            if existing.key != artifact.key:
                raise RuntimeError(f"Upload cleanup artifact key mismatch for {artifact.id}")
            continue
        cleanup = UploadFileCleanupOutbox(
            tenant_id=workspace_id,
            upload_file_id=artifact.id,
            key=artifact.key,
        )
        if session.get_bind().dialect.name == "sqlite":
            # Releasing SQLite's first savepoint can escape a later outer
            # rollback when no write transaction preceded it. Keep local/test
            # registration in the caller transaction; its next query/commit
            # flushes the unique row normally.
            session.add(cleanup)
            pending[artifact.id] = cleanup
            continue
        try:
            with session.begin_nested():
                session.add(cleanup)
        except IntegrityError:
            # Another resource transaction registered the same global file.
            # The savepoint preserves this caller's unrelated state.
            continue


def begin_created_upload_artifact_write_in_session(
    session: Session,
    *,
    workspace_id: str,
    artifact: CreatedUploadArtifact,
    writer_id: str,
) -> None:
    """Persist exclusive ownership before an external object write starts."""
    existing = session.scalar(
        select(UploadFileCleanupOutbox)
        .where(
            UploadFileCleanupOutbox.tenant_id == workspace_id,
            UploadFileCleanupOutbox.upload_file_id == artifact.id,
        )
        .with_for_update()
    )
    if existing is not None:
        raise RuntimeError(f"Upload artifact {artifact.id} already has cleanup ownership")
    session.add(
        UploadFileCleanupOutbox(
            tenant_id=workspace_id,
            upload_file_id=artifact.id,
            key=artifact.key,
            writer_id=writer_id,
            writer_expires_at=naive_utc_now() + timedelta(seconds=_UPLOAD_WRITER_LEASE_SECONDS),
        )
    )
    session.flush()


def finish_created_upload_artifact_writes_in_session(
    session: Session,
    *,
    workspace_id: str,
    writer_ids: Mapping[str, str],
) -> None:
    """Fence cleanup and record successful PUTs in the caller's SQL transaction."""
    if not writer_ids:
        return
    for upload_file_id, writer_id in sorted(writer_ids.items()):
        result = session.execute(
            update(UploadFileCleanupOutbox)
            .where(
                UploadFileCleanupOutbox.tenant_id == workspace_id,
                UploadFileCleanupOutbox.upload_file_id == upload_file_id,
                UploadFileCleanupOutbox.writer_id == writer_id,
                UploadFileCleanupOutbox.writer_finished_at.is_(None),
                UploadFileCleanupOutbox.cleanup_started_at.is_(None),
            )
            .values(
                writer_finished_at=func.current_timestamp(),
                writer_expires_at=naive_utc_now() + timedelta(seconds=_UPLOAD_WRITER_LEASE_SECONDS),
            )
        )
        if cast(CursorResult[object], result).rowcount != 1:
            raise UploadFilePublicationLostError(f"Upload artifact writer marker was lost: {upload_file_id}")


def renew_created_upload_artifact_writes_in_session(
    session: Session,
    *,
    workspace_id: str,
    upload_file_ids: Sequence[str],
) -> None:
    """Extend active writer ownership without reviving cleanup-fenced rows."""
    candidate_ids = sorted({file_id for file_id in upload_file_ids if file_id})
    if not candidate_ids:
        return
    result = session.execute(
        update(UploadFileCleanupOutbox)
        .where(
            UploadFileCleanupOutbox.tenant_id == workspace_id,
            UploadFileCleanupOutbox.upload_file_id.in_(candidate_ids),
            UploadFileCleanupOutbox.writer_id.is_not(None),
            UploadFileCleanupOutbox.cleanup_started_at.is_(None),
        )
        .values(writer_expires_at=naive_utc_now() + timedelta(seconds=_UPLOAD_WRITER_LEASE_SECONDS))
    )
    if cast(CursorResult[object], result).rowcount != len(candidate_ids):
        raise UploadFilePublicationLostError("One or more upload artifact writer leases were lost")


def abandon_created_upload_artifact_writes_in_session(
    session: Session,
    *,
    workspace_id: str,
    upload_file_ids: Sequence[str],
) -> None:
    """Make stopped writers immediately eligible for cleanup takeover."""
    candidate_ids = sorted({file_id for file_id in upload_file_ids if file_id})
    if not candidate_ids:
        return
    session.execute(
        update(UploadFileCleanupOutbox)
        .where(
            UploadFileCleanupOutbox.tenant_id == workspace_id,
            UploadFileCleanupOutbox.upload_file_id.in_(candidate_ids),
            UploadFileCleanupOutbox.cleanup_started_at.is_(None),
        )
        .values(writer_expires_at=naive_utc_now())
    )


def lock_created_upload_artifact_publication_in_session(
    session: Session,
    *,
    workspace_id: str,
    upload_file_ids: Sequence[str],
) -> None:
    """Lock and validate every extractor-created object before binding publication."""
    candidate_ids = sorted({file_id for file_id in upload_file_ids if file_id})
    if not candidate_ids:
        return
    rows = session.scalars(
        select(UploadFileCleanupOutbox)
        .where(
            UploadFileCleanupOutbox.tenant_id == workspace_id,
            UploadFileCleanupOutbox.upload_file_id.in_(candidate_ids),
        )
        .order_by(UploadFileCleanupOutbox.upload_file_id)
        .with_for_update()
    ).all()
    rows_by_id = {row.upload_file_id: row for row in rows}
    for upload_file_id in candidate_ids:
        row = rows_by_id.get(upload_file_id)
        if row is None or row.writer_id is None or row.writer_finished_at is None or row.cleanup_started_at is not None:
            raise UploadFilePublicationLostError(f"Upload artifact is no longer publishable: {upload_file_id}")


def complete_created_upload_artifact_publication_in_session(
    session: Session,
    *,
    workspace_id: str,
    upload_file_ids: Sequence[str],
) -> None:
    """Consume writer markers only for files now owned by durable references."""
    candidate_ids = sorted({file_id for file_id in upload_file_ids if file_id})
    if not candidate_ids:
        return
    referenced_ids = get_upload_file_reference_ids(
        session,
        tenant_id=workspace_id,
        candidate_ids=candidate_ids,
    )
    if referenced_ids:
        session.execute(
            delete(UploadFileCleanupOutbox).where(
                UploadFileCleanupOutbox.tenant_id == workspace_id,
                UploadFileCleanupOutbox.upload_file_id.in_(referenced_ids),
                UploadFileCleanupOutbox.writer_id.is_not(None),
                UploadFileCleanupOutbox.writer_finished_at.is_not(None),
                UploadFileCleanupOutbox.cleanup_started_at.is_(None),
            )
        )


def publish_created_upload_file_in_session(
    session: Session,
    *,
    workspace_id: str,
    artifact: CreatedUploadArtifact,
    writer_id: str,
) -> None:
    """Transfer a completed object into its public UploadFile record atomically."""
    result = session.execute(
        delete(UploadFileCleanupOutbox).where(
            UploadFileCleanupOutbox.tenant_id == workspace_id,
            UploadFileCleanupOutbox.upload_file_id == artifact.id,
            UploadFileCleanupOutbox.key == artifact.key,
            UploadFileCleanupOutbox.writer_id == writer_id,
            UploadFileCleanupOutbox.cleanup_started_at.is_(None),
        )
    )
    if cast(CursorResult[object], result).rowcount != 1:
        raise UploadFilePublicationLostError(f"Upload artifact writer marker was lost: {artifact.id}")


def claim_upload_files_for_cleanup_in_session(
    session: Session,
    *,
    workspace_id: str,
    file_ids: Sequence[str],
) -> tuple[UploadFileCleanupArtifact, ...]:
    """Claim candidates under the global UploadFile-then-outbox lock order."""
    candidate_ids = sorted({file_id for file_id in file_ids if file_id})
    if not candidate_ids:
        return ()
    uploads = session.scalars(
        select(UploadFile)
        .where(UploadFile.tenant_id == workspace_id, UploadFile.id.in_(candidate_ids))
        .order_by(UploadFile.id)
        .with_for_update()
    ).all()
    pending = session.scalars(
        select(UploadFileCleanupOutbox)
        .where(
            UploadFileCleanupOutbox.tenant_id == workspace_id,
            UploadFileCleanupOutbox.upload_file_id.in_(candidate_ids),
        )
        .order_by(UploadFileCleanupOutbox.upload_file_id, UploadFileCleanupOutbox.id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).all()
    pending_by_upload_id = {artifact.upload_file_id: artifact for artifact in pending}
    liveness = _get_upload_file_liveness_or_defer(
        session,
        tenant_id=workspace_id,
        candidate_ids=candidate_ids,
    )
    database_now = naive_utc_now()
    active_writer_ids = {
        artifact.upload_file_id
        for artifact in pending
        if artifact.writer_id is not None
        and artifact.cleanup_started_at is None
        and artifact.writer_expires_at is not None
        and artifact.writer_expires_at > database_now
    }
    eligible_ids = set(candidate_ids) - liveness.retained_ids - active_writer_ids
    deferred_ids = set(liveness.deferred_ids)
    claimed_uploads = [upload for upload in uploads if upload.id in eligible_ids]
    intent_uploads = [upload for upload in uploads if upload.id in eligible_ids or upload.id in deferred_ids]
    for upload in intent_uploads:
        if upload.id in pending_by_upload_id:
            continue
        pending_artifact = UploadFileCleanupOutbox(
            tenant_id=workspace_id,
            upload_file_id=upload.id,
            key=upload.key,
        )
        session.add(pending_artifact)
        pending_by_upload_id[upload.id] = pending_artifact
    if claimed_uploads:
        session.flush()
        session.execute(
            delete(UploadFile).where(
                UploadFile.tenant_id == workspace_id,
                UploadFile.id.in_([upload.id for upload in claimed_uploads]),
            )
        )
    cleanup_started_at = naive_utc_now()
    for file_id in eligible_ids:
        pending_artifact = pending_by_upload_id.get(file_id)
        if pending_artifact is not None and pending_artifact.cleanup_started_at is None:
            pending_artifact.cleanup_started_at = cleanup_started_at
    return tuple(
        _to_cleanup_artifact(pending_by_upload_id[file_id])
        for file_id in candidate_ids
        if file_id in eligible_ids and file_id in pending_by_upload_id
    )


class SQLAlchemyUploadFileCleanupRepository:
    """File-domain persistence for durable UploadFile blob cleanup."""

    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def claim_unreferenced_files_for_cleanup(
        self,
        *,
        workspace_id: str,
        file_ids: Sequence[str],
    ) -> tuple[UploadFileCleanupArtifact, ...]:
        with self._session_factory.begin() as session:
            return self.claim_unreferenced_files_for_cleanup_in_session(
                session,
                workspace_id=workspace_id,
                file_ids=file_ids,
            )

    def register_created_upload_artifacts_for_cleanup(
        self,
        *,
        workspace_id: str,
        artifacts: Sequence[CreatedUploadArtifact],
    ) -> None:
        """Commit cleanup evidence independently of the failed writer session."""
        with self._session_factory.begin() as session:
            register_created_upload_artifacts_for_cleanup_in_session(
                session,
                workspace_id=workspace_id,
                artifacts=artifacts,
            )

    def begin_created_upload_artifact_write(
        self,
        *,
        workspace_id: str,
        artifact: CreatedUploadArtifact,
    ) -> str:
        """Commit an unfinished writer marker before storage I/O begins."""
        writer_id = str(uuid4())
        with self._session_factory.begin() as session:
            begin_created_upload_artifact_write_in_session(
                session,
                workspace_id=workspace_id,
                artifact=artifact,
                writer_id=writer_id,
            )
        return writer_id

    def finish_created_upload_artifact_writes_in_session(
        self,
        session: Session,
        *,
        workspace_id: str,
        writer_ids: Mapping[str, str],
    ) -> None:
        finish_created_upload_artifact_writes_in_session(
            session,
            workspace_id=workspace_id,
            writer_ids=writer_ids,
        )

    def finish_created_upload_artifact_write(
        self,
        *,
        workspace_id: str,
        upload_file_id: str,
        writer_id: str,
    ) -> None:
        """Resolve a known-complete PUT after its publishing transaction failed."""
        with self._session_factory.begin() as session:
            finish_created_upload_artifact_writes_in_session(
                session,
                workspace_id=workspace_id,
                writer_ids={upload_file_id: writer_id},
            )

    def renew_created_upload_artifact_writes(
        self,
        *,
        workspace_id: str,
        upload_file_ids: Sequence[str],
    ) -> None:
        with self._session_factory.begin() as session:
            renew_created_upload_artifact_writes_in_session(
                session,
                workspace_id=workspace_id,
                upload_file_ids=upload_file_ids,
            )

    def abandon_created_upload_artifact_writes(
        self,
        *,
        workspace_id: str,
        upload_file_ids: Sequence[str],
    ) -> None:
        with self._session_factory.begin() as session:
            abandon_created_upload_artifact_writes_in_session(
                session,
                workspace_id=workspace_id,
                upload_file_ids=upload_file_ids,
            )

    def publish_created_upload_file_in_session(
        self,
        session: Session,
        *,
        workspace_id: str,
        artifact: CreatedUploadArtifact,
        writer_id: str,
    ) -> None:
        publish_created_upload_file_in_session(
            session,
            workspace_id=workspace_id,
            artifact=artifact,
            writer_id=writer_id,
        )

    def claim_unreferenced_files_for_cleanup_in_session(
        self,
        session: Session,
        *,
        workspace_id: str,
        file_ids: Sequence[str],
    ) -> tuple[UploadFileCleanupArtifact, ...]:
        return claim_upload_files_for_cleanup_in_session(
            session,
            workspace_id=workspace_id,
            file_ids=file_ids,
        )

    def list_pending_cleanup_artifacts(self, *, limit: int) -> tuple[UploadFileCleanupArtifact, ...]:
        """Claim a fair retry batch after current-read liveness validation.

        Discovery never holds queue locks. The claim transaction takes every
        existing UploadFile lock before any cleanup-outbox lock, then rechecks
        writer leases and all durable owners. Missing UploadFile rows remain
        claimable because the outbox key is the durable blob-cleanup evidence.
        """
        if limit <= 0:
            return ()

        # Candidate discovery is deliberately non-locking and ends before the
        # claim transaction starts. Besides keeping the queue scan short, the
        # separate Session prevents this REPEATABLE READ snapshot from hiding
        # an owner row that commits while the drainer waits for UploadFile X.
        with self._session_factory() as candidate_session:
            candidates = candidate_session.execute(
                select(
                    UploadFileCleanupOutbox.id,
                    UploadFileCleanupOutbox.tenant_id,
                    UploadFileCleanupOutbox.upload_file_id,
                    UploadFile.id.label("existing_upload_file_id"),
                )
                .outerjoin(
                    UploadFile,
                    and_(
                        UploadFile.tenant_id == UploadFileCleanupOutbox.tenant_id,
                        UploadFile.id == UploadFileCleanupOutbox.upload_file_id,
                    ),
                )
                .order_by(
                    case((UploadFileCleanupOutbox.last_attempted_at.is_(None), 0), else_=1),
                    UploadFileCleanupOutbox.last_attempted_at,
                    UploadFileCleanupOutbox.created_at,
                    UploadFileCleanupOutbox.id,
                )
                .limit(max(limit, 100))
            ).all()
        if not candidates:
            return ()

        candidate_cleanup_ids = tuple(row.id for row in candidates)
        upload_lock_hints: dict[str, list[str]] = {}
        for row in candidates:
            if row.existing_upload_file_id is not None:
                upload_lock_hints.setdefault(row.tenant_id, []).append(row.upload_file_id)

        with self._session_factory.begin() as session:
            # Every cleanup path takes UploadFile before its outbox row. Lock
            # only rows observed by discovery: asking FOR UPDATE for a missing
            # UUID can take an InnoDB gap lock and deadlock an unfinished writer
            # that already owns the outbox row but has not inserted UploadFile.
            locked_uploads: set[tuple[str, str]] = set()
            for tenant_id in sorted(upload_lock_hints):
                upload_file_ids = sorted(set(upload_lock_hints[tenant_id]))
                uploads = session.scalars(
                    select(UploadFile)
                    .where(
                        UploadFile.tenant_id == tenant_id,
                        UploadFile.id.in_(upload_file_ids),
                    )
                    .order_by(UploadFile.id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                ).all()
                locked_uploads.update((tenant_id, upload.id) for upload in uploads)

            # Lock one intent at a time in owner order. An IN query ordered by
            # non-PK columns may still acquire InnoDB row locks in its chosen
            # index scan order before sorting, which is not a lock-order
            # guarantee for two overlapping drainer batches.
            current_pending: list[UploadFileCleanupOutbox] = []
            for candidate in sorted(
                candidates,
                key=lambda row: (row.tenant_id, row.upload_file_id, row.id),
            ):
                artifact = session.scalar(
                    select(UploadFileCleanupOutbox)
                    .where(
                        UploadFileCleanupOutbox.id == candidate.id,
                        UploadFileCleanupOutbox.tenant_id == candidate.tenant_id,
                        UploadFileCleanupOutbox.upload_file_id == candidate.upload_file_id,
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                if artifact is not None:
                    current_pending.append(artifact)
            pending_by_id = {artifact.id: artifact for artifact in current_pending}
            # Preserve discovery priority after acquiring locks in the global
            # stable order. Rows completed by another worker simply disappear.
            pending = tuple(
                artifact
                for cleanup_id in candidate_cleanup_ids
                if (artifact := pending_by_id.get(cleanup_id)) is not None
            )
            retained: set[tuple[str, str]] = set()
            database_now = naive_utc_now()
            active_writers = {
                (artifact.tenant_id, artifact.upload_file_id)
                for artifact in pending
                if artifact.writer_id is not None
                and artifact.cleanup_started_at is None
                and artifact.writer_expires_at is not None
                and artifact.writer_expires_at > database_now
            }
            tenant_candidates: dict[str, list[str]] = {}
            for artifact in pending:
                tenant_candidates.setdefault(artifact.tenant_id, []).append(artifact.upload_file_id)
            for tenant_id in sorted(tenant_candidates):
                candidate_ids = sorted(set(tenant_candidates[tenant_id]))
                liveness = _get_upload_file_liveness_or_defer(
                    session,
                    tenant_id=tenant_id,
                    candidate_ids=candidate_ids,
                )
                retained.update((tenant_id, upload_file_id) for upload_file_id in liveness.retained_ids)
            retained_cleanup_ids = [
                artifact.id
                for artifact in pending
                if (artifact.tenant_id, artifact.upload_file_id) in retained
                or (artifact.tenant_id, artifact.upload_file_id) in active_writers
            ]
            if retained_cleanup_ids:
                session.execute(
                    update(UploadFileCleanupOutbox)
                    .where(UploadFileCleanupOutbox.id.in_(retained_cleanup_ids))
                    .values(last_attempted_at=func.current_timestamp())
                )
            claimable = tuple(
                artifact
                for artifact in pending
                if (artifact.tenant_id, artifact.upload_file_id) not in retained
                and (artifact.tenant_id, artifact.upload_file_id) not in active_writers
            )[:limit]
            claimable_by_tenant: dict[str, list[str]] = {}
            for artifact in claimable:
                if (artifact.tenant_id, artifact.upload_file_id) in locked_uploads:
                    claimable_by_tenant.setdefault(artifact.tenant_id, []).append(artifact.upload_file_id)
            for tenant_id in sorted(claimable_by_tenant):
                upload_file_ids = sorted(set(claimable_by_tenant[tenant_id]))
                session.execute(
                    delete(UploadFile).where(
                        UploadFile.tenant_id == tenant_id,
                        UploadFile.id.in_(upload_file_ids),
                    )
                )
            cleanup_started_at = naive_utc_now()
            for artifact in claimable:
                if artifact.cleanup_started_at is None:
                    artifact.cleanup_started_at = cleanup_started_at
            return tuple(_to_cleanup_artifact(artifact) for artifact in claimable)

    def cleanup_artifact_is_referenced(self, *, tenant_id: str, upload_file_id: str) -> bool:
        """Recheck liveness immediately before an external blob deletion."""
        with self._session_factory() as session:
            return upload_file_id in get_upload_file_reference_ids(
                session,
                tenant_id=tenant_id,
                candidate_ids=(upload_file_id,),
            )

    def mark_cleanup_attempted(self, *, tenant_id: str, cleanup_id: str) -> None:
        with self._session_factory.begin() as session:
            session.execute(
                update(UploadFileCleanupOutbox)
                .where(
                    UploadFileCleanupOutbox.id == cleanup_id,
                    UploadFileCleanupOutbox.tenant_id == tenant_id,
                )
                .values(last_attempted_at=func.current_timestamp())
            )

    def complete_cleanup(self, *, tenant_id: str, cleanup_id: str) -> None:
        with self._session_factory.begin() as session:
            session.execute(
                delete(UploadFileCleanupOutbox).where(
                    UploadFileCleanupOutbox.id == cleanup_id,
                    UploadFileCleanupOutbox.tenant_id == tenant_id,
                    or_(
                        UploadFileCleanupOutbox.writer_id.is_(None),
                        UploadFileCleanupOutbox.writer_finished_at.is_not(None),
                    ),
                )
            )

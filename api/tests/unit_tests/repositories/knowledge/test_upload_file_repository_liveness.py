import json
import uuid
from collections.abc import Sequence

import pytest
from pymysql.err import OperationalError as PyMySQLOperationalError
from sqlalchemy import Engine, event, select
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import ORMExecuteState, Session, sessionmaker
from sqlalchemy.sql import ClauseElement, Executable

import repositories.upload_file_repository as upload_repository
from core.workflow.file_reference import build_file_reference
from models.dataset import ChildChunk, Document, DocumentSegment, SegmentAttachmentBinding
from models.enums import DataSourceType
from models.model import UploadFile, UploadFileCleanupOutbox
from tests.unit_tests.model_factories import make_dataset, make_document, make_upload_file


def _new_id() -> str:
    return str(uuid.uuid4())


def _segment(
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    content: str,
    answer: str | None = None,
    position: int = 1,
) -> DocumentSegment:
    return DocumentSegment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        position=position,
        content=content,
        answer=answer,
        word_count=1,
        tokens=1,
        created_by=_new_id(),
    )


def _child(*, segment: DocumentSegment, content: str, position: int = 1) -> ChildChunk:
    return ChildChunk(
        tenant_id=segment.tenant_id,
        dataset_id=segment.dataset_id,
        document_id=segment.document_id,
        segment_id=segment.id,
        position=position,
        content=content,
        word_count=1,
        created_by=_new_id(),
    )


def _preview(file_id: str, *, relative: bool = False) -> str:
    prefix = "" if relative else "https://files.example"
    return f"![diagram]({prefix}/files/{file_id}/file-preview)"


def test_knowledge_owner_current_reads_compile_to_cross_database_nowait(sqlite_session: Session) -> None:
    statements: list[Executable] = []

    def record_statement(state: ORMExecuteState) -> None:
        statements.append(state.statement)

    event.listen(sqlite_session, "do_orm_execute", record_statement)
    try:
        assert (
            upload_repository.get_upload_file_reference_ids(
                sqlite_session,
                tenant_id=_new_id(),
                candidate_ids=(_new_id(),),
            )
            == set()
        )
    finally:
        event.remove(sqlite_session, "do_orm_execute", record_statement)

    owner_tables = {
        SegmentAttachmentBinding.__tablename__,
        DocumentSegment.__tablename__,
        ChildChunk.__tablename__,
        Document.__tablename__,
    }
    owner_statements = [statement for statement in statements if owner_tables.intersection(str(statement).split())]
    assert len(owner_statements) == 4
    for statement in owner_statements:
        assert isinstance(statement, ClauseElement)
        mysql_sql = str(statement.compile(dialect=mysql.dialect())).upper()
        postgres_sql = str(statement.compile(dialect=postgresql.dialect())).upper()
        assert mysql_sql.endswith(" FOR UPDATE NOWAIT")
        assert postgres_sql.endswith(" FOR UPDATE NOWAIT")


def test_direct_cleanup_nowait_conflict_defers_within_outer_transaction(
    sqlite_engine: Engine,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    tenant_id = _new_id()
    upload = make_upload_file(file_id=_new_id(), tenant_id=tenant_id, key="uploads/deferred")
    with sqlite_session_factory.begin() as session:
        session.add(upload)

    raised = False

    def fail_owner_current_read(
        _connection,
        _cursor,
        statement: str,
        _parameters: object,
        _context,
        _executemany: bool,
    ) -> None:
        nonlocal raised
        if not raised and f"FROM {DocumentSegment.__tablename__}" in statement:
            raised = True
            raise OperationalError(
                statement,
                None,
                PyMySQLOperationalError(3572, "NOWAIT owner lock unavailable"),
            )

    event.listen(sqlite_engine, "before_cursor_execute", fail_owner_current_read)
    try:
        with sqlite_session_factory.begin() as session:
            assert (
                upload_repository.claim_upload_files_for_cleanup_in_session(
                    session,
                    workspace_id=tenant_id,
                    file_ids=(upload.id,),
                )
                == ()
            )
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", fail_owner_current_read)

    assert raised
    with sqlite_session_factory() as session:
        assert session.get(UploadFile, upload.id) is not None
        marker = session.scalar(
            select(UploadFileCleanupOutbox).where(
                UploadFileCleanupOutbox.tenant_id == tenant_id,
                UploadFileCleanupOutbox.upload_file_id == upload.id,
            )
        )
        assert marker is not None
        assert marker.cleanup_started_at is None


def test_reference_lookup_preserves_all_supported_liveness_sources(sqlite_session: Session) -> None:
    tenant_id = _new_id()
    foreign_tenant_id = _new_id()
    dataset_id = _new_id()
    foreign_dataset_id = _new_id()
    document_id = _new_id()
    foreign_document_id = _new_id()
    candidate_names = (
        "binding",
        "segment_content",
        "segment_answer",
        "child",
        "upload_source",
        "local_real",
        "local_canonical",
        "local_reference",
        "local_related",
        "plain_text",
        "wrong_source_type",
        "source_false_positive",
        "foreign_only",
        "unreferenced",
    )
    uploads = {
        name: make_upload_file(file_id=_new_id(), tenant_id=tenant_id, key=f"uploads/{name}")
        for name in candidate_names
    }
    dataset = make_dataset(dataset_id=dataset_id, tenant_id=tenant_id, data_source_type=DataSourceType.UPLOAD_FILE)
    foreign_dataset = make_dataset(
        dataset_id=foreign_dataset_id,
        tenant_id=foreign_tenant_id,
        data_source_type=DataSourceType.UPLOAD_FILE,
    )
    document = make_document(document_id=document_id, tenant_id=tenant_id, dataset_id=dataset_id)
    foreign_document = make_document(
        document_id=foreign_document_id,
        tenant_id=foreign_tenant_id,
        dataset_id=foreign_dataset_id,
    )
    content_segment = _segment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        content=_preview(uploads["segment_content"].id),
        answer=_preview(uploads["segment_answer"].id, relative=True),
    )
    child_segment = _segment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        content="parent text",
        position=2,
    )
    binding_segment = _segment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        content="bound attachment",
        position=3,
    )
    false_positive_segment = _segment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        content=f"plain /files/{uploads['plain_text'].id}/file-preview text",
        position=4,
    )
    foreign_segment = _segment(
        tenant_id=foreign_tenant_id,
        dataset_id=foreign_dataset_id,
        document_id=foreign_document_id,
        content=_preview(uploads["foreign_only"].id),
    )
    child = _child(segment=child_segment, content=_preview(uploads["child"].id, relative=True))
    binding = SegmentAttachmentBinding(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        segment_id=binding_segment.id,
        attachment_id=uploads["binding"].id,
    )
    source_documents = [
        make_document(
            document_id=_new_id(),
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=position,
            data_source_type=data_source_type,
            data_source_info=json.dumps(data_source_info),
        )
        for position, data_source_type, data_source_info in (
            (2, DataSourceType.UPLOAD_FILE, {"upload_file_id": uploads["upload_source"].id}),
            (3, DataSourceType.LOCAL_FILE, {"real_file_id": uploads["local_real"].id}),
            (
                4,
                DataSourceType.LOCAL_FILE,
                {
                    "reference": build_file_reference(
                        record_id=uploads["local_canonical"].id,
                        storage_key="uploads/canonical",
                    )
                },
            ),
            (5, DataSourceType.LOCAL_FILE, {"reference": uploads["local_reference"].id}),
            (6, DataSourceType.LOCAL_FILE, {"related_id": uploads["local_related"].id}),
            (7, DataSourceType.WEBSITE_CRAWL, {"upload_file_id": uploads["wrong_source_type"].id}),
            (
                8,
                DataSourceType.UPLOAD_FILE,
                {
                    "upload_file_id": _new_id(),
                    "name": uploads["source_false_positive"].id,
                },
            ),
        )
    ]
    foreign_source_document = make_document(
        document_id=_new_id(),
        tenant_id=foreign_tenant_id,
        dataset_id=foreign_dataset_id,
        position=2,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info=json.dumps({"upload_file_id": uploads["foreign_only"].id}),
    )
    sqlite_session.add_all(
        [
            dataset,
            foreign_dataset,
            document,
            foreign_document,
            *uploads.values(),
            content_segment,
            child_segment,
            binding_segment,
            false_positive_segment,
            foreign_segment,
            child,
            binding,
            *source_documents,
            foreign_source_document,
        ]
    )
    sqlite_session.commit()

    referenced = upload_repository.get_upload_file_reference_ids(
        sqlite_session,
        tenant_id=tenant_id,
        candidate_ids=[upload.id for upload in uploads.values()],
    )

    assert referenced == {
        uploads[name].id
        for name in (
            "binding",
            "segment_content",
            "segment_answer",
            "child",
            "upload_source",
            "local_real",
            "local_canonical",
            "local_reference",
            "local_related",
        )
    }


def test_reference_lookup_pushes_candidates_into_sql_before_parsing_rows(
    sqlite_session: Session,
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id = _new_id()
    dataset_id = _new_id()
    document_id = _new_id()
    content_upload = make_upload_file(file_id=_new_id(), tenant_id=tenant_id, key="uploads/content")
    source_upload = make_upload_file(file_id=_new_id(), tenant_id=tenant_id, key="uploads/source")
    missing_upload = make_upload_file(file_id=_new_id(), tenant_id=tenant_id, key="uploads/missing")
    dataset = make_dataset(dataset_id=dataset_id, tenant_id=tenant_id, data_source_type=DataSourceType.UPLOAD_FILE)
    document = make_document(document_id=document_id, tenant_id=tenant_id, dataset_id=dataset_id)
    matching_segment = _segment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        content=_preview(content_upload.id),
    )
    matching_source_info = json.dumps({"upload_file_id": source_upload.id})
    matching_source = make_document(
        document_id=_new_id(),
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        position=2,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info=matching_source_info,
    )

    unrelated_segments: list[DocumentSegment] = []
    unrelated_children: list[ChildChunk] = []
    unrelated_documents = []
    unrelated_ids: list[str] = []
    for position in range(3, 153):
        unrelated_id = _new_id()
        unrelated_ids.append(unrelated_id)
        segment = _segment(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            content=f"unrelated-marker-{position} {_preview(unrelated_id)}",
            position=position,
        )
        unrelated_segments.append(segment)
        unrelated_children.append(
            _child(
                segment=segment,
                content=f"unrelated-child-{position} {_preview(unrelated_id, relative=True)}",
                position=position,
            )
        )
        unrelated_documents.append(
            make_document(
                document_id=_new_id(),
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                position=position + 200,
                data_source_type=DataSourceType.UPLOAD_FILE,
                data_source_info=json.dumps({"upload_file_id": unrelated_id}),
            )
        )

    sqlite_session.add_all(
        [
            dataset,
            document,
            content_upload,
            source_upload,
            missing_upload,
            matching_segment,
            matching_source,
            *unrelated_segments,
            *unrelated_children,
            *unrelated_documents,
        ]
    )
    sqlite_session.commit()

    parsed_markdown: list[str] = []
    parsed_sources: list[str | None] = []
    original_markdown_parser = upload_repository._markdown_upload_file_ids
    original_source_parser = upload_repository._source_upload_file_id

    def track_markdown(content: str) -> set[str]:
        parsed_markdown.append(content)
        return original_markdown_parser(content)

    def track_source(*, data_source_type: str, data_source_info: str | None) -> str | None:
        parsed_sources.append(data_source_info)
        return original_source_parser(data_source_type=data_source_type, data_source_info=data_source_info)

    monkeypatch.setattr(upload_repository, "_markdown_upload_file_ids", track_markdown)
    monkeypatch.setattr(upload_repository, "_source_upload_file_id", track_source)

    executed: list[tuple[str, object]] = []

    def record_statement(
        _connection,
        _cursor,
        statement: str,
        parameters: object,
        _context,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            executed.append((statement, parameters))

    event.listen(sqlite_engine, "before_cursor_execute", record_statement)
    try:
        referenced = upload_repository.get_upload_file_reference_ids(
            sqlite_session,
            tenant_id=tenant_id,
            candidate_ids=(content_upload.id, source_upload.id, missing_upload.id),
        )
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", record_statement)

    assert referenced == {content_upload.id, source_upload.id}
    assert parsed_markdown == [matching_segment.content]
    assert parsed_sources == [matching_source_info]

    reference_queries = [
        (statement, parameters)
        for statement, parameters in executed
        if any(table in statement for table in ("document_segments", "child_chunks", "documents"))
    ]
    assert len(reference_queries) == 3
    for _statement, parameters in reference_queries:
        serialized_parameters = repr(parameters)
        assert any(
            candidate_id in serialized_parameters
            for candidate_id in (content_upload.id, source_upload.id, missing_upload.id)
        )
        assert all(unrelated_id not in serialized_parameters for unrelated_id in unrelated_ids)
        assert "%/files/%" not in serialized_parameters


def test_binding_lookup_batches_large_candidate_sets(sqlite_session: Session, sqlite_engine: Engine) -> None:
    tenant_id = _new_id()
    dataset_id = _new_id()
    document_id = _new_id()
    dataset = make_dataset(dataset_id=dataset_id, tenant_id=tenant_id, data_source_type=DataSourceType.UPLOAD_FILE)
    document = make_document(document_id=document_id, tenant_id=tenant_id, dataset_id=dataset_id)
    segment = _segment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        content="binding owner",
    )
    candidate_ids = [_new_id() for _ in range(205)]
    bindings = [
        SegmentAttachmentBinding(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            segment_id=segment.id,
            attachment_id=candidate_id,
        )
        for candidate_id in candidate_ids
    ]
    sqlite_session.add_all([dataset, document, segment, *bindings])
    sqlite_session.commit()

    binding_query_parameter_counts: list[int] = []

    def record_binding_query(
        _connection,
        _cursor,
        statement: str,
        parameters: Sequence[object],
        _context,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT") and "segment_attachment_bindings" in statement:
            binding_query_parameter_counts.append(len(parameters))

    event.listen(sqlite_engine, "before_cursor_execute", record_binding_query)
    try:
        referenced = upload_repository.get_upload_file_reference_ids(
            sqlite_session,
            tenant_id=tenant_id,
            candidate_ids=candidate_ids,
        )
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", record_binding_query)

    assert referenced == set(candidate_ids)
    assert binding_query_parameter_counts == [101, 101, 6]

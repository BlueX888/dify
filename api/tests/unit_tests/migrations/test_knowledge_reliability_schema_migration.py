from __future__ import annotations

import importlib.util
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector

from models.model import (
    AttachmentVectorCleanupOutbox,
    ChildVectorCleanupOutbox,
    DatasetIndexingOutbox,
    DocumentIndexingOutbox,
    KeywordFileCleanupOutbox,
    ResourceCleanupOutbox,
    SegmentStateOutbox,
    SummaryVectorCleanupOutbox,
    UploadFileCleanupOutbox,
)
from models.source import WebsiteCrawlJob

_MIGRATION_DIR = Path(__file__).resolve().parents[3] / "migrations/versions"
_MIGRATION_PATH = _MIGRATION_DIR / "2026_10_25_0600-f9d6e0c35a48_add_knowledge_reliability_schema.py"
_WORKFLOW_SEQUENCE_MIGRATION_PATH = (
    _MIGRATION_DIR / "2025_06_19_1633-0ab65e1cc7fa_remove_sequence_number_from_workflow_.py"
)
_NEW_TABLE_MODELS = (
    UploadFileCleanupOutbox,
    KeywordFileCleanupOutbox,
    ResourceCleanupOutbox,
    AttachmentVectorCleanupOutbox,
    SummaryVectorCleanupOutbox,
    ChildVectorCleanupOutbox,
    DocumentIndexingOutbox,
    DatasetIndexingOutbox,
    SegmentStateOutbox,
    WebsiteCrawlJob,
)
_ALTERED_COLUMNS = {
    "documents": {
        "indexing_claim_id",
        "indexing_claim_expires_at",
        "indexing_request_id",
        "dataset_indexing_request_id",
        "indexing_dataset_generation_id",
    },
    "workflow_trigger_logs": {
        "execution_owner_id",
        "execution_lease_expires_at",
        "terminal_owner_id",
        "pending_dispatch_abandoned_at",
        "dispatch_prepared_at",
        "dispatch_published_at",
        "dispatch_abandoned_at",
        "quota_reservation_id",
        "quota_settlement_status",
        "quota_settlement_attempted_at",
        "quota_sync_pending",
    },
    "workflow_runs": {
        "history_revision",
        "history_published_revision",
        "history_claim_id",
        "history_claim_expires_at",
        "history_last_attempted_at",
        "history_sync_pending",
    },
    "datasets": {"indexing_request_id", "indexing_applied_representation"},
    "document_pipeline_execution_logs": {"workflow_id", "original_document_id"},
    "document_segments": {"indexing_request_id"},
    "document_segment_summaries": {"version", "vector_route", "vector_route_hash"},
    "child_chunks": {"vector_route", "vector_route_hash"},
}


def _load_migration_module(path: Path = _MIGRATION_PATH) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load migration module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_migration(module: ModuleType, operations: Operations, migration: Callable[[], None]) -> None:
    original_op = module.__dict__["op"]
    module.__dict__["op"] = operations
    try:
        migration()
    finally:
        module.__dict__["op"] = original_op


def _create_pre_migration_schema(engine: Engine) -> None:
    metadata = sa.MetaData()
    for table_name in _ALTERED_COLUMNS:
        columns = [sa.Column("id", sa.String(36), primary_key=True)]
        if table_name == "workflow_trigger_logs":
            columns.append(sa.Column("status", sa.String(50), nullable=False))
        sa.Table(table_name, metadata, *columns)
    metadata.create_all(engine)


def _index_signatures(inspector: Inspector, table_name: str) -> set[tuple[str, tuple[str, ...]]]:
    signatures: set[tuple[str, tuple[str, ...]]] = set()
    for index in inspector.get_indexes(table_name):
        name = index["name"]
        column_names = index["column_names"]
        assert name is not None
        assert all(column_name is not None for column_name in column_names)
        signatures.add((name, tuple(cast(str, column_name) for column_name in column_names)))
    return signatures


def _unique_signatures(inspector: Inspector, table_name: str) -> set[tuple[str, tuple[str, ...]]]:
    signatures: set[tuple[str, tuple[str, ...]]] = set()
    for constraint in inspector.get_unique_constraints(table_name):
        name = constraint["name"]
        assert name is not None
        signatures.add((name, tuple(constraint["column_names"])))
    return signatures


def test_single_squashed_migration_matches_final_models_and_downgrades() -> None:
    late_migrations = sorted(_MIGRATION_DIR.glob("2026_10_2[2-5]_*.py"))
    assert late_migrations == [_MIGRATION_PATH]

    module = _load_migration_module()
    assert module.revision == "f9d6e0c35a48"
    assert module.down_revision == "a8c3e7f4b912"

    engine = sa.create_engine("sqlite:///:memory:")
    _create_pre_migration_schema(engine)
    with engine.begin() as connection:
        connection.execute(
            sa.text("INSERT INTO document_segment_summaries (id) VALUES (:id)"),
            {"id": "summary-1"},
        )
        operations = Operations(MigrationContext.configure(connection))
        _run_migration(module, operations, module.upgrade)

    inspector = sa.inspect(engine)
    for model in _NEW_TABLE_MODELS:
        table = cast(sa.Table, model.__table__)
        reflected_columns = {column["name"]: column for column in inspector.get_columns(table.name)}
        assert set(reflected_columns) == set(table.columns.keys())
        assert {name: column["nullable"] for name, column in reflected_columns.items()} == {
            column.name: column.nullable for column in table.columns
        }
        expected_indexes: set[tuple[str, tuple[str, ...]]] = set()
        for index in table.indexes:
            assert index.name is not None
            expected_indexes.add((index.name, tuple(column.name for column in index.columns)))
        assert _index_signatures(inspector, table.name) == expected_indexes

        expected_uniques: set[tuple[str, tuple[str, ...]]] = set()
        for constraint in table.constraints:
            if isinstance(constraint, sa.UniqueConstraint):
                assert constraint.name is not None
                expected_uniques.add((cast(str, constraint.name), tuple(column.name for column in constraint.columns)))
        assert _unique_signatures(inspector, table.name) == expected_uniques

    for table_name, added_columns in _ALTERED_COLUMNS.items():
        assert added_columns <= {column["name"] for column in inspector.get_columns(table_name)}

    assert _index_signatures(inspector, "documents") >= {
        ("document_indexing_request_idx", ("indexing_request_id",)),
        ("document_dataset_indexing_request_idx", ("dataset_indexing_request_id",)),
        ("document_indexing_dataset_generation_idx", ("indexing_dataset_generation_id",)),
    }
    assert _index_signatures(inspector, "document_segments") >= {
        ("document_segment_indexing_request_idx", ("indexing_request_id",)),
    }
    assert {constraint["name"] for constraint in inspector.get_check_constraints("workflow_trigger_logs")} >= {
        "workflow_trigger_log_pending_dispatch_abandoned_check",
        "workflow_trigger_log_dispatch_receipt_check",
        "workflow_trigger_log_queued_dispatch_abandoned_check",
        "workflow_trigger_log_quota_settlement_status_check",
        "workflow_trigger_log_quota_sync_pending_check",
    }
    assert _index_signatures(inspector, "workflow_trigger_logs") >= {
        (
            "workflow_trigger_log_quota_sync_pending_idx",
            ("quota_sync_pending", "quota_settlement_attempted_at", "dispatch_prepared_at", "id"),
        ),
    }
    assert _index_signatures(inspector, "workflow_runs") >= {
        (
            "workflow_run_history_claim_idx",
            (
                "history_sync_pending",
                "history_claim_expires_at",
                "history_last_attempted_at",
            ),
        ),
    }
    assert {constraint["name"] for constraint in inspector.get_check_constraints("workflow_runs")} >= {
        "workflow_run_history_revision_check",
        "workflow_run_history_claim_check",
        "workflow_run_history_pending_check",
    }
    with engine.connect() as connection:
        assert connection.scalar(sa.text("SELECT version FROM document_segment_summaries WHERE id = 'summary-1'")) == 1

    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        _run_migration(module, operations, module.downgrade)

    inspector = sa.inspect(engine)
    assert not ({model.__tablename__ for model in _NEW_TABLE_MODELS} & set(inspector.get_table_names()))
    for table_name, added_columns in _ALTERED_COLUMNS.items():
        assert not (added_columns & {column["name"] for column in inspector.get_columns(table_name)})


@pytest.mark.parametrize("dialect_name", ["postgresql", "mysql"])
def test_upgrade_and_downgrade_render_offline_sql(dialect_name: str) -> None:
    module = _load_migration_module()
    output = StringIO()
    operations = Operations(
        MigrationContext.configure(
            dialect_name=dialect_name,
            opts={"as_sql": True, "output_buffer": output},
        )
    )

    _run_migration(module, operations, module.upgrade)
    _run_migration(module, operations, module.downgrade)

    generated_sql = output.getvalue().upper()
    assert "CREATE TABLE DATASET_INDEXING_OUTBOX" in generated_sql
    assert "SOURCE_REPRESENTATION" in generated_sql
    assert "TARGET_REPRESENTATION" in generated_sql
    assert "ALTER TABLE DOCUMENT_PIPELINE_EXECUTION_LOGS ADD COLUMN WORKFLOW_ID" in generated_sql
    assert "ALTER TABLE DOCUMENT_PIPELINE_EXECUTION_LOGS ADD COLUMN ORIGINAL_DOCUMENT_ID" in generated_sql
    assert "PIPELINE_EXECUTION_LOG_ID" in generated_sql
    assert "CREATE TABLE ATTACHMENT_VECTOR_CLEANUP_OUTBOX" in generated_sql
    assert "ATTACHMENT_VECTOR_CLEANUP_OUTBOX_NODE_ROUTE_IDX" in generated_sql
    assert "ATTACHMENT_VECTOR_CLEANUP_OUTBOX_RETRY_IDX" in generated_sql
    assert "ATTACHMENT_VECTOR_CLEANUP_OUTBOX_DATASET_WRITER_IDX" in generated_sql
    assert "DROP TABLE ATTACHMENT_VECTOR_CLEANUP_OUTBOX" in generated_sql
    assert "CREATE TABLE WEBSITE_CRAWL_JOBS" in generated_sql
    assert "PROVIDER_INSTANCE_ID" in generated_sql
    assert "DROP TABLE WEBSITE_CRAWL_JOBS" in generated_sql
    assert "WORKFLOW_TRIGGER_LOG_PENDING_DISPATCH_ABANDONED_CHECK" in generated_sql
    assert "STATUS <> 'QUEUED' OR PENDING_DISPATCH_ABANDONED_AT IS NULL" in generated_sql
    assert "WORKFLOW_TRIGGER_LOG_DISPATCH_RECEIPT_CHECK" in generated_sql
    assert "DISPATCH_PUBLISHED_AT IS NULL OR DISPATCH_PREPARED_AT IS NOT NULL" in generated_sql
    assert "DISPATCH_ABANDONED_AT IS NULL OR DISPATCH_PREPARED_AT IS NOT NULL" in generated_sql
    assert "DISPATCH_PUBLISHED_AT IS NULL OR DISPATCH_ABANDONED_AT IS NULL" in generated_sql
    assert "WORKFLOW_TRIGGER_LOG_QUEUED_DISPATCH_ABANDONED_CHECK" in generated_sql
    assert "STATUS <> 'QUEUED' OR DISPATCH_ABANDONED_AT IS NULL" in generated_sql
    assert "WORKFLOW_TRIGGER_LOG_QUOTA_SETTLEMENT_STATUS_CHECK" in generated_sql
    assert "QUOTA_SETTLEMENT_STATUS IN ('COMMITTED', 'REFUNDED', 'PERMANENT_ERROR')" in generated_sql
    assert "WORKFLOW_TRIGGER_LOG_QUOTA_SYNC_PENDING_CHECK" in generated_sql
    assert "QUOTA_SYNC_PENDING = FALSE OR QUOTA_SETTLEMENT_STATUS IS NULL" in generated_sql
    assert "WORKFLOW_TRIGGER_LOG_QUOTA_SYNC_PENDING_IDX" in generated_sql
    assert "WORKFLOW_RUN_HISTORY_CLAIM_IDX" in generated_sql
    assert "WORKFLOW_RUN_HISTORY_REVISION_CHECK" in generated_sql
    assert "HISTORY_PUBLISHED_REVISION <= HISTORY_REVISION" in generated_sql
    assert "WORKFLOW_RUN_HISTORY_CLAIM_CHECK" in generated_sql
    assert "HISTORY_CLAIM_ID IS NULL AND HISTORY_CLAIM_EXPIRES_AT IS NULL" in generated_sql
    assert "WORKFLOW_RUN_HISTORY_PENDING_CHECK" in generated_sql
    assert "HISTORY_PUBLISHED_REVISION = HISTORY_REVISION AND HISTORY_CLAIM_ID IS NULL" in generated_sql
    assert "SELECT " not in generated_sql


def test_legacy_workflow_sequence_downgrade_renders_mysql_column_change() -> None:
    module = _load_migration_module(_WORKFLOW_SEQUENCE_MIGRATION_PATH)
    output = StringIO()
    operations = Operations(
        MigrationContext.configure(
            dialect_name="mysql",
            opts={"as_sql": True, "output_buffer": output},
        )
    )

    _run_migration(module, operations, module.downgrade)

    assert "ALTER TABLE workflow_runs MODIFY sequence_number INTEGER NOT NULL" in output.getvalue()

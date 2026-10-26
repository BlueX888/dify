"""Add the durable knowledge indexing and cleanup schema.

Revision ID: f9d6e0c35a48
Revises: a8c3e7f4b912
Create Date: 2026-10-25 06:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from models.types import LongText, StringUUID

revision: str = "f9d6e0c35a48"
down_revision: str | None = "a8c3e7f4b912"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(sa.Column("indexing_claim_id", StringUUID(), nullable=True))
        batch_op.add_column(sa.Column("indexing_claim_expires_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("indexing_request_id", StringUUID(), nullable=True))
        batch_op.add_column(sa.Column("dataset_indexing_request_id", StringUUID(), nullable=True))
        batch_op.add_column(sa.Column("indexing_dataset_generation_id", StringUUID(), nullable=True))
        batch_op.create_index("document_indexing_request_idx", ["indexing_request_id"], unique=False)
        batch_op.create_index(
            "document_dataset_indexing_request_idx",
            ["dataset_indexing_request_id"],
            unique=False,
        )
        batch_op.create_index(
            "document_indexing_dataset_generation_idx",
            ["indexing_dataset_generation_id"],
            unique=False,
        )

    with op.batch_alter_table("workflow_trigger_logs") as batch_op:
        batch_op.add_column(sa.Column("execution_owner_id", StringUUID(), nullable=True))
        batch_op.add_column(sa.Column("execution_lease_expires_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("terminal_owner_id", StringUUID(), nullable=True))
        batch_op.add_column(sa.Column("pending_dispatch_abandoned_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("dispatch_prepared_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("dispatch_published_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("dispatch_abandoned_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("quota_reservation_id", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("quota_settlement_status", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("quota_settlement_attempted_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("quota_sync_pending", sa.Boolean(), server_default=sa.false(), nullable=False))
        batch_op.create_check_constraint(
            "workflow_trigger_log_pending_dispatch_abandoned_check",
            "status <> 'queued' OR pending_dispatch_abandoned_at IS NULL",
        )
        batch_op.create_check_constraint(
            "workflow_trigger_log_dispatch_receipt_check",
            "(dispatch_published_at IS NULL OR dispatch_prepared_at IS NOT NULL) "
            "AND (dispatch_abandoned_at IS NULL OR dispatch_prepared_at IS NOT NULL) "
            "AND (dispatch_published_at IS NULL OR dispatch_abandoned_at IS NULL)",
        )
        batch_op.create_check_constraint(
            "workflow_trigger_log_queued_dispatch_abandoned_check",
            "status <> 'queued' OR dispatch_abandoned_at IS NULL",
        )
        batch_op.create_check_constraint(
            "workflow_trigger_log_quota_settlement_status_check",
            "quota_settlement_status IS NULL "
            "OR quota_settlement_status IN ('committed', 'refunded', 'permanent_error')",
        )
        batch_op.create_check_constraint(
            "workflow_trigger_log_quota_sync_pending_check",
            "quota_sync_pending = false OR quota_settlement_status IS NULL",
        )
        batch_op.create_index(
            "workflow_trigger_log_quota_sync_pending_idx",
            ["quota_sync_pending", "quota_settlement_attempted_at", "dispatch_prepared_at", "id"],
            unique=False,
        )

    with op.batch_alter_table("workflow_runs") as batch_op:
        batch_op.add_column(sa.Column("history_revision", sa.BigInteger(), server_default=sa.text("0"), nullable=False))
        batch_op.add_column(
            sa.Column("history_published_revision", sa.BigInteger(), server_default=sa.text("0"), nullable=False)
        )
        batch_op.add_column(sa.Column("history_claim_id", StringUUID(), nullable=True))
        batch_op.add_column(sa.Column("history_claim_expires_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("history_last_attempted_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("history_sync_pending", sa.Boolean(), server_default=sa.false(), nullable=False))
        batch_op.create_index(
            "workflow_run_history_claim_idx",
            [
                "history_sync_pending",
                "history_claim_expires_at",
                "history_last_attempted_at",
            ],
            unique=False,
        )
        batch_op.create_check_constraint(
            "workflow_run_history_revision_check",
            "history_revision >= 0 AND history_published_revision >= 0 "
            "AND history_published_revision <= history_revision",
        )
        batch_op.create_check_constraint(
            "workflow_run_history_claim_check",
            "(history_claim_id IS NULL AND history_claim_expires_at IS NULL) "
            "OR (history_claim_id IS NOT NULL AND history_claim_expires_at IS NOT NULL)",
        )
        batch_op.create_check_constraint(
            "workflow_run_history_pending_check",
            "history_sync_pending OR (history_published_revision = history_revision AND history_claim_id IS NULL)",
        )

    with op.batch_alter_table("datasets") as batch_op:
        batch_op.add_column(sa.Column("indexing_request_id", StringUUID(), nullable=True))
        batch_op.add_column(sa.Column("indexing_applied_representation", LongText(), nullable=True))

    with op.batch_alter_table("document_pipeline_execution_logs") as batch_op:
        batch_op.add_column(sa.Column("workflow_id", StringUUID(), nullable=True))
        batch_op.add_column(sa.Column("original_document_id", StringUUID(), nullable=True))

    with op.batch_alter_table("document_segments") as batch_op:
        batch_op.add_column(sa.Column("indexing_request_id", StringUUID(), nullable=True))
        batch_op.create_index("document_segment_indexing_request_idx", ["indexing_request_id"], unique=False)

    with op.batch_alter_table("document_segment_summaries") as batch_op:
        batch_op.add_column(sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False))
        batch_op.add_column(sa.Column("vector_route", LongText(), nullable=True))
        batch_op.add_column(sa.Column("vector_route_hash", sa.String(length=64), nullable=True))

    with op.batch_alter_table("child_chunks") as batch_op:
        batch_op.add_column(sa.Column("vector_route", LongText(), nullable=True))
        batch_op.add_column(sa.Column("vector_route_hash", sa.String(length=64), nullable=True))

    op.create_table(
        "upload_file_cleanup_outbox",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("upload_file_id", StringUUID(), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("writer_id", StringUUID(), nullable=True),
        sa.Column("writer_finished_at", sa.DateTime(), nullable=True),
        sa.Column("writer_expires_at", sa.DateTime(), nullable=True),
        sa.Column("cleanup_started_at", sa.DateTime(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="upload_file_cleanup_outbox_pkey"),
        sa.UniqueConstraint(
            "tenant_id",
            "upload_file_id",
            name="upload_file_cleanup_outbox_tenant_upload_unique",
        ),
    )
    op.create_index(
        "upload_file_cleanup_outbox_retry_idx",
        "upload_file_cleanup_outbox",
        ["last_attempted_at", "created_at", "id"],
        unique=False,
    )

    op.create_table(
        "keyword_file_cleanup_outbox",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("dataset_id", StringUUID(), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("writer_id", StringUUID(), nullable=True),
        sa.Column("writer_finished_at", sa.DateTime(), nullable=True),
        sa.Column("cleanup_started_at", sa.DateTime(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="keyword_file_cleanup_outbox_pkey"),
        sa.UniqueConstraint(
            "tenant_id",
            "dataset_id",
            "key",
            name="keyword_file_cleanup_outbox_owner_key_unique",
        ),
    )
    op.create_index(
        "keyword_file_cleanup_outbox_retry_idx",
        "keyword_file_cleanup_outbox",
        ["last_attempted_at", "created_at", "id"],
        unique=False,
    )

    op.create_table(
        "resource_cleanup_outbox",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("dataset_id", StringUUID(), nullable=False),
        sa.Column("resource_type", sa.String(length=16), nullable=False),
        sa.Column("resource_id", StringUUID(), nullable=False),
        sa.Column("doc_form", sa.String(length=255), nullable=True),
        sa.Column("source_file_id", StringUUID(), nullable=True),
        sa.Column("indexing_technique", sa.String(length=255), nullable=True),
        sa.Column("index_struct", LongText(), nullable=True),
        sa.Column("collection_binding_id", StringUUID(), nullable=True),
        sa.Column("pipeline_id", StringUUID(), nullable=True),
        sa.Column("claim_id", StringUUID(), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="resource_cleanup_outbox_pkey"),
        sa.UniqueConstraint(
            "tenant_id",
            "dataset_id",
            "resource_type",
            "resource_id",
            name="resource_cleanup_outbox_owner_resource_unique",
        ),
    )
    op.create_index(
        "resource_cleanup_outbox_retry_idx",
        "resource_cleanup_outbox",
        ["last_attempted_at", "created_at", "id"],
        unique=False,
    )
    op.create_index(
        "resource_cleanup_outbox_claim_idx",
        "resource_cleanup_outbox",
        ["claim_expires_at"],
        unique=False,
    )

    op.create_table(
        "summary_vector_cleanup_outbox",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("dataset_id", StringUUID(), nullable=False),
        sa.Column("document_id", StringUUID(), nullable=False),
        sa.Column("segment_id", StringUUID(), nullable=False),
        sa.Column("summary_id", StringUUID(), nullable=False),
        sa.Column("summary_version", sa.Integer(), nullable=False),
        sa.Column("node_id", StringUUID(), nullable=False),
        sa.Column("writer_id", StringUUID(), nullable=False),
        sa.Column("vector_route", LongText(), nullable=True),
        sa.Column("vector_route_hash", sa.String(length=64), nullable=True),
        sa.Column("writer_finished_at", sa.DateTime(), nullable=True),
        sa.Column("cleanup_started_at", sa.DateTime(), nullable=True),
        sa.Column("claim_id", StringUUID(), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="summary_vector_cleanup_outbox_pkey"),
        sa.UniqueConstraint(
            "tenant_id",
            "node_id",
            "vector_route_hash",
            name="summary_vector_cleanup_outbox_tenant_node_route_unique",
        ),
    )
    op.create_index(
        "summary_vector_cleanup_outbox_retry_idx",
        "summary_vector_cleanup_outbox",
        ["claim_expires_at", "last_attempted_at", "created_at", "id"],
        unique=False,
    )

    op.create_table(
        "attachment_vector_cleanup_outbox",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("dataset_id", StringUUID(), nullable=False),
        sa.Column("attachment_id", StringUUID(), nullable=False),
        sa.Column("writer_id", StringUUID(), nullable=False),
        sa.Column("vector_route", LongText(), nullable=False),
        sa.Column("vector_route_hash", sa.String(length=64), nullable=False),
        sa.Column("writer_finished_at", sa.DateTime(), nullable=True),
        sa.Column("cleanup_started_at", sa.DateTime(), nullable=True),
        sa.Column("claim_id", StringUUID(), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="attachment_vector_cleanup_outbox_pkey"),
    )
    op.create_index(
        "attachment_vector_cleanup_outbox_node_route_idx",
        "attachment_vector_cleanup_outbox",
        ["tenant_id", "dataset_id", "attachment_id", "vector_route_hash"],
        unique=False,
    )
    op.create_index(
        "attachment_vector_cleanup_outbox_retry_idx",
        "attachment_vector_cleanup_outbox",
        ["claim_expires_at", "last_attempted_at", "created_at", "id"],
        unique=False,
    )
    op.create_index(
        "attachment_vector_cleanup_outbox_dataset_writer_idx",
        "attachment_vector_cleanup_outbox",
        ["tenant_id", "dataset_id", "writer_finished_at"],
        unique=False,
    )

    op.create_table(
        "child_vector_cleanup_outbox",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("dataset_id", StringUUID(), nullable=False),
        sa.Column("document_id", StringUUID(), nullable=False),
        sa.Column("segment_id", StringUUID(), nullable=False),
        sa.Column("child_id", StringUUID(), nullable=False),
        sa.Column("node_id", StringUUID(), nullable=False),
        sa.Column("writer_id", StringUUID(), nullable=False),
        sa.Column("vector_route", LongText(), nullable=True),
        sa.Column("vector_route_hash", sa.String(length=64), nullable=True),
        sa.Column("writer_finished_at", sa.DateTime(), nullable=True),
        sa.Column("cleanup_started_at", sa.DateTime(), nullable=True),
        sa.Column("claim_id", StringUUID(), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="child_vector_cleanup_outbox_pkey"),
        sa.UniqueConstraint(
            "tenant_id",
            "node_id",
            "vector_route_hash",
            name="child_vector_cleanup_outbox_tenant_node_route_unique",
        ),
    )
    op.create_index(
        "child_vector_cleanup_outbox_retry_idx",
        "child_vector_cleanup_outbox",
        ["claim_expires_at", "last_attempted_at", "created_at", "id"],
        unique=False,
    )

    op.create_table(
        "document_indexing_outbox",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("dataset_id", StringUUID(), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("document_ids", LongText(), nullable=False),
        sa.Column("actor_id", StringUUID(), nullable=True),
        sa.Column("pipeline_execution_log_id", StringUUID(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="document_indexing_outbox_pkey"),
    )
    op.create_index(
        "document_indexing_outbox_retry_idx",
        "document_indexing_outbox",
        ["last_attempted_at", "created_at", "id"],
        unique=False,
    )

    op.create_table(
        "dataset_indexing_outbox",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("dataset_id", StringUUID(), nullable=False),
        sa.Column("generation_id", StringUUID(), nullable=False),
        sa.Column("operation", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=True),
        sa.Column("source_representation", LongText(), nullable=False),
        sa.Column("target_representation", LongText(), nullable=False),
        sa.Column("claim_id", StringUUID(), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("vector_completed_at", sa.DateTime(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="dataset_indexing_outbox_pkey"),
        sa.UniqueConstraint(
            "generation_id",
            "operation",
            name="dataset_indexing_outbox_generation_operation_unique",
        ),
    )
    op.create_index(
        "dataset_indexing_outbox_retry_idx",
        "dataset_indexing_outbox",
        ["claim_expires_at", "last_attempted_at", "created_at", "id"],
        unique=False,
    )
    op.create_index(
        "dataset_indexing_outbox_owner_idx",
        "dataset_indexing_outbox",
        ["tenant_id", "dataset_id", "generation_id"],
        unique=False,
    )

    op.create_table(
        "segment_state_outbox",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("dataset_id", StringUUID(), nullable=False),
        sa.Column("document_id", StringUUID(), nullable=False),
        sa.Column("segment_id", StringUUID(), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("actor_id", StringUUID(), nullable=True),
        sa.Column("index_node_id", sa.String(length=255), nullable=True),
        sa.Column("child_node_ids", LongText(), nullable=True),
        sa.Column("doc_form", sa.String(length=255), nullable=True),
        sa.Column("summary_id", StringUUID(), nullable=True),
        sa.Column("summary_version", sa.Integer(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="segment_state_outbox_pkey"),
    )
    op.create_index(
        "segment_state_outbox_retry_idx",
        "segment_state_outbox",
        ["last_attempted_at", "created_at", "id"],
        unique=False,
    )
    op.create_index(
        "segment_state_outbox_owner_idx",
        "segment_state_outbox",
        ["tenant_id", "dataset_id", "document_id", "segment_id"],
        unique=False,
    )

    op.create_table(
        "website_crawl_jobs",
        sa.Column("id", StringUUID(), nullable=False),
        sa.Column("tenant_id", StringUUID(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("provider_instance_id", sa.String(length=64), nullable=False),
        sa.Column("job_id", sa.String(length=255), nullable=False),
        sa.Column("exclusive_job_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="website_crawl_job_pkey"),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_instance_id",
            "job_id",
            name="website_crawl_job_owner_job_key",
        ),
        sa.UniqueConstraint(
            "provider",
            "provider_instance_id",
            "exclusive_job_id",
            name="website_crawl_job_exclusive_job_key",
        ),
    )


def downgrade() -> None:
    op.drop_table("website_crawl_jobs")

    op.drop_index("segment_state_outbox_owner_idx", table_name="segment_state_outbox")
    op.drop_index("segment_state_outbox_retry_idx", table_name="segment_state_outbox")
    op.drop_table("segment_state_outbox")

    op.drop_index("dataset_indexing_outbox_owner_idx", table_name="dataset_indexing_outbox")
    op.drop_index("dataset_indexing_outbox_retry_idx", table_name="dataset_indexing_outbox")
    op.drop_table("dataset_indexing_outbox")

    op.drop_index("document_indexing_outbox_retry_idx", table_name="document_indexing_outbox")
    op.drop_table("document_indexing_outbox")

    op.drop_index("child_vector_cleanup_outbox_retry_idx", table_name="child_vector_cleanup_outbox")
    op.drop_table("child_vector_cleanup_outbox")

    op.drop_index(
        "attachment_vector_cleanup_outbox_dataset_writer_idx",
        table_name="attachment_vector_cleanup_outbox",
    )
    op.drop_index("attachment_vector_cleanup_outbox_retry_idx", table_name="attachment_vector_cleanup_outbox")
    op.drop_index("attachment_vector_cleanup_outbox_node_route_idx", table_name="attachment_vector_cleanup_outbox")
    op.drop_table("attachment_vector_cleanup_outbox")

    op.drop_index("summary_vector_cleanup_outbox_retry_idx", table_name="summary_vector_cleanup_outbox")
    op.drop_table("summary_vector_cleanup_outbox")

    op.drop_index("resource_cleanup_outbox_claim_idx", table_name="resource_cleanup_outbox")
    op.drop_index("resource_cleanup_outbox_retry_idx", table_name="resource_cleanup_outbox")
    op.drop_table("resource_cleanup_outbox")

    op.drop_index("keyword_file_cleanup_outbox_retry_idx", table_name="keyword_file_cleanup_outbox")
    op.drop_table("keyword_file_cleanup_outbox")

    op.drop_index("upload_file_cleanup_outbox_retry_idx", table_name="upload_file_cleanup_outbox")
    op.drop_table("upload_file_cleanup_outbox")

    with op.batch_alter_table("child_chunks") as batch_op:
        batch_op.drop_column("vector_route_hash")
        batch_op.drop_column("vector_route")

    with op.batch_alter_table("document_segment_summaries") as batch_op:
        batch_op.drop_column("vector_route_hash")
        batch_op.drop_column("vector_route")
        batch_op.drop_column("version")

    with op.batch_alter_table("document_segments") as batch_op:
        batch_op.drop_index("document_segment_indexing_request_idx")
        batch_op.drop_column("indexing_request_id")

    with op.batch_alter_table("document_pipeline_execution_logs") as batch_op:
        batch_op.drop_column("original_document_id")
        batch_op.drop_column("workflow_id")

    with op.batch_alter_table("datasets") as batch_op:
        batch_op.drop_column("indexing_applied_representation")
        batch_op.drop_column("indexing_request_id")

    with op.batch_alter_table("workflow_runs") as batch_op:
        batch_op.drop_constraint("workflow_run_history_pending_check", type_="check")
        batch_op.drop_constraint("workflow_run_history_claim_check", type_="check")
        batch_op.drop_constraint("workflow_run_history_revision_check", type_="check")
        batch_op.drop_index("workflow_run_history_claim_idx")
        batch_op.drop_column("history_sync_pending")
        batch_op.drop_column("history_last_attempted_at")
        batch_op.drop_column("history_claim_expires_at")
        batch_op.drop_column("history_claim_id")
        batch_op.drop_column("history_published_revision")
        batch_op.drop_column("history_revision")

    with op.batch_alter_table("workflow_trigger_logs") as batch_op:
        batch_op.drop_index("workflow_trigger_log_quota_sync_pending_idx")
        batch_op.drop_constraint("workflow_trigger_log_quota_sync_pending_check", type_="check")
        batch_op.drop_constraint("workflow_trigger_log_quota_settlement_status_check", type_="check")
        batch_op.drop_constraint("workflow_trigger_log_queued_dispatch_abandoned_check", type_="check")
        batch_op.drop_constraint("workflow_trigger_log_dispatch_receipt_check", type_="check")
        batch_op.drop_constraint("workflow_trigger_log_pending_dispatch_abandoned_check", type_="check")
        batch_op.drop_column("quota_sync_pending")
        batch_op.drop_column("quota_settlement_attempted_at")
        batch_op.drop_column("quota_settlement_status")
        batch_op.drop_column("quota_reservation_id")
        batch_op.drop_column("dispatch_abandoned_at")
        batch_op.drop_column("dispatch_published_at")
        batch_op.drop_column("dispatch_prepared_at")
        batch_op.drop_column("pending_dispatch_abandoned_at")
        batch_op.drop_column("terminal_owner_id")
        batch_op.drop_column("execution_lease_expires_at")
        batch_op.drop_column("execution_owner_id")

    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_index("document_indexing_dataset_generation_idx")
        batch_op.drop_index("document_dataset_indexing_request_idx")
        batch_op.drop_index("document_indexing_request_idx")
        batch_op.drop_column("indexing_dataset_generation_id")
        batch_op.drop_column("dataset_indexing_request_id")
        batch_op.drop_column("indexing_request_id")
        batch_op.drop_column("indexing_claim_expires_at")
        batch_op.drop_column("indexing_claim_id")

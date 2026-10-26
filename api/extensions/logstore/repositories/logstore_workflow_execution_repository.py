import logging
from typing import override

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from configs import dify_config
from core.repositories.factory import WorkflowExecutionRepository
from core.repositories.sqlalchemy_workflow_execution_repository import SQLAlchemyWorkflowExecutionRepository
from extensions.logstore.aliyun_logstore import AliyunLogStore
from graphon.entities import WorkflowExecution
from graphon.enums import WorkflowExecutionStatus
from models import (
    Account,
    EndUser,
    WorkflowRun,
)
from models.enums import WorkflowRunTriggeredFrom
from models.workflow import WorkflowPause
from repositories.workflow_run_history import (
    WorkflowRunHistorySnapshot,
    advance_and_claim_workflow_run_history,
    confirm_workflow_run_history_publication,
    release_workflow_run_history_claim,
    snapshot_claimed_workflow_run_history,
    workflow_history_database_now,
)

logger = logging.getLogger(__name__)


def _to_logstore_model(snapshot: WorkflowRunHistorySnapshot, *, enable_graph: bool) -> list[tuple[str, str]]:
    """Convert one exact SQL projection into an append-only record."""
    return [
        ("id", snapshot.id),
        ("log_version", str(snapshot.log_version)),
        ("tenant_id", snapshot.tenant_id),
        ("app_id", snapshot.app_id),
        ("workflow_id", snapshot.workflow_id),
        ("triggered_from", snapshot.triggered_from),
        ("type", snapshot.type),
        ("version", snapshot.version),
        ("graph", snapshot.graph if enable_graph else "{}"),
        ("inputs", snapshot.inputs),
        ("outputs", snapshot.outputs),
        ("status", snapshot.status),
        ("error_message", snapshot.error),
        ("total_tokens", str(snapshot.total_tokens)),
        ("total_steps", str(snapshot.total_steps)),
        ("exceptions_count", str(snapshot.exceptions_count)),
        ("elapsed_time", str(snapshot.elapsed_time)),
        ("created_by_role", snapshot.created_by_role),
        ("created_by", snapshot.created_by),
        ("started_at", snapshot.started_at),
        ("finished_at", snapshot.finished_at),
    ]


def publish_workflow_run_history_snapshot(
    *,
    session_factory: sessionmaker,
    logstore_client: AliyunLogStore,
    snapshot: WorkflowRunHistorySnapshot,
    enable_graph: bool,
) -> None:
    """Publish and exact-CAS acknowledge one durable history claim."""
    try:
        logstore_client.put_log(
            AliyunLogStore.workflow_execution_logstore,
            _to_logstore_model(snapshot, enable_graph=enable_graph),
        )
        with session_factory.begin() as session:
            confirm_workflow_run_history_publication(session, snapshot)
    except Exception:
        logger.exception("Failed to publish workflow execution history: id=%s", snapshot.id)
        try:
            with session_factory.begin() as session:
                release_workflow_run_history_claim(session, snapshot)
        except Exception:
            logger.exception("Failed to release workflow history claim: id=%s", snapshot.id)
        raise


class LogstoreWorkflowExecutionRepository(WorkflowExecutionRepository):
    def __init__(
        self,
        session_factory: sessionmaker | Engine,
        tenant_id: str,
        user: Account | EndUser,
        app_id: str | None,
        triggered_from: WorkflowRunTriggeredFrom | None,
    ):
        """
        Initialize the repository with a SQLAlchemy sessionmaker or engine and context information.

        Args:
            session_factory: SQLAlchemy sessionmaker or engine for creating sessions
            tenant_id: Tenant that owns the workflow execution
            user: Account or EndUser used for creator attribution
            app_id: App ID for filtering by application (can be None)
            triggered_from: Source of the execution trigger (DEBUGGING or APP_RUN)
        """
        logger.debug(
            "LogstoreWorkflowExecutionRepository.__init__: app_id=%s, triggered_from=%s", app_id, triggered_from
        )
        # Initialize LogStore client
        # Note: Project/logstore/index initialization is done at app startup via ext_logstore
        self.logstore_client = AliyunLogStore()

        if not tenant_id:
            raise ValueError("tenant_id is required")
        self._tenant_id = tenant_id

        # Store app context
        self._app_id = app_id

        # Initialize SQL repository for dual-write support
        self.sql_repository = SQLAlchemyWorkflowExecutionRepository(
            session_factory=session_factory,
            tenant_id=tenant_id,
            user=user,
            app_id=app_id,
            triggered_from=triggered_from,
        )

        self._enable_dual_write = dify_config.LOGSTORE_DUAL_WRITE_ENABLED
        self._session_factory = (
            sessionmaker(bind=session_factory) if isinstance(session_factory, Engine) else session_factory
        )

        # Control flag for whether to write the `graph` field to LogStore.
        # If LOGSTORE_ENABLE_PUT_GRAPH_FIELD is "true", write the full `graph` field;
        # otherwise write an empty {} instead. Defaults to writing the `graph` field.
        self._enable_put_graph_field = dify_config.LOGSTORE_ENABLE_PUT_GRAPH_FIELD

    @override
    def save_synchronously(self, execution: WorkflowExecution) -> None:
        self._save_and_publish(execution)

    @override
    def publish_committed_pause(
        self,
        execution: WorkflowExecution,
        *,
        history_revision: int,
        history_claim_id: str,
    ) -> None:
        """Append PAUSED history without issuing a second SQL control write."""
        if execution.status != WorkflowExecutionStatus.PAUSED:
            raise ValueError("committed pause history requires a paused execution")

        conditions = [
            WorkflowRun.id == execution.id_,
            WorkflowRun.tenant_id == self._tenant_id,
            WorkflowRun.history_revision == history_revision,
            WorkflowRun.history_claim_id == history_claim_id,
            WorkflowRun.status == WorkflowExecutionStatus.PAUSED,
            WorkflowPause.resumed_at.is_(None),
        ]
        if self._app_id is not None:
            conditions.append(WorkflowRun.app_id == self._app_id)
        with self._session_factory() as session:
            workflow_run = session.scalar(
                select(WorkflowRun)
                .join(WorkflowPause, WorkflowRun.id == WorkflowPause.workflow_run_id)
                .where(*conditions)
            )
            snapshot = snapshot_claimed_workflow_run_history(workflow_run) if workflow_run is not None else None
        if snapshot is None:
            logger.info(
                "Skipped superseded workflow pause history: tenant_id=%s, app_id=%s, workflow_run_id=%s",
                self._tenant_id,
                self._app_id,
                execution.id_,
            )
            return

        self._publish_snapshot(snapshot)

    @override
    def save(self, execution: WorkflowExecution) -> None:
        """
        Save or update a WorkflowExecution domain entity to the logstore.

        This method serves as a domain-to-logstore adapter that:
        1. Converts the domain entity to its logstore representation
        2. Persists the logstore model using Aliyun SLS
        3. Maintains proper multi-tenancy by including tenant context during conversion
        4. Optionally writes to SQL database for dual-write support (controlled by LOGSTORE_DUAL_WRITE_ENABLED)

        Args:
            execution: The WorkflowExecution domain entity to persist
        """
        self._save_and_publish(execution)

    def _save_and_publish(self, execution: WorkflowExecution) -> None:
        snapshot = self.sql_repository.save_with_model_callback(
            execution,
            lambda session, model: advance_and_claim_workflow_run_history(
                model,
                now=workflow_history_database_now(session),
            ),
        )
        self._publish_snapshot(snapshot)

    def _publish_snapshot(self, snapshot: WorkflowRunHistorySnapshot) -> None:
        publish_workflow_run_history_snapshot(
            session_factory=self._session_factory,
            logstore_client=self.logstore_client,
            snapshot=snapshot,
            enable_graph=self._enable_put_graph_field,
        )

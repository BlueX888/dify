"""Reconcile durable SQL workflow projections into append-only LogStore history."""

import logging

from celery import shared_task
from sqlalchemy.orm import Session, sessionmaker

from configs import dify_config
from core.db.session_factory import session_factory
from extensions.logstore.aliyun_logstore import AliyunLogStore
from extensions.logstore.repositories.logstore_workflow_execution_repository import (
    publish_workflow_run_history_snapshot,
)
from repositories.workflow_run_history import claim_pending_workflow_run_histories

logger = logging.getLogger(__name__)

_LOGSTORE_EXECUTION_REPOSITORY = (
    "extensions.logstore.repositories.logstore_workflow_execution_repository.LogstoreWorkflowExecutionRepository"
)
_BATCH_SIZE = 100


def drain_pending_workflow_run_histories(
    *,
    database_client: sessionmaker[Session],
    logstore_client: AliyunLogStore,
    enable_graph: bool,
    limit: int = _BATCH_SIZE,
) -> int:
    """Publish one fair batch, leaving failed rows eligible for a later pass."""
    with database_client.begin() as session:
        snapshots = claim_pending_workflow_run_histories(session, limit=limit)

    published = 0
    for snapshot in snapshots:
        try:
            publish_workflow_run_history_snapshot(
                session_factory=database_client,
                logstore_client=logstore_client,
                snapshot=snapshot,
                enable_graph=enable_graph,
            )
        except Exception:
            logger.exception("Workflow history reconciliation failed: workflow_run_id=%s", snapshot.id)
        else:
            published += 1
    return published


@shared_task(queue="workflow_storage")
def drain_workflow_run_history_task() -> None:
    """Periodically repair LogStore history only when that backend is configured."""
    if dify_config.CORE_WORKFLOW_EXECUTION_REPOSITORY != _LOGSTORE_EXECUTION_REPOSITORY:
        return
    drain_pending_workflow_run_histories(
        database_client=session_factory.get_session_maker(),
        logstore_client=AliyunLogStore(),
        enable_graph=dify_config.LOGSTORE_ENABLE_PUT_GRAPH_FIELD,
    )

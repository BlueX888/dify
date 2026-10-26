import logging
from datetime import UTC, datetime
from typing import Any, ClassVar, override

from pydantic import TypeAdapter

from core.db.session_factory import session_factory
from core.workflow.system_variables import SystemVariableKey, get_system_text
from graphon.engine.layer import Layer
from graphon.engine_events import (
    EngineEvent,
    GraphRunAbortedEvent,
    GraphRunFailedEvent,
    GraphRunPartialSucceededEvent,
    GraphRunPausedEvent,
    GraphRunSucceededEvent,
)
from models.enums import WorkflowTriggerStatus
from repositories.entities.workflow_pause import WorkflowTriggerPausePublication
from repositories.sqlalchemy_workflow_trigger_log_repository import SQLAlchemyWorkflowTriggerLogRepository

logger = logging.getLogger(__name__)


class TriggerPostLayer(Layer):
    """
    Trigger post layer.
    """

    _STATUS_MAP: ClassVar[dict[type[EngineEvent], WorkflowTriggerStatus]] = {
        GraphRunSucceededEvent: WorkflowTriggerStatus.SUCCEEDED,
        GraphRunPartialSucceededEvent: WorkflowTriggerStatus.SUCCEEDED,
        GraphRunFailedEvent: WorkflowTriggerStatus.FAILED,
        GraphRunAbortedEvent: WorkflowTriggerStatus.FAILED,
        GraphRunPausedEvent: WorkflowTriggerStatus.PAUSED,
    }

    def __init__(
        self,
        start_time: datetime,
        trigger_log_id: str,
        execution_owner_id: str | None,
    ):
        super().__init__()
        self.trigger_log_id = trigger_log_id
        self.start_time = start_time
        self.execution_owner_id = execution_owner_id
        self._defer_owned_pause = False
        self._pause_publication_outcome_unknown = False

    def defer_owned_pause(self) -> bool:
        """Keep an owned trigger RUNNING until its pause snapshot is atomically published."""
        if self.execution_owner_id is None:
            return False
        self._defer_owned_pause = True
        return True

    def prepare_deferred_pause(self) -> WorkflowTriggerPausePublication:
        """Capture the exact owner transition without releasing it in a separate transaction."""
        owner_id = self.execution_owner_id
        if not self._defer_owned_pause or owner_id is None:
            raise RuntimeError("trigger pause is not deferred")
        now = datetime.now(UTC)
        workflow_run_id = get_system_text(
            self.runtime_state.variable_pool,
            SystemVariableKey.WORKFLOW_EXECUTION_ID,
        )
        if not workflow_run_id:
            raise ValueError("Workflow run id is not set")
        return WorkflowTriggerPausePublication(
            trigger_log_id=self.trigger_log_id,
            owner_id=owner_id,
            workflow_run_id=workflow_run_id,
            outputs=TypeAdapter(dict[str, Any]).dump_json(self.runtime_state.outputs).decode(),
            elapsed_time=(now - self.start_time).total_seconds(),
            total_tokens=self.runtime_state.total_tokens,
            finished_at=now,
        )

    def confirm_deferred_pause(self, publication: WorkflowTriggerPausePublication) -> None:
        """Reflect a committed pause so a later failure can use the PAUSED ownerless transition."""
        if publication.trigger_log_id != self.trigger_log_id:
            raise ValueError("trigger pause publication does not belong to this layer")
        self.execution_owner_id = None
        self.start_time = publication.finished_at
        self._pause_publication_outcome_unknown = False

    def mark_pause_publication_outcome_unknown(self) -> None:
        if not self._defer_owned_pause or self.execution_owner_id is None:
            raise RuntimeError("trigger pause is not awaiting publication")
        self._pause_publication_outcome_unknown = True

    @property
    def pause_publication_outcome_unknown(self) -> bool:
        return self._pause_publication_outcome_unknown

    @override
    def on_event(self, event: EngineEvent):
        """
        Update trigger log with success or failure.
        """
        if isinstance(event, tuple(self._STATUS_MAP.keys())):
            if isinstance(event, GraphRunPausedEvent) and self._defer_owned_pause:
                return
            with session_factory.create_session() as session:
                repo = SQLAlchemyWorkflowTriggerLogRepository(session)
                # Calculate elapsed time
                now = datetime.now(UTC)
                elapsed_time = (now - self.start_time).total_seconds()

                # Extract relevant data from result
                outputs = self.runtime_state.outputs

                # BASICLY, workflow_execution_id is the same as workflow_run_id
                workflow_run_id = get_system_text(
                    self.runtime_state.variable_pool,
                    SystemVariableKey.WORKFLOW_EXECUTION_ID,
                )
                assert workflow_run_id, "Workflow run id is not set"

                total_tokens = self.runtime_state.total_tokens

                updated = repo.finish_execution(
                    self.trigger_log_id,
                    owner_id=self.execution_owner_id,
                    status=self._STATUS_MAP[type(event)],
                    workflow_run_id=workflow_run_id,
                    outputs=TypeAdapter(dict[str, Any]).dump_json(outputs).decode(),
                    error=(event.reason or "Workflow execution aborted")
                    if isinstance(event, GraphRunAbortedEvent)
                    else None,
                    elapsed_time=elapsed_time,
                    total_tokens=total_tokens,
                    finished_at=now,
                )
                session.commit()
                if not updated:
                    logger.error(
                        "Trigger log execution ownership lost before terminal update: %s",
                        self.trigger_log_id,
                    )
                    return
                if self._STATUS_MAP[type(event)] == WorkflowTriggerStatus.PAUSED:
                    # A pause-persistence failure is reported as a later failed
                    # event by the same layer after the lease has been released.
                    self.execution_owner_id = None
                self.start_time = now

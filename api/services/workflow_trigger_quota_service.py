"""Quota settlement for durable async-workflow dispatch receipts."""

from typing import Literal

from enums import QuotaType
from repositories.workflow_trigger_log_repository import WorkflowQuotaSettlementCandidate
from services.quota_service import QuotaService, QuotaSettlementResult


class WorkflowTriggerQuotaService:
    """Resolve one claimed settlement without holding a database transaction."""

    @staticmethod
    def settle(
        candidate: WorkflowQuotaSettlementCandidate,
    ) -> Literal["committed", "refunded", "permanent_error"] | None:
        if candidate.reservation_id is None:
            # Community/unlimited reservations require no external I/O, but
            # still receive an observable terminal receipt in the trigger log.
            return candidate.outcome

        result = QuotaService.settle_reservation(
            QuotaType.WORKFLOW,
            candidate.tenant_id,
            candidate.reservation_id,
            settlement=candidate.outcome,
        )
        if result == QuotaSettlementResult.CONFIRMED:
            return candidate.outcome
        if result == QuotaSettlementResult.PERMANENT_ERROR:
            return "permanent_error"
        return None

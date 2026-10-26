import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, override

from core.entities.document_task import DocumentTask
from core.rag.pipeline.queue import TenantIsolatedTaskQueue

from .base import DocumentTaskProxyBase

logger = logging.getLogger(__name__)


class BatchDocumentIndexingProxy(DocumentTaskProxyBase):
    """
    Base proxy for batch document indexing tasks (document_ids in plural).

    Adds:
    - Tenant isolated queue management
    - Batch document handling
    """

    def __init__(
        self,
        tenant_id: str,
        dataset_id: str,
        document_ids: Sequence[str],
        *,
        indexing_request_id: str | None = None,
    ):
        """
        Initialize with batch documents.

        Args:
            tenant_id: Tenant identifier
            dataset_id: Dataset identifier
            document_ids: List of document IDs to process
        """
        super().__init__(tenant_id, dataset_id)
        self._document_ids = document_ids
        self._indexing_request_id = indexing_request_id
        self._tenant_isolated_task_queue = TenantIsolatedTaskQueue(tenant_id, self.QUEUE_NAME)

    def _task_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "tenant_id": self._tenant_id,
            "dataset_id": self._dataset_id,
            "document_ids": self._document_ids,
        }
        if self._indexing_request_id is not None:
            kwargs["indexing_request_id"] = self._indexing_request_id
        return kwargs

    @override
    def _send_to_direct_queue(self, task_func: Callable[[str, str, Sequence[str]], Any]) -> bool:
        """
        Send batch task to direct queue.

        Args:
            task_func: The Celery task function to call with (tenant_id, dataset_id, document_ids)
        """
        logger.info("tenant %s send documents %s to direct queue", self._tenant_id, self._document_ids)
        task_func.delay(**self._task_kwargs())  # type: ignore
        return True

    @override
    def _send_to_tenant_queue(self, task_func: Callable[[str, str, Sequence[str]], Any]) -> bool:
        """
        Send batch task to tenant-isolated queue.

        Args:
            task_func: The Celery task function to call with (tenant_id, dataset_id, document_ids)
        """
        logger.info(
            "tenant %s send documents %s to tenant queue %s", self._tenant_id, self._document_ids, self.QUEUE_NAME
        )
        if self._indexing_request_id is not None and not self._tenant_isolated_task_queue.mark_request_queued(
            self._indexing_request_id
        ):
            logger.info(
                "tenant %s indexing request %s is already queued",
                self._tenant_id,
                self._indexing_request_id,
            )
            return False

        document_task = DocumentTask(
            tenant_id=self._tenant_id,
            dataset_id=self._dataset_id,
            document_ids=self._document_ids,
            indexing_request_id=self._indexing_request_id,
        )
        if self._tenant_isolated_task_queue.get_task_key():
            try:
                # Redis waiting-list insertion is not broker acceptance. The durable
                # request remains until the worker finishes its exact generation.
                self._tenant_isolated_task_queue.push_tasks([asdict(document_task)])
            except Exception:
                if self._indexing_request_id is not None:
                    self._tenant_isolated_task_queue.clear_request_marker(self._indexing_request_id)
                raise
            logger.info("tenant %s push tasks: %s - %s", self._tenant_id, self._dataset_id, self._document_ids)
            return False

        try:
            self._tenant_isolated_task_queue.set_task_waiting_time()
            task_func.delay(**self._task_kwargs())  # type: ignore
        except Exception:
            self._tenant_isolated_task_queue.delete_task_key()
            if self._indexing_request_id is not None:
                self._tenant_isolated_task_queue.clear_request_marker(self._indexing_request_id)
            raise
        logger.info("tenant %s init tasks: %s - %s", self._tenant_id, self._dataset_id, self._document_ids)
        return True

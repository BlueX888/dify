import logging
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import click
from celery import shared_task  # type: ignore
from flask import current_app
from pydantic import TypeAdapter

from configs import dify_config
from core.app.apps.pipeline.pipeline_generator import PipelineGenerator
from core.rag.pipeline.queue import TenantIsolatedTaskQueue
from extensions.ext_database import db
from services.file_service import FileService
from tasks.rag_pipeline.rag_pipeline_run_task import run_single_rag_pipeline_task as _run_owned_pipeline_task

logger = logging.getLogger(__name__)

# Each entry is validated into RagPipelineInvokeEntity inside the worker.
_INVOKE_ENTITIES_ADAPTER = TypeAdapter(list[dict[str, Any]])


@shared_task(queue="priority_pipeline")
def priority_rag_pipeline_run_task(
    rag_pipeline_invoke_entities_file_id: str,
    tenant_id: str,
):
    """
    Async Run rag pipeline task using high priority queue.

    :param rag_pipeline_invoke_entities_file_id: File ID containing serialized RAG pipeline invoke entities
    :param tenant_id: Tenant ID for the pipeline execution
    """
    from extensions.ext_application_services import application_services

    # run with threading, thread pool size is 10

    try:
        pipeline_generator = application_services().knowledge.pipeline_generator
        start_at = time.perf_counter()
        rag_pipeline_invoke_entities_content = FileService(db.engine).get_file_content(
            rag_pipeline_invoke_entities_file_id
        )
        rag_pipeline_invoke_entities = _INVOKE_ENTITIES_ADAPTER.validate_json(rag_pipeline_invoke_entities_content)

        logger.info("tenant %s received %d rag pipeline invoke entities", tenant_id, len(rag_pipeline_invoke_entities))

        # Get Flask app object for thread context
        flask_app = current_app._get_current_object()  # type: ignore

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for rag_pipeline_invoke_entity in rag_pipeline_invoke_entities:
                # Submit task to thread pool with Flask app
                future = executor.submit(
                    run_single_rag_pipeline_task,
                    rag_pipeline_invoke_entity,
                    flask_app,
                    pipeline_generator=pipeline_generator,
                )
                futures.append(future)

            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()  # This will raise any exceptions that occurred in the thread
                except Exception:
                    logging.exception("Error in pipeline task")
        end_at = time.perf_counter()
        logging.info(
            click.style(
                f"tenant_id: {tenant_id}, Rag pipeline run completed. Latency: {end_at - start_at}s", fg="green"
            )
        )
    except Exception:
        logging.exception(click.style(f"Error running rag pipeline, tenant_id: {tenant_id}", fg="red"))
        raise
    finally:
        tenant_isolated_task_queue = TenantIsolatedTaskQueue(tenant_id, "pipeline")

        # Check if there are waiting tasks in the queue
        # Use rpop to get the next task from the queue (FIFO order)
        next_file_ids = tenant_isolated_task_queue.pull_tasks(count=dify_config.TENANT_ISOLATED_TASK_CONCURRENCY)
        logger.info("priority rag pipeline tenant isolation queue %s next files: %s", tenant_id, next_file_ids)

        if next_file_ids:
            for next_file_id in next_file_ids:
                # Process the next waiting task
                # Keep the flag set to indicate a task is running
                tenant_isolated_task_queue.set_task_waiting_time()
                priority_rag_pipeline_run_task.delay(  # type: ignore
                    rag_pipeline_invoke_entities_file_id=next_file_id.decode("utf-8")
                    if isinstance(next_file_id, bytes)
                    else next_file_id,
                    tenant_id=tenant_id,
                )
        else:
            # No more waiting tasks, clear the flag
            tenant_isolated_task_queue.delete_task_key()
        file_service = FileService(db.engine)
        file_service.delete_file(rag_pipeline_invoke_entities_file_id)
        db.session.close()


def run_single_rag_pipeline_task(
    rag_pipeline_invoke_entity: Mapping[str, Any], flask_app, *, pipeline_generator: PipelineGenerator
):
    """Run through the same durable owner path as the regular queue."""
    return _run_owned_pipeline_task(
        rag_pipeline_invoke_entity,
        flask_app,
        pipeline_generator=pipeline_generator,
    )

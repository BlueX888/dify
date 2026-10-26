import json
from collections.abc import Sequence
from typing import Protocol, cast
from unittest.mock import create_autospec

from core.entities.document_task import DocumentTask
from services.document_indexing_proxy.document_indexing_task_proxy import DocumentIndexingTaskProxy
from tasks.document_indexing_task import normal_document_indexing_task


class _RecordedCall(Protocol):
    args: tuple[object, ...]


class _CommandRecorder(Protocol):
    side_effect: object
    call_args_list: Sequence[_RecordedCall]


def test_request_marker_suppresses_republication_until_its_ttl_expires(
    tenant_queue_commands: _CommandRecorder,
) -> None:
    request_id = "request-1"
    proxy = DocumentIndexingTaskProxy(
        "tenant-123",
        "dataset-456",
        ["doc-1"],
        indexing_request_id=request_id,
    )
    task = create_autospec(normal_document_indexing_task, spec_set=True)
    tenant_queue_commands.side_effect = [True, None, None, None, True, None, None]

    assert proxy._send_to_tenant_queue(task) is True
    assert proxy._send_to_tenant_queue(task) is False
    assert proxy._send_to_tenant_queue(task) is True

    assert task.delay.call_count == 2
    task.delay.assert_called_with(
        tenant_id="tenant-123",
        dataset_id="dataset-456",
        document_ids=["doc-1"],
        indexing_request_id=request_id,
    )
    marker_key = f"tenant_document_indexing_request:tenant-123:{request_id}"
    marker_sets = [entry for entry in tenant_queue_commands.call_args_list if entry.args[0] == "SET"]
    assert len(marker_sets) == 3
    assert all(entry.args[1] == marker_key for entry in marker_sets)
    assert all("EX" in entry.args and "NX" in entry.args for entry in marker_sets)


def test_waiting_list_keeps_request_id_and_is_not_reported_as_broker_acceptance(
    tenant_queue_commands: _CommandRecorder,
) -> None:
    request_id = "request-1"
    proxy = DocumentIndexingTaskProxy(
        "tenant-123",
        "dataset-456",
        ["doc-1"],
        indexing_request_id=request_id,
    )
    task = create_autospec(normal_document_indexing_task, spec_set=True)
    tenant_queue_commands.side_effect = [True, "active-task", None]

    assert proxy._send_to_tenant_queue(task) is False

    task.delay.assert_not_called()
    push = next(entry for entry in tenant_queue_commands.call_args_list if entry.args[0] == "LPUSH")
    payload = json.loads(cast(str, push.args[2]))["data"]
    assert DocumentTask(**payload).indexing_request_id == request_id

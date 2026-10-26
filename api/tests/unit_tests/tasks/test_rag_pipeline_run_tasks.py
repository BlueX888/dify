import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import pytest
from flask import Flask
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from core.app.apps.pipeline.pipeline_config_manager import PipelineConfig
from core.app.apps.pipeline.pipeline_generator import PipelineGenerator
from core.app.entities.app_invoke_entities import InvokeFrom, RagPipelineGenerateEntity
from core.app.entities.rag_pipeline_invoke_entities import RagPipelineInvokeEntity
from models.dataset import Document, DocumentPipelineExecutionLog
from models.enums import IndexingStatus
from models.model import AppMode, DocumentIndexingOutbox
from models.workflow import WorkflowType
from repositories.knowledge.document_indexing_outbox_repository import create_pipeline_document_indexing_intents
from tests.unit_tests.model_factories import make_dataset, make_document, make_tenant, make_workflow


@dataclass(slots=True)
class _DatabaseSessionStub:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _DatabaseStub:
    engine: object
    session: _DatabaseSessionStub


class _FileServiceStub:
    def __init__(self, _engine: object) -> None:
        pass

    def get_file_content(self, _file_id: str) -> str:
        return "[{}, {}]"

    def delete_file(self, _file_id: str) -> None:
        pass


class _TenantQueueStub:
    def __init__(self, _tenant_id: str, _queue: str) -> None:
        pass

    def pull_tasks(self, *, count: int) -> list[str]:
        del count
        return []

    def delete_task_key(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class _KnowledgeServicesStub:
    pipeline_generator: PipelineGenerator


@dataclass(frozen=True, slots=True)
class _ApplicationServicesStub:
    knowledge: _KnowledgeServicesStub


class _PipelineGeneratorStub:
    pass


class _RunnableTask(Protocol):
    def run(self, invoke_entities_file_id: str, tenant_id: str) -> None: ...


@pytest.mark.parametrize(
    ("module_name", "task_name"),
    [
        ("tasks.rag_pipeline.rag_pipeline_run_task", "rag_pipeline_run_task"),
        ("tasks.rag_pipeline.priority_rag_pipeline_run_task", "priority_rag_pipeline_run_task"),
    ],
)
def test_task_passes_composed_pipeline_generator_to_every_worker(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    task_name: str,
) -> None:
    module = importlib.import_module(module_name)
    generator = cast(PipelineGenerator, _PipelineGeneratorStub())
    services = _ApplicationServicesStub(knowledge=_KnowledgeServicesStub(pipeline_generator=generator))
    worker_calls: list[dict[str, object]] = []
    service_lookups = 0

    class _CompletedFuture:
        def result(self) -> None:
            return None

    class _InlineExecutor:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 10

        def __enter__(self) -> "_InlineExecutor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: Callable[..., object], *args: object, **kwargs: object) -> _CompletedFuture:
            function(*args, **kwargs)
            return _CompletedFuture()

    def _record_worker(*_args: object, **kwargs: object) -> None:
        worker_calls.append(kwargs)

    def _application_services() -> _ApplicationServicesStub:
        nonlocal service_lookups
        service_lookups += 1
        return services

    monkeypatch.setattr(module, "ThreadPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(module, "run_single_rag_pipeline_task", _record_worker)
    monkeypatch.setattr(module, "FileService", _FileServiceStub)
    monkeypatch.setattr(module, "TenantIsolatedTaskQueue", _TenantQueueStub)
    monkeypatch.setattr(module, "db", _DatabaseStub(engine=object(), session=_DatabaseSessionStub()))
    monkeypatch.setattr("extensions.ext_application_services.application_services", _application_services)

    app = Flask(__name__)
    with app.app_context():
        task = cast(_RunnableTask, vars(module)[task_name])
        task.run("invoke-entities-file", "tenant-id")

    assert service_lookups == 1
    assert len(worker_calls) == 2
    assert all(call["pipeline_generator"] is generator for call in worker_calls)


def test_priority_worker_delegates_to_the_regular_owned_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("tasks.rag_pipeline.priority_rag_pipeline_run_task")
    generator = cast(PipelineGenerator, _PipelineGeneratorStub())
    calls: list[tuple[object, object, PipelineGenerator]] = []

    def _run(payload: object, app: object, *, pipeline_generator: PipelineGenerator) -> None:
        calls.append((payload, app, pipeline_generator))

    monkeypatch.setattr(module, "_run_owned_pipeline_task", _run)
    app = Flask(__name__)
    payload = {"request": "one"}

    module.run_single_rag_pipeline_task(payload, app, pipeline_generator=generator)

    assert calls == [(payload, app, generator)]


def _pipeline_entity(
    *, request_id: str, pipeline_id: str = "pipeline-1"
) -> tuple[RagPipelineInvokeEntity, RagPipelineGenerateEntity]:
    config = PipelineConfig(
        app_id=pipeline_id,
        workflow_id="workflow-1",
        tenant_id="tenant-1",
        app_mode=AppMode.RAG_PIPELINE,
    )
    entity = RagPipelineGenerateEntity(
        task_id="task-1",
        stream=True,
        app_config=config,
        pipeline_config=config,
        invoke_from=InvokeFrom.PUBLISHED_PIPELINE,
        user_id="actor-1",
        trace_manager=None,
        inputs={"input": "value"},
        files=[],
        workflow_execution_id="run-1",
        document_id="document-1",
        original_document_id=None,
        indexing_request_id=request_id,
        indexing_claim_id=None,
        batch="batch-1",
        dataset_id="dataset-1",
        datasource_type="local_file",
        datasource_info={"name": "file"},
        start_node_id="start",
        call_depth=0,
        single_iteration_run=None,
        single_loop_run=None,
    )
    return (
        RagPipelineInvokeEntity(
            pipeline_id=pipeline_id,
            application_generate_entity=entity.model_dump(),
            user_id="actor-1",
            tenant_id="tenant-1",
            workflow_id="workflow-1",
            streaming=True,
            workflow_execution_id="run-1",
        ),
        entity,
    )


def _persist_pipeline_request(session_factory: sessionmaker[Session]) -> str:
    with session_factory.begin() as session:
        tenant = make_tenant(tenant_id="tenant-1")
        dataset = make_dataset(dataset_id="dataset-1", tenant_id=tenant.id, pipeline_id="pipeline-1")
        document = make_document(
            document_id="document-1",
            dataset_id=dataset.id,
            tenant_id=tenant.id,
            batch="batch-1",
        )
        document.indexing_status = IndexingStatus.WAITING
        workflow = make_workflow(
            workflow_id="workflow-1",
            tenant_id=tenant.id,
            app_id="pipeline-1",
            workflow_type=WorkflowType.RAG_PIPELINE,
            version="v1",
        )
        execution_log = DocumentPipelineExecutionLog(
            pipeline_id="pipeline-1",
            document_id=document.id,
            datasource_type="local_file",
            datasource_info=json.dumps({"name": "file"}),
            datasource_node_id="start",
            input_data={},
            created_by="actor-1",
            workflow_id=workflow.id,
            original_document_id=None,
        )
        session.add_all([tenant, dataset, document, workflow, execution_log])
        session.flush()
        requests = create_pipeline_document_indexing_intents(
            session,
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            pipeline_id="pipeline-1",
            pipeline_execution_log_ids={document.id: execution_log.id},
            actor_id="actor-1",
        )
    return requests["document-1"]


def test_pipeline_worker_claims_only_the_persisted_pipeline_owner(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tasks.rag_pipeline.rag_pipeline_run_task")
    request_id = _persist_pipeline_request(sqlite_session_factory)
    invoke_entity, entity = _pipeline_entity(request_id=request_id)
    monkeypatch.setattr(module.session_factory, "get_session_maker", lambda: sqlite_session_factory)
    monkeypatch.setattr(module, "pipeline_indexing_admission_error", lambda _tenant_id: None)

    owner = module._claim_pipeline_entity(invoke_entity, entity)

    assert owner is not None
    assert owner.request_id == request_id
    assert owner.claim.ref.document_id == "document-1"
    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.indexing_status == IndexingStatus.PARSING
        assert document.indexing_request_id == request_id
        assert document.indexing_claim_id == owner.claim.claim_id


def test_pipeline_worker_rejects_a_same_tenant_foreign_pipeline_before_claim(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tasks.rag_pipeline.rag_pipeline_run_task")
    request_id = _persist_pipeline_request(sqlite_session_factory)
    invoke_entity, entity = _pipeline_entity(request_id=request_id, pipeline_id="pipeline-2")
    monkeypatch.setattr(module.session_factory, "get_session_maker", lambda: sqlite_session_factory)

    with pytest.raises(ValueError, match="does not own"):
        module._claim_pipeline_entity(invoke_entity, entity)

    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.indexing_status == IndexingStatus.WAITING
        assert document.indexing_claim_id is None
        assert document.indexing_request_id == request_id
        assert session.scalar(select(DocumentIndexingOutbox.id)) == request_id


def test_pipeline_worker_uses_the_same_preclaim_quota_admission_as_replay(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tasks.rag_pipeline.rag_pipeline_run_task")
    request_id = _persist_pipeline_request(sqlite_session_factory)
    invoke_entity, entity = _pipeline_entity(request_id=request_id)
    monkeypatch.setattr(module.session_factory, "get_session_maker", lambda: sqlite_session_factory)
    monkeypatch.setattr(module, "pipeline_indexing_admission_error", lambda _tenant_id: "quota full")

    assert module._claim_pipeline_entity(invoke_entity, entity) is None

    with sqlite_session_factory() as session:
        document = session.get_one(Document, "document-1")
        assert document.indexing_status == IndexingStatus.ERROR
        assert document.error == "quota full"
        assert document.indexing_request_id is None
        assert session.scalar(select(DocumentIndexingOutbox.id)) is None

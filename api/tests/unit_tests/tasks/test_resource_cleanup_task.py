import json
from contextlib import nullcontext
from datetime import datetime, timedelta
from threading import Event
from time import monotonic

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

import tasks.clean_dataset_task as clean_dataset_module
import tasks.resource_cleanup_task as task_module
from constants.resource_access_token import ResourceAccessTokenResourceType
from extensions.storage.storage_type import StorageType
from models.account import Tenant
from models.dataset import Dataset, DatasetKeywordTable, DatasetMetadata, DatasetMetadataBinding, Document
from models.enums import DatasetMetadataType, DataSourceType
from models.model import (
    ChildVectorCleanupOutbox,
    DatasetIndexingOutbox,
    ResourceCleanupOutbox,
    SummaryVectorCleanupOutbox,
    UploadFile,
    UploadFileCleanupOutbox,
)
from repositories.knowledge.resource_cleanup_repository import (
    ResourceCleanupIntent,
    ResourceCleanupType,
    SQLAlchemyResourceCleanupRepository,
    create_dataset_cleanup_intent,
    create_document_cleanup_intent,
    delete_dataset_for_cleanup,
)
from services.knowledge.entities.datasets import DatasetIndexingRepresentation
from services.knowledge.entities.segments import ChildVectorRoute
from tests.unit_tests.model_factories import make_dataset, make_document, make_upload_file


def _child_route(collection_name: str, *, doc_form: str = "hierarchical_model") -> ChildVectorRoute:
    return ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name=collection_name,
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form=doc_form,
    )


class _ResourceRelationsBoundary:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, ResourceAccessTokenResourceType, str]] = []

    def delete_resource_relations(
        self,
        *,
        tenant_id: str,
        resource_type: ResourceAccessTokenResourceType,
        resource_id: str,
    ) -> None:
        self.deleted.append((tenant_id, resource_type, resource_id))


def _representation(
    route: ChildVectorRoute | None,
    *,
    vector_route: ChildVectorRoute | None = None,
) -> DatasetIndexingRepresentation:
    return DatasetIndexingRepresentation(
        indexing_technique="high_quality",
        index_struct=route.index_struct if route is not None else None,
        collection_binding_id=route.collection_binding_id if route is not None else None,
        embedding_model="embedding",
        embedding_model_provider="provider",
        keyword_number=10,
        is_multimodal=False,
        doc_form=route.doc_form if route is not None else "hierarchical_model",
        summary_index_setting=None,
        vector_route=vector_route if vector_route is not None else route,
    )


def _create_document_intent(
    session_factory: sessionmaker[Session],
    *,
    document_id: str = "document-1",
    source_file_id: str | None = "file-1",
) -> str:
    with session_factory.begin() as session:
        return create_document_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id=document_id,
            doc_form="text_model",
            source_file_id=source_file_id,
        )


def _use_repository(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
) -> SQLAlchemyResourceCleanupRepository:
    repository = SQLAlchemyResourceCleanupRepository(session_factory=session_factory)
    monkeypatch.setattr(task_module, "_repository", lambda: repository)
    return repository


def test_process_cleanup_acknowledges_only_after_success(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_id = _create_document_intent(sqlite_session_factory)
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    cleaned: list[ResourceCleanupIntent] = []
    monkeypatch.setattr(task_module, "_run_cleanup", cleaned.append)

    task_module.process_resource_cleanup_intent_task.run(intent_id)

    assert [(intent.resource_id, intent.source_file_id) for intent in cleaned] == [("document-1", "file-1")]
    assert repository.get_intent(intent_id) is None


def test_dataset_delete_hands_off_applied_and_partial_routes_for_independent_cleanup(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied_route = _child_route("collection-a")
    partial_route = _child_route("collection-b")
    desired_route = _child_route("collection-c")
    tenant = Tenant(name="Tenant")
    tenant.id = "tenant-1"
    dataset = make_dataset(
        dataset_id="dataset-1",
        tenant_id=tenant.id,
        indexing_technique="high_quality",
        index_struct=desired_route.index_struct,
    )
    dataset.indexing_applied_representation = _representation(applied_route).model_dump_json()
    dataset.indexing_request_id = "generation-1"
    pending = DatasetIndexingOutbox(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        generation_id="generation-1",
        operation="rebuild",
        action="update",
        source_representation=_representation(applied_route).model_dump_json(),
        target_representation=_representation(desired_route, vector_route=partial_route).model_dump_json(),
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, dataset, pending])
    pending_id = pending.id

    relations = _ResourceRelationsBoundary()
    with sqlite_session_factory.begin() as session:
        deletion = delete_dataset_for_cleanup(
            session,
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            resource_relations=relations,
        )
        assert deletion is not None
    assert relations.deleted == [("tenant-1", ResourceAccessTokenResourceType.KNOWLEDGE, "dataset-1")]

    repository = _use_repository(monkeypatch, sqlite_session_factory)
    parent_id, *vector_ids = deletion.cleanup_intent_ids
    assert len(vector_ids) == 2
    parent = repository.get_intent(parent_id)
    assert parent is not None
    assert parent.resource_type == ResourceCleanupType.DATASET_RELATIONAL
    with sqlite_session_factory() as session:
        assert session.get(Dataset, dataset.id) is None
        assert session.get(DatasetIndexingOutbox, pending_id) is None
        rows = session.scalars(select(ResourceCleanupOutbox).order_by(ResourceCleanupOutbox.resource_type)).all()
        assert len(rows) == 3
    persisted_routes = [repository.get_intent(intent_id) for intent_id in vector_ids]
    assert all(intent is not None for intent in persisted_routes)
    assert {
        intent.vector_route.collection_name
        for intent in persisted_routes
        if intent is not None and intent.vector_route is not None
    } == {"collection-a", "collection-b"}

    cleaned_routes: list[ChildVectorRoute] = []

    def clean_vector(route: ChildVectorRoute) -> ChildVectorRoute:
        cleaned_routes.append(route)
        return route

    parent_calls: list[str] = []
    monkeypatch.setattr(task_module, "run_dataset_vector_cleanup", clean_vector)
    monkeypatch.setattr(
        task_module,
        "run_dataset_relational_cleanup",
        lambda **kwargs: parent_calls.append(str(kwargs["dataset_id"])),
    )
    for intent_id in vector_ids:
        task_module.process_resource_cleanup_intent_task.run(intent_id)
        assert repository.get_intent(intent_id) is None
    task_module.process_resource_cleanup_intent_task.run(parent_id)

    assert {route.collection_name for route in cleaned_routes} == {"collection-a", "collection-b"}
    assert all(route.collection_name != "collection-c" for route in cleaned_routes)
    assert parent_calls == ["dataset-1"]
    with sqlite_session_factory() as session:
        assert session.scalar(select(ResourceCleanupOutbox.id)) is None


def test_qa_dataset_exact_route_and_relational_parent_cleanup_are_independent(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A route-backed QA deletion must never resolve a default vector backend from its parent."""

    route = _child_route("qa-collection-a", doc_form="qa_model")
    tenant = Tenant(name="Tenant")
    tenant.id = "tenant-1"
    dataset = make_dataset(
        dataset_id="dataset-1",
        tenant_id=tenant.id,
        indexing_technique="high_quality",
        index_struct=route.index_struct,
        chunk_structure="qa_model",
    )
    dataset.indexing_applied_representation = _representation(route).model_dump_json()
    source_file = make_upload_file(
        file_id="source-file-1",
        tenant_id=tenant.id,
        storage_type=StorageType.LOCAL,
        key="upload_files/source-file-1.pdf",
        name="source.pdf",
        extension="pdf",
        mime_type="application/pdf",
    )
    document = make_document(
        document_id="document-1",
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info=json.dumps({"upload_file_id": source_file.id}),
        doc_form="qa_model",
    )
    metadata = DatasetMetadata(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        type=DatasetMetadataType.STRING,
        name="category",
        created_by="account-1",
    )
    metadata_binding = DatasetMetadataBinding(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        metadata_id=metadata.id,
        document_id=document.id,
        created_by="account-1",
    )
    keyword_table = DatasetKeywordTable(
        dataset_id=dataset.id,
        keyword_table=json.dumps({"keyword": ["node-1"]}),
        data_source_type="database",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, dataset, source_file, document, metadata, metadata_binding, keyword_table])
    metadata_id = metadata.id
    binding_id = metadata_binding.id

    with sqlite_session_factory.begin() as session:
        deletion = delete_dataset_for_cleanup(
            session,
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            resource_relations=_ResourceRelationsBoundary(),
        )
        assert deletion is not None

    repository = _use_repository(monkeypatch, sqlite_session_factory)
    intents = tuple(repository.get_intent(intent_id) for intent_id in deletion.cleanup_intent_ids)
    parent = next(
        intent
        for intent in intents
        if intent is not None and intent.resource_type == ResourceCleanupType.DATASET_RELATIONAL
    )
    vector_intent = next(
        intent
        for intent in intents
        if intent is not None and intent.resource_type == ResourceCleanupType.DATASET_VECTOR
    )
    assert vector_intent.vector_route == route

    class _ExactVectorBoundary:
        def __init__(self) -> None:
            self.deleted = False

        def delete(self) -> None:
            self.deleted = True

    class _StorageBoundary:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def load_once(self, _key: str) -> bytes:
            raise AssertionError("keyword storage read is not expected")

        def save(self, _key: str, _data: bytes) -> None:
            raise AssertionError("keyword storage write is not expected")

        def delete(self, key: str) -> None:
            self.deleted.append(key)

    class _RedisBoundary:
        def lock(self, _name: str, *, timeout: int) -> nullcontext[None]:
            assert timeout == 600
            return nullcontext()

    exact_vector = _ExactVectorBoundary()
    resolved_routes: list[ChildVectorRoute] = []
    storage = _StorageBoundary()
    billing_refreshes: list[str] = []

    def resolve_exact(candidate: ChildVectorRoute) -> _ExactVectorBoundary:
        resolved_routes.append(candidate)
        return exact_vector

    def reject_default_backend(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("relational parent attempted default vector backend resolution")

    monkeypatch.setattr(clean_dataset_module, "vector_for_route", resolve_exact)
    monkeypatch.setattr(clean_dataset_module, "Vector", reject_default_backend)
    monkeypatch.setattr(clean_dataset_module, "IndexProcessorFactory", reject_default_backend)
    monkeypatch.setattr(clean_dataset_module, "storage", storage)
    monkeypatch.setattr(clean_dataset_module, "redis_client", _RedisBoundary())
    monkeypatch.setattr(clean_dataset_module, "schedule_billing_vector_space_refresh", billing_refreshes.append)

    task_module.process_resource_cleanup_intent_task.run(vector_intent.id)
    task_module.process_resource_cleanup_intent_task.run(parent.id)

    assert resolved_routes == [route]
    assert exact_vector.deleted
    assert storage.deleted == [source_file.key]
    assert billing_refreshes == [tenant.id]
    with sqlite_session_factory() as session:
        assert session.get(Document, document.id) is None
        assert session.get(UploadFile, source_file.id) is None
        assert (
            session.scalar(
                select(UploadFileCleanupOutbox.id).where(
                    UploadFileCleanupOutbox.tenant_id == tenant.id,
                    UploadFileCleanupOutbox.upload_file_id == source_file.id,
                )
            )
            is None
        )
        assert session.get(DatasetMetadata, metadata_id) is None
        assert session.get(DatasetMetadataBinding, binding_id) is None
        assert session.get(DatasetKeywordTable, keyword_table.id) is None
        assert session.scalar(select(ResourceCleanupOutbox.id)) is None


def test_dataset_delete_during_economy_transition_cleans_applied_high_quality_route(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied_route = _child_route("collection-a")
    applied = _representation(applied_route)
    economy = applied.model_copy(
        update={
            "indexing_technique": "economy",
            "index_struct": None,
            "collection_binding_id": None,
            "embedding_model": None,
            "embedding_model_provider": None,
            "vector_route": None,
            "cleanup_routes": (),
        }
    )
    tenant = Tenant(name="Tenant")
    tenant.id = "tenant-1"
    dataset = make_dataset(dataset_id="dataset-1", tenant_id=tenant.id, indexing_technique="economy")
    dataset.indexing_applied_representation = applied.model_dump_json()
    dataset.indexing_request_id = "generation-1"
    pending = DatasetIndexingOutbox(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        generation_id="generation-1",
        operation="rebuild",
        action="remove",
        source_representation=applied.model_dump_json(),
        target_representation=economy.model_dump_json(),
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, dataset, pending])
    with sqlite_session_factory.begin() as session:
        deletion = delete_dataset_for_cleanup(
            session,
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            resource_relations=_ResourceRelationsBoundary(),
        )
        assert deletion is not None

    repository = _use_repository(monkeypatch, sqlite_session_factory)
    vector_intents = tuple(
        intent
        for intent_id in deletion.cleanup_intent_ids
        if (intent := repository.get_intent(intent_id)) is not None
        and intent.resource_type == ResourceCleanupType.DATASET_VECTOR
    )
    assert len(vector_intents) == 1
    assert vector_intents[0].vector_route == applied_route
    cleaned: list[ChildVectorRoute] = []

    def clean_vector(route: ChildVectorRoute) -> ChildVectorRoute:
        cleaned.append(route)
        return route

    monkeypatch.setattr(task_module, "run_dataset_vector_cleanup", clean_vector)
    task_module.process_resource_cleanup_intent_task.run(vector_intents[0].id)

    assert cleaned == [applied_route]
    assert repository.get_intent(vector_intents[0].id) is None


def test_dataset_cleanup_acknowledges_routed_summary_but_retains_legacy_child_cleanup(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted_route = _child_route("collection-a")
    with sqlite_session_factory.begin() as session:
        intent_id = create_dataset_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="hierarchical_model",
            indexing_technique="high_quality",
            index_struct=deleted_route.index_struct,
            collection_binding_id=None,
            pipeline_id=None,
        )
        session.add(
            SummaryVectorCleanupOutbox(
                tenant_id="tenant-1",
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="segment-1",
                summary_id="summary-1",
                summary_version=1,
                node_id="finished-node",
                writer_id="writer-1",
                vector_route=deleted_route.serialized,
                vector_route_hash=deleted_route.key,
                writer_finished_at=datetime(2026, 10, 24),
            )
        )
        session.add(
            ChildVectorCleanupOutbox(
                tenant_id="tenant-1",
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="segment-1",
                child_id="child-1",
                node_id="finished-child-node",
                writer_id="child-writer-1",
                writer_finished_at=datetime(2026, 10, 24),
            )
        )
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    monkeypatch.setattr(task_module, "_run_cleanup", lambda _intent: deleted_route)

    task_module.process_resource_cleanup_intent_task.run(intent_id)

    assert repository.get_intent(intent_id) is None
    with sqlite_session_factory() as session:
        assert session.scalar(select(SummaryVectorCleanupOutbox.id)) is None
        assert session.scalar(select(ChildVectorCleanupOutbox.id)) is not None


def test_dataset_cleanup_acknowledges_only_children_in_the_collection_it_deleted(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_route = _child_route("collection-a")
    deleted_route = _child_route("collection-b")
    with sqlite_session_factory.begin() as session:
        intent_id = create_dataset_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="hierarchical_model",
            indexing_technique="high_quality",
            index_struct=deleted_route.index_struct,
            collection_binding_id=None,
            pipeline_id=None,
        )
        rows = [
            ChildVectorCleanupOutbox(
                tenant_id="tenant-1",
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="segment-1",
                child_id=f"child-{position}",
                node_id="same-node",
                writer_id=f"writer-{position}",
                writer_finished_at=datetime(2026, 10, 24),
                vector_route=route.serialized,
                vector_route_hash=route.key,
            )
            for position, route in enumerate((old_route, deleted_route), start=1)
        ]
        session.add_all(rows)
        old_cleanup_id, deleted_cleanup_id = (row.id for row in rows)
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    monkeypatch.setattr(task_module, "_run_cleanup", lambda _intent: deleted_route)

    task_module.process_resource_cleanup_intent_task.run(intent_id)

    assert repository.get_intent(intent_id) is None
    with sqlite_session_factory() as session:
        assert session.get(ChildVectorCleanupOutbox, old_cleanup_id) is not None
        assert session.get(ChildVectorCleanupOutbox, deleted_cleanup_id) is None


def test_dataset_vector_cleanup_is_not_blocked_by_unfinished_writer_on_another_route(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_a = _child_route("collection-a")
    route_b = _child_route("collection-b")
    with sqlite_session_factory.begin() as session:
        route_intent = ResourceCleanupOutbox(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            resource_type=ResourceCleanupType.DATASET_VECTOR.value,
            resource_id="00000000-0000-0000-0000-000000000001",
            doc_form=route_a.doc_form,
            indexing_technique=route_a.indexing_technique,
            index_struct=route_a.index_struct,
            collection_binding_id=route_a.collection_binding_id,
        )
        unrelated_writer = SummaryVectorCleanupOutbox(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            summary_id="summary-1",
            summary_version=1,
            node_id="late-node",
            writer_id="writer-1",
            vector_route=route_b.serialized,
            vector_route_hash=route_b.key,
        )
        session.add_all([route_intent, unrelated_writer])
        route_intent_id = route_intent.id
        writer_id = unrelated_writer.id
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    cleaned: list[ChildVectorRoute] = []

    def clean_vector(route: ChildVectorRoute) -> ChildVectorRoute:
        cleaned.append(route)
        return route

    monkeypatch.setattr(task_module, "run_dataset_vector_cleanup", clean_vector)
    task_module.process_resource_cleanup_intent_task.run(route_intent_id)

    assert cleaned == [route_a]
    assert repository.get_intent(route_intent_id) is None
    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, writer_id) is not None


def test_dataset_cleanup_retains_parent_and_unfinished_child_writer(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        intent_id = create_dataset_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="hierarchical_model",
            indexing_technique="high_quality",
            index_struct='{"type":"test"}',
            collection_binding_id="collection-1",
            pipeline_id=None,
        )
        writer = ChildVectorCleanupOutbox(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            child_id="child-1",
            node_id="unknown-child-node",
            writer_id="child-writer-1",
        )
        session.add(writer)
        writer_id = writer.id
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    monkeypatch.setattr(task_module, "_run_cleanup", lambda _intent: None)

    task_module.process_resource_cleanup_intent_task.run(intent_id)

    retained = repository.get_intent(intent_id)
    assert retained is not None
    assert retained.claim_id is None
    with sqlite_session_factory() as session:
        assert session.get(ChildVectorCleanupOutbox, writer_id) is not None


def test_dataset_cleanup_repeats_when_a_writer_finishes_after_collection_deletion(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted_route = _child_route("collection-a")
    with sqlite_session_factory.begin() as session:
        intent_id = create_dataset_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="hierarchical_model",
            indexing_technique="high_quality",
            index_struct=deleted_route.index_struct,
            collection_binding_id=None,
            pipeline_id=None,
        )
        writer = SummaryVectorCleanupOutbox(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            summary_id="summary-1",
            summary_version=1,
            node_id="late-node",
            writer_id="writer-1",
            vector_route=deleted_route.serialized,
            vector_route_hash=deleted_route.key,
        )
        session.add(writer)
        writer_id = writer.id
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    nodes = {"original-node"}

    def delete_then_finish_writer(_intent: ResourceCleanupIntent) -> ChildVectorRoute:
        nodes.clear()
        # The remote write can finish after the provider deleted the collection
        # but before its response lets this task acknowledge the parent intent.
        nodes.add("late-node")
        with sqlite_session_factory.begin() as session:
            persisted = session.get_one(SummaryVectorCleanupOutbox, writer_id)
            persisted.writer_finished_at = datetime(2026, 10, 24)
        return deleted_route

    monkeypatch.setattr(task_module, "_run_cleanup", delete_then_finish_writer)
    task_module.process_resource_cleanup_intent_task.run(intent_id)

    retained = repository.get_intent(intent_id)
    assert retained is not None
    assert retained.claim_id is None
    assert retained.index_struct == deleted_route.index_struct
    assert nodes == {"late-node"}
    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, writer_id) is not None

    def delete_finished_writer(_intent: ResourceCleanupIntent) -> ChildVectorRoute:
        nodes.clear()
        return deleted_route

    monkeypatch.setattr(task_module, "_run_cleanup", delete_finished_writer)
    task_module.process_resource_cleanup_intent_task.run(intent_id)

    assert nodes == set()
    assert repository.get_intent(intent_id) is None
    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, writer_id) is None


def test_dataset_cleanup_retains_parent_and_unfinished_summary_writer(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory.begin() as session:
        intent_id = create_dataset_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="text_model",
            indexing_technique="high_quality",
            index_struct='{"type":"test"}',
            collection_binding_id="collection-1",
            pipeline_id=None,
        )
        writer = SummaryVectorCleanupOutbox(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            summary_id="summary-1",
            summary_version=1,
            node_id="unknown-node",
            writer_id="writer-1",
        )
        session.add(writer)
        writer_id = writer.id
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    monkeypatch.setattr(task_module, "_run_cleanup", lambda _intent: None)

    task_module.process_resource_cleanup_intent_task.run(intent_id)

    retained = repository.get_intent(intent_id)
    assert retained is not None
    assert retained.claim_id is None
    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, writer_id) is not None


@pytest.mark.parametrize(
    ("writer_tenant", "writer_dataset"),
    [("tenant-1", "other-dataset"), ("other-tenant", "dataset-1")],
)
def test_unrelated_summary_writers_do_not_keep_dataset_cleanup_pending(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    writer_tenant: str,
    writer_dataset: str,
) -> None:
    with sqlite_session_factory.begin() as session:
        intent_id = create_dataset_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form=None,
            indexing_technique=None,
            index_struct=None,
            collection_binding_id=None,
            pipeline_id=None,
        )
        writer = SummaryVectorCleanupOutbox(
            tenant_id=writer_tenant,
            dataset_id=writer_dataset,
            document_id="document-1",
            segment_id="segment-1",
            summary_id="summary-1",
            summary_version=1,
            node_id="node-1",
            writer_id="writer-1",
            writer_finished_at=datetime(2026, 10, 24),
        )
        session.add(writer)
        writer_id = writer.id
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    monkeypatch.setattr(task_module, "_run_cleanup", lambda _intent: None)

    task_module.process_resource_cleanup_intent_task.run(intent_id)

    assert repository.get_intent(intent_id) is None
    with sqlite_session_factory() as session:
        assert session.get(SummaryVectorCleanupOutbox, writer_id) is not None


def test_committed_dataset_ack_is_not_revived_when_commit_result_is_unknown(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted_route = _child_route("collection-a")
    with sqlite_session_factory.begin() as session:
        intent_id = create_dataset_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            doc_form="hierarchical_model",
            indexing_technique="high_quality",
            index_struct=deleted_route.index_struct,
            collection_binding_id=None,
            pipeline_id=None,
        )
        session.add(
            SummaryVectorCleanupOutbox(
                tenant_id="tenant-1",
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="segment-1",
                summary_id="summary-1",
                summary_version=1,
                node_id="finished-node",
                writer_id="writer-1",
                vector_route=deleted_route.serialized,
                vector_route_hash=deleted_route.key,
                writer_finished_at=datetime(2026, 10, 24),
            )
        )
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    complete = repository.complete

    def commit_then_report_unknown(
        intent: ResourceCleanupIntent,
        *,
        claim_id: str,
        deleted_child_route: ChildVectorRoute | None,
    ) -> None:
        complete(intent, claim_id=claim_id, deleted_child_route=deleted_child_route)
        raise RuntimeError("commit outcome unknown")

    monkeypatch.setattr(task_module, "_run_cleanup", lambda _intent: deleted_route)
    monkeypatch.setattr(repository, "complete", commit_then_report_unknown)

    with pytest.raises(RuntimeError, match="commit outcome unknown"):
        task_module.process_resource_cleanup_intent_task.run(intent_id)
    task_module.process_resource_cleanup_intent_task.run(intent_id)

    assert repository.get_intent(intent_id) is None
    with sqlite_session_factory() as session:
        assert session.scalar(select(SummaryVectorCleanupOutbox.id)) is None


def test_document_intent_routes_captured_tenant_and_source_without_dataset_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(task_module, "run_document_cleanup", lambda **kwargs: calls.append(kwargs))
    intent = ResourceCleanupIntent(
        id="intent-1",
        tenant_id="tenant-1",
        dataset_id="deleted-dataset",
        resource_type=ResourceCleanupType.DOCUMENT,
        resource_id="deleted-document",
        doc_form="qa_model",
        source_file_id="file-1",
        indexing_technique=None,
        index_struct=None,
        collection_binding_id=None,
        pipeline_id=None,
    )

    task_module._run_cleanup(intent)

    assert calls == [
        {
            "tenant_id": "tenant-1",
            "dataset_id": "deleted-dataset",
            "document_id": "deleted-document",
            "doc_form": "qa_model",
            "file_id": "file-1",
        }
    ]


def test_dataset_intent_without_index_configuration_still_routes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(task_module, "run_dataset_cleanup", lambda **kwargs: calls.append(kwargs))
    intent = ResourceCleanupIntent(
        id="intent-1",
        tenant_id="tenant-1",
        dataset_id="deleted-dataset",
        resource_type=ResourceCleanupType.DATASET,
        resource_id="deleted-dataset",
        doc_form=None,
        source_file_id=None,
        indexing_technique=None,
        index_struct=None,
        collection_binding_id=None,
        pipeline_id=None,
    )

    task_module._run_cleanup(intent)

    assert calls == [
        {
            "tenant_id": "tenant-1",
            "dataset_id": "deleted-dataset",
            "indexing_technique": None,
            "index_struct": None,
            "collection_binding_id": None,
            "doc_form": "text_model",
            "pipeline_id": None,
        }
    ]


@pytest.mark.parametrize("failure_stage", ["cleanup", "ack"])
def test_process_cleanup_retains_intent_after_retryable_failure(
    failure_stage: str,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_id = _create_document_intent(sqlite_session_factory)
    repository = _use_repository(monkeypatch, sqlite_session_factory)

    def fail(_intent: ResourceCleanupIntent, **_kwargs: object) -> None:
        raise RuntimeError(f"{failure_stage} failed")

    if failure_stage == "cleanup":
        monkeypatch.setattr(task_module, "_run_cleanup", fail)
    else:
        monkeypatch.setattr(task_module, "_run_cleanup", lambda _intent: None)
        monkeypatch.setattr(repository, "complete", fail)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        task_module.process_resource_cleanup_intent_task.run(intent_id)

    assert repository.get_intent(intent_id) is not None
    with sqlite_session_factory() as session:
        row = session.get(ResourceCleanupOutbox, intent_id)
        assert row is not None
        assert row.last_attempted_at is not None
        assert row.claim_id is None
        assert row.claim_expires_at is None


def test_process_cleanup_rejects_an_intent_whose_source_still_exists(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit_tests.model_factories import make_dataset, make_document

    dataset = make_dataset()
    document = make_document()
    with sqlite_session_factory.begin() as session:
        session.add_all([dataset, document])
        intent_id = create_document_cleanup_intent(
            session,
            tenant_id=document.tenant_id,
            dataset_id=document.dataset_id,
            document_id=document.id,
            doc_form=document.doc_form,
            source_file_id=None,
        )
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    cleaned: list[ResourceCleanupIntent] = []
    monkeypatch.setattr(task_module, "_run_cleanup", cleaned.append)

    with pytest.raises(task_module.CleanupSourceStillExistsError):
        task_module.process_resource_cleanup_intent_task.run(intent_id)

    assert cleaned == []
    assert repository.get_intent(intent_id) is not None


def test_enqueue_swallows_broker_failure_for_committed_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_publish(_intent_id: str) -> None:
        raise OSError("broker unavailable")

    monkeypatch.setattr(task_module.process_resource_cleanup_intent_task, "delay", fail_publish)

    task_module.enqueue_resource_cleanup_intent("intent-1")


def test_periodic_drain_rotates_a_full_batch_before_newer_work(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_ids = [
        _create_document_intent(
            sqlite_session_factory,
            document_id=f"document-{position:03d}",
            source_file_id=None,
        )
        for position in range(task_module._DRAIN_BATCH_SIZE + 1)
    ]
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    first_batch = repository.list_pending_intent_ids(limit=task_module._DRAIN_BATCH_SIZE)
    remaining = set(intent_ids).difference(first_batch)
    published: list[str] = []
    monkeypatch.setattr(task_module, "enqueue_resource_cleanup_intent", published.append)

    task_module.drain_resource_cleanup_outbox_task.run()

    assert len(published) == task_module._DRAIN_BATCH_SIZE
    assert tuple(published) == first_batch
    assert set(repository.list_pending_intent_ids(limit=1)) == remaining


def test_periodic_drain_skips_active_cleanup_and_recovers_expired_lease(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_id = _create_document_intent(sqlite_session_factory)
    repository = _use_repository(monkeypatch, sqlite_session_factory)
    claim = repository.claim(intent_id)
    assert claim is not None
    published: list[str] = []
    monkeypatch.setattr(task_module, "enqueue_resource_cleanup_intent", published.append)

    task_module.drain_resource_cleanup_outbox_task.run()
    assert published == []

    with sqlite_session_factory.begin() as session:
        row = session.get_one(ResourceCleanupOutbox, intent_id)
        database_now = session.scalar(select(func.current_timestamp()))
        assert isinstance(database_now, datetime)
        row.claim_expires_at = database_now - timedelta(seconds=1)

    task_module.drain_resource_cleanup_outbox_task.run()
    assert published == [intent_id]


def test_cleanup_heartbeat_renews_claim_until_work_finishes() -> None:
    renewed = Event()

    class Repository:
        def renew_claim(self, intent_id: str, *, claim_id: str) -> None:
            assert (intent_id, claim_id) == ("intent-1", "claim-1")
            renewed.set()

    intent = ResourceCleanupIntent(
        id="intent-1",
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        resource_type=ResourceCleanupType.DOCUMENT,
        resource_id="document-1",
        doc_form="text_model",
        source_file_id=None,
        indexing_technique=None,
        index_struct=None,
        collection_binding_id=None,
        pipeline_id=None,
        claim_id="claim-1",
    )

    with task_module._maintain_cleanup_claim(Repository(), intent, interval_seconds=0.001):
        assert renewed.wait(timeout=1)


@pytest.mark.parametrize("operation_fails", [False, True])
def test_cleanup_heartbeat_shutdown_does_not_wait_for_blocked_renewal(operation_fails: bool) -> None:
    renewal_started = Event()
    release_renewal = Event()
    renewal_finished = Event()
    exit_started_at: list[float] = []

    class Repository:
        def renew_claim(self, intent_id: str, *, claim_id: str) -> None:
            assert (intent_id, claim_id) == ("intent-1", "claim-1")
            renewal_started.set()
            release_renewal.wait(timeout=2)
            renewal_finished.set()

    intent = ResourceCleanupIntent(
        id="intent-1",
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        resource_type=ResourceCleanupType.DOCUMENT,
        resource_id="document-1",
        doc_form="text_model",
        source_file_id=None,
        indexing_technique=None,
        index_struct=None,
        collection_binding_id=None,
        pipeline_id=None,
        claim_id="claim-1",
    )

    def run_context() -> None:
        with task_module._maintain_cleanup_claim(
            Repository(),
            intent,
            interval_seconds=0.001,
        ):
            assert renewal_started.wait(timeout=1)
            exit_started_at.append(monotonic())
            if operation_fails:
                raise RuntimeError("cleanup failed")

    try:
        if operation_fails:
            with pytest.raises(RuntimeError, match="cleanup failed"):
                run_context()
        else:
            run_context()
        assert monotonic() - exit_started_at[0] < 0.5
    finally:
        release_renewal.set()
    assert renewal_finished.wait(timeout=1)

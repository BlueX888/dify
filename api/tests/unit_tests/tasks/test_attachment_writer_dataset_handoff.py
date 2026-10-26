from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import tasks.attachment_vector_cleanup_task as attachment_cleanup_task
from constants.resource_access_token import ResourceAccessTokenResourceType
from models.account import Tenant
from models.dataset import Dataset, SegmentAttachmentBinding
from models.model import AttachmentVectorCleanupOutbox, ResourceCleanupOutbox
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.resource_cleanup_repository import (
    ResourceCleanupIntent,
    ResourceCleanupType,
    SQLAlchemyResourceCleanupRepository,
    delete_dataset_for_cleanup,
)
from services.knowledge.entities.datasets import DatasetIndexingRepresentation
from services.knowledge.entities.vector_route import VectorRoute
from tasks import resource_cleanup_task
from tests.unit_tests.model_factories import make_dataset


def _route(collection_name: str) -> VectorRoute:
    return VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name=collection_name,
        collection_binding_id=f"binding-{collection_name}",
        indexing_technique="high_quality",
        doc_form="attachment",
    )


def _representation(route: VectorRoute) -> DatasetIndexingRepresentation:
    return DatasetIndexingRepresentation(
        indexing_technique=route.indexing_technique,
        index_struct=route.index_struct,
        collection_binding_id=route.collection_binding_id,
        embedding_model="embedding",
        embedding_model_provider="provider",
        keyword_number=10,
        is_multimodal=True,
        doc_form=route.doc_form,
        summary_index_setting=None,
        vector_route=route,
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


def _use_resource_repository(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
) -> SQLAlchemyResourceCleanupRepository:
    repository = SQLAlchemyResourceCleanupRepository(session_factory=session_factory)
    monkeypatch.setattr(resource_cleanup_task, "_repository", lambda: repository)
    return repository


def _use_attachment_repository(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
) -> SQLAlchemyAttachmentVectorStore:
    repository = SQLAlchemyAttachmentVectorStore(session_factory=session_factory)
    monkeypatch.setattr(attachment_cleanup_task, "_repository", lambda: repository)
    return repository


def _terminal_receipt(
    session_factory: sessionmaker[Session],
    *,
    route: VectorRoute,
    attachment_id: str,
) -> str:
    with session_factory.begin() as session:
        receipt = AttachmentVectorCleanupOutbox(
            tenant_id=route.tenant_id,
            dataset_id=route.dataset_id,
            attachment_id=attachment_id,
            writer_id=f"writer-{attachment_id}",
            vector_route=route.serialized,
            vector_route_hash=route.key,
            writer_finished_at=datetime(2026, 10, 25),
            cleanup_started_at=datetime(2026, 10, 25),
        )
        session.add(receipt)
        session.flush()
        return receipt.id


def _install_routed_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[str, ...]]]:
    deleted: list[tuple[str, tuple[str, ...]]] = []

    class RoutedVector:
        @staticmethod
        def resolve_vector_type(dataset: Dataset, *, session: Session) -> str:
            assert session is not None
            index_struct = dataset.index_struct_dict
            assert index_struct is not None
            return str(index_struct["type"])

        def __init__(self, dataset: Dataset, *, session: Session | None, vector_type: str) -> None:
            del session
            index_struct = dataset.index_struct_dict
            assert index_struct is not None
            self._target = (vector_type, str(index_struct["vector_store"]["class_prefix"]))

        def get_routing_target(self) -> tuple[str, str]:
            return self._target

        def delete_by_ids(self, node_ids: list[str]) -> None:
            deleted.append((self._target[1], tuple(node_ids)))

    monkeypatch.setattr(attachment_cleanup_task, "Vector", RoutedVector)
    return deleted


def test_exact_dataset_cleanup_waits_for_unfinished_attachment_writer_then_acknowledges_terminal_receipt(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_a = _route("attachments-a")
    tenant = Tenant(name="Tenant")
    tenant.id = route_a.tenant_id
    dataset = make_dataset(
        dataset_id=route_a.dataset_id,
        tenant_id=route_a.tenant_id,
        indexing_technique=route_a.indexing_technique,
        index_struct=route_a.index_struct,
    )
    dataset.collection_binding_id = route_a.collection_binding_id
    dataset.is_multimodal = True
    dataset.indexing_applied_representation = _representation(route_a).model_dump_json()
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, dataset])

    attachment_store = SQLAlchemyAttachmentVectorStore(session_factory=sqlite_session_factory)
    (writer,) = attachment_store.begin_writes(
        tenant_id=route_a.tenant_id,
        dataset_id=route_a.dataset_id,
        attachment_ids=("attachment-1",),
        route=route_a,
    )

    relations = _ResourceRelationsBoundary()
    with sqlite_session_factory.begin() as session:
        deletion = delete_dataset_for_cleanup(
            session,
            tenant_id=route_a.tenant_id,
            dataset_id=route_a.dataset_id,
            resource_relations=relations,
        )
        assert deletion is not None
    assert relations.deleted == [(route_a.tenant_id, ResourceAccessTokenResourceType.KNOWLEDGE, route_a.dataset_id)]

    parent_repository = _use_resource_repository(monkeypatch, sqlite_session_factory)
    intents = tuple(
        intent
        for intent_id in deletion.cleanup_intent_ids
        if (intent := parent_repository.get_intent(intent_id)) is not None
    )
    exact_parent = next(intent for intent in intents if intent.resource_type == ResourceCleanupType.DATASET_VECTOR)
    relational_parent = next(
        intent for intent in intents if intent.resource_type == ResourceCleanupType.DATASET_RELATIONAL
    )
    cleaned: list[VectorRoute] = []

    def clean_exact_route(intent: ResourceCleanupIntent) -> VectorRoute | None:
        if intent.resource_type == ResourceCleanupType.DATASET_VECTOR:
            assert intent.vector_route is not None
            cleaned.append(intent.vector_route)
            return intent.vector_route
        return None

    monkeypatch.setattr(resource_cleanup_task, "_run_cleanup", clean_exact_route)

    resource_cleanup_task.process_resource_cleanup_intent_task.run(exact_parent.id)

    retained = parent_repository.get_intent(exact_parent.id)
    assert retained is not None
    assert retained.claim_id is None
    assert cleaned == [route_a]
    with sqlite_session_factory() as session:
        receipt = session.get_one(AttachmentVectorCleanupOutbox, writer.id)
        assert receipt.writer_finished_at is None
        assert session.get(Dataset, route_a.dataset_id) is None

    (terminal_claim,) = attachment_store.finish_writes((writer,))
    assert terminal_claim.writer_finished
    resource_cleanup_task.process_resource_cleanup_intent_task.run(exact_parent.id)
    resource_cleanup_task.process_resource_cleanup_intent_task.run(relational_parent.id)

    assert cleaned == [route_a, route_a]
    assert parent_repository.get_intent(exact_parent.id) is None
    assert parent_repository.get_intent(relational_parent.id) is None
    with sqlite_session_factory() as session:
        assert session.get(AttachmentVectorCleanupOutbox, writer.id) is None
        assert session.scalar(select(ResourceCleanupOutbox.id)) is None


def test_attachment_cleanup_uses_persisted_route_after_dataset_and_parent_are_gone(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_a = _route("attachments-a")
    _use_attachment_repository(monkeypatch, sqlite_session_factory)
    cleanup_id = _terminal_receipt(
        sqlite_session_factory,
        route=route_a,
        attachment_id="attachment-1",
    )
    deleted = _install_routed_vector(monkeypatch)

    attachment_cleanup_task.process_attachment_vector_cleanup_task.run(cleanup_id)

    assert deleted == [(route_a.collection_name, ("attachment-1",))]
    with sqlite_session_factory() as session:
        assert session.get(Dataset, route_a.dataset_id) is None
        assert session.scalar(select(ResourceCleanupOutbox.id)) is None
        assert session.get(AttachmentVectorCleanupOutbox, cleanup_id) is None


def test_current_route_b_binding_cannot_ack_old_route_a_without_deleting_route_a(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_a = _route("attachments-a")
    route_b = _route("attachments-b")
    dataset = make_dataset(
        dataset_id=route_b.dataset_id,
        tenant_id=route_b.tenant_id,
        indexing_technique=route_b.indexing_technique,
        index_struct=route_b.index_struct,
    )
    dataset.collection_binding_id = route_b.collection_binding_id
    dataset.is_multimodal = True
    binding = SegmentAttachmentBinding(
        tenant_id=route_b.tenant_id,
        dataset_id=route_b.dataset_id,
        document_id="document-current",
        segment_id="segment-current",
        attachment_id="attachment-1",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([dataset, binding])

    _use_attachment_repository(monkeypatch, sqlite_session_factory)
    cleanup_id = _terminal_receipt(
        sqlite_session_factory,
        route=route_a,
        attachment_id=binding.attachment_id,
    )
    deleted = _install_routed_vector(monkeypatch)

    attachment_cleanup_task.process_attachment_vector_cleanup_task.run(cleanup_id)

    assert deleted == [(route_a.collection_name, (binding.attachment_id,))]
    with sqlite_session_factory() as session:
        assert session.get(AttachmentVectorCleanupOutbox, cleanup_id) is None
        assert session.get(SegmentAttachmentBinding, binding.id) is not None
        current = session.get_one(Dataset, route_b.dataset_id)
        assert current.index_struct == route_b.index_struct

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session, sessionmaker

from constants.resource_access_token import ResourceAccessTokenResourceType
from models.account import Tenant
from models.dataset import Dataset
from models.model import DatasetIndexingOutbox, ResourceCleanupOutbox, SummaryVectorCleanupOutbox
from repositories.knowledge.resource_cleanup_repository import (
    ResourceCleanupClaimLostError,
    ResourceCleanupType,
    SQLAlchemyResourceCleanupRepository,
    _vector_route_from_cleanup_row,
    create_dataset_cleanup_intent,
    create_document_cleanup_intent,
    delete_dataset_for_cleanup,
)
from services.knowledge.entities.datasets import DatasetIndexingRepresentation
from services.knowledge.entities.vector_route import VectorRoute
from tests.unit_tests.model_factories import make_dataset, make_document


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


def _route(collection_name: str) -> VectorRoute:
    return VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name=collection_name,
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="text_model",
    )


def _representation(
    route: VectorRoute | None,
    *,
    indexing_technique: str = "high_quality",
    cleanup_routes: tuple[VectorRoute, ...] = (),
) -> DatasetIndexingRepresentation:
    return DatasetIndexingRepresentation(
        indexing_technique=indexing_technique,
        index_struct=route.index_struct if route is not None else None,
        collection_binding_id=route.collection_binding_id if route is not None else None,
        embedding_model="embedding" if indexing_technique == "high_quality" else None,
        embedding_model_provider="provider" if indexing_technique == "high_quality" else None,
        keyword_number=10,
        is_multimodal=False,
        doc_form="text_model",
        summary_index_setting=None,
        vector_route=route,
        cleanup_routes=cleanup_routes,
    )


def test_dataset_cleanup_row_restores_lindorm_shared_collection_without_losing_routing() -> None:
    route = VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="lindorm",
        collection_name="ugc_index_1536_hnsw_l2",
        routing_index_struct=(
            '{"type":"lindorm","vector_store":{"class_prefix":"dataset_route"},'
            '"using_ugc":true,"dimension":1536,"index_type":"hnsw","distance_type":"l2"}'
        ),
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="text_model",
    )
    legacy_index_struct = json.loads(route.index_struct)
    legacy_index_struct.pop("_dify_resolved_collection_name")
    row = ResourceCleanupOutbox(
        tenant_id=route.tenant_id,
        dataset_id=route.dataset_id,
        resource_type=ResourceCleanupType.DATASET_VECTOR.value,
        resource_id="cleanup-1",
        index_struct=json.dumps(legacy_index_struct),
        indexing_technique=route.indexing_technique,
        doc_form=route.doc_form,
    )

    restored = _vector_route_from_cleanup_row(row)
    assert restored is not None
    assert restored.collection_name == route.collection_name
    assert json.loads(restored.index_struct)["vector_store"]["class_prefix"] == "dataset_route"


def test_cleanup_intent_rolls_back_with_resource_deletion(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset()
    sqlite_session.add(dataset)
    sqlite_session.commit()

    intent_id = create_dataset_cleanup_intent(
        sqlite_session,
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        doc_form="text_model",
        indexing_technique=dataset.indexing_technique,
        index_struct=dataset.index_struct,
        collection_binding_id=dataset.collection_binding_id,
        pipeline_id=dataset.pipeline_id,
    )
    sqlite_session.delete(dataset)
    sqlite_session.rollback()

    repository = SQLAlchemyResourceCleanupRepository(session_factory=sqlite_session_factory)
    assert repository.get_intent(intent_id) is None
    with sqlite_session_factory() as observer:
        assert observer.get(type(dataset), dataset.id) is not None


def test_dataset_route_handoff_rolls_back_with_dataset_and_indexing_receipt(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    applied_route = _route("collection-a")
    partial_route = _route("collection-b")
    tenant = Tenant(name="Tenant")
    tenant.id = "tenant-1"
    dataset = make_dataset(
        dataset_id="dataset-1",
        tenant_id=tenant.id,
        indexing_technique="high_quality",
        index_struct=_route("desired-c").index_struct,
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
        target_representation=_representation(partial_route).model_dump_json(),
    )
    with sqlite_session_factory.begin() as session:
        session.add_all([tenant, dataset, pending])
    pending_id = pending.id

    relations = _ResourceRelationsBoundary()
    with sqlite_session_factory() as session:
        deletion = delete_dataset_for_cleanup(
            session,
            tenant_id=tenant.id,
            dataset_id=dataset.id,
            resource_relations=relations,
        )
        assert deletion is not None
        assert len(deletion.cleanup_intent_ids) == 3
        session.rollback()

    with sqlite_session_factory() as session:
        assert session.get(Dataset, dataset.id) is not None
        assert session.get(DatasetIndexingOutbox, pending_id) is not None
        assert session.scalar(select(ResourceCleanupOutbox.id)) is None


def test_document_intent_keeps_explicit_owner_chain_and_guards_live_source(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    dataset = make_dataset()
    document = make_document(doc_form="text_model")
    with sqlite_session_factory.begin() as session:
        session.add_all([dataset, document])
        intent_id = create_document_cleanup_intent(
            session,
            tenant_id=document.tenant_id,
            dataset_id=document.dataset_id,
            document_id=document.id,
            doc_form=document.doc_form,
            source_file_id="file-1",
        )

    repository = SQLAlchemyResourceCleanupRepository(session_factory=sqlite_session_factory)
    intent = repository.get_intent(intent_id)
    assert intent is not None
    assert intent.resource_type == ResourceCleanupType.DOCUMENT
    assert (intent.tenant_id, intent.dataset_id, intent.resource_id) == (
        document.tenant_id,
        document.dataset_id,
        document.id,
    )
    assert intent.source_file_id == "file-1"
    assert repository.source_exists(intent)

    with sqlite_session_factory.begin() as session:
        session.delete(session.get_one(type(document), document.id))
    assert not repository.source_exists(intent)

    claimed = repository.claim(intent.id)
    assert claimed is not None
    assert claimed.claim_id is not None
    repository.complete(claimed, claim_id=claimed.claim_id)
    assert repository.get_intent(intent.id) is None


def test_dataset_completion_rolls_back_parent_when_child_ack_fails(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route = VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="dataset-1-summary",
        collection_binding_id="collection-1",
        indexing_technique="high_quality",
        doc_form="text_model",
    )
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
        cleanup = SummaryVectorCleanupOutbox(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            summary_id="summary-1",
            summary_version=1,
            node_id="node-1",
            writer_id="writer-1",
            vector_route=route.serialized,
            vector_route_hash=route.key,
            writer_finished_at=datetime(2026, 10, 24),
        )
        session.add(cleanup)
        cleanup_id = cleanup.id
    repository = SQLAlchemyResourceCleanupRepository(session_factory=sqlite_session_factory)
    claimed = repository.claim(intent_id)
    assert claimed is not None
    assert claimed.claim_id is not None
    engine = sqlite_session_factory.kw["bind"]

    def reject_child_delete(_connection, _cursor, statement: str, _parameters, _context, _executemany) -> None:
        if statement.lstrip().startswith("DELETE FROM summary_vector_cleanup_outbox"):
            raise RuntimeError("child acknowledgement failed")

    event.listen(engine, "before_cursor_execute", reject_child_delete)
    try:
        with pytest.raises(RuntimeError, match="child acknowledgement failed"):
            repository.complete(claimed, claim_id=claimed.claim_id, deleted_child_route=route)
    finally:
        event.remove(engine, "before_cursor_execute", reject_child_delete)

    with sqlite_session_factory() as session:
        assert session.get(ResourceCleanupOutbox, intent_id) is not None
        assert session.get(SummaryVectorCleanupOutbox, cleanup_id) is not None


def test_pending_intents_prioritize_new_work_before_old_failures(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        first_id = create_document_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            doc_form="text_model",
            source_file_id=None,
        )
    repository = SQLAlchemyResourceCleanupRepository(session_factory=sqlite_session_factory)
    repository.mark_attempted(first_id)
    with sqlite_session_factory.begin() as session:
        second_id = create_document_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-2",
            doc_form="text_model",
            source_file_id=None,
        )

    assert repository.list_pending_intent_ids(limit=1) == (second_id,)
    with sqlite_session_factory() as session:
        rows = session.scalars(select(ResourceCleanupOutbox)).all()
        assert {row.id for row in rows} == {first_id, second_id}


def test_cleanup_claim_is_atomic_and_expired_claim_can_be_recovered(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        intent_id = create_document_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            doc_form="text_model",
            source_file_id=None,
        )
    repository = SQLAlchemyResourceCleanupRepository(session_factory=sqlite_session_factory)

    first = repository.claim(intent_id)
    assert first is not None
    assert first.claim_id is not None
    assert repository.claim(intent_id) is None
    assert repository.list_pending_intent_ids(limit=10) == ()

    with sqlite_session_factory.begin() as session:
        row = session.get_one(ResourceCleanupOutbox, intent_id)
        database_now = session.scalar(select(func.current_timestamp()))
        assert isinstance(database_now, datetime)
        row.claim_expires_at = database_now - timedelta(seconds=1)

    second = repository.claim(intent_id)
    assert second is not None
    assert second.claim_id is not None
    assert second.claim_id != first.claim_id
    with sqlite_session_factory() as session:
        takeover_expiry = session.get_one(ResourceCleanupOutbox, intent_id).claim_expires_at
    with pytest.raises(ResourceCleanupClaimLostError):
        repository.renew_claim(intent_id, claim_id=first.claim_id)
    with sqlite_session_factory() as session:
        persisted = session.get_one(ResourceCleanupOutbox, intent_id)
        assert persisted.claim_id == second.claim_id
        assert persisted.claim_expires_at == takeover_expiry
    assert not repository.release_claim(intent_id, claim_id=first.claim_id)
    with pytest.raises(ResourceCleanupClaimLostError):
        repository.complete(first, claim_id=first.claim_id)
    assert repository.get_intent(intent_id) is not None

    repository.complete(second, claim_id=second.claim_id)
    assert repository.get_intent(intent_id) is None


def test_concurrent_cleanup_claims_have_one_winner(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        intent_id = create_document_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            doc_form="text_model",
            source_file_id=None,
        )
    repository = SQLAlchemyResourceCleanupRepository(session_factory=sqlite_session_factory)

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda _position: repository.claim(intent_id), range(2)))

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert winners[0].claim_id is not None


def test_cleanup_claim_renewal_requires_the_exact_live_owner(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        intent_id = create_document_cleanup_intent(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            doc_form="text_model",
            source_file_id=None,
        )
    repository = SQLAlchemyResourceCleanupRepository(session_factory=sqlite_session_factory)
    claimed = repository.claim(intent_id)
    assert claimed is not None
    assert claimed.claim_id is not None

    repository.renew_claim(intent_id, claim_id=claimed.claim_id)
    with pytest.raises(ResourceCleanupClaimLostError):
        repository.renew_claim(intent_id, claim_id="stale-claim")

import json
from datetime import datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from models.dataset import ChildChunk, Dataset, Document, DocumentSegment
from models.enums import DocumentCreatedFrom
from models.model import ChildVectorCleanupOutbox
from repositories.knowledge.child_vector_cleanup_repository import (
    ChildVectorCleanupClaimLostError,
    SQLAlchemyChildVectorCleanupRepository,
)
from repositories.knowledge.child_vector_route_repository import SQLAlchemyChildVectorRouteStore, stored_vector_route
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.segments import ChildVectorRoute, ChildVectorWriteIntent


def _seed_child(session: Session) -> ChildChunk:
    dataset = Dataset(
        id="dataset-1",
        tenant_id="tenant-1",
        name="Dataset",
        created_by="account-1",
        indexing_technique="high_quality",
    )
    document = Document(
        id="document-1",
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type="upload_file",
        batch="batch-1",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by="account-1",
        doc_form="hierarchical_model",
    )
    segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="parent",
        word_count=1,
        tokens=1,
        created_by="account-1",
    )
    segment.id = "segment-1"
    child = ChildChunk(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        position=1,
        content="child",
        word_count=1,
        created_by="account-1",
        index_node_id="node-1",
    )
    child.id = "child-1"
    session.add_all([dataset, document, segment, child])
    return child


def test_stored_route_rejects_inconsistent_index_structure() -> None:
    route = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    payload = json.loads(route.serialized)
    payload["index_struct"] = '{"type":"weaviate","vector_store":{"class_prefix":"collection-a"}}'

    assert (
        stored_vector_route(
            route_serialized=json.dumps(payload),
            route_hash=route.key,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
        )
        is None
    )


def test_route_store_commits_actual_route_in_caller_publication_transaction(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory.begin() as session:
        _seed_child(session)

    store = SQLAlchemyChildVectorRouteStore()
    with sqlite_session_factory.begin() as session:
        document = session.get(Document, "document-1")
        assert document is not None
        document.error = "caller already owns a write transaction"
        session.flush()
        store.persist(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            collection_binding_id="binding-b",
            indexing_technique="high_quality",
            doc_form="hierarchical_model",
            node_ids=("node-1",),
            vector_type="qdrant",
            collection_name="collection-b",
        )

    with sqlite_session_factory() as session:
        child = session.get(ChildChunk, "child-1")
        assert child is not None
        assert child.vector_route is not None
        route = ChildVectorRoute.model_validate_json(child.vector_route)
        assert route.vector_type == "qdrant"
        assert route.collection_name == "collection-b"
        assert route.collection_binding_id == "binding-b"
        assert child.vector_route_hash == route.key


def test_route_store_persists_lindorm_ugc_routing_and_shared_collection(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    source_index_struct = json.dumps(
        {
            "type": "lindorm",
            "vector_store": {"class_prefix": "dataset_route"},
            "using_ugc": True,
            "dimension": 1536,
            "index_type": "hnsw",
            "distance_type": "l2",
        }
    )
    with sqlite_session_factory.begin() as session:
        _seed_child(session)
        SQLAlchemyChildVectorRouteStore.persist(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            collection_binding_id=None,
            indexing_technique="high_quality",
            doc_form="hierarchical_model",
            node_ids=("node-1",),
            vector_type="lindorm",
            collection_name="ugc_index_1536_hnsw_l2",
            routing_index_struct=source_index_struct,
        )

    with sqlite_session_factory() as session:
        child = session.get(ChildChunk, "child-1")
        assert child is not None
        assert child.vector_route is not None
        route = ChildVectorRoute.model_validate_json(child.vector_route)
        assert json.loads(route.index_struct)["vector_store"]["class_prefix"] == "dataset_route"
        assert route.collection_name == "ugc_index_1536_hnsw_l2"
        assert child.vector_route_hash == route.key
        assert session.query(ChildVectorCleanupOutbox).count() == 0


def test_route_store_rejects_partial_target_set_without_changing_existing_route(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route_a = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    with sqlite_session_factory.begin() as session:
        child = _seed_child(session)
        child.vector_route = route_a.serialized
        child.vector_route_hash = route_a.key

    store = SQLAlchemyChildVectorRouteStore()
    with sqlite_session_factory.begin() as session, pytest.raises(LookupError):
        store.persist(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            collection_binding_id="binding-b",
            indexing_technique="high_quality",
            doc_form="hierarchical_model",
            node_ids=("node-1", "missing-node"),
            vector_type="qdrant",
            collection_name="collection-b",
        )

    with sqlite_session_factory() as session:
        child = session.get(ChildChunk, "child-1")
        assert child is not None
        assert child.vector_route == route_a.serialized
        assert child.vector_route_hash == route_a.key


def test_route_store_enqueues_the_proven_old_route_when_switching_collections(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route_a = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    with sqlite_session_factory.begin() as session:
        child = _seed_child(session)
        child.vector_route = route_a.serialized
        child.vector_route_hash = route_a.key

    store = SQLAlchemyChildVectorRouteStore()
    with sqlite_session_factory.begin() as session:
        store.persist(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            collection_binding_id="binding-b",
            indexing_technique="high_quality",
            doc_form="hierarchical_model",
            node_ids=("node-1",),
            vector_type="qdrant",
            collection_name="collection-b",
        )

    with sqlite_session_factory() as session:
        child = session.get(ChildChunk, "child-1")
        cleanup = session.query(ChildVectorCleanupOutbox).one()
        assert child is not None
        assert child.vector_route is not None
        route_b = ChildVectorRoute.model_validate_json(child.vector_route)
        assert route_b.collection_name == "collection-b"
        assert cleanup.node_id == "node-1"
        assert cleanup.vector_route == route_a.serialized
        assert cleanup.vector_route_hash == route_a.key
        assert cleanup.writer_finished_at is not None


def test_route_store_does_not_cleanup_same_physical_collection_after_binding_change(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route_a = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="shared-collection",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    route_b = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="shared-collection",
        collection_binding_id="binding-b",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    assert route_a.key == route_b.key
    with sqlite_session_factory.begin() as session:
        child = _seed_child(session)
        child.vector_route = route_a.serialized
        child.vector_route_hash = route_a.key

    with sqlite_session_factory.begin() as session:
        SQLAlchemyChildVectorRouteStore().persist(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            collection_binding_id="binding-b",
            indexing_technique="high_quality",
            doc_form="hierarchical_model",
            node_ids=("node-1",),
            vector_type="qdrant",
            collection_name="shared-collection",
        )

    with sqlite_session_factory() as session:
        child = session.get(ChildChunk, "child-1")
        assert child is not None
        assert child.vector_route is not None
        route = ChildVectorRoute.model_validate_json(child.vector_route)
        assert route.collection_binding_id == "binding-b"
        assert route.key == route_a.key
        assert session.query(ChildVectorCleanupOutbox).count() == 0


def test_route_switch_rolls_back_with_the_caller_transaction(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route_a = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    with sqlite_session_factory.begin() as session:
        child = _seed_child(session)
        child.vector_route = route_a.serialized
        child.vector_route_hash = route_a.key

    store = SQLAlchemyChildVectorRouteStore()
    with sqlite_session_factory() as session:
        store.persist(
            session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            collection_binding_id="binding-b",
            indexing_technique="high_quality",
            doc_form="hierarchical_model",
            node_ids=("node-1",),
            vector_type="qdrant",
            collection_name="collection-b",
        )
        session.rollback()

    with sqlite_session_factory() as session:
        child = session.get(ChildChunk, "child-1")
        assert child is not None
        assert child.vector_route == route_a.serialized
        assert child.vector_route_hash == route_a.key
        assert session.query(ChildVectorCleanupOutbox).count() == 0


def test_route_switch_refreshes_a_stale_identity_map_before_recording_cleanup(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    route_a = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    route_b = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="collection-b",
        collection_binding_id="binding-b",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    with sqlite_session_factory.begin() as session:
        child = _seed_child(session)
        child.vector_route = route_a.serialized
        child.vector_route_hash = route_a.key

    with sqlite_session_factory() as stale_session:
        stale_child = stale_session.get(ChildChunk, "child-1")
        assert stale_child is not None
        assert stale_child.vector_route_hash == route_a.key
        with sqlite_session_factory.begin() as current_session:
            current_child = current_session.get(ChildChunk, "child-1")
            assert current_child is not None
            current_child.vector_route = route_b.serialized
            current_child.vector_route_hash = route_b.key

        SQLAlchemyChildVectorRouteStore().persist(
            stale_session,
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            collection_binding_id="binding-c",
            indexing_technique="high_quality",
            doc_form="hierarchical_model",
            node_ids=("node-1",),
            vector_type="qdrant",
            collection_name="collection-c",
        )
        stale_session.commit()

    with sqlite_session_factory() as session:
        cleanup = session.query(ChildVectorCleanupOutbox).one()
        assert cleanup.vector_route_hash == route_b.key
        assert cleanup.vector_route == route_b.serialized


@pytest.mark.parametrize("writer_outcome", ["late_finish", "discard_unsubmitted"])
def test_terminal_handoff_preserves_unfinished_child_writer_until_safe_ack(
    sqlite_session_factory: sessionmaker[Session],
    writer_outcome: str,
) -> None:
    route = ChildVectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="qdrant",
        collection_name="collection-a",
        collection_binding_id="binding-a",
        indexing_technique="high_quality",
        doc_form="hierarchical_model",
    )
    with sqlite_session_factory.begin() as session:
        child = _seed_child(session)
        child.vector_route = route.serialized
        child.vector_route_hash = route.key
        cleanup = ChildVectorCleanupOutbox(
            tenant_id=child.tenant_id,
            dataset_id=child.dataset_id,
            document_id=child.document_id,
            segment_id=child.segment_id,
            child_id=child.id,
            node_id=child.index_node_id or "",
            writer_id="writer-1",
            vector_route=route.serialized,
            vector_route_hash=route.key,
            claim_id="writer-claim-1",
            claim_expires_at=datetime(2030, 1, 1),
        )
        session.add(cleanup)
        session.flush()
        writer = ChildVectorWriteIntent(
            cleanup.id,
            cleanup.tenant_id,
            cleanup.dataset_id,
            cleanup.node_id,
            cleanup.writer_id,
            cleanup.claim_id or "",
            route,
        )
        child_id = child.id
        node_id = cleanup.node_id
        document_id = child.document_id
        segment_id = child.segment_id

    repository = SQLAlchemyChildVectorCleanupRepository(session_factory=sqlite_session_factory)
    ref = DatasetRef("tenant-1", "dataset-1").document(document_id).segment(segment_id)
    with sqlite_session_factory.begin() as terminal:
        repository.add_terminal_cleanups(
            terminal,
            ref=ref,
            obsolete_nodes=[(child_id, node_id, route)],
        )
        terminal.delete(terminal.get_one(ChildChunk, child_id))
    with sqlite_session_factory() as observer:
        handed_off = observer.get_one(ChildVectorCleanupOutbox, writer.id)
        cleanup_started_at = handed_off.cleanup_started_at
        assert handed_off.writer_id == writer.writer_id
        assert handed_off.writer_finished_at is None
        assert cleanup_started_at is not None
        assert handed_off.claim_id is None
        assert handed_off.vector_route == route.serialized
        assert handed_off.vector_route_hash == route.key

    with sqlite_session_factory.begin() as terminal:
        repository.add_terminal_cleanups(
            terminal,
            ref=ref,
            obsolete_nodes=[(child_id, node_id, route)],
        )
    with sqlite_session_factory() as observer:
        handed_off = observer.get_one(ChildVectorCleanupOutbox, writer.id)
        assert handed_off.writer_id == writer.writer_id
        assert handed_off.writer_finished_at is None
        assert handed_off.cleanup_started_at == cleanup_started_at

    vector_nodes = {node_id}
    first_cleanup = repository.claim_pending_cleanup(writer.id)
    assert first_cleanup is not None
    assert first_cleanup.writer_finished is False
    with repository.guard_cleanup_delete(first_cleanup) as referenced:
        assert referenced is False
        vector_nodes.discard(node_id)
    assert repository.release_cleanup(first_cleanup) is True
    with sqlite_session_factory() as observer:
        assert observer.get(ChildVectorCleanupOutbox, writer.id) is not None

    if writer_outcome == "late_finish":
        vector_nodes.add(node_id)
        with pytest.raises(ChildVectorCleanupClaimLostError, match="another worker"):
            repository.finish_writes((writer,))
    else:
        repository.discard_unsubmitted_writes((writer,))

    final_cleanup = repository.claim_pending_cleanup(writer.id)
    assert final_cleanup is not None
    assert final_cleanup.writer_finished is True
    with repository.guard_cleanup_delete(final_cleanup) as referenced:
        assert referenced is False
        vector_nodes.discard(node_id)

    with sqlite_session_factory() as observer:
        assert observer.get(ChildVectorCleanupOutbox, writer.id) is None
    assert vector_nodes == set()

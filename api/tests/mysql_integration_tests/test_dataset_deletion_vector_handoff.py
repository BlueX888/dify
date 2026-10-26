import json
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from constants.resource_access_token import ResourceAccessTokenResourceType
from core.rag.index_processor.constant.index_type import IndexStructureType
from models.account import Tenant
from models.dataset import Dataset
from models.enums import DataSourceType
from models.model import DatasetIndexingOutbox, ResourceCleanupOutbox
from repositories.knowledge.dataset_indexing_outbox_repository import (
    SQLAlchemyDatasetIndexingOutboxRepository,
    create_dataset_indexing_intents,
    lock_dataset_indexing_execution,
)
from repositories.knowledge.resource_cleanup_repository import (
    ResourceCleanupType,
    SQLAlchemyResourceCleanupRepository,
    delete_dataset_for_cleanup,
)
from services.knowledge.entities.datasets import DatasetIndexingJob, DatasetIndexingRepresentation
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.vector_route import VectorRoute


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


def _repeatable_read_sessions(mysql_session: Session) -> sessionmaker[Session]:
    engine = mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ")
    return sessionmaker(bind=engine, expire_on_commit=False)


def _route(*, tenant_id: str, dataset_id: str, collection: str) -> VectorRoute:
    return VectorRoute.capture(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type="qdrant",
        collection_name=collection,
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )


def _representation(
    *,
    collection: str,
    route: VectorRoute | None = None,
) -> DatasetIndexingRepresentation:
    return DatasetIndexingRepresentation(
        indexing_technique="high_quality",
        index_struct=json.dumps({"type": "qdrant", "vector_store": {"class_prefix": collection}}),
        collection_binding_id=None,
        embedding_model="embedding-model",
        embedding_model_provider="embedding-provider",
        keyword_number=10,
        is_multimodal=False,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        summary_index_setting=None,
        vector_route=route,
    )


def test_delete_handoffs_applied_and_late_partial_routes_after_repeatable_read_snapshot(
    mysql_session: Session,
) -> None:
    """Deletion must refresh route receipts before retiring their only durable source."""

    sessions = _repeatable_read_sessions(mysql_session)
    tenant_id = str(uuid4())
    dataset_id = str(uuid4())
    actor_id = str(uuid4())
    route_a = _route(tenant_id=tenant_id, dataset_id=dataset_id, collection="collection-a")
    route_b = _route(tenant_id=tenant_id, dataset_id=dataset_id, collection="collection-b")
    with sessions.begin() as seed:
        tenant = Tenant(name=f"route-handoff-{uuid4()}")
        tenant.id = tenant_id
        dataset = Dataset(
            id=dataset_id,
            tenant_id=tenant_id,
            name=f"route-handoff-{uuid4()}",
            data_source_type=DataSourceType.UPLOAD_FILE,
            indexing_technique="high_quality",
            # Desired C has never executed and is not physical evidence.
            index_struct=_representation(collection="collection-c").index_struct,
            created_by=actor_id,
        )
        dataset.indexing_applied_representation = _representation(
            collection="collection-a",
            route=route_a,
        ).model_dump_json()
        seed.add_all([tenant, dataset])

    with sessions.begin() as create_session:
        (intent_b_id,) = create_dataset_indexing_intents(
            create_session,
            DatasetIndexingJob(
                DatasetRef(tenant_id, dataset_id),
                _representation(collection="collection-a", route=route_a),
                _representation(collection="collection-b"),
            ),
        )
    indexing = SQLAlchemyDatasetIndexingOutboxRepository(session_factory=sessions)
    intent_b = indexing.claim(intent_b_id)
    assert intent_b is not None
    relations = _ResourceRelationsBoundary()

    try:
        with sessions.begin() as deletion_session:
            # Establish an old RR snapshot in the exact transaction that will
            # delete the Dataset. At this point B has no physical receipt.
            snapshot = deletion_session.scalar(
                select(DatasetIndexingOutbox).where(DatasetIndexingOutbox.id == intent_b_id)
            )
            assert snapshot is not None
            snapshot_target = DatasetIndexingRepresentation.model_validate_json(snapshot.target_representation)
            assert snapshot_target.vector_route is None

            # A real second Session records that B was physically reached and
            # commits after deletion_session's snapshot.
            with sessions.begin() as writer_session:
                locked = lock_dataset_indexing_execution(
                    writer_session,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    generation_id=intent_b.generation_id,
                    intent_id=intent_b.id,
                    claim_id=intent_b.claim_id,
                )
                assert locked is not None
                recorded = indexing.record_target_route_locked(
                    writer_session,
                    intent_id=intent_b.id,
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    generation_id=intent_b.generation_id,
                    claim_id=intent_b.claim_id,
                    route=route_b,
                )
                assert recorded.vector_route == route_b

            deletion = delete_dataset_for_cleanup(
                deletion_session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                resource_relations=relations,
            )
            assert deletion is not None

        cleanup_repository = SQLAlchemyResourceCleanupRepository(session_factory=sessions)
        intents = tuple(cleanup_repository.get_intent(intent_id) for intent_id in deletion.cleanup_intent_ids)
        assert all(intent is not None for intent in intents)
        parent = tuple(
            intent
            for intent in intents
            if intent is not None and intent.resource_type == ResourceCleanupType.DATASET_RELATIONAL
        )
        vector_intents = tuple(
            intent
            for intent in intents
            if intent is not None and intent.resource_type == ResourceCleanupType.DATASET_VECTOR
        )
        assert len(parent) == 1
        assert parent[0].vector_route is None
        assert {intent.vector_route for intent in vector_intents} == {route_a, route_b}
        assert relations.deleted == [
            (tenant_id, ResourceAccessTokenResourceType.KNOWLEDGE, dataset_id),
        ]

        with sessions() as verify:
            assert verify.get(Dataset, dataset_id) is None
            assert verify.get(DatasetIndexingOutbox, intent_b_id) is None
            # C was desired only; no cleanup row may invent a route for it.
            assert all(
                intent.vector_route is None or intent.vector_route.collection_name != "collection-c"
                for intent in intents
                if intent is not None
            )
    finally:
        with sessions.begin() as cleanup:
            cleanup.execute(
                delete(ResourceCleanupOutbox).where(
                    ResourceCleanupOutbox.tenant_id == tenant_id,
                    ResourceCleanupOutbox.dataset_id == dataset_id,
                )
            )
            cleanup.execute(
                delete(DatasetIndexingOutbox).where(
                    DatasetIndexingOutbox.tenant_id == tenant_id,
                    DatasetIndexingOutbox.dataset_id == dataset_id,
                )
            )
            cleanup.execute(delete(Dataset).where(Dataset.tenant_id == tenant_id, Dataset.id == dataset_id))
            cleanup.execute(delete(Tenant).where(Tenant.id == tenant_id))

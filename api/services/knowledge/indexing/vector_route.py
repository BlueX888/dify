"""Adapters for immutable, non-secret vector routing snapshots."""

from functools import partial

from core.db.session_factory import session_factory
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.index_processor_base import KnowledgeIndexRuntime
from models.dataset import Dataset
from repositories.knowledge.attachment_vector_repository import SQLAlchemyAttachmentVectorStore
from repositories.knowledge.child_vector_route_repository import SQLAlchemyChildVectorRouteStore
from repositories.knowledge.pipeline_indexing_repository import (
    build_pipeline_indexing_guard_from_ids,
    build_pipeline_indexing_route_recorder_from_ids,
)
from services.knowledge.entities.vector_route import VectorRoute
from services.knowledge.retrieval.finalization import KnowledgeRetrievalFinalizer


def build_knowledge_index_runtime() -> KnowledgeIndexRuntime:
    """Bind workflow knowledge-node ports to persistence-aware adapters."""
    return KnowledgeIndexRuntime(
        attachment_vectors=SQLAlchemyAttachmentVectorStore(),
        child_vector_routes=SQLAlchemyChildVectorRouteStore(),
        indexing_guard_factory=build_pipeline_indexing_guard_from_ids,
        indexing_route_recorder_factory=partial(
            build_pipeline_indexing_route_recorder_from_ids,
            new_session=session_factory.create_session,
        ),
        retrieval_finalizer=KnowledgeRetrievalFinalizer(),
    )


def capture_vector_route(dataset: Dataset, vector: Vector, *, doc_form: str) -> VectorRoute:
    vector_type, collection_name = vector.get_routing_target()
    return VectorRoute.capture(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        vector_type=vector_type,
        collection_name=collection_name,
        routing_index_struct=dataset.index_struct,
        collection_binding_id=dataset.collection_binding_id,
        indexing_technique=str(dataset.indexing_technique),
        doc_form=doc_form,
    )


def vector_for_route(route: VectorRoute) -> Vector:
    """Resolve only the persisted physical target, independent of live Dataset state."""
    dataset = Dataset(
        id=route.dataset_id,
        tenant_id=route.tenant_id,
        indexing_technique=route.indexing_technique,
        index_struct=route.index_struct,
        collection_binding_id=None,
    )
    vector = Vector(dataset, session=None, vector_type=route.vector_type)
    if vector.get_routing_target() != (route.vector_type, route.collection_name):
        raise RuntimeError("Persisted vector route resolved to a different physical target")
    return vector

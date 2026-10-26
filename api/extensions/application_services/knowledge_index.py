"""Composition facade for persistence-aware knowledge workflow ports."""

from core.rag.embedding.retrieval import RetrievalFinalizer
from core.rag.index_processor.index_processor_base import KnowledgeIndexRuntime
from services.knowledge.indexing.vector_route import build_knowledge_index_runtime
from services.knowledge.retrieval.finalization import KnowledgeRetrievalFinalizer


def build_retrieval_finalizer() -> RetrievalFinalizer:
    """Bind the core retrieval consumer contract to its application owner."""
    return KnowledgeRetrievalFinalizer()


__all__ = ["KnowledgeIndexRuntime", "build_knowledge_index_runtime", "build_retrieval_finalizer"]

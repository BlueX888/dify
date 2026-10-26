from typing import TYPE_CHECKING

from . import errors

if TYPE_CHECKING:
    from core.rag.embedding.retrieval import RetrievalFinalizer
    from core.rag.index_processor.index_processor_base import KnowledgeIndexRuntime


def build_knowledge_index_runtime() -> "KnowledgeIndexRuntime":
    """Compose persistence-aware knowledge dependencies at the legacy runtime boundary."""
    from services.knowledge.indexing.vector_route import build_knowledge_index_runtime as build

    return build()


def build_retrieval_finalizer() -> "RetrievalFinalizer":
    """Compose the canonical retrieval final-result owner."""
    from services.knowledge.retrieval.finalization import KnowledgeRetrievalFinalizer

    return KnowledgeRetrievalFinalizer()


__all__ = ["build_knowledge_index_runtime", "build_retrieval_finalizer", "errors"]

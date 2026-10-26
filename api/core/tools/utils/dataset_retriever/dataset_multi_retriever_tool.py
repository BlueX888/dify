import threading
from typing import override

from flask import Flask, current_app
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.callback_handler.index_tool_callback_handler import DatasetIndexToolCallbackHandler
from core.model_manager import ModelManager
from core.rag.datasource.retrieval_service import DefaultRetrievalModelDict, RetrievalService
from core.rag.embedding.retrieval import FinalizedInternalRetrieval
from core.rag.entities import RetrievalSourceMetadata
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from core.rag.models.document import Document as RagDocument
from core.rag.rerank.rerank_model import RerankModelRunner
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from core.tools.utils.dataset_retriever.dataset_retriever_base_tool import DatasetRetrieverBaseTool
from extensions.ext_database import db
from graphon.model_runtime.entities.model_entities import ModelType
from models.dataset import Dataset
from repositories.knowledge.retrieval_source_repository import get_retrieval_source_documents

default_retrieval_model: DefaultRetrievalModelDict = {
    "search_method": RetrievalMethod.SEMANTIC_SEARCH,
    "reranking_enable": False,
    "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
    "top_k": 2,
    "score_threshold_enabled": False,
}


class DatasetMultiRetrieverToolInput(BaseModel):
    query: str = Field(..., description="dataset multi retriever and rerank")


class DatasetMultiRetrieverTool(DatasetRetrieverBaseTool):
    """Tool for querying multi dataset."""

    name: str = "dataset_"
    args_schema: type[BaseModel] = DatasetMultiRetrieverToolInput
    description: str = "dataset multi retriever and rerank. "
    dataset_ids: list[str]
    reranking_provider_name: str
    reranking_model_name: str

    @classmethod
    def from_dataset(cls, dataset_ids: list[str], tenant_id: str, **kwargs):
        return cls(
            name=f"dataset_{tenant_id.replace('-', '_')}", tenant_id=tenant_id, dataset_ids=dataset_ids, **kwargs
        )

    @override
    def _run(self, session: Session, query: str) -> str:
        threads = []
        all_documents: list[RagDocument] = []
        for dataset_id in self.dataset_ids:
            retrieval_thread = threading.Thread(
                target=self._retriever,
                kwargs={
                    "flask_app": current_app._get_current_object(),  # type: ignore
                    "dataset_id": dataset_id,
                    "query": query,
                    "all_documents": all_documents,
                    "hit_callbacks": self.hit_callbacks,
                },
            )
            threads.append(retrieval_thread)
            retrieval_thread.start()
        for thread in threads:
            thread.join()
        # do rerank for searched documents
        model_manager = ModelManager.for_tenant(tenant_id=self.tenant_id)
        rerank_model_instance = model_manager.get_model_instance(
            tenant_id=self.tenant_id,
            provider=self.reranking_provider_name,
            model_type=ModelType.RERANK,
            model=self.reranking_model_name,
        )

        rerank_runner = RerankModelRunner(rerank_model_instance, session=session)
        all_documents = rerank_runner.run(query, all_documents, self.score_threshold, len(all_documents))
        with Session(bind=session.get_bind()) as format_session:
            finalized = self.retrieval_finalizer.finalize(
                format_session=format_session,
                authorization_session=format_session,
                candidates=all_documents,
                tenant_id=self.tenant_id,
                dataset_ids=self.dataset_ids,
                top_k=self.top_k,
            )

        if finalized.segment_ids:
            for hit_callback in self.hit_callbacks:
                hit_callback.on_tool_end(finalized.trace_documents, session)

        internal_items = [item for item in finalized.items if isinstance(item, FinalizedInternalRetrieval)]
        datasets = session.scalars(
            select(Dataset).where(Dataset.tenant_id == self.tenant_id, Dataset.id.in_(self.dataset_ids))
        ).all()
        dataset_map = {dataset.id: dataset for dataset in datasets}
        documents = get_retrieval_source_documents(
            tenant_id=self.tenant_id,
            dataset_ids=self.dataset_ids,
            document_ids=[item.record.segment.document_id for item in internal_items],
            session=session,
        )
        document_map = {document.id: document for document in documents}
        contents: list[str] = []
        resources: list[RetrievalSourceMetadata] = []
        for position, item in enumerate(finalized.items, start=1):
            if isinstance(item, RagDocument):
                contents.append(item.page_content)
                continue
            record = item.record
            segment = record.segment
            segment_content = f"question:{item.content} answer:{segment.answer}" if segment.answer else item.content
            contents.append(f"{record.summary}\n{segment_content}" if record.summary else segment_content)
            dataset = dataset_map.get(segment.dataset_id)
            document = document_map.get(segment.document_id)
            if dataset is None or document is None:
                continue
            source = RetrievalSourceMetadata(
                position=position,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                document_id=document.id,
                document_name=document.name,
                data_source_type=document.data_source_type,
                segment_id=segment.id,
                retriever_from=self.retriever_from,
                score=record.score or 0.0,
                doc_metadata=document.doc_metadata,
                content=(
                    f"question:{segment.content} \nanswer:{segment.answer}" if segment.answer else segment.content
                ),
                summary=record.summary,
            )
            if self.retriever_from == "dev":
                source.hit_count = segment.hit_count
                source.word_count = segment.word_count
                source.segment_position = segment.position
                source.index_node_hash = segment.index_node_hash
            resources.append(source)

        if self.return_resource and resources:
            for hit_callback in self.hit_callbacks:
                hit_callback.return_retriever_resource_info(resources)
        return "\n".join(contents)

    def _retriever(
        self,
        flask_app: Flask,
        dataset_id: str,
        query: str,
        all_documents: list,
        hit_callbacks: list[DatasetIndexToolCallbackHandler],
    ):
        with flask_app.app_context():
            stmt = select(Dataset).where(Dataset.tenant_id == self.tenant_id, Dataset.id == dataset_id)
            dataset = db.session.scalar(stmt)

            if not dataset:
                return []

            for hit_callback in hit_callbacks:
                hit_callback.on_query(query, dataset.id, db.session())

            # get retrieval model , if the model is not setting , using default
            retrieval_model = dataset.retrieval_model or default_retrieval_model

            if dataset.indexing_technique == IndexTechniqueType.ECONOMY:
                # use keyword table query
                documents = RetrievalService.retrieve(
                    retrieval_method=RetrievalMethod.KEYWORD_SEARCH,
                    dataset_id=dataset.id,
                    query=query,
                    top_k=retrieval_model.get("top_k") or 4,
                )
                if documents:
                    all_documents.extend(documents)
            else:
                if self.top_k > 0:
                    # retrieval source
                    documents = RetrievalService.retrieve(
                        retrieval_method=retrieval_model["search_method"],
                        dataset_id=dataset.id,
                        query=query,
                        top_k=retrieval_model.get("top_k") or 4,
                        score_threshold=retrieval_model.get("score_threshold", 0.0)
                        if retrieval_model["score_threshold_enabled"]
                        else 0.0,
                        reranking_model=retrieval_model.get("reranking_model", None)
                        if retrieval_model["reranking_enable"]
                        else None,
                        reranking_mode=retrieval_model.get("reranking_mode") or "reranking_model",
                        weights=retrieval_model.get("weights", None),
                    )

                    all_documents.extend(documents)

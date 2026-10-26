from typing import Any, cast, override

from pydantic import BaseModel, Field, SkipValidation
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.app.app_config.entities import DatasetRetrieveConfigEntity, ModelConfig
from core.rag.datasource.retrieval_service import DefaultRetrievalModelDict, RetrievalService
from core.rag.embedding.retrieval import FinalizedInternalRetrieval
from core.rag.entities import RetrievalSourceMetadata
from core.rag.entities.retrieval_settings import ExternalDatasetRetriever
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from core.rag.models.document import Document as RetrievalDocument
from core.rag.retrieval.dataset_retrieval import DatasetRetrieval
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from core.tools.utils.dataset_retriever.dataset_retriever_base_tool import DatasetRetrieverBaseTool
from models.dataset import Dataset
from repositories.knowledge.retrieval_source_repository import get_retrieval_source_documents

default_retrieval_model: DefaultRetrievalModelDict = {
    "search_method": RetrievalMethod.SEMANTIC_SEARCH,
    "reranking_enable": False,
    "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
    "reranking_mode": "reranking_model",
    "top_k": 2,
    "score_threshold_enabled": False,
}


class DatasetRetrieverToolInput(BaseModel):
    query: str = Field(..., description="Query for the dataset to be used to retrieve the dataset.")


class DatasetRetrieverTool(DatasetRetrieverBaseTool):
    """Tool for querying a Dataset."""

    name: str = "dataset"
    args_schema: type[BaseModel] = DatasetRetrieverToolInput
    description: str = "use this to retrieve a dataset. "
    dataset_id: str
    user_id: str | None = None
    retrieve_config: DatasetRetrieveConfigEntity
    inputs: dict[str, Any]
    external_retrieval: SkipValidation[ExternalDatasetRetriever]

    @classmethod
    def from_dataset(cls, dataset: Dataset, **kwargs):
        description = dataset.description
        if not description:
            description = "useful for when you want to answer queries about the " + dataset.name

        description = description.replace("\n", "").replace("\r", "")
        return cls(
            name=f"dataset_{dataset.id.replace('-', '_')}",
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            description=description,
            **kwargs,
        )

    @override
    def _run(self, session: Session, query: str) -> str:
        with Session(bind=session.get_bind(), expire_on_commit=False) as read_session:
            dataset_stmt = select(Dataset).where(Dataset.tenant_id == self.tenant_id, Dataset.id == self.dataset_id)
            dataset = read_session.scalar(dataset_stmt)

        if not dataset:
            return ""
        for hit_callback in self.hit_callbacks:
            hit_callback.on_query(query, dataset.id, session)
        dataset_retrieval = DatasetRetrieval(
            external_retrieval=self.external_retrieval,
            retrieval_finalizer=self.retrieval_finalizer,
        )
        metadata_filter_document_ids, metadata_condition = dataset_retrieval.get_metadata_filter_condition(
            session,
            [dataset.id],
            query,
            self.tenant_id,
            self.user_id or "unknown",
            cast(str, self.retrieve_config.metadata_filtering_mode),
            cast(ModelConfig, self.retrieve_config.metadata_model_config),
            self.retrieve_config.metadata_filtering_conditions,
            self.inputs,
        )
        document_ids_filter = metadata_filter_document_ids.get(dataset.id, []) if metadata_filter_document_ids else None
        candidates: list[RetrievalDocument] = []
        if dataset.provider == "external":
            external_documents = self.external_retrieval.retrieve(
                tenant_id=dataset.tenant_id,
                dataset_id=dataset.id,
                query=query,
                retrieval_model=dataset.retrieval_model,
                metadata_filters=metadata_condition.model_dump() if metadata_condition else None,
            )
            for external_document in external_documents:
                document = RetrievalDocument.model_validate(
                    {
                        "page_content": external_document.get("content"),
                        "metadata": external_document.get("metadata"),
                        "provider": "external",
                    }
                )
                if document.metadata is not None:
                    document.metadata["score"] = external_document.get("score")
                    document.metadata["title"] = external_document.get("title")
                    document.metadata["dataset_id"] = dataset.id
                    document.metadata["dataset_name"] = dataset.name
                    candidates.append(document)
        else:
            if metadata_condition and not document_ids_filter:
                return ""
            allowed_document_ids_filter = tuple(document_ids_filter) if document_ids_filter is not None else None
            if allowed_document_ids_filter is not None:
                with Session(bind=session.get_bind()) as filter_session:
                    document_ids_filter = list(
                        self.retrieval_finalizer.expand_document_ids_filter(
                            session=filter_session,
                            tenant_id=self.tenant_id,
                            dataset_id=dataset.id,
                            document_ids=allowed_document_ids_filter,
                        )
                    )
            # get retrieval model , if the model is not setting , using default
            retrieval_model = dataset.retrieval_model or default_retrieval_model
            if dataset.indexing_technique == IndexTechniqueType.ECONOMY:
                # use keyword table query
                candidates = RetrievalService.retrieve(
                    retrieval_method=RetrievalMethod.KEYWORD_SEARCH,
                    dataset_id=dataset.id,
                    query=query,
                    top_k=self.top_k,
                    document_ids_filter=document_ids_filter,
                    allowed_document_ids_filter=allowed_document_ids_filter,
                )
            else:
                if self.top_k > 0:
                    # retrieval source
                    candidates = RetrievalService.retrieve(
                        retrieval_method=retrieval_model.get("search_method", "semantic_search"),
                        dataset_id=dataset.id,
                        query=query,
                        top_k=self.top_k,
                        score_threshold=retrieval_model.get("score_threshold", 0.0)
                        if retrieval_model["score_threshold_enabled"]
                        else 0.0,
                        reranking_model=retrieval_model.get("reranking_model")
                        if retrieval_model["reranking_enable"]
                        else None,
                        reranking_mode=retrieval_model.get("reranking_mode") or "reranking_model",
                        weights=retrieval_model.get("weights"),
                        document_ids_filter=document_ids_filter,
                        allowed_document_ids_filter=allowed_document_ids_filter,
                    )

        with Session(bind=session.get_bind()) as format_session:
            finalized = self.retrieval_finalizer.finalize(
                format_session=format_session,
                authorization_session=format_session,
                candidates=candidates,
                tenant_id=self.tenant_id,
                dataset_ids=[dataset.id],
                top_k=self.top_k,
            )

        if finalized.segment_ids:
            for hit_callback in self.hit_callbacks:
                hit_callback.on_tool_end(finalized.trace_documents, session)

        internal_items = [item for item in finalized.items if isinstance(item, FinalizedInternalRetrieval)]
        source_documents = get_retrieval_source_documents(
            tenant_id=self.tenant_id,
            dataset_ids=[dataset.id],
            document_ids=[item.record.segment.document_id for item in internal_items],
            session=session,
        )
        source_document_map = {document.id: document for document in source_documents}
        contents: list[str] = []
        resources: list[RetrievalSourceMetadata] = []
        for position, item in enumerate(finalized.items, start=1):
            if isinstance(item, RetrievalDocument):
                metadata = item.metadata
                contents.append(item.page_content)
                resources.append(
                    RetrievalSourceMetadata(
                        position=position,
                        dataset_id=metadata.get("dataset_id"),
                        dataset_name=metadata.get("dataset_name"),
                        document_id=metadata.get("document_id") or metadata.get("title"),
                        document_name=metadata.get("title"),
                        data_source_type="external",
                        retriever_from=self.retriever_from,
                        score=metadata.get("score"),
                        title=metadata.get("title"),
                        content=item.page_content,
                    )
                )
                continue

            record = item.record
            segment = record.segment
            segment_content = f"question:{item.content} answer:{segment.answer}" if segment.answer else item.content
            contents.append(f"{record.summary}\n{segment_content}" if record.summary else segment_content)
            document = source_document_map.get(segment.document_id)
            if document is None:
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

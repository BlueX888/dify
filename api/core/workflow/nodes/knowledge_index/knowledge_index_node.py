import logging
from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, override

from sqlalchemy.orm import Session

from core.app.entities.app_invoke_entities import DIFY_RUN_CONTEXT_KEY, DifyRunContext
from core.db.session_factory import session_factory
from core.model_context import use_credit_usage_metadata
from core.rag.index_processor.index_processor import IndexProcessor
from core.rag.index_processor.index_processor_base import (
    AttachmentVectorStore,
    ChildVectorRouteStore,
    PipelineIndexingGuardFactory,
    PipelineIndexingRouteRecorderFactory,
    SummaryIndexSettingDict,
)
from core.rag.summary_index.summary_index import SummaryIndex
from core.workflow.nodes.knowledge_index import KNOWLEDGE_INDEX_NODE_TYPE
from core.workflow.system_variables import SystemVariableKey, get_system_segment, get_system_text
from graphon.enums import NodeExecutionType, WorkflowNodeExecutionStatus
from graphon.node_events import NodeRunResult
from graphon.nodes.base.node import Node
from graphon.nodes.base.template import Template

from .entities import KnowledgeIndexNodeData
from .exc import (
    KnowledgeIndexNodeError,
)

if TYPE_CHECKING:
    from graphon.runtime import InitParams, RuntimeState

logger = logging.getLogger(__name__)
_INVOKE_FROM_DEBUGGER = "debugger"


class KnowledgeIndexNode(Node[KnowledgeIndexNodeData]):
    node_type = KNOWLEDGE_INDEX_NODE_TYPE
    execution_type = NodeExecutionType.RESPONSE

    def __init__(
        self,
        node_id: str,
        data: KnowledgeIndexNodeData,
        *,
        init_params: "InitParams",
        runtime_state: "RuntimeState",
        attachment_vectors: AttachmentVectorStore | None = None,
        child_vector_routes: ChildVectorRouteStore | None = None,
        indexing_guard_factory: PipelineIndexingGuardFactory | None = None,
        indexing_route_recorder_factory: PipelineIndexingRouteRecorderFactory | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            data=data,
            init_params=init_params,
            runtime_state=runtime_state,
        )
        self.index_processor = IndexProcessor(
            attachment_vectors=attachment_vectors,
            child_vector_routes=child_vector_routes,
            indexing_guard_factory=indexing_guard_factory,
            indexing_route_recorder_factory=indexing_route_recorder_factory,
        )
        self._indexing_guard_factory = indexing_guard_factory
        self.summary_index_service = SummaryIndex()

    @override
    def _run(self) -> NodeRunResult:
        node_data = self.node_data
        variable_pool = self.runtime_state.variable_pool

        # get dataset id as string
        dataset_id_segment = get_system_segment(variable_pool, SystemVariableKey.DATASET_ID)
        if not dataset_id_segment:
            raise KnowledgeIndexNodeError("Dataset ID is required.")
        dataset_id: str = dataset_id_segment.value

        # get document id as string (may be empty when not provided)
        document_id_segment = get_system_segment(variable_pool, SystemVariableKey.DOCUMENT_ID)
        document_id: str = document_id_segment.value if document_id_segment else ""

        # extract variables
        variable = variable_pool.get(node_data.index_chunk_variable_selector)
        if not variable:
            raise KnowledgeIndexNodeError("Index chunk variable is required.")
        invoke_from_value = get_system_text(variable_pool, SystemVariableKey.INVOKE_FROM)
        is_preview = invoke_from_value == _INVOKE_FROM_DEBUGGER

        chunks = variable.value
        variables = {"chunks": chunks}
        if not chunks:
            return NodeRunResult(
                status=WorkflowNodeExecutionStatus.FAILED, inputs=variables, error="Chunks is required."
            )

        try:
            summary_index_setting = node_data.summary_index_setting
            if is_preview:
                # Preview mode: generate summaries for chunks directly without saving to database
                # Format preview and generate summaries on-the-fly
                # Get indexing_technique and summary_index_setting from node_data (workflow graph config)
                # or fallback to dataset if not available in node_data

                with session_factory.create_session() as session:
                    outputs = self.index_processor.get_preview_output(
                        chunks,
                        dataset_id,
                        document_id,
                        node_data.chunk_structure,
                        summary_index_setting,
                        session=session,
                    )
                return NodeRunResult(
                    status=WorkflowNodeExecutionStatus.SUCCEEDED,
                    inputs=variables,
                    outputs=outputs.model_dump(exclude_none=True),
                )

            original_document_id_segment = get_system_segment(variable_pool, SystemVariableKey.ORIGINAL_DOCUMENT_ID)
            capacity_segment = get_system_segment(variable_pool, SystemVariableKey.ENFORCE_PROJECTED_CAPACITY)
            enforce_projected_capacity = capacity_segment.value if capacity_segment else None
            if enforce_projected_capacity is not None and not isinstance(enforce_projected_capacity, bool):
                raise KnowledgeIndexNodeError("Projected capacity policy must be a boolean.")
            indexing_request_id = get_system_text(variable_pool, SystemVariableKey.INDEXING_REQUEST_ID)
            indexing_claim_id = get_system_text(variable_pool, SystemVariableKey.INDEXING_CLAIM_ID)
            batch = get_system_segment(variable_pool, SystemVariableKey.BATCH)
            if not batch:
                raise KnowledgeIndexNodeError("Batch is required.")

            with session_factory.create_session() as session:
                results = self._invoke_knowledge_index(
                    session=session,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    original_document_id=original_document_id_segment.value if original_document_id_segment else "",
                    is_preview=is_preview,
                    batch=batch.value,
                    chunks=chunks,
                    summary_index_setting=summary_index_setting,
                    enforce_projected_capacity=enforce_projected_capacity,
                    indexing_request_id=indexing_request_id,
                    claim_id=indexing_claim_id,
                )
            return NodeRunResult(status=WorkflowNodeExecutionStatus.SUCCEEDED, inputs=variables, outputs=results)

        except KnowledgeIndexNodeError as e:
            logger.warning("Error when running knowledge index node", exc_info=True)
            return NodeRunResult(
                status=WorkflowNodeExecutionStatus.FAILED,
                inputs=variables,
                error=str(e),
                error_type=type(e).__name__,
            )
        except Exception as e:
            logger.error(e, exc_info=True)
            return NodeRunResult(
                status=WorkflowNodeExecutionStatus.FAILED,
                inputs=variables,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _invoke_knowledge_index(
        self,
        dataset_id: str,
        document_id: str,
        original_document_id: str,
        is_preview: bool,
        batch: Any,
        chunks: Mapping[str, Any],
        summary_index_setting: SummaryIndexSettingDict | None = None,
        indexing_request_id: str | None = None,
        claim_id: str | None = None,
        enforce_projected_capacity: bool | None = None,
        *,
        session: Session,
    ):
        if not document_id:
            raise KnowledgeIndexNodeError("document_id is required.")
        dify_ctx = DifyRunContext.model_validate(self.require_run_context_value(DIFY_RUN_CONTEXT_KEY))
        request_metadata = {"app_type": dify_ctx.app_type} if dify_ctx.app_type is not None else None
        with use_credit_usage_metadata(request_metadata):
            if indexing_request_id is None and claim_id is None:
                rst = self.index_processor.index_and_clean(
                    dataset_id,
                    document_id,
                    original_document_id,
                    chunks,
                    batch,
                    summary_index_setting,
                    session=session,
                    enforce_projected_capacity=enforce_projected_capacity,
                )
            else:
                rst = self.index_processor.index_and_clean(
                    dataset_id,
                    document_id,
                    original_document_id,
                    chunks,
                    batch,
                    summary_index_setting,
                    session=session,
                    enforce_projected_capacity=enforce_projected_capacity,
                    indexing_request_id=indexing_request_id,
                    claim_id=claim_id,
                )
            # Summary generation opens independent sessions and must see the indexed rows.
            session.commit()
            if indexing_request_id is None and claim_id is None:
                mutation_guard = None
            else:
                if self._indexing_guard_factory is None:
                    raise RuntimeError("pipeline indexing guard factory is required for claimed indexing")
                mutation_guard = self._indexing_guard_factory(
                    session,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    indexing_request_id=indexing_request_id,
                    claim_id=claim_id,
                )
            with mutation_guard(session) if mutation_guard is not None else nullcontext():
                self.summary_index_service.generate_and_vectorize_summary(
                    dataset_id, document_id, is_preview, summary_index_setting
                )
            if mutation_guard is not None:
                # A claimed summary can exceed the nominal lease. Publish the
                # refreshed expiry before releasing the row lock to a takeover.
                session.commit()
        return rst

    @classmethod
    @override
    def version(cls) -> str:
        return "1"

    def get_streaming_template(self) -> Template:
        """
        Get the template for streaming.

        Returns:
            Template instance for this knowledge index node
        """
        return Template(segments=[])

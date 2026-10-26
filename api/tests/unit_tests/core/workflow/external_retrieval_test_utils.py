from collections.abc import Mapping
from typing import cast
from unittest.mock import create_autospec

from core.rag.entities.retrieval_settings import ExternalDatasetRetriever
from graphon.entities.base_node_data import BaseNodeData
from graphon.entities.graph_config import NodeConfigDict
from graphon.enums import NodeExecutionType
from graphon.graph import NodeFactory
from graphon.nodes.base.node import Node
from graphon.runtime import RuntimeState


class ExternalRetrievalTestNodeFactory:
    """NodeFactory capability for tests whose parent graph is already built."""

    def __init__(self, external_retrieval: ExternalDatasetRetriever) -> None:
        self.external_retrieval = external_retrieval

    def with_runtime_state(self, runtime_state: RuntimeState) -> NodeFactory:
        del runtime_state
        raise AssertionError("The already-built parent graph must not recreate nodes")

    def with_graph_config(self, graph_config: Mapping[str, object]) -> NodeFactory:
        del graph_config
        raise AssertionError("The already-built parent graph must not recreate nodes")

    def validate_node(self, node_config: NodeConfigDict) -> NodeExecutionType:
        del node_config
        raise AssertionError("The already-built parent graph must not validate nodes")

    def create_node(self, node_config: NodeConfigDict) -> Node[BaseNodeData]:
        del node_config
        raise AssertionError("The already-built parent graph must not recreate nodes")


def make_external_dataset_retriever() -> ExternalDatasetRetriever:
    return cast(
        ExternalDatasetRetriever,
        create_autospec(ExternalDatasetRetriever, instance=True, spec_set=True),
    )

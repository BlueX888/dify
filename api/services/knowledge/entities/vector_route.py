"""Persistence-neutral identity for an actual vector backend target."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

_LINDORM_VECTOR_TYPE = "lindorm"
_LINDORM_UGC_INDEX_PREFIX = "ugc_index"
_RESOLVED_COLLECTION_NAME = "_dify_resolved_collection_name"


def _load_index_struct(index_struct: str) -> dict[str, Any]:
    try:
        value = json.loads(index_struct)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Vector route has an invalid index structure") from error
    if not isinstance(value, dict):
        raise ValueError("Vector route has an invalid index structure")
    return value


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Vector route has an invalid {field}")
    return value


def _lindorm_ugc_collection_name(index_struct: Mapping[str, object]) -> str:
    dimension = index_struct.get("dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("Vector route has an invalid Lindorm dimension")
    index_type = _required_string(index_struct.get("index_type"), field="Lindorm index type")
    distance_type = _required_string(index_struct.get("distance_type"), field="Lindorm distance type")
    return f"{_LINDORM_UGC_INDEX_PREFIX}_{dimension}_{index_type}_{distance_type}".lower()


def vector_route_target_from_index_struct(index_struct: str) -> tuple[str, str]:
    """Return a trustworthy backend target from persisted operational evidence.

    Lindorm UGC uses one shared physical index and a dataset-specific routing
    value. The routing value stays in ``vector_store.class_prefix`` for the
    provider factory, while the resolved shared collection is recorded
    separately. Legacy minimal snapshots that only name a UGC collection do
    not prove the routing value and therefore fail closed.
    """
    value = _load_index_struct(index_struct)
    try:
        vector_type = _required_string(value["type"], field="vector type")
        vector_store = value["vector_store"]
        if not isinstance(vector_store, Mapping):
            raise ValueError("Vector route has an invalid vector store")
        class_prefix = _required_string(vector_store["class_prefix"], field="collection prefix")
    except KeyError as error:
        raise ValueError("Vector route has an invalid index structure") from error

    if vector_type != _LINDORM_VECTOR_TYPE:
        return vector_type, class_prefix

    using_ugc = value.get("using_ugc")
    if using_ugc is True:
        expected_collection = _lindorm_ugc_collection_name(value)
        resolved_collection = value.get(_RESOLVED_COLLECTION_NAME)
        if resolved_collection is not None and (
            _required_string(resolved_collection, field="Lindorm resolved collection").lower() != expected_collection
        ):
            raise ValueError("Vector route has inconsistent Lindorm UGC evidence")
        return vector_type, expected_collection
    if using_ugc is not None and using_ugc is not False:
        raise ValueError("Vector route has an invalid Lindorm UGC marker")
    if class_prefix.lower().startswith(f"{_LINDORM_UGC_INDEX_PREFIX}_"):
        raise ValueError("Lindorm shared collection has no trustworthy UGC routing evidence")
    return vector_type, class_prefix.lower()


class VectorRoute(BaseModel):
    """Non-secret routing captured from the vector processor used for I/O."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    vector_type: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    index_struct: str
    collection_binding_id: str | None
    indexing_technique: str = Field(min_length=1)
    doc_form: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_index_struct(self) -> VectorRoute:
        route_type, collection_name = vector_route_target_from_index_struct(self.index_struct)
        if route_type != self.vector_type or collection_name != self.collection_name:
            raise ValueError("Vector route does not match its index structure")
        return self

    @classmethod
    def capture(
        cls,
        *,
        tenant_id: str,
        dataset_id: str,
        vector_type: str,
        collection_name: str,
        routing_index_struct: str | None = None,
        collection_binding_id: str | None,
        indexing_technique: str,
        doc_form: str,
    ) -> VectorRoute:
        if vector_type == _LINDORM_VECTOR_TYPE and routing_index_struct is not None:
            source_struct = _load_index_struct(routing_index_struct)
            if source_struct.get("type") != vector_type:
                raise ValueError("Lindorm route does not match its source index structure")
            vector_store = source_struct.get("vector_store")
            if not isinstance(vector_store, Mapping):
                raise ValueError("Lindorm route has an invalid vector store")
            class_prefix = _required_string(vector_store.get("class_prefix"), field="Lindorm routing value")
            using_ugc = source_struct.get("using_ugc")
            route_struct: dict[str, object] = {
                "type": vector_type,
                "vector_store": {"class_prefix": class_prefix},
            }
            if using_ugc is True:
                expected_collection = _lindorm_ugc_collection_name(source_struct)
                if expected_collection != collection_name:
                    raise ValueError("Lindorm route resolved to an unexpected UGC collection")
                route_struct.update(
                    {
                        "using_ugc": True,
                        "dimension": source_struct["dimension"],
                        "index_type": source_struct["index_type"],
                        "distance_type": source_struct["distance_type"],
                    }
                )
                route_struct[_RESOLVED_COLLECTION_NAME] = collection_name
            elif using_ugc is False:
                if collection_name.lower().startswith(f"{_LINDORM_UGC_INDEX_PREFIX}_"):
                    raise ValueError("Lindorm shared collection cannot be captured as a dedicated index")
                route_struct["using_ugc"] = False
            else:
                if collection_name.lower().startswith(f"{_LINDORM_UGC_INDEX_PREFIX}_"):
                    raise ValueError("Legacy Lindorm UGC route has no trustworthy routing evidence")
            index_struct = json.dumps(route_struct, separators=(",", ":"), sort_keys=True)
        else:
            if vector_type == _LINDORM_VECTOR_TYPE and collection_name.lower().startswith(
                f"{_LINDORM_UGC_INDEX_PREFIX}_"
            ):
                raise ValueError("Lindorm UGC route requires its operational index structure")
            index_struct = json.dumps(
                {
                    "type": vector_type,
                    "vector_store": {"class_prefix": collection_name},
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        return cls(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            vector_type=vector_type,
            collection_name=collection_name,
            index_struct=index_struct,
            collection_binding_id=collection_binding_id,
            indexing_technique=indexing_technique,
            doc_form=doc_form,
        )

    @property
    def serialized(self) -> str:
        return self.model_dump_json()

    @property
    def key(self) -> str:
        physical_identity_values = {
            "tenant_id": self.tenant_id,
            "dataset_id": self.dataset_id,
            "vector_type": self.vector_type,
            "collection_name": self.collection_name,
        }
        # Old dedicated-index routes retain their historical hash. New shared
        # Lindorm routes include the routing evidence so a later routing change
        # in the same dataset cannot reuse an earlier cleanup receipt.
        if self.vector_type == _LINDORM_VECTOR_TYPE and json.loads(self.index_struct).get("using_ugc") is True:
            physical_identity_values["index_struct"] = self.index_struct
        physical_identity = json.dumps(physical_identity_values, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(physical_identity.encode()).hexdigest()

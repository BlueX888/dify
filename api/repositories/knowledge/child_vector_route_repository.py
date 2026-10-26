"""Persist the physical route selected for parent-child vectors."""

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, func, select, update
from sqlalchemy.orm import Session

from models.dataset import ChildChunk, Document
from models.model import ChildVectorCleanupOutbox

_LINDORM_VECTOR_TYPE = "lindorm"
_LINDORM_UGC_INDEX_PREFIX = "ugc_index"
_RESOLVED_COLLECTION_NAME = "_dify_resolved_collection_name"


def _database_now(session: Session) -> datetime:
    value = session.scalar(select(func.current_timestamp(type_=DateTime())))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value.replace(tzinfo=None, microsecond=0)


def _route_payload(
    *,
    tenant_id: str,
    dataset_id: str,
    collection_binding_id: str | None,
    indexing_technique: str,
    doc_form: str,
    vector_type: str,
    collection_name: str,
    routing_index_struct: str | None,
) -> tuple[str, str]:
    index_struct = _route_index_struct(
        vector_type=vector_type,
        collection_name=collection_name,
        routing_index_struct=routing_index_struct,
    )
    serialized = json.dumps(
        {
            "tenant_id": tenant_id,
            "dataset_id": dataset_id,
            "vector_type": vector_type,
            "collection_name": collection_name,
            "index_struct": index_struct,
            "collection_binding_id": collection_binding_id,
            "indexing_technique": indexing_technique,
            "doc_form": doc_form,
        },
        separators=(",", ":"),
    )
    return serialized, _physical_route_hash(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        vector_type=vector_type,
        collection_name=collection_name,
        index_struct=index_struct,
    )


def _route_index_struct(*, vector_type: str, collection_name: str, routing_index_struct: str | None) -> str:
    if vector_type != _LINDORM_VECTOR_TYPE or routing_index_struct is None:
        if vector_type == _LINDORM_VECTOR_TYPE and collection_name.lower().startswith(f"{_LINDORM_UGC_INDEX_PREFIX}_"):
            raise ValueError("Lindorm UGC route requires its operational index structure")
        return json.dumps(
            {"type": vector_type, "vector_store": {"class_prefix": collection_name}},
            separators=(",", ":"),
            sort_keys=True,
        )
    try:
        source = json.loads(routing_index_struct)
        class_prefix = source["vector_store"]["class_prefix"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Lindorm route has an invalid source index structure") from error
    if source.get("type") != vector_type or not isinstance(class_prefix, str) or not class_prefix:
        raise ValueError("Lindorm route has an invalid source index structure")
    route_struct: dict[str, object] = {
        "type": vector_type,
        "vector_store": {"class_prefix": class_prefix},
    }
    using_ugc = source.get("using_ugc")
    if using_ugc is True:
        expected_collection = _lindorm_ugc_collection(source)
        if expected_collection != collection_name:
            raise ValueError("Lindorm route resolved to an unexpected UGC collection")
        route_struct.update(
            {
                "using_ugc": True,
                "dimension": source["dimension"],
                "index_type": source["index_type"],
                "distance_type": source["distance_type"],
                _RESOLVED_COLLECTION_NAME: collection_name,
            }
        )
    elif using_ugc is False:
        if collection_name.lower().startswith(f"{_LINDORM_UGC_INDEX_PREFIX}_"):
            raise ValueError("Lindorm shared collection cannot be captured as a dedicated index")
        route_struct["using_ugc"] = False
    elif collection_name.lower().startswith(f"{_LINDORM_UGC_INDEX_PREFIX}_"):
        raise ValueError("Legacy Lindorm UGC route has no trustworthy routing evidence")
    return json.dumps(route_struct, separators=(",", ":"), sort_keys=True)


def _lindorm_ugc_collection(index_struct: dict[str, object]) -> str:
    dimension = index_struct.get("dimension")
    index_type = index_struct.get("index_type")
    distance_type = index_struct.get("distance_type")
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension <= 0
        or not isinstance(index_type, str)
        or not index_type
        or not isinstance(distance_type, str)
        or not distance_type
    ):
        raise ValueError("Lindorm route has incomplete UGC evidence")
    return f"{_LINDORM_UGC_INDEX_PREFIX}_{dimension}_{index_type}_{distance_type}".lower()


def _target_from_index_struct(index_struct_json: str) -> tuple[str, str, bool]:
    try:
        index_struct = json.loads(index_struct_json)
        vector_type = index_struct["type"]
        class_prefix = index_struct["vector_store"]["class_prefix"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Vector route has an invalid index structure") from error
    if not isinstance(vector_type, str) or not vector_type or not isinstance(class_prefix, str) or not class_prefix:
        raise ValueError("Vector route has an invalid index structure")
    if vector_type != _LINDORM_VECTOR_TYPE:
        return vector_type, class_prefix, False
    using_ugc = index_struct.get("using_ugc")
    if using_ugc is True:
        collection_name = _lindorm_ugc_collection(index_struct)
        resolved_collection = index_struct.get(_RESOLVED_COLLECTION_NAME)
        if resolved_collection is not None and resolved_collection != collection_name:
            raise ValueError("Vector route has inconsistent Lindorm UGC evidence")
        return vector_type, collection_name, True
    if using_ugc is not None and using_ugc is not False:
        raise ValueError("Vector route has an invalid Lindorm UGC marker")
    if class_prefix.lower().startswith(f"{_LINDORM_UGC_INDEX_PREFIX}_"):
        raise ValueError("Lindorm shared collection has no trustworthy UGC routing evidence")
    return vector_type, class_prefix.lower(), False


def _physical_route_hash(
    *, tenant_id: str, dataset_id: str, vector_type: str, collection_name: str, index_struct: str
) -> str:
    identity: dict[str, str] = {
        "tenant_id": tenant_id,
        "dataset_id": dataset_id,
        "vector_type": vector_type,
        "collection_name": collection_name,
    }
    _, _, is_shared_lindorm = _target_from_index_struct(index_struct)
    if is_shared_lindorm:
        identity["index_struct"] = index_struct
    return hashlib.sha256(json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def stored_vector_route(
    *,
    route_serialized: str | None,
    route_hash: str | None,
    tenant_id: str,
    dataset_id: str,
) -> tuple[str, str] | None:
    """Validate persisted route evidence without importing application DTOs."""
    if route_serialized is None or route_hash is None:
        return None
    try:
        payload = json.loads(route_serialized)
        stored_tenant_id = payload["tenant_id"]
        stored_dataset_id = payload["dataset_id"]
        vector_type = payload["vector_type"]
        collection_name = payload["collection_name"]
        index_struct = payload["index_struct"]
        indexing_technique = payload["indexing_technique"]
        doc_form = payload["doc_form"]
        collection_binding_id = payload["collection_binding_id"]
        if not all(
            isinstance(value, str) and value
            for value in (
                stored_tenant_id,
                stored_dataset_id,
                vector_type,
                collection_name,
                index_struct,
                indexing_technique,
                doc_form,
            )
        ):
            return None
        if collection_binding_id is not None and not isinstance(collection_binding_id, str):
            return None
        route_type, resolved_collection, _ = _target_from_index_struct(index_struct)
        expected_hash = _physical_route_hash(
            tenant_id=stored_tenant_id,
            dataset_id=stored_dataset_id,
            vector_type=vector_type,
            collection_name=collection_name,
            index_struct=index_struct,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        stored_tenant_id != tenant_id
        or stored_dataset_id != dataset_id
        or route_type != vector_type
        or resolved_collection != collection_name
        or expected_hash != route_hash
    ):
        return None
    return route_serialized, route_hash


def _stored_route(child: ChildChunk) -> tuple[str, str] | None:
    return stored_vector_route(
        route_serialized=child.vector_route,
        route_hash=child.vector_route_hash,
        tenant_id=child.tenant_id,
        dataset_id=child.dataset_id,
    )


def persist_child_vector_routes(
    session: Session,
    *,
    tenant_id: str,
    dataset_id: str,
    node_ids: Sequence[str],
    route_serialized: str,
    route_hash: str,
    route_doc_form: str,
) -> None:
    """Switch exact child rows and durably retain every proven old route."""
    normalized_node_ids = tuple(dict.fromkeys(node_ids))
    if not normalized_node_ids:
        return
    rows = session.execute(
        select(ChildChunk, Document.doc_form)
        .join(
            Document,
            (Document.id == ChildChunk.document_id)
            & (Document.dataset_id == ChildChunk.dataset_id)
            & (Document.tenant_id == ChildChunk.tenant_id),
        )
        .where(
            ChildChunk.tenant_id == tenant_id,
            ChildChunk.dataset_id == dataset_id,
            ChildChunk.index_node_id.in_(normalized_node_ids),
        )
        .with_for_update(of=ChildChunk)
        .execution_options(populate_existing=True)
    ).all()
    children = [child for child, _ in rows]
    if len(children) != len(normalized_node_ids) or {child.index_node_id for child in children} != set(
        normalized_node_ids
    ):
        raise LookupError("A child vector route target no longer exists or is ambiguous")
    if any(str(doc_form) != route_doc_form for _, doc_form in rows):
        raise ValueError("Child vector route document form does not match its child rows")

    old_routes = {
        (child.index_node_id, stored[1])
        for child in children
        if (stored := _stored_route(child)) is not None and stored[1] != route_hash
    }
    existing = set()
    if old_routes:
        existing = set(
            session.execute(
                select(ChildVectorCleanupOutbox.node_id, ChildVectorCleanupOutbox.vector_route_hash)
                .where(
                    ChildVectorCleanupOutbox.tenant_id == tenant_id,
                    ChildVectorCleanupOutbox.node_id.in_([node_id for node_id, _ in old_routes]),
                )
                .with_for_update()
            )
        )
        database_now = _database_now(session)
        session.add_all(
            [
                ChildVectorCleanupOutbox(
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    document_id=child.document_id,
                    segment_id=child.segment_id,
                    child_id=child.id,
                    node_id=child.index_node_id,
                    writer_id=str(uuid4()),
                    vector_route=stored[0],
                    vector_route_hash=stored[1],
                    writer_finished_at=database_now,
                )
                for child in children
                if (stored := _stored_route(child)) is not None
                and stored[1] != route_hash
                and (child.index_node_id, stored[1]) not in existing
            ]
        )

    session.execute(
        update(ChildChunk)
        .where(
            ChildChunk.tenant_id == tenant_id,
            ChildChunk.dataset_id == dataset_id,
            ChildChunk.index_node_id.in_(normalized_node_ids),
        )
        .values(vector_route=route_serialized, vector_route_hash=route_hash)
    )


class SQLAlchemyChildVectorRouteStore:
    """Persist route evidence in the caller-owned publication transaction."""

    @staticmethod
    def persist(
        session: Session,
        *,
        tenant_id: str,
        dataset_id: str,
        collection_binding_id: str | None,
        indexing_technique: str,
        doc_form: str,
        node_ids: Sequence[str],
        vector_type: str,
        collection_name: str,
        routing_index_struct: str | None = None,
    ) -> None:
        route_serialized, route_hash = _route_payload(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            collection_binding_id=collection_binding_id,
            indexing_technique=indexing_technique,
            doc_form=doc_form,
            vector_type=vector_type,
            collection_name=collection_name,
            routing_index_struct=routing_index_struct,
        )
        persist_child_vector_routes(
            session,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            node_ids=node_ids,
            route_serialized=route_serialized,
            route_hash=route_hash,
            route_doc_form=doc_form,
        )

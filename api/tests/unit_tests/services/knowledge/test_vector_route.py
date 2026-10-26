import hashlib
import json

import pytest
from pydantic import ValidationError

from services.knowledge.entities.vector_route import VectorRoute, vector_route_target_from_index_struct


def _lindorm_ugc_index_struct(*, routing_value: str = "dataset_route") -> str:
    return json.dumps(
        {
            "type": "lindorm",
            "vector_store": {"class_prefix": routing_value},
            "using_ugc": True,
            "dimension": 1536,
            "index_type": "hnsw",
            "distance_type": "l2",
            # Provider-specific additions must not silently enter the durable
            # routing identity or a future cleanup credential snapshot.
            "ignored_extension": "not-persisted",
        }
    )


def _capture_ugc(*, routing_value: str = "dataset_route") -> VectorRoute:
    return VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="lindorm",
        collection_name="ugc_index_1536_hnsw_l2",
        routing_index_struct=_lindorm_ugc_index_struct(routing_value=routing_value),
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="text_model",
    )


def test_lindorm_ugc_route_round_trips_complete_nonsecret_operational_evidence() -> None:
    route = _capture_ugc()

    restored = VectorRoute.model_validate_json(route.serialized)
    persisted_index_struct = json.loads(restored.index_struct)

    assert restored == route
    assert vector_route_target_from_index_struct(restored.index_struct) == (
        "lindorm",
        "ugc_index_1536_hnsw_l2",
    )
    assert persisted_index_struct == {
        "_dify_resolved_collection_name": "ugc_index_1536_hnsw_l2",
        "dimension": 1536,
        "distance_type": "l2",
        "index_type": "hnsw",
        "type": "lindorm",
        "using_ugc": True,
        "vector_store": {"class_prefix": "dataset_route"},
    }


def test_lindorm_ugc_route_key_includes_dataset_routing_identity() -> None:
    route_a = _capture_ugc(routing_value="route-a")
    route_b = _capture_ugc(routing_value="route-b")

    assert route_a.collection_name == route_b.collection_name
    assert route_a.key != route_b.key


def test_complete_legacy_lindorm_ugc_route_derives_shared_collection_without_internal_marker() -> None:
    legacy_index_struct = _lindorm_ugc_index_struct()
    route = VectorRoute(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="lindorm",
        collection_name="ugc_index_1536_hnsw_l2",
        index_struct=legacy_index_struct,
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="text_model",
    )

    assert vector_route_target_from_index_struct(route.index_struct) == (
        "lindorm",
        "ugc_index_1536_hnsw_l2",
    )


@pytest.mark.parametrize(
    "index_struct",
    [
        json.dumps(
            {
                "type": "lindorm",
                "vector_store": {"class_prefix": "ugc_index_1536_hnsw_l2"},
            }
        ),
        json.dumps(
            {
                "type": "lindorm",
                "vector_store": {"class_prefix": "ugc_index_1536_hnsw_l2"},
                "using_ugc": False,
            }
        ),
        json.dumps(
            {
                "type": "lindorm",
                "vector_store": {"class_prefix": "dataset_route"},
                "using_ugc": True,
                "dimension": 1536,
                "index_type": "hnsw",
                "distance_type": "l2",
                "_dify_resolved_collection_name": "ugc_index_1536_hnsw_cosine",
            }
        ),
    ],
)
def test_untrusted_or_incomplete_lindorm_ugc_route_fails_closed(index_struct: str) -> None:
    with pytest.raises(ValidationError):
        VectorRoute(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            vector_type="lindorm",
            collection_name="ugc_index_1536_hnsw_l2",
            index_struct=index_struct,
            collection_binding_id=None,
            indexing_technique="high_quality",
            doc_form="text_model",
        )


def test_dedicated_legacy_route_keeps_historical_key() -> None:
    route = VectorRoute.capture(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        vector_type="lindorm",
        collection_name="dedicated_index",
        collection_binding_id=None,
        indexing_technique="high_quality",
        doc_form="text_model",
    )
    historical_identity = json.dumps(
        {
            "tenant_id": route.tenant_id,
            "dataset_id": route.dataset_id,
            "vector_type": route.vector_type,
            "collection_name": route.collection_name,
        },
        separators=(",", ":"),
        sort_keys=True,
    )

    assert route.key == hashlib.sha256(historical_identity.encode()).hexdigest()


def test_capture_rejects_shared_lindorm_collection_marked_as_dedicated() -> None:
    with pytest.raises(ValueError, match="shared collection"):
        VectorRoute.capture(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            vector_type="lindorm",
            collection_name="ugc_index_1536_hnsw_l2",
            routing_index_struct=json.dumps(
                {
                    "type": "lindorm",
                    "vector_store": {"class_prefix": "ugc_index_1536_hnsw_l2"},
                    "using_ugc": False,
                }
            ),
            collection_binding_id=None,
            indexing_technique="high_quality",
            doc_form="text_model",
        )

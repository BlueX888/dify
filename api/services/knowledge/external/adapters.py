"""External knowledge HTTP and access-policy adapters."""

import json
from collections.abc import Mapping
from typing import Any

import httpx

from core.helper import ssrf_proxy
from machinery.context import RequestContext
from services.enterprise import rbac_service
from services.errors.knowledge_retrieval import ExternalKnowledgeRetrievalError
from services.knowledge.entities.external import (
    ExternalKnowledgeApiSetting,
    ExternalKnowledgeRetrievalRequest,
    validate_external_knowledge_api_settings,
)
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.external.application import ExternalKnowledgeRepository
from services.rbac.contracts import RBACResourceType


def sync_external_dataset_creator_access(context: RequestContext, ref: DatasetRef) -> None:
    rbac_service.try_sync_creator_access_policy_member_bindings(
        context.active_workspace_id, context.account_id, RBACResourceType.DATASET, ref.dataset_id
    )


class ExternalKnowledgeGateway:
    def check_settings(self, settings: Mapping[str, Any]) -> None:
        self.check_endpoint_and_api_key(settings)

    def retrieve(self, request: ExternalKnowledgeRetrievalRequest) -> list[dict[str, Any]]:
        """Send the query unchanged; JSON serialization handles transport escaping."""
        http_request = self.build_external_knowledge_retrieval_request(request)
        return self.execute_external_knowledge_retrieval(http_request)

    @staticmethod
    def check_endpoint_and_api_key(settings: Mapping[str, Any]) -> None:
        validate_external_knowledge_api_settings(settings)
        endpoint = f"{settings['endpoint']}/retrieval"
        api_key = settings["api_key"]
        # Send a minimal body shaped like the External Knowledge API retrieval contract so providers
        # that require a JSON payload (e.g. RAGFlow) accept the validation probe instead of rejecting
        # a body-less POST. Mirrors the request built in build_external_knowledge_retrieval_request.
        validation_payload = {
            "knowledge_id": "",
            "query": "",
            "retrieval_setting": {"top_k": 1, "score_threshold": 0.0},
        }
        try:
            response = ssrf_proxy.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data=json.dumps(validation_payload),
            )
        except Exception as e:
            raise ValueError(f"failed to connect to the endpoint: {endpoint}") from e
        if response.status_code == 502:
            raise ValueError(f"Bad Gateway: failed to connect to the endpoint: {endpoint}")
        if response.status_code == 404:
            raise ValueError(f"Not Found: failed to connect to the endpoint: {endpoint}")
        if response.status_code == 401:
            raise ValueError("Unauthorized: Authorization failed with the provided api_key")
        if response.status_code == 403:
            raise ValueError("Forbidden: Authorization failed with the provided api_key")

    @staticmethod
    def send_retrieval_request(request: ExternalKnowledgeApiSetting) -> httpx.Response:
        """POST a retrieval request through the SSRF-safe transport."""
        return ssrf_proxy.post(
            request.url,
            headers=request.headers,
            data=json.dumps(request.params),
            follow_redirects=True,
        )

    @staticmethod
    def build_external_knowledge_retrieval_request(
        request: ExternalKnowledgeRetrievalRequest,
    ) -> ExternalKnowledgeApiSetting:
        """Adapt domain retrieval input to the external provider's HTTP contract."""
        headers = {"Content-Type": "application/json"}
        if request.api_key:
            headers["Authorization"] = f"Bearer {request.api_key}"
        parameters = request.retrieval_model
        threshold = parameters.get("score_threshold", 0.0) if parameters.get("score_threshold_enabled") else 0.0
        return ExternalKnowledgeApiSetting(
            url=f"{request.endpoint}/retrieval",
            headers=headers,
            params={
                "retrieval_setting": {"top_k": parameters.get("top_k"), "score_threshold": threshold},
                "query": request.query,
                "knowledge_id": request.knowledge_id,
                "metadata_condition": request.metadata_filters,
            },
        )

    @staticmethod
    def execute_external_knowledge_retrieval(request: ExternalKnowledgeApiSetting) -> list[dict[str, Any]]:
        """Execute and validate a prepared request without holding a database session."""
        try:
            response = ExternalKnowledgeGateway.send_retrieval_request(request)
        except Exception as exc:
            raise ExternalKnowledgeRetrievalError(str(exc)) from exc

        if response.status_code != 200:
            raise ExternalKnowledgeRetrievalError(response.text)

        try:
            response_payload = response.json()
        except Exception as exc:
            raise ExternalKnowledgeRetrievalError("invalid external knowledge response") from exc

        if not isinstance(response_payload, dict):
            raise ExternalKnowledgeRetrievalError("invalid external knowledge response")

        records = response_payload.get("records")
        if not isinstance(records, list):
            raise ExternalKnowledgeRetrievalError("invalid external knowledge response: records must be a list")

        validated_records: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ExternalKnowledgeRetrievalError(
                    f"invalid external knowledge response: records[{index}] must be an object"
                )

            if "content" in record and not isinstance(record["content"], str):
                raise ExternalKnowledgeRetrievalError(
                    f"invalid external knowledge response: records[{index}].content must be a string"
                )

            if "score" in record:
                score = record["score"]
                if isinstance(score, bool) or not isinstance(score, int | float):
                    raise ExternalKnowledgeRetrievalError(
                        f"invalid external knowledge response: records[{index}].score must be a number"
                    )

            if "title" in record and not isinstance(record["title"], str):
                raise ExternalKnowledgeRetrievalError(
                    f"invalid external knowledge response: records[{index}].title must be a string"
                )

            if "metadata" in record and not isinstance(record["metadata"], dict):
                raise ExternalKnowledgeRetrievalError(
                    f"invalid external knowledge response: records[{index}].metadata must be an object"
                )

            # The public provider contract permits missing fields. Normalize
            # the runtime-sensitive values to finite, sortable representations
            # while retaining any provider-specific fields.
            normalized_record = dict(record)
            normalized_record.setdefault("content", "")
            normalized_record.setdefault("score", 0.0)
            normalized_record.setdefault("metadata", {})
            validated_records.append(normalized_record)

        return validated_records


class ExternalDatasetRetrievalAdapter:
    """Combine a short configuration read with external provider I/O."""

    def __init__(self, *, repository: ExternalKnowledgeRepository, gateway: ExternalKnowledgeGateway) -> None:
        self._repository = repository
        self._gateway = gateway

    def retrieve(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        query: str,
        retrieval_model: dict[str, Any] | None,
        metadata_filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        request = self._repository.prepare_retrieval(
            DatasetRef(tenant_id, dataset_id),
            query=query,
            retrieval_model=retrieval_model,
            metadata_filters=metadata_filters,
        )
        if request is None:
            return []
        return self._gateway.retrieve(request)

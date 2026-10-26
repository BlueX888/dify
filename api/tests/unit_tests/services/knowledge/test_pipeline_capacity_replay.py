import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import ClassVar
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.index_processor import index_processor as index_processor_module
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.index_processor.index_processor import IndexProcessor
from core.rag.index_processor.index_processor_base import AttachmentRef, IndexingMutationGuard
from enums import CloudPlan, DeploymentEdition
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary, SegmentAttachmentBinding
from services.billing_service import BillingInfo, _VectorSpaceQuota
from services.vector_space_admission_service import VectorSpaceAdmissionError, VectorSpaceAdmissionService
from tests.unit_tests.model_factories import make_dataset, make_document, make_upload_file

_OVERSIZED_CHUNKS = {"general_chunks": [{"content": "chunk"}] * 5_000}


class _FakeRedisLock:
    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock

    def __enter__(self) -> "_FakeRedisLock":
        self._lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._lock.release()


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}

    def lock(self, key: str, **_kwargs: object) -> _FakeRedisLock:
        return _FakeRedisLock(self._locks.setdefault(key, threading.Lock()))

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def setex(self, key: str, _ttl: int, value: str) -> None:
        self.values[key] = value


@dataclass
class _IndexRecorder:
    indexed_document_ids: list[str] = field(default_factory=list)
    new_segment_ids: list[str] = field(default_factory=list)
    cleaned_node_ids: list[tuple[str, ...]] = field(default_factory=list)


class _RecordingIndexProcessor:
    def __init__(self, recorder: _IndexRecorder) -> None:
        self._recorder = recorder

    def index(
        self,
        _dataset: Dataset,
        document: Document,
        _chunks: object,
        session: Session,
        *,
        mutation_guard: IndexingMutationGuard | None = None,
    ) -> None:
        assert mutation_guard is None
        self._recorder.indexed_document_ids.append(document.id)
        new_segment = DocumentSegment(
            tenant_id=document.tenant_id,
            dataset_id=document.dataset_id,
            document_id=document.id,
            position=1,
            content="new result",
            word_count=2,
            tokens=2,
            created_by=document.created_by,
            index_node_id="new-node",
        )
        session.add(new_segment)
        session.flush()
        self._recorder.new_segment_ids.append(new_segment.id)

    def clean(
        self,
        _dataset: Dataset,
        node_ids: list[str] | None,
        _with_keywords: bool = True,
        *,
        session: Session,
        **_kwargs: object,
    ) -> None:
        assert session.in_transaction()
        self._recorder.cleaned_node_ids.append(tuple(node_ids or ()))


class _RecordingIndexProcessorFactory:
    recorder: ClassVar[_IndexRecorder]

    def __init__(self, _index_type: str | None, **_kwargs: object) -> None:
        pass

    def init_index_processor(self) -> _RecordingIndexProcessor:
        return _RecordingIndexProcessor(self.recorder)


@dataclass
class _VectorRecorder:
    deleted_node_ids: list[tuple[str, ...]] = field(default_factory=list)


class _RecordingVector:
    recorder: ClassVar[_VectorRecorder]

    def __init__(self, _dataset: Dataset, *, session: Session | None, vector_type: str | None = None) -> None:
        assert session is None
        assert vector_type == VectorType.TIDB_ON_QDRANT

    @staticmethod
    def resolve_vector_type(_dataset: Dataset, *, session: Session) -> str:
        assert session.in_transaction()
        return VectorType.TIDB_ON_QDRANT

    def delete_by_ids(self, node_ids: Sequence[str]) -> None:
        self.recorder.deleted_node_ids.append(tuple(node_ids))

    def get_routing_target(self) -> tuple[str, str]:
        return VectorType.TIDB_ON_QDRANT, "pipeline-test-collection"


@dataclass
class _AttachmentVectorStore:
    deleted_document_ids: list[tuple[str, ...]] = field(default_factory=list)

    def publish_bound(
        self,
        session: Session,
        *,
        tenant_id: str,
        dataset_id: str,
        collection_binding_id: str | None,
        indexing_technique: str,
        attachment_refs: Sequence[AttachmentRef],
        vector_type: str,
        collection_name: str,
        routing_index_struct: str | None,
        publish: Callable[
            [frozenset[AttachmentRef], Callable[[Sequence[str]], None], Callable[[Sequence[str]], None]], None
        ],
        mutation_guard: IndexingMutationGuard | None = None,
    ) -> frozenset[AttachmentRef]:
        assert session.in_transaction()
        assert tenant_id
        assert dataset_id
        assert collection_binding_id is None
        assert indexing_technique
        assert attachment_refs
        assert vector_type
        assert collection_name
        _ = routing_index_struct
        assert publish
        assert mutation_guard is None
        raise AssertionError("attachment publication is not expected")

    def delete_unreferenced(
        self,
        session: Session,
        *,
        tenant_id: str,
        dataset_id: str,
        collection_binding_id: str | None,
        indexing_technique: str,
        vector_type: str,
        collection_name: str,
        routing_index_struct: str | None,
        document_ids: Sequence[str] | None = None,
        segment_ids: Sequence[str] | None = None,
        candidate_attachment_ids: Sequence[str] | None = None,
        mutation_guard: IndexingMutationGuard | None = None,
        validate: Callable[[Session], None] | None = None,
    ) -> tuple[str, ...]:
        assert session.in_transaction()
        assert tenant_id
        assert dataset_id
        assert collection_binding_id is None
        assert indexing_technique
        assert vector_type == VectorType.TIDB_ON_QDRANT
        assert collection_name == "pipeline-test-collection"
        _ = routing_index_struct
        assert segment_ids is None
        assert candidate_attachment_ids is None
        assert mutation_guard is None
        assert validate is None
        self.deleted_document_ids.append(tuple(document_ids or ()))
        return ()


def _billing_info(_tenant_id: str, exclude_vector_space: bool = False) -> BillingInfo:
    assert exclude_vector_space is True
    return {
        "subscription": {"plan": CloudPlan.SANDBOX.value, "interval": "month", "education": False},
        "members": {"size": 1, "limit": 1},
        "apps": {"size": 1, "limit": 1},
        "documents_upload_quota": {"size": 1, "limit": 1},
        "annotation_quota_limit": {"size": 1, "limit": 1},
        "can_replace_logo": False,
        "model_load_balancing_enabled": False,
        "knowledge_pipeline_publish_enabled": True,
    }


def _vector_space(_tenant_id: str, bypass_cache: bool = False) -> _VectorSpaceQuota:
    assert bypass_cache is False
    return {"size": 1, "limit": 50}


def _embedding_dimension(_service: VectorSpaceAdmissionService, _dataset: Dataset, _probe_text: str) -> int:
    return 1_536


@pytest.fixture(autouse=True)
def _capacity_config(config_overrides: Callable[..., None]) -> None:
    config_overrides(
        DEPLOYMENT_EDITION=DeploymentEdition.CLOUD,
        TIDB_ON_QDRANT_ESTIMATED_STORAGE_LIMITS_MB="sandbox:60,professional:6400,team:25600",
    )


@dataclass(frozen=True)
class _PipelineGraph:
    dataset: Dataset
    document: Document
    partial_segment_id: str
    partial_summary_id: str
    attachment_binding_id: str
    upload_file_id: str


@pytest.fixture
def pipeline_graph(sqlite_session: Session) -> _PipelineGraph:
    dataset = make_dataset(
        dataset_id=str(uuid4()),
        tenant_id=str(uuid4()),
        created_by=str(uuid4()),
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        embedding_model_provider="provider",
        embedding_model="model",
        index_struct=json.dumps({"type": VectorType.TIDB_ON_QDRANT}),
        chunk_structure=IndexStructureType.PARAGRAPH_INDEX,
    )
    document = make_document(
        document_id=str(uuid4()),
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        created_by=dataset.created_by,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    partial_segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="partial",
        word_count=1,
        tokens=1,
        created_by=dataset.created_by,
        index_node_id="partial-node",
    )
    partial_summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=partial_segment.id,
        summary_content="partial summary",
        summary_index_node_id="partial-summary-node",
    )
    upload_file = make_upload_file(
        tenant_id=dataset.tenant_id,
        key="pipeline/partial-attachment.png",
        name="partial-attachment.png",
        extension="png",
        mime_type="image/png",
        created_by=dataset.created_by,
    )
    attachment_binding = SegmentAttachmentBinding(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=partial_segment.id,
        attachment_id=upload_file.id,
    )
    sqlite_session.add_all([dataset, document, partial_segment, partial_summary, upload_file, attachment_binding])
    sqlite_session.commit()
    return _PipelineGraph(
        dataset=dataset,
        document=document,
        partial_segment_id=partial_segment.id,
        partial_summary_id=partial_summary.id,
        attachment_binding_id=attachment_binding.id,
        upload_file_id=upload_file.id,
    )


@dataclass(frozen=True)
class _RunProbes:
    index: _IndexRecorder
    vector: _VectorRecorder
    attachments: _AttachmentVectorStore
    redis: _FakeRedis


def _run_indexing(
    sqlite_session: Session,
    pipeline_graph: _PipelineGraph,
    *,
    original_document_id: str,
    enforce_projected_capacity: bool,
    chunks: Mapping[str, object] = _OVERSIZED_CHUNKS,
) -> _RunProbes:
    index_recorder = _IndexRecorder()
    vector_recorder = _VectorRecorder()
    attachment_vectors = _AttachmentVectorStore()
    _RecordingIndexProcessorFactory.recorder = index_recorder
    _RecordingVector.recorder = vector_recorder
    redis = _FakeRedis()

    with (
        patch.object(index_processor_module, "IndexProcessorFactory", new=_RecordingIndexProcessorFactory),
        patch.object(index_processor_module, "Vector", new=_RecordingVector),
        patch("services.vector_space_admission_service.BillingService.get_info", new=_billing_info),
        patch("services.vector_space_admission_service.BillingService.get_vector_space", new=_vector_space),
        patch.object(VectorSpaceAdmissionService, "_get_embedding_dimension", new=_embedding_dimension),
        patch("services.vector_space_admission_service.redis_client", new=redis),
    ):
        IndexProcessor(attachment_vectors=attachment_vectors).index_and_clean(
            dataset_id=pipeline_graph.dataset.id,
            document_id=pipeline_graph.document.id,
            original_document_id=original_document_id,
            chunks=chunks,
            batch="batch-1",
            enforce_projected_capacity=enforce_projected_capacity,
            session=sqlite_session,
        )

    return _RunProbes(index=index_recorder, vector=vector_recorder, attachments=attachment_vectors, redis=redis)


@pytest.mark.parametrize("cleanup_target", ["", "recovery"], ids=["fresh", "recovery"])
def test_first_submission_enforces_projection_before_publish_or_partial_cleanup(
    sqlite_session: Session,
    pipeline_graph: _PipelineGraph,
    cleanup_target: str,
) -> None:
    original_document_id = pipeline_graph.document.id if cleanup_target else ""

    with pytest.raises(VectorSpaceAdmissionError, match="exceeding the 50 MB limit"):
        _run_indexing(
            sqlite_session,
            pipeline_graph,
            original_document_id=original_document_id,
            enforce_projected_capacity=True,
        )

    sqlite_session.expire_all()
    assert sqlite_session.get(DocumentSegment, pipeline_graph.partial_segment_id) is not None
    assert sqlite_session.get(DocumentSegmentSummary, pipeline_graph.partial_summary_id) is not None
    assert sqlite_session.get(SegmentAttachmentBinding, pipeline_graph.attachment_binding_id) is not None
    assert _RecordingIndexProcessorFactory.recorder.indexed_document_ids == []
    assert _RecordingVector.recorder.deleted_node_ids == []


def test_replacement_keeps_v1_projection_exemption(
    sqlite_session: Session,
    pipeline_graph: _PipelineGraph,
) -> None:
    probes = _run_indexing(
        sqlite_session,
        pipeline_graph,
        original_document_id=pipeline_graph.document.id,
        enforce_projected_capacity=False,
    )

    sqlite_session.expire_all()
    assert (
        sqlite_session.scalar(select(DocumentSegment.id).where(DocumentSegment.id == pipeline_graph.partial_segment_id))
        is None
    )
    assert probes.index.indexed_document_ids == [pipeline_graph.document.id]
    assert probes.vector.deleted_node_ids == [("partial-node", "partial-summary-node")]
    assert probes.attachments.deleted_document_ids == [(pipeline_graph.document.id,)]


def test_first_submission_recovery_reserves_projection_then_replaces_partial_result_once(
    sqlite_session: Session,
    pipeline_graph: _PipelineGraph,
) -> None:
    probes = _run_indexing(
        sqlite_session,
        pipeline_graph,
        original_document_id=pipeline_graph.document.id,
        enforce_projected_capacity=True,
        chunks={"general_chunks": [{"content": "new result"}]},
    )

    sqlite_session.expire_all()
    segments = sqlite_session.scalars(
        select(DocumentSegment).where(
            DocumentSegment.tenant_id == pipeline_graph.dataset.tenant_id,
            DocumentSegment.dataset_id == pipeline_graph.dataset.id,
            DocumentSegment.document_id == pipeline_graph.document.id,
        )
    ).all()
    assert [(segment.id, segment.content, segment.index_node_id) for segment in segments] == [
        (probes.index.new_segment_ids[0], "new result", "new-node")
    ]
    assert sqlite_session.get(DocumentSegmentSummary, pipeline_graph.partial_summary_id) is None
    assert sqlite_session.get(SegmentAttachmentBinding, pipeline_graph.attachment_binding_id) is None
    assert probes.index.indexed_document_ids == [pipeline_graph.document.id]
    assert probes.vector.deleted_node_ids == [("partial-node", "partial-summary-node")]
    assert probes.attachments.deleted_document_ids == [(pipeline_graph.document.id,)]
    watermark = json.loads(
        probes.redis.values[f"tenant:{pipeline_graph.dataset.tenant_id}:vector_space_estimate_watermark"]
    )
    assert watermark["document_ids"] == [pipeline_graph.document.id]

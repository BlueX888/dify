from dataclasses import replace
from threading import Event
from time import monotonic
from unittest.mock import MagicMock

import pytest

from core.rag.extractor.entity.extract_setting import CreatedUploadArtifact, ExtractionResult
from core.rag.models.document import Document
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.errors import (
    DocumentIndexingClaimLostError,
    DocumentIsDeletedPausedError,
    DocumentIsPausedError,
)
from services.knowledge.indexing.estimate import StoredSource
from services.knowledge.indexing.execution import (
    DocumentIndexingService,
    IndexingDocument,
    IndexingTransformResult,
    maintain_indexing_claim,
)

ExecutionFixture = tuple[DocumentIndexingService, MagicMock, IndexingDocument, list[Document]]


@pytest.fixture
def execution() -> ExecutionFixture:
    ref = DatasetRef("tenant", "dataset").document("document")
    document = IndexingDocument(
        StoredSource(ref, "upload_file", {}, "text_model"), {"mode": "automatic"}, "English", False
    )
    ports = MagicMock()
    ports.documents.get_indexing_document.return_value = document
    chunks = [Document(page_content="one"), Document(page_content="two")]
    ports.backend.extract.return_value = ExtractionResult(documents=chunks)
    ports.backend.transform.return_value = IndexingTransformResult(chunks=chunks)
    ports.backend.count_tokens.return_value = [7, 11]
    ports.backend.describe_error.side_effect = str
    ports.segments.resume_indexing.return_value = (chunks[1:], 18)
    service = DocumentIndexingService(
        documents=ports.documents,
        segments=ports.segments,
        backend=ports.backend,
        enforce_vector_space_admission=True,
        clock=iter([10, 13]).__next__,
    )
    return service, ports, document, chunks


@pytest.mark.parametrize("start", ["parsing", "splitting", "indexing"])
def test_completion_waits_for_persisted_segments_and_all_index_writes(execution: ExecutionFixture, start: str) -> None:
    service, ports, document, chunks = execution
    if start == "parsing":
        service.run([document.ref])
    elif start == "splitting":
        service.run_in_splitting_status(document.ref)
    else:
        service.run_in_indexing_status(document.ref)
    calls = [call[0] for call in ports.mock_calls]
    assert calls.index("backend.load") < calls.index("documents.complete_indexing")
    ports.documents.complete_indexing.assert_called_once_with(
        document.ref,
        claim_id=None,
        tokens=18,
        latency=3,
        dataset_generation_id=document.dataset_generation_id,
    )
    if start == "indexing":
        ports.backend.extract.assert_not_called()
        ports.segments.save_for_indexing.assert_not_called()
        ports.backend.load.assert_called_once_with(document, chunks[1:])
    else:
        assert calls.index("segments.save_for_indexing") < calls.index("backend.load")
        ports.segments.save_for_indexing.assert_called_once_with(document, chunks, [7, 11], created_upload_file_ids=())
        if start == "splitting":
            assert calls.index("documents.mark_cleaning") < calls.index("backend.clean_for_reindex")
            assert calls.index("backend.clean_for_reindex") < calls.index("backend.extract")
    assert ports.backend.ensure_admission.call_count == (start == "parsing")


def test_source_version_is_published_only_with_successful_completion(execution: ExecutionFixture) -> None:
    service, ports, document, chunks = execution
    ports.backend.extract.return_value = ExtractionResult(
        documents=chunks,
        source_version="2026-10-24T00:00:00Z",
    )

    service.run_claimed(document.ref, claim_id="claim-1")

    ports.documents.complete_indexing.assert_called_once_with(
        document.ref,
        claim_id="claim-1",
        tokens=18,
        latency=3,
        source_version="2026-10-24T00:00:00Z",
        dataset_generation_id=document.dataset_generation_id,
    )


def test_extractor_artifacts_are_fenced_in_segment_publication(execution: ExecutionFixture) -> None:
    service, ports, document, chunks = execution
    artifact = CreatedUploadArtifact("upload-1", "workspace-1/extracted.png")
    ports.backend.extract.return_value = ExtractionResult(
        documents=chunks,
        created_upload_artifacts=(artifact,),
    )
    ports.backend.transform.return_value = IndexingTransformResult(
        chunks=chunks,
        created_upload_file_ids=("transformed-upload",),
    )

    service.run([document.ref])

    ports.segments.save_for_indexing.assert_called_once_with(
        document,
        chunks,
        [7, 11],
        created_upload_file_ids=(artifact.id, "transformed-upload"),
    )


@pytest.mark.parametrize("stage", ["extract", "transform", "ensure_admission", "count_tokens", "load"])
def test_failed_phase_records_error_without_completing(execution: ExecutionFixture, stage: str) -> None:
    service, ports, document, _ = execution
    operation = {
        "extract": ports.backend.extract,
        "transform": ports.backend.transform,
        "ensure_admission": ports.backend.ensure_admission,
        "count_tokens": ports.backend.count_tokens,
        "load": ports.backend.load,
    }[stage]
    operation.side_effect = RuntimeError("index unavailable")
    service.run([document.ref])
    ports.documents.complete_indexing.assert_not_called()
    ports.documents.fail_indexing.assert_called_once_with(
        document.ref,
        "index unavailable",
        claim_id=None,
        dataset_generation_id=document.dataset_generation_id,
    )
    if stage != "load":
        ports.backend.load.assert_not_called()


@pytest.mark.parametrize("stage", ["transform", "load"])
def test_failed_indexing_cleans_files_created_by_extraction(execution: ExecutionFixture, stage: str) -> None:
    service, ports, document, chunks = execution
    ports.backend.extract.return_value = ExtractionResult(
        documents=chunks,
        created_upload_artifacts=(CreatedUploadArtifact("created-upload-1", "created-key"),),
    )
    operation = ports.backend.transform if stage == "transform" else ports.backend.load
    operation.side_effect = RuntimeError(f"{stage} failed")

    service.run([document.ref])

    ports.backend.cleanup_created_upload_files.assert_called_once_with(
        document,
        ("created-upload-1",),
        discard_persisted_attempt=False,
    )


def test_retry_keeps_only_the_successful_extractions_files(execution: ExecutionFixture) -> None:
    service, ports, document, chunks = execution
    ports.backend.extract.return_value = ExtractionResult(
        documents=chunks,
        created_upload_artifacts=(CreatedUploadArtifact("failed-upload", "failed-key"),),
    )
    ports.backend.transform.side_effect = RuntimeError("transform failed")
    service.run([document.ref])

    ports.backend.extract.return_value = ExtractionResult(
        documents=chunks,
        created_upload_artifacts=(CreatedUploadArtifact("successful-upload", "successful-key"),),
    )
    ports.backend.transform.side_effect = None
    ports.backend.transform.return_value = IndexingTransformResult(chunks=chunks)
    service.run([document.ref])

    ports.backend.cleanup_created_upload_files.assert_called_once_with(
        document,
        ("failed-upload",),
        discard_persisted_attempt=False,
    )
    ports.documents.complete_indexing.assert_called_once()


def test_failed_indexing_cleans_files_created_during_extraction_and_transform(
    execution: ExecutionFixture,
) -> None:
    service, ports, document, chunks = execution
    ports.backend.extract.return_value = ExtractionResult(
        documents=chunks,
        created_upload_artifacts=(CreatedUploadArtifact("extract-upload", "extract-key"),),
    )
    ports.backend.transform.return_value = IndexingTransformResult(
        chunks=chunks,
        created_upload_file_ids=("transform-upload",),
    )
    ports.backend.load.side_effect = DocumentIndexingClaimLostError()

    with pytest.raises(DocumentIndexingClaimLostError):
        service.run_claimed(document.ref, claim_id="claim-1")

    ports.backend.cleanup_created_upload_files.assert_called_once_with(
        document,
        ("extract-upload", "transform-upload"),
        discard_persisted_attempt=True,
    )


@pytest.mark.parametrize("stage", ["get_indexing_document", "mark_splitting", "complete_indexing"])
@pytest.mark.parametrize("error", [DocumentIsPausedError, DocumentIsDeletedPausedError])
def test_pause_and_deletion_do_not_become_indexing_errors(
    execution: ExecutionFixture, stage: str, error: type[Exception]
) -> None:
    service, ports, document, _ = execution
    operation = {
        "get_indexing_document": ports.documents.get_indexing_document,
        "mark_splitting": ports.documents.mark_splitting,
        "complete_indexing": ports.documents.complete_indexing,
    }[stage]
    operation.side_effect = error()
    if error is DocumentIsPausedError:
        with pytest.raises(DocumentIsPausedError):
            service.run([document.ref])
    else:
        service.run([document.ref])
    ports.documents.fail_indexing.assert_not_called()


def test_pause_after_worker_completion_prevents_document_completion(execution: ExecutionFixture) -> None:
    service, ports, document, _ = execution
    ports.backend.check_paused.side_effect = [None, None, DocumentIsPausedError()]
    with pytest.raises(DocumentIsPausedError):
        service.run([document.ref])
    ports.documents.complete_indexing.assert_not_called()
    ports.documents.fail_indexing.assert_not_called()


def test_missing_document_is_skipped(execution: ExecutionFixture) -> None:
    service, ports, document, _ = execution
    ports.documents.get_indexing_document.return_value = None
    service.run([document.ref])
    ports.backend.extract.assert_not_called()
    ports.documents.fail_indexing.assert_not_called()


def test_claimed_run_propagates_the_same_owner_to_every_persistence_phase(execution: ExecutionFixture) -> None:
    service, ports, document, chunks = execution
    claim_id = "11111111-1111-1111-1111-111111111111"
    claimed_document = replace(document, claim_id=claim_id)
    ports.documents.get_indexing_document.return_value = claimed_document

    service.run_claimed(document.ref, claim_id=claim_id)

    ports.documents.get_indexing_document.assert_called_once_with(document.ref, claim_id=claim_id)
    ports.documents.mark_splitting.assert_called_once_with(document.ref, claim_id=claim_id)
    ports.segments.save_for_indexing.assert_called_once_with(
        claimed_document, chunks, [7, 11], created_upload_file_ids=()
    )
    ports.documents.complete_indexing.assert_called_once_with(
        document.ref,
        claim_id=claim_id,
        tokens=18,
        latency=3,
        dataset_generation_id=claimed_document.dataset_generation_id,
    )


def test_claimed_restart_persists_cleaning_phase_with_exact_owner(execution: ExecutionFixture) -> None:
    service, ports, document, _chunks = execution
    claim_id = "11111111-1111-1111-1111-111111111111"
    claimed_document = replace(document, claim_id=claim_id)
    ports.documents.get_indexing_document.return_value = claimed_document

    service.restart_claimed(document.ref, claim_id=claim_id)

    ports.documents.mark_cleaning.assert_called_once_with(document.ref, claim_id=claim_id)
    calls = [call[0] for call in ports.mock_calls]
    assert calls.index("documents.mark_cleaning") < calls.index("backend.check_paused")
    assert calls.index("documents.mark_cleaning") < calls.index("backend.clean_for_reindex")


def test_claimed_run_renews_lease_during_long_execution(execution: ExecutionFixture) -> None:
    _service, ports, document, chunks = execution
    claim_id = "11111111-1111-1111-1111-111111111111"
    claimed_document = replace(document, claim_id=claim_id)
    renewed = Event()
    ports.documents.get_indexing_document.return_value = claimed_document

    def renew(ref: object, *, claim_id: str) -> None:
        assert ref == document.ref
        assert claim_id == claimed_document.claim_id
        renewed.set()

    ports.documents.renew_indexing_claim.side_effect = renew

    def wait_for_heartbeat(_document: IndexingDocument) -> ExtractionResult:
        assert renewed.wait(timeout=1)
        return ExtractionResult(documents=chunks)

    ports.backend.extract.side_effect = wait_for_heartbeat
    service = DocumentIndexingService(
        documents=ports.documents,
        segments=ports.segments,
        backend=ports.backend,
        claim_heartbeat_interval=0.001,
        clock=iter([10, 13]).__next__,
    )

    service.run_claimed(document.ref, claim_id=claim_id)

    ports.documents.renew_indexing_claim.assert_called_with(document.ref, claim_id=claim_id)
    ports.documents.complete_indexing.assert_called_once_with(
        document.ref,
        claim_id=claim_id,
        tokens=18,
        latency=3,
        dataset_generation_id=claimed_document.dataset_generation_id,
    )


@pytest.mark.parametrize("operation_fails", [False, True])
def test_indexing_heartbeat_shutdown_does_not_wait_for_blocked_renewal(
    execution: ExecutionFixture,
    operation_fails: bool,
) -> None:
    _service, ports, document, _chunks = execution
    renewal_started = Event()
    release_renewal = Event()
    renewal_finished = Event()
    exit_started_at: list[float] = []

    def blocked_renewal(_ref: object, *, claim_id: str) -> None:
        assert claim_id == "claim-1"
        renewal_started.set()
        release_renewal.wait(timeout=2)
        renewal_finished.set()

    ports.documents.renew_indexing_claim.side_effect = blocked_renewal

    def run_context() -> None:
        with maintain_indexing_claim(
            ports.documents,
            document.ref,
            claim_id="claim-1",
            interval_seconds=0.001,
        ):
            assert renewal_started.wait(timeout=1)
            exit_started_at.append(monotonic())
            if operation_fails:
                raise RuntimeError("indexing failed")

    try:
        if operation_fails:
            with pytest.raises(RuntimeError, match="indexing failed"):
                run_context()
        else:
            run_context()
        assert monotonic() - exit_started_at[0] < 0.5
    finally:
        release_renewal.set()
    assert renewal_finished.wait(timeout=1)


def test_indexing_persists_the_backend_error_description(execution: ExecutionFixture) -> None:
    service, ports, document, _ = execution
    error = RuntimeError("backend details")
    ports.backend.extract.side_effect = error
    ports.backend.describe_error.side_effect = None
    ports.backend.describe_error.return_value = "source file missing"

    service.run([document.ref])

    ports.backend.describe_error.assert_called_once_with(error)
    ports.documents.fail_indexing.assert_called_once_with(
        document.ref,
        "source file missing",
        claim_id=None,
        dataset_generation_id=document.dataset_generation_id,
    )
    ports.documents.complete_indexing.assert_not_called()

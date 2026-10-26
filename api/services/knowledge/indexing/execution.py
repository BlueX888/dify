"""Document indexing phases and their application-owned persistence and execution ports."""

import logging
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from dataclasses import dataclass
from threading import Event, Thread
from typing import Literal, Protocol

from core.credit_usage import CreditUsageCreatedBy
from core.model_context import with_credit_usage_created_by
from core.rag.extractor.entity.extract_setting import ExtractionResult
from core.rag.models.document import Document
from services.knowledge.entities.resource_scope import DocumentRef
from services.knowledge.indexing.errors import (
    DatasetIndexingGenerationLostError,
    DocumentIndexingClaimLostError,
    DocumentIsDeletedPausedError,
    DocumentIsPausedError,
)
from services.knowledge.indexing.estimate import StoredSource

logger = logging.getLogger(__name__)

INDEXING_CLAIM_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class IndexingDocument:
    source: StoredSource
    processing_rule: Mapping[str, object]
    doc_language: str
    need_summary: bool
    claim_id: str | None = None
    indexing_request_id: str | None = None
    dataset_generation_id: str | None = None

    @property
    def ref(self) -> DocumentRef:
        return self.source.document_ref

    @property
    def doc_form(self) -> str:
        return self.source.document_model


@dataclass(frozen=True)
class IndexingTransformResult:
    chunks: list[Document]
    created_upload_file_ids: tuple[str, ...] = ()


class IndexingDocumentStore(Protocol):
    def get_indexing_document(self, ref: DocumentRef, *, claim_id: str | None) -> IndexingDocument | None: ...

    def renew_indexing_claim(self, ref: DocumentRef, *, claim_id: str) -> None: ...

    def release_dataset_generation_owner(
        self,
        ref: DocumentRef,
        *,
        indexing_request_id: str | None,
        claim_id: str,
        dataset_generation_id: str | None,
    ) -> None: ...

    def mark_cleaning(self, ref: DocumentRef, *, claim_id: str | None) -> None: ...

    def mark_splitting(self, ref: DocumentRef, *, claim_id: str | None) -> None: ...

    def complete_indexing(
        self,
        ref: DocumentRef,
        *,
        claim_id: str | None,
        tokens: int,
        latency: float,
        source_version: str | None = None,
        dataset_generation_id: str | None = None,
    ) -> None: ...

    def fail_indexing(
        self,
        ref: DocumentRef,
        error: str,
        *,
        claim_id: str | None,
        dataset_generation_id: str | None = None,
    ) -> None: ...


@contextmanager
def maintain_indexing_claim(
    documents: IndexingDocumentStore,
    ref: DocumentRef,
    *,
    claim_id: str,
    interval_seconds: float = INDEXING_CLAIM_HEARTBEAT_SECONDS,
) -> Generator[None]:
    """Renew one exact claim while its blocking indexing work is running."""
    if interval_seconds <= 0:
        raise ValueError("claim heartbeat interval must be positive")
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                documents.renew_indexing_claim(ref, claim_id=claim_id)
            except DocumentIndexingClaimLostError:
                logger.info("Document indexing heartbeat lost ownership: %s", ref)
                return
            except Exception:
                # A transient renewal failure still leaves several heartbeat
                # periods before expiry. Persistence fences reject the worker
                # if subsequent renewals cannot restore the lease.
                logger.exception("Document indexing heartbeat failed: %s", ref)

    thread = Thread(target=heartbeat, name=f"indexing-heartbeat-{ref.document_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, _HEARTBEAT_SHUTDOWN_TIMEOUT_SECONDS))
        if thread.is_alive():
            logger.warning("Document indexing heartbeat did not stop promptly: %s", ref)


class IndexingSegmentStore(Protocol):
    def save_for_indexing(
        self,
        document: IndexingDocument,
        chunks: list[Document],
        token_counts: list[int],
        *,
        created_upload_file_ids: Sequence[str] = (),
    ) -> None: ...

    def resume_indexing(self, document: IndexingDocument) -> tuple[list[Document], int]: ...


class IndexingBackend(Protocol):
    def describe_error(self, error: Exception) -> str:
        """Translate backend-specific exceptions into the persisted error message."""
        ...

    def check_paused(self, ref: DocumentRef) -> None: ...

    def clean_for_reindex(self, document: IndexingDocument) -> None: ...

    def extract(self, document: IndexingDocument) -> ExtractionResult: ...

    def maintain_created_upload_file_writers(
        self,
        document: IndexingDocument,
        file_ids: Sequence[str],
    ) -> AbstractContextManager[None]: ...

    def cleanup_created_upload_files(
        self,
        document: IndexingDocument,
        file_ids: Sequence[str],
        *,
        discard_persisted_attempt: bool = False,
    ) -> None: ...

    def transform(self, document: IndexingDocument, texts: list[Document]) -> IndexingTransformResult: ...

    def ensure_admission(self, document: IndexingDocument, chunks: list[Document]) -> None: ...

    def count_tokens(self, document: IndexingDocument, chunks: list[Document]) -> list[int]: ...

    def load(self, document: IndexingDocument, chunks: list[Document]) -> None: ...


class DocumentIndexingService:
    """Coordinate committed phases; no caller session is held across execution."""

    def __init__(
        self,
        *,
        documents: IndexingDocumentStore,
        segments: IndexingSegmentStore,
        backend: IndexingBackend,
        enforce_vector_space_admission: bool = False,
        clock: Callable[[], float] = time.perf_counter,
        claim_heartbeat_interval: float = INDEXING_CLAIM_HEARTBEAT_SECONDS,
    ) -> None:
        self._documents = documents
        self._segments = segments
        self._backend = backend
        self._enforce_vector_space_admission = enforce_vector_space_admission
        self._clock = clock
        if claim_heartbeat_interval <= 0:
            raise ValueError("claim heartbeat interval must be positive")
        self._claim_heartbeat_interval = claim_heartbeat_interval

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def run(self, document_refs: Sequence[DocumentRef]) -> None:
        for ref in document_refs:
            self._run(ref, claim_id=None, start="parsing")

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def run_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
        self._run(ref, claim_id=claim_id, start="parsing")

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def restart_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
        """Clean an interrupted or replacement generation before indexing it."""
        self._run(ref, claim_id=claim_id, start="splitting")

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def run_in_splitting_status(self, ref: DocumentRef, *, claim_id: str | None = None) -> None:
        self._run(ref, claim_id=claim_id, start="splitting")

    @with_credit_usage_created_by(CreditUsageCreatedBy.KNOWLEDGE_INDEXING)
    def run_in_indexing_status(self, ref: DocumentRef, *, claim_id: str | None = None) -> None:
        self._run(ref, claim_id=claim_id, start="indexing")

    def _run(
        self,
        ref: DocumentRef,
        *,
        claim_id: str | None,
        start: Literal["parsing", "splitting", "indexing"],
    ) -> None:
        document: IndexingDocument | None = None
        created_upload_file_ids: tuple[str, ...] = ()
        created_extraction_upload_file_ids: tuple[str, ...] = ()
        source_version: str | None = None
        try:
            if start == "splitting":
                # Persist restart intent before loading the paused-guarded
                # snapshot, so an already-paused worker remains recoverable as
                # cleanup rather than a fresh parse.
                self._documents.mark_cleaning(ref, claim_id=claim_id)
            document = self._documents.get_indexing_document(ref, claim_id=claim_id)
            if document is None:
                logger.info("Document not found, skipping indexing: %s", ref)
                return
            claim_context = (
                maintain_indexing_claim(
                    self._documents,
                    ref,
                    claim_id=claim_id,
                    interval_seconds=self._claim_heartbeat_interval,
                )
                if claim_id is not None
                else nullcontext()
            )
            with claim_context:
                self._backend.check_paused(ref)
                if start == "indexing":
                    chunks, total_tokens = self._segments.resume_indexing(document)
                else:
                    if start == "splitting":
                        self._backend.clean_for_reindex(document)
                    extraction = self._backend.extract(document)
                    source_version = extraction.source_version
                    created_extraction_upload_file_ids = extraction.created_upload_file_ids
                    created_upload_file_ids = created_extraction_upload_file_ids
                    artifact_writer_context = (
                        self._backend.maintain_created_upload_file_writers(
                            document,
                            created_extraction_upload_file_ids,
                        )
                        if created_extraction_upload_file_ids
                        else nullcontext()
                    )
                    with artifact_writer_context:
                        self._documents.mark_splitting(ref, claim_id=claim_id)
                        transformation = self._backend.transform(document, extraction.documents)
                        created_upload_file_ids += transformation.created_upload_file_ids
                        chunks = transformation.chunks
                        transform_writer_context = (
                            self._backend.maintain_created_upload_file_writers(
                                document,
                                transformation.created_upload_file_ids,
                            )
                            if transformation.created_upload_file_ids
                            else nullcontext()
                        )
                        with transform_writer_context:
                            # Recovery retains its existing admission policy; only normal
                            # indexing callers that explicitly opt in reserve vector space.
                            if start == "parsing" and self._enforce_vector_space_admission:
                                self._backend.ensure_admission(document, chunks)
                            token_counts = self._backend.count_tokens(document, chunks)
                            total_tokens = sum(token_counts)
                            self._segments.save_for_indexing(
                                document,
                                chunks,
                                token_counts,
                                created_upload_file_ids=created_upload_file_ids,
                            )

                started = self._clock()
                self._backend.check_paused(ref)
                self._backend.load(document, chunks)
                self._backend.check_paused(ref)
            latency = self._clock() - started
            if source_version is None:
                self._documents.complete_indexing(
                    ref,
                    claim_id=claim_id,
                    tokens=total_tokens,
                    latency=latency,
                    dataset_generation_id=document.dataset_generation_id,
                )
            else:
                self._documents.complete_indexing(
                    ref,
                    claim_id=claim_id,
                    tokens=total_tokens,
                    latency=latency,
                    source_version=source_version,
                    dataset_generation_id=document.dataset_generation_id,
                )
        except DocumentIsPausedError:
            self._cleanup_created_upload_files(document, created_upload_file_ids)
            raise
        except DatasetIndexingGenerationLostError as error:
            self._cleanup_created_upload_files(
                document,
                created_upload_file_ids,
                discard_persisted_attempt=True,
            )
            if claim_id is not None:
                with suppress(DocumentIndexingClaimLostError):
                    self._documents.release_dataset_generation_owner(
                        ref,
                        indexing_request_id=document.indexing_request_id if document is not None else None,
                        claim_id=claim_id,
                        dataset_generation_id=document.dataset_generation_id if document is not None else None,
                    )
            raise DocumentIndexingClaimLostError() from error
        except DocumentIndexingClaimLostError:
            self._cleanup_created_upload_files(
                document,
                created_upload_file_ids,
                discard_persisted_attempt=True,
            )
            raise
        except DocumentIsDeletedPausedError:
            self._cleanup_created_upload_files(document, created_upload_file_ids)
            logger.info("Document deleted during indexing: %s", ref)
        except Exception as error:
            self._cleanup_created_upload_files(document, created_upload_file_ids)
            logger.exception("Document indexing failed: %s", ref)
            try:
                self._documents.fail_indexing(
                    ref,
                    self._backend.describe_error(error),
                    claim_id=claim_id,
                    dataset_generation_id=document.dataset_generation_id if document is not None else None,
                )
            except DatasetIndexingGenerationLostError as generation_error:
                if claim_id is not None:
                    with suppress(DocumentIndexingClaimLostError):
                        self._documents.release_dataset_generation_owner(
                            ref,
                            indexing_request_id=document.indexing_request_id if document is not None else None,
                            claim_id=claim_id,
                            dataset_generation_id=document.dataset_generation_id if document is not None else None,
                        )
                raise DocumentIndexingClaimLostError() from generation_error

    def _cleanup_created_upload_files(
        self,
        document: IndexingDocument | None,
        file_ids: Sequence[str],
        *,
        discard_persisted_attempt: bool = False,
    ) -> None:
        if document is None or not file_ids:
            return
        try:
            self._backend.cleanup_created_upload_files(
                document,
                file_ids,
                discard_persisted_attempt=discard_persisted_attempt,
            )
        except Exception:
            logger.exception("Failed to clean extraction files after document indexing stopped: %s", document.ref)

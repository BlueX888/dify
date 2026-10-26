import json
from datetime import datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from models.dataset import Dataset, Document
from models.enums import IndexingStatus
from models.model import DocumentIndexingOutbox
from repositories.knowledge.document_indexing_outbox_repository import (
    SQLAlchemyDocumentIndexingOutboxRepository,
    create_document_indexing_intents,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository, refresh_locked_indexing_claim
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError
from tests.unit_tests.model_factories import make_dataset, make_document, make_tenant


def _create_request(session_factory: sessionmaker[Session]) -> tuple[str, DatasetRef, str]:
    ref = DatasetRef("tenant-1", "dataset-1")
    document_id = "document-1"
    with session_factory.begin() as session:
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                make_document(document_id=document_id, dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
            ]
        )
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, created=(document_id,)),
        )
    return request_id, ref, document_id


def test_claim_uses_a_worker_lease_and_recovers_only_after_expiry(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    request_id, ref, document_id = _create_request(sqlite_session_factory)
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)

    (first_claim,) = repository.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )
    assert first_claim.ref == ref.document(document_id)
    assert not first_claim.recovery_required
    assert (
        repository.claim_indexing_request(
            request_id,
            expected_action="create",
            processing_started_at=datetime(2026, 1, 2),
            admission_error=lambda _count: None,
        )
        == ()
    )

    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, document_id)
        document.indexing_claim_expires_at = datetime(2000, 1, 1)

    (recovery_claim,) = repository.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 2),
        admission_error=lambda _count: None,
    )
    assert recovery_claim.recovery_required
    assert recovery_claim.claim_id != first_claim.claim_id

    with sqlite_session_factory() as session:
        takeover_expiry = session.get_one(Document, document_id).indexing_claim_expires_at
    with pytest.raises(DocumentIndexingClaimLostError):
        repository.renew_indexing_claim(first_claim.ref, claim_id=first_claim.claim_id)
    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document_id)
        assert persisted.indexing_claim_id == recovery_claim.claim_id
        assert persisted.indexing_claim_expires_at == takeover_expiry


def test_pipeline_retry_reclaims_an_inner_completed_phase_until_outer_success_acknowledges(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document_id = "document-1"
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                make_document(document_id=document_id, dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
            ]
        )
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, retried=(document_id,), actor_id="actor-1"),
        )
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (first_claim,) = repository.claim_indexing_request(
        request_id,
        expected_action="retry",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )
    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, document_id)
        document.indexing_status = IndexingStatus.COMPLETED
        document.indexing_claim_expires_at = datetime(2000, 1, 1)

    (recovery_claim,) = repository.claim_indexing_request(
        request_id,
        expected_action="retry",
        processing_started_at=datetime(2026, 1, 2),
        admission_error=lambda _count: None,
    )

    assert recovery_claim.recovery_required
    assert recovery_claim.claim_id != first_claim.claim_id
    with sqlite_session_factory() as session:
        assert session.get(DocumentIndexingOutbox, request_id) is not None
        persisted = session.get_one(Document, document_id)
        assert persisted.indexing_status == IndexingStatus.PARSING
        assert persisted.indexing_request_id == request_id
    repository.complete_indexing(
        recovery_claim.ref,
        claim_id=recovery_claim.claim_id,
        tokens=7,
        latency=1.5,
        dataset_generation_id=recovery_claim.dataset_generation_id,
    )
    with sqlite_session_factory() as session:
        assert session.get(DocumentIndexingOutbox, request_id) is None


def test_parent_generation_loss_hands_the_same_request_to_a_full_restart(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    request_id, ref, document_id = _create_request(sqlite_session_factory)
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    started_at = datetime(2026, 1, 1)
    (old_claim,) = repository.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=started_at,
        admission_error=lambda _count: None,
    )
    with sqlite_session_factory.begin() as session:
        dataset = session.get_one(Dataset, ref.dataset_id)
        dataset.indexing_request_id = "dataset-generation-1"
        document = session.get_one(Document, document_id)
        document.indexing_status = IndexingStatus.SPLITTING
        document.parsing_completed_at = datetime(2026, 1, 1, 0, 1)
        document.cleaning_completed_at = datetime(2026, 1, 1, 0, 2)
        document.splitting_completed_at = datetime(2026, 1, 1, 0, 3)
        document.completed_at = datetime(2026, 1, 1, 0, 4)
        document.stopped_at = datetime(2026, 1, 1, 0, 5)
        document.error = "stale"

    repository.release_dataset_generation_owner(
        old_claim.ref,
        indexing_request_id=request_id,
        claim_id=old_claim.claim_id,
        dataset_generation_id=old_claim.dataset_generation_id,
    )

    with sqlite_session_factory() as session:
        released = session.get_one(Document, document_id)
        assert released.indexing_status == IndexingStatus.WAITING
        assert released.indexing_request_id == request_id
        assert released.indexing_claim_id is None
        assert released.indexing_claim_expires_at is None
        assert released.indexing_dataset_generation_id is None
        assert released.processing_started_at == started_at
        assert released.parsing_completed_at is None
        assert released.cleaning_completed_at is None
        assert released.splitting_completed_at is None
        assert released.completed_at is None
        assert released.stopped_at is None
        assert released.error is None

    (successor,) = repository.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 2),
        admission_error=lambda _count: None,
    )
    assert successor.recovery_required
    assert successor.dataset_generation_id == "dataset-generation-1"
    assert successor.claim_id != old_claim.claim_id

    with pytest.raises(DocumentIndexingClaimLostError):
        repository.release_dataset_generation_owner(
            old_claim.ref,
            indexing_request_id=request_id,
            claim_id=old_claim.claim_id,
            dataset_generation_id=old_claim.dataset_generation_id,
        )
    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document_id)
        assert persisted.indexing_claim_id == successor.claim_id
        assert persisted.indexing_dataset_generation_id == "dataset-generation-1"
        assert persisted.indexing_status == IndexingStatus.PARSING


@pytest.mark.parametrize("action", ["update", "retry"])
def test_existing_waiting_restart_marker_is_conservatively_restarted(
    sqlite_session_factory: sessionmaker[Session],
    action: str,
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document_id = "document-1"
    prior_started_at = datetime(2025, 12, 31)
    with sqlite_session_factory.begin() as session:
        document = make_document(document_id=document_id, dataset_id=ref.dataset_id, tenant_id=ref.tenant_id)
        # Configuration/service preparation has moved a previously completed
        # document back to WAITING. Retain the old attempt marker so claiming
        # cannot mistake its segments for a first-time create.
        document.indexing_status = IndexingStatus.WAITING
        document.processing_started_at = prior_started_at
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                document,
            ]
        )
        jobs = (
            DocumentIndexingJobs(ref, updated=document_id)
            if action == "update"
            else DocumentIndexingJobs(ref, retried=(document_id,), actor_id="actor-1")
        )
        (request_id,) = create_document_indexing_intents(session, jobs)

    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (claim,) = repository.claim_indexing_request(
        request_id,
        expected_action=action,
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )

    assert claim.recovery_required
    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document_id)
        assert persisted.indexing_status == IndexingStatus.PARSING
        assert persisted.indexing_claim_id == claim.claim_id


def test_pause_release_preserves_request_and_resumable_phase(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    request_id, ref, document_id = _create_request(sqlite_session_factory)
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (claim,) = repository.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )
    with sqlite_session_factory.begin() as session:
        session.get_one(Document, document_id).is_paused = True

    assert repository.release_indexing_claim_for_pause(claim.ref, claim_id=claim.claim_id)

    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document_id)
        assert persisted.indexing_status == IndexingStatus.PARSING
        assert persisted.indexing_request_id == request_id
        assert persisted.indexing_claim_id is None
        assert persisted.indexing_claim_expires_at is None
        assert persisted.error is None
        assert session.get(DocumentIndexingOutbox, request_id) is not None


def test_stale_pause_release_cannot_clear_successor_claim(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    request_id, ref, document_id = _create_request(sqlite_session_factory)
    repository = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (old_claim,) = repository.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )
    successor_expiry = datetime(2099, 1, 1)
    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, document_id)
        document.is_paused = True
        document.indexing_request_id = "successor-request"
        document.indexing_claim_id = "successor-claim"
        document.indexing_claim_expires_at = successor_expiry
        document.indexing_status = IndexingStatus.INDEXING

    with pytest.raises(DocumentIndexingClaimLostError):
        repository.release_indexing_claim_for_pause(old_claim.ref, claim_id=old_claim.claim_id)

    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document_id)
        assert persisted.indexing_request_id == "successor-request"
        assert persisted.indexing_claim_id == "successor-claim"
        assert persisted.indexing_claim_expires_at == successor_expiry
        assert persisted.indexing_status == IndexingStatus.INDEXING


@pytest.mark.parametrize(
    ("stale_request_id", "stale_claim_id", "successor_status"),
    [
        ("request-old", "claim-new", IndexingStatus.INDEXING),
        ("request-new", "claim-old", IndexingStatus.INDEXING),
        ("request-new", "claim-new", IndexingStatus.SPLITTING),
    ],
)
def test_locked_refresh_does_not_extend_a_replaced_generation_or_phase(
    sqlite_session_factory: sessionmaker[Session],
    stale_request_id: str,
    stale_claim_id: str,
    successor_status: IndexingStatus,
) -> None:
    _request_id, ref, document_id = _create_request(sqlite_session_factory)
    successor_expiry = datetime(2099, 1, 1)
    with sqlite_session_factory.begin() as session:
        document = session.get_one(Document, document_id)
        document.indexing_status = successor_status
        document.indexing_request_id = "request-new"
        document.indexing_claim_id = "claim-new"
        document.indexing_claim_expires_at = successor_expiry

    with pytest.raises(DocumentIndexingClaimLostError), sqlite_session_factory.begin() as session:
        refresh_locked_indexing_claim(
            session,
            ref.document(document_id),
            indexing_request_id=stale_request_id,
            claim_id=stale_claim_id,
            allowed_statuses=(IndexingStatus.INDEXING,),
        )

    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document_id)
        assert persisted.indexing_request_id == "request-new"
        assert persisted.indexing_claim_id == "claim-new"
        assert persisted.indexing_claim_expires_at == successor_expiry
        assert persisted.indexing_status == successor_status


def test_old_request_cannot_claim_a_new_waiting_generation(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    old_request_id, ref, document_id = _create_request(sqlite_session_factory)
    with sqlite_session_factory.begin() as session:
        (new_request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, updated=document_id),
        )
    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)

    assert (
        documents.claim_indexing_request(
            old_request_id,
            expected_action="create",
            processing_started_at=datetime(2026, 1, 1),
            admission_error=lambda _count: None,
        )
        == ()
    )

    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert outbox.get_intent(old_request_id) is None
    assert outbox.get_intent(new_request_id) is not None
    with sqlite_session_factory() as session:
        document = session.get_one(Document, document_id)
        assert document.indexing_status == IndexingStatus.WAITING
        assert document.indexing_request_id == new_request_id


def test_completion_acknowledges_the_request_only_after_the_claimed_document_finishes(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    request_id, ref, document_id = _create_request(sqlite_session_factory)
    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (claim,) = documents.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )

    documents.complete_indexing(claim.ref, claim_id=claim.claim_id, tokens=3, latency=0.5)

    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert outbox.get_intent(request_id) is None
    with sqlite_session_factory() as session:
        document = session.get_one(Document, document_id)
        assert document.indexing_status == IndexingStatus.COMPLETED
        assert document.indexing_request_id is None
        assert document.indexing_claim_id is None


def test_batch_request_leases_only_one_document_at_a_time(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document_ids = ("document-1", "document-2")
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                *(
                    make_document(document_id=document_id, dataset_id=ref.dataset_id, tenant_id=ref.tenant_id)
                    for document_id in document_ids
                ),
            ]
        )
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, created=document_ids),
        )
    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)

    (first_claim,) = documents.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )
    assert (
        documents.claim_indexing_request(
            request_id,
            expected_action="create",
            processing_started_at=datetime(2026, 1, 1),
            admission_error=lambda _count: None,
        )
        == ()
    )
    with sqlite_session_factory() as session:
        unclaimed = session.get_one(Document, "document-2")
        assert unclaimed.indexing_status == IndexingStatus.WAITING
        assert unclaimed.indexing_claim_id is None

    documents.complete_indexing(first_claim.ref, claim_id=first_claim.claim_id, tokens=1, latency=0.1)
    (second_claim,) = documents.claim_indexing_request(
        request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )
    assert second_claim.ref.document_id == "document-2"


def test_recover_request_preserves_the_phase_for_claim_aware_resume(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document_id = "document-1"
    with sqlite_session_factory.begin() as session:
        document = make_document(document_id=document_id, dataset_id=ref.dataset_id, tenant_id=ref.tenant_id)
        document.indexing_status = IndexingStatus.SPLITTING
        request = DocumentIndexingOutbox(
            tenant_id=ref.tenant_id,
            dataset_id=ref.dataset_id,
            action="recover",
            document_ids=json.dumps([document_id]),
        )
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                document,
                request,
            ]
        )
        session.flush()
        document.indexing_request_id = request.id
        request_id = request.id
    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)

    (claim,) = documents.claim_indexing_request(
        request_id,
        expected_action="recover",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )

    assert claim.ref == ref.document(document_id)
    assert claim.resume_status == IndexingStatus.SPLITTING
    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document_id)
        assert persisted.indexing_status == IndexingStatus.SPLITTING


def test_pipeline_inner_completed_document_does_not_settle_the_outer_request(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document_id = "document-1"
    with sqlite_session_factory.begin() as session:
        document = make_document(document_id=document_id, dataset_id=ref.dataset_id, tenant_id=ref.tenant_id)
        document.indexing_status = IndexingStatus.COMPLETED
        document.indexing_claim_id = "expired-worker"
        document.indexing_claim_expires_at = datetime(2000, 1, 1)
        request = DocumentIndexingOutbox(
            tenant_id=ref.tenant_id,
            dataset_id=ref.dataset_id,
            action="retry",
            document_ids=json.dumps([document_id]),
        )
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                document,
                request,
            ]
        )
        session.flush()
        document.indexing_request_id = request.id
        request_id = request.id
    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)

    def reject_recovery_admission(_count: int) -> str | None:
        raise AssertionError("recovery reran quota admission")

    (claim,) = documents.claim_indexing_request(
        request_id,
        expected_action="retry",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=reject_recovery_admission,
    )
    assert claim.recovery_required

    with sqlite_session_factory() as session:
        persisted = session.get_one(Document, document_id)
        assert persisted.indexing_status == IndexingStatus.PARSING
        assert persisted.indexing_request_id == request_id
        assert persisted.indexing_claim_id == claim.claim_id
        assert session.get(DocumentIndexingOutbox, request_id) is not None

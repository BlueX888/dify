from datetime import datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

import tasks.clean_notion_document_task as task_module
from models.dataset import DatasetMetadataBinding, Document
from repositories.knowledge.document_indexing_outbox_repository import (
    SQLAlchemyDocumentIndexingOutboxRepository,
    create_document_indexing_intents,
    invalidate_notion_removal_intents,
)
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.indexing.errors import DocumentIndexingClaimLostError
from tests.unit_tests.model_factories import make_dataset, make_document, make_tenant


def test_successful_notion_cleanup_acknowledges_its_durable_request(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document = make_document(document_id="document-1", tenant_id=ref.tenant_id, dataset_id=ref.dataset_id)
    binding = DatasetMetadataBinding(
        tenant_id=ref.tenant_id,
        dataset_id=ref.dataset_id,
        metadata_id="metadata-1",
        document_id=document.id,
        created_by="account-1",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                document,
                binding,
            ]
        )
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, removed_notion=(document.id,)),
        )

    with sqlite_session_factory() as session:
        stored = session.get(Document, document.id)
        assert stored is not None
        assert stored.indexing_request_id == request_id

    monkeypatch.setattr(task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(task_module.session_factory, "get_session_maker", lambda: sqlite_session_factory)
    monkeypatch.setattr(task_module, "clean_document_for_reindex", lambda **_kwargs: None)

    task_module.clean_notion_document_task.run(
        [document.id],
        ref.dataset_id,
        indexing_request_id=request_id,
    )

    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert outbox.get_intent(request_id) is None
    with sqlite_session_factory() as session:
        assert session.get(Document, document.id) is None
        assert session.get(DatasetMetadataBinding, binding.id) is None


def test_notion_removal_fences_old_index_worker_and_still_completes(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document = make_document(document_id="document-1", tenant_id=ref.tenant_id, dataset_id=ref.dataset_id)
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                document,
            ]
        )
        (index_request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, created=(document.id,)),
        )

    documents = SQLAlchemyDocumentRepository(session_factory=sqlite_session_factory)
    (old_claim,) = documents.claim_indexing_request(
        index_request_id,
        expected_action="create",
        processing_started_at=datetime(2026, 1, 1),
        admission_error=lambda _count: None,
    )
    with sqlite_session_factory.begin() as session:
        (remove_request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, removed_notion=(document.id,)),
        )
        current = session.get_one(Document, document.id)
        current.is_paused = True

    with pytest.raises(DocumentIndexingClaimLostError):
        documents.complete_indexing(old_claim.ref, claim_id=old_claim.claim_id, tokens=3, latency=0.1)
    with pytest.raises(DocumentIndexingClaimLostError):
        documents.release_indexing_claim_for_pause(old_claim.ref, claim_id=old_claim.claim_id)
    documents.fail_indexing(old_claim.ref, "late old-worker failure", claim_id=old_claim.claim_id)

    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert outbox.get_intent(remove_request_id) is not None
    with sqlite_session_factory() as session:
        current = session.get_one(Document, document.id)
        assert current.indexing_request_id == remove_request_id
        assert current.indexing_claim_id is None
        assert current.indexing_claim_expires_at is None
        assert current.error is None

    monkeypatch.setattr(task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(task_module.session_factory, "get_session_maker", lambda: sqlite_session_factory)
    monkeypatch.setattr(task_module, "clean_document_for_reindex", lambda **_kwargs: None)
    task_module.clean_notion_document_task.run(
        [document.id],
        ref.dataset_id,
        indexing_request_id=remove_request_id,
    )

    assert outbox.get_intent(remove_request_id) is None
    with sqlite_session_factory() as session:
        assert session.get(Document, document.id) is None


def test_reselected_notion_document_invalidates_stale_removal(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document = make_document(document_id="document-1", tenant_id=ref.tenant_id, dataset_id=ref.dataset_id)
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                document,
            ]
        )
        (request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, removed_notion=(document.id,)),
        )
        invalidate_notion_removal_intents(
            session,
            tenant_id=ref.tenant_id,
            dataset_id=ref.dataset_id,
            document_ids=(document.id,),
        )

    monkeypatch.setattr(task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(task_module.session_factory, "get_session_maker", lambda: sqlite_session_factory)
    cleanup_calls: list[object] = []
    monkeypatch.setattr(task_module, "clean_document_for_reindex", lambda **kwargs: cleanup_calls.append(kwargs))

    task_module.clean_notion_document_task.run(
        [document.id],
        ref.dataset_id,
        indexing_request_id=request_id,
    )

    assert cleanup_calls == []
    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert outbox.get_intent(request_id) is None
    with sqlite_session_factory() as session:
        stored = session.get(Document, document.id)
        assert stored is not None
        assert stored.indexing_request_id is None


def test_older_notion_removal_cannot_delete_document_owned_by_new_generation(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document = make_document(document_id="document-1", tenant_id=ref.tenant_id, dataset_id=ref.dataset_id)
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                document,
            ]
        )
        (old_request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, removed_notion=(document.id,)),
        )
        (new_request_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, removed_notion=(document.id,)),
        )

    monkeypatch.setattr(task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(task_module.session_factory, "get_session_maker", lambda: sqlite_session_factory)
    cleanup_calls: list[object] = []
    monkeypatch.setattr(task_module, "clean_document_for_reindex", lambda **kwargs: cleanup_calls.append(kwargs))

    task_module.clean_notion_document_task.run(
        [document.id],
        ref.dataset_id,
        indexing_request_id=old_request_id,
    )

    assert cleanup_calls == []
    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert outbox.get_intent(old_request_id) is None
    assert outbox.get_intent(new_request_id) is not None
    with sqlite_session_factory() as session:
        stored = session.get(Document, document.id)
        assert stored is not None
        assert stored.indexing_request_id == new_request_id


def test_reselection_during_cleanup_preserves_document_for_replacement_indexing(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = DatasetRef("tenant-1", "dataset-1")
    document = make_document(document_id="document-1", tenant_id=ref.tenant_id, dataset_id=ref.dataset_id)
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                make_tenant(tenant_id=ref.tenant_id),
                make_dataset(dataset_id=ref.dataset_id, tenant_id=ref.tenant_id),
                document,
            ]
        )
        (removal_id,) = create_document_indexing_intents(
            session,
            DocumentIndexingJobs(ref, removed_notion=(document.id,)),
        )

    replacement_ids: list[str] = []

    def supersede_cleanup(**kwargs) -> None:
        with sqlite_session_factory.begin() as session:
            invalidated = invalidate_notion_removal_intents(
                session,
                tenant_id=ref.tenant_id,
                dataset_id=ref.dataset_id,
                document_ids=(document.id,),
            )
            assert invalidated == (document.id,)
            (replacement_id,) = create_document_indexing_intents(
                session,
                DocumentIndexingJobs(ref, duplicated=invalidated),
            )
            replacement_ids.append(replacement_id)
        with sqlite_session_factory.begin() as session:
            kwargs["mutation_guard"](session)

    monkeypatch.setattr(task_module.session_factory, "create_session", sqlite_session_factory)
    monkeypatch.setattr(task_module.session_factory, "get_session_maker", lambda: sqlite_session_factory)
    monkeypatch.setattr(task_module, "clean_document_for_reindex", supersede_cleanup)

    task_module.clean_notion_document_task.run(
        [document.id],
        ref.dataset_id,
        indexing_request_id=removal_id,
    )

    assert len(replacement_ids) == 1
    outbox = SQLAlchemyDocumentIndexingOutboxRepository(session_factory=sqlite_session_factory)
    assert outbox.get_intent(removal_id) is None
    replacement = outbox.get_intent(replacement_ids[0])
    assert replacement is not None
    with sqlite_session_factory() as session:
        stored = session.get(Document, document.id)
        assert stored is not None
        assert stored.indexing_request_id == replacement.id

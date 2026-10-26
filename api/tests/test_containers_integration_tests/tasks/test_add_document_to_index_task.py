from collections.abc import Sequence

import pytest
from faker import Faker
from sqlalchemy.orm import Session

from extensions.ext_redis import redis_client
from models import Account, AccountStatus, Tenant, TenantAccountJoin, TenantAccountRole, TenantStatus
from models.dataset import Dataset, Document, DocumentSegment
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus, SegmentStatus
from models.model import SegmentStateOutbox
from tasks.add_document_to_index_task import add_document_to_index_task


def _owner_graph(session: Session) -> tuple[Dataset, Document]:
    fake = Faker()
    account = Account(
        email=fake.email(),
        name=fake.name(),
        interface_language="en-US",
        status=AccountStatus.ACTIVE,
    )
    tenant = Tenant(name=fake.company(), status=TenantStatus.NORMAL)
    session.add_all([account, tenant])
    session.flush()
    session.add(
        TenantAccountJoin(
            tenant_id=tenant.id,
            account_id=account.id,
            role=TenantAccountRole.OWNER,
            current=True,
        )
    )
    dataset = Dataset(
        id=fake.uuid4(),
        tenant_id=tenant.id,
        name=fake.company(),
        description="",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique="economy",
        created_by=account.id,
    )
    document = Document(
        id=fake.uuid4(),
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="document-enable",
        name=fake.file_name(),
        created_from=DocumentCreatedFrom.WEB,
        created_by=account.id,
        indexing_status=IndexingStatus.COMPLETED,
        enabled=True,
        archived=False,
        doc_form="text_model",
    )
    session.add_all([dataset, document])
    session.commit()
    return dataset, document


def _enable_intents(
    session: Session,
    dataset: Dataset,
    document: Document,
    *,
    count: int = 2,
) -> tuple[SegmentStateOutbox, ...]:
    intents = []
    for position in range(count):
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            position=position + 1,
            content=f"content-{position}",
            word_count=1,
            tokens=1,
            index_node_id=f"node-{position}",
            index_node_hash=f"hash-{position}",
            enabled=True,
            status=SegmentStatus.COMPLETED,
            created_by=document.created_by,
        )
        intent = SegmentStateOutbox(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            segment_id=segment.id,
            action="enable",
            actor_id=document.created_by,
            summary_id=None,
            summary_version=0,
        )
        segment.indexing_request_id = intent.id
        session.add_all([segment, intent])
        intents.append(intent)
    session.commit()
    return tuple(intents)


def test_add_document_forwards_only_committed_current_enable_intents(
    db_session_with_containers: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document = _owner_graph(db_session_with_containers)
    intents = _enable_intents(db_session_with_containers, dataset, document)
    published: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "tasks.add_document_to_index_task.publish_segment_state_intents",
        lambda values: published.append(tuple(values)),
    )
    cache_key = f"document_{document.id}_indexing"
    redis_client.set(cache_key, "processing", ex=300)

    assert add_document_to_index_task(document.id) is False

    assert published == [tuple(sorted(intent.id for intent in intents))]
    assert redis_client.exists(cache_key) == 0
    for intent in intents:
        assert db_session_with_containers.get(SegmentStateOutbox, intent.id) is not None


def test_add_document_keeps_intents_when_broker_publish_fails(
    db_session_with_containers: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, document = _owner_graph(db_session_with_containers)
    (intent,) = _enable_intents(db_session_with_containers, dataset, document, count=1)

    def reject_publish(_intent_ids: Sequence[str]) -> None:
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr("tasks.add_document_to_index_task.publish_segment_state_intents", reject_publish)

    assert add_document_to_index_task(document.id) is False
    assert db_session_with_containers.get(SegmentStateOutbox, intent.id) is not None


@pytest.mark.parametrize("stale_kind", ["document-disabled", "opposite-intent", "unreferenced-intent"])
def test_add_document_does_not_reauthorize_stale_enable_work(
    db_session_with_containers: Session,
    monkeypatch: pytest.MonkeyPatch,
    stale_kind: str,
) -> None:
    dataset, document = _owner_graph(db_session_with_containers)
    (intent,) = _enable_intents(db_session_with_containers, dataset, document, count=1)
    segment = db_session_with_containers.get_one(DocumentSegment, intent.segment_id)
    if stale_kind == "document-disabled":
        document.enabled = False
    elif stale_kind == "opposite-intent":
        intent.action = "disable"
        segment.enabled = False
    else:
        segment.indexing_request_id = None
    db_session_with_containers.commit()
    published: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "tasks.add_document_to_index_task.publish_segment_state_intents",
        lambda values: published.append(tuple(values)),
    )

    assert add_document_to_index_task(document.id) is False
    assert published == []


def test_add_document_missing_owner_is_a_noop_and_clears_legacy_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = Faker().uuid4()
    cache_key = f"document_{document_id}_indexing"
    redis_client.set(cache_key, "processing", ex=300)
    published: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "tasks.add_document_to_index_task.publish_segment_state_intents",
        lambda values: published.append(tuple(values)),
    )

    assert add_document_to_index_task(document_id) is False
    assert published == []
    assert redis_client.exists(cache_key) == 0

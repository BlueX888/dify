import types

import pytest
from redis.exceptions import LockNotOwnedError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from models.account import Account, Tenant
from models.dataset import Dataset, DatasetProcessRule, Document
from models.enums import ProcessRuleMode
from services.knowledge.dataset_service import DocumentService
from services.knowledge.entities.document_creation import DocumentIndexingJobs
from services.knowledge.entities.knowledge_entities import DataSource, InfoList, KnowledgeConfig
from services.knowledge.entities.resource_scope import DatasetRef

TENANT_ID = "11111111-1111-1111-1111-111111111111"
USER_ID = "22222222-2222-2222-2222-222222222222"
DATASET_ID = "33333333-3333-3333-3333-333333333333"
DOCUMENT_ID = "44444444-4444-4444-4444-444444444444"


class FakeLock:
    """Lock that fails while being acquired."""

    def __enter__(self) -> None:
        raise LockNotOwnedError("simulated")

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: types.TracebackType | None
    ) -> bool:
        return False


class FakeReleaseLock:
    """Lock whose ownership is lost while releasing it."""

    def __enter__(self) -> "FakeReleaseLock":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: types.TracebackType | None
    ) -> bool:
        raise LockNotOwnedError("simulated")


@pytest.fixture
def fake_current_user(monkeypatch: pytest.MonkeyPatch) -> Account:
    tenant = Tenant(name="Test Tenant")
    tenant.id = TENANT_ID
    user = Account(name="Test User", email="test@example.com")
    user.id = USER_ID
    user._current_tenant = tenant
    monkeypatch.setattr("services.knowledge.dataset_service.current_user", user)
    return user


@pytest.fixture
def fake_features(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    features = types.SimpleNamespace(
        billing=types.SimpleNamespace(subscription=types.SimpleNamespace(plan="ENTERPRISE")),
        documents_upload_quota=types.SimpleNamespace(limit=10_000, size=0),
    )
    monkeypatch.setattr(
        "services.knowledge.dataset_service.FeatureService.get_features",
        lambda _tenant_id, **_kwargs: features,
    )
    return features


@pytest.fixture
def fake_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch redis_client.lock to fail while being acquired."""

    def _fake_lock(*_args: object, **_kwargs: object) -> FakeLock:
        return FakeLock()

    # DatasetService imports redis_client directly from extensions.ext_redis
    monkeypatch.setattr("services.knowledge.dataset_service.redis_client.lock", _fake_lock)


@pytest.fixture
def fake_release_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_lock(*_args: object, **_kwargs: object) -> FakeReleaseLock:
        return FakeReleaseLock()

    monkeypatch.setattr("services.knowledge.dataset_service.redis_client.lock", _fake_lock)


# ---------------------------------------------------------------------------
# 1. Knowledge Pipeline document creation (save_document_with_dataset_id)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sqlite_session", [(Account, Tenant, Dataset, DatasetProcessRule, Document)], indirect=True)
@pytest.mark.usefixtures("fake_features", "fake_release_lock")
def test_save_document_with_dataset_id_ignores_lock_not_owned(
    monkeypatch: pytest.MonkeyPatch,
    fake_current_user: Account,
    sqlite_session: Session,
) -> None:
    # Arrange
    dataset = Dataset(
        id=DATASET_ID,
        tenant_id=TENANT_ID,
        name="Test Dataset",
        description="",
        created_by=USER_ID,
        data_source_type="upload_file",
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
    )
    process_rule = DatasetProcessRule(
        dataset_id=DATASET_ID,
        mode=ProcessRuleMode.AUTOMATIC,
        rules=None,
        created_by=USER_ID,
    )
    sqlite_session.add_all([fake_current_user._current_tenant, fake_current_user, dataset, process_rule])
    sqlite_session.commit()

    knowledge_config = KnowledgeConfig(
        doc_form=IndexStructureType.QA_INDEX,
        original_document_id=None,  # go into "new document" branch
        data_source=DataSource(info_list=InfoList(data_source_type="upload_file")),
        indexing_technique="high_quality",
        duplicate=False,
        doc_language="en",
    )

    account = fake_current_user

    # Avoid touching real doc_form logic
    monkeypatch.setattr(
        "services.knowledge.dataset_service.DatasetService.check_doc_form", lambda *_args, **_kwargs: None
    )

    monkeypatch.setattr(
        DocumentService,
        "prepare_document_creation",
        lambda *_args, **_kwargs: knowledge_config,
    )
    document = Document(
        id=DOCUMENT_ID,
        tenant_id=TENANT_ID,
        dataset_id=DATASET_ID,
        position=1,
        data_source_type="upload_file",
        data_source_info="{}",
        batch="batch-1",
        name="test.txt",
        created_from="api",
        created_by=USER_ID,
        indexing_status="waiting",
        doc_form=IndexStructureType.QA_INDEX,
        word_count=0,
    )

    def save_prepared_documents(*_args: object, **kwargs: object) -> tuple[list[Document], str, DocumentIndexingJobs]:
        session = kwargs["session"]
        assert isinstance(session, Session)
        session.add(document)
        session.flush()
        return [document], "batch-1", DocumentIndexingJobs(DatasetRef(TENANT_ID, DATASET_ID), created=(document.id,))

    monkeypatch.setattr(DocumentService, "save_prepared_documents", save_prepared_documents)
    dispatched: list[object] = []
    monkeypatch.setattr(DocumentService, "dispatch_document_indexing", dispatched.append)

    # The write succeeds, then releasing the lock reports that ownership expired.
    with DocumentService.save_document_with_dataset_id(
        dataset=dataset,
        knowledge_config=knowledge_config,
        account=account,
        dataset_process_rule=process_rule,
        session=sqlite_session,
    ) as (documents, batch):
        pass

    # Assert
    # We mainly care that:
    # - No exception is raised
    # - The function returns a sensible tuple
    assert documents == [document]
    assert batch == "batch-1"
    assert len(dispatched) == 1
    assert not sqlite_session.in_transaction()
    assert sqlite_session.scalar(select(func.count(Document.id))) == 1
    assert sqlite_session.get(Document, DOCUMENT_ID) is document
    assert sqlite_session.get(DatasetProcessRule, process_rule.id) is process_rule


@pytest.mark.parametrize("sqlite_session", [(Account, Tenant, Dataset, DatasetProcessRule, Document)], indirect=True)
@pytest.mark.usefixtures("fake_features", "fake_release_lock")
def test_save_document_with_dataset_id_preserves_caller_error_when_lock_release_fails(
    monkeypatch: pytest.MonkeyPatch,
    fake_current_user: Account,
    sqlite_session: Session,
) -> None:
    class ResponseError(Exception):
        pass

    dataset = Dataset(
        id=DATASET_ID,
        tenant_id=TENANT_ID,
        name="Test Dataset",
        description="",
        created_by=USER_ID,
        data_source_type="upload_file",
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
    )
    sqlite_session.add_all([fake_current_user._current_tenant, fake_current_user, dataset])
    sqlite_session.commit()
    knowledge_config = KnowledgeConfig(
        doc_form=IndexStructureType.QA_INDEX,
        original_document_id=None,
        data_source=DataSource(info_list=InfoList(data_source_type="upload_file")),
        indexing_technique="high_quality",
        duplicate=False,
        doc_language="en",
    )
    monkeypatch.setattr(
        "services.knowledge.dataset_service.DatasetService.check_doc_form", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        DocumentService,
        "prepare_document_creation",
        lambda *_args, **_kwargs: knowledge_config,
    )
    monkeypatch.setattr(
        DocumentService,
        "save_prepared_documents",
        lambda *_args, **_kwargs: ([], "batch-1", DocumentIndexingJobs(DatasetRef(TENANT_ID, DATASET_ID))),
    )
    dispatched: list[object] = []
    monkeypatch.setattr(DocumentService, "dispatch_document_indexing", dispatched.append)

    with pytest.raises(ResponseError, match="response failed"):
        with DocumentService.save_document_with_dataset_id(
            dataset=dataset,
            knowledge_config=knowledge_config,
            account=fake_current_user,
            session=sqlite_session,
        ):
            raise ResponseError("response failed")

    assert dispatched == []

"""Unit tests for document sync persistence and external collaborator wiring.

The task's Dataset and Document lookups use real SQLite transactions. Datasource,
Notion extraction, index cleanup, and indexing remain mocked I/O boundaries.
"""

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timedelta
from unittest.mock import ANY, MagicMock, patch

import httpx
import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

import tasks.document_indexing_sync_task as document_indexing_sync_task_module
from core.db import session_factory as session_factory_module
from core.rag.index_processor.constant.index_type import IndexStructureType
from models.account import Tenant
from models.dataset import Dataset, Document, DocumentSegment
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus, PermissionEnum, SegmentStatus
from models.oauth import DatasourceProvider
from repositories.data_source.credential_repository import SQLAlchemyDatasourceCredentialRepository
from repositories.knowledge.document_repository import SQLAlchemyDocumentRepository
from services.data_source.credential_gateway import (
    RefreshedDatasourceCredential,
    TrustedStoredDatasourceCredentialGateway,
)
from services.data_source.entities.credential import DatasourceCredentialRecord
from services.knowledge.document_sync import ClaimedNotionDocument
from services.knowledge.entities.resource_scope import DatasetRef, DocumentRef
from tasks.document_indexing_sync_task import _NotionSyncLeaseHeartbeat, document_indexing_sync_task

pytestmark = pytest.mark.parametrize(
    "sqlite_session",
    [(Tenant, Dataset, Document, DocumentSegment, DatasourceProvider)],
    indirect=True,
)


class _UnexpectedActorCredentials:
    def resolve(self, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("stored sync must not use actor credential resolution")


class _RecordingStoredCredentials:
    def __init__(self, credentials: Mapping[str, object] | None = None) -> None:
        self._credentials = dict(credentials or {"integration_secret": "test_token"})
        self.calls: list[dict[str, object]] = []

    def resolve_for_document(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return dict(self._credentials)


class _TaskCredentials:
    def __init__(self, stored: object) -> None:
        self.actor = _UnexpectedActorCredentials()
        self.stored = stored


class _PlainCredentialCodec:
    def decrypt(self, record: DatasourceCredentialRecord) -> dict[str, object]:
        return dict(record.encrypted_credentials)

    def encrypt(self, record: DatasourceCredentialRecord, credentials: Mapping[str, object]) -> dict[str, object]:
        del record
        return dict(credentials)


class _UnexpectedCredentialRefresher:
    def refresh(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        record: DatasourceCredentialRecord,
        credentials: Mapping[str, object],
    ) -> RefreshedDatasourceCredential:
        del workspace_id, actor_id, record, credentials
        raise AssertionError("non-expiring test credentials must not refresh")


def _notion_credential(
    credential_id: str,
    *,
    tenant_id: str,
    integration_secret: str,
    is_default: bool = False,
) -> DatasourceProvider:
    credential = DatasourceProvider(
        tenant_id=tenant_id,
        name=credential_id,
        provider="notion_datasource",
        plugin_id="langgenius/notion_datasource",
        auth_type="oauth2",
        encrypted_credentials={"integration_secret": integration_secret},
        user_id=str(uuid.uuid4()),
        is_default=is_default,
        expires_at=-1,
        visibility=PermissionEnum.ONLY_ME,
    )
    credential.id = credential_id
    return credential


def _trusted_stored_credentials(sqlite_session: Session) -> TrustedStoredDatasourceCredentialGateway:
    return TrustedStoredDatasourceCredentialGateway(
        credentials=SQLAlchemyDatasourceCredentialRepository(
            session_factory=sessionmaker(bind=sqlite_session.get_bind(), expire_on_commit=False)
        ),
        codec=_PlainCredentialCodec(),
        refresher=_UnexpectedCredentialRefresher(),
    )


@pytest.fixture
def dataset_id() -> str:
    """Generate a dataset id."""
    return str(uuid.uuid4())


@pytest.fixture
def document_id() -> str:
    """Generate a document id."""
    return str(uuid.uuid4())


@pytest.fixture
def notion_workspace_id() -> str:
    """Generate a notion workspace id."""
    return str(uuid.uuid4())


@pytest.fixture
def notion_page_id() -> str:
    """Generate a notion page id."""
    return str(uuid.uuid4())


@pytest.fixture
def credential_id() -> str:
    """Generate a credential id."""
    return str(uuid.uuid4())


@pytest.fixture
def tenant_id() -> str:
    """Generate the tenant that owns the persisted dataset and document."""
    return str(uuid.uuid4())


@pytest.fixture
def dataset(sqlite_session: Session, dataset_id: str, tenant_id: str) -> Dataset:
    """Persist the dataset resolved by the task's initial and cleanup transactions."""
    tenant = Tenant(name="Workspace")
    tenant.id = tenant_id
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name="Notion dataset",
        data_source_type=DataSourceType.NOTION_IMPORT,
        created_by=str(uuid.uuid4()),
    )
    sqlite_session.add_all([tenant, dataset])
    sqlite_session.commit()
    return dataset


@pytest.fixture
def document(
    sqlite_session: Session,
    dataset: Dataset,
    document_id: str,
    tenant_id: str,
    notion_workspace_id: str,
    notion_page_id: str,
    credential_id: str,
) -> Document:
    """Persist a completed Notion document for collaborator and update assertions."""
    document = Document(
        id=document_id,
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.NOTION_IMPORT,
        data_source_info=json.dumps(
            {
                "notion_workspace_id": notion_workspace_id,
                "notion_page_id": notion_page_id,
                "type": "page",
                "last_edited_time": "2024-01-01T00:00:00Z",
                "credential_id": credential_id,
            }
        ),
        batch="batch-1",
        name="Notion page",
        created_from=DocumentCreatedFrom.API,
        created_by=str(uuid.uuid4()),
        indexing_status=IndexingStatus.COMPLETED,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    sqlite_session.add(document)
    sqlite_session.commit()
    return document


@pytest.fixture(autouse=True)
def _bind_sqlite_session_factory(monkeypatch: pytest.MonkeyPatch, sqlite_session: Session) -> None:
    """Bind each task-owned session to the test's isolated SQLite database."""
    monkeypatch.setattr(
        session_factory_module,
        "_session_maker",
        sessionmaker(bind=sqlite_session.get_bind(), expire_on_commit=False),
    )


@pytest.fixture
def stored_credentials(monkeypatch: pytest.MonkeyPatch) -> _RecordingStoredCredentials:
    """Record exact stored-document credential resolution at the task boundary."""
    stored = _RecordingStoredCredentials()
    monkeypatch.setattr(
        "tasks.document_indexing_sync_task.build_data_source_credentials",
        lambda **_kwargs: _TaskCredentials(stored),
    )
    return stored


@pytest.fixture
def mock_notion_extractor() -> Iterator[dict[str, MagicMock]]:
    """Mock notion extractor class and instance."""
    with patch("tasks.document_indexing_sync_task.NotionExtractor", autospec=True) as mock_extractor_class:
        mock_extractor = MagicMock()
        mock_extractor.get_notion_last_edited_time.return_value = "2024-01-01T00:00:00Z"
        mock_extractor_class.return_value = mock_extractor
        yield {"class": mock_extractor_class, "instance": mock_extractor}


class TestDocumentIndexingSyncTaskCollaboratorParams:
    """Unit tests for collaborator parameter passing in document_indexing_sync_task."""

    def test_notion_extractor_initialized_with_correct_params(
        self,
        stored_credentials: _RecordingStoredCredentials,
        mock_notion_extractor: dict[str, MagicMock],
        document: Document,
        dataset_id: str,
        document_id: str,
        notion_workspace_id: str,
        notion_page_id: str,
    ) -> None:
        """Test that NotionExtractor is initialized with expected arguments."""
        # Arrange
        expected_token = "test_token"

        # Act
        document_indexing_sync_task(dataset_id, document_id)

        # Assert
        mock_notion_extractor["class"].assert_called_once_with(
            notion_token_loader=ANY,
            notion_workspace_id=notion_workspace_id,
            notion_obj_id=notion_page_id,
            notion_page_type="page",
            notion_access_token=expected_token,
            tenant_id=document.tenant_id,
        )

    def test_datasource_credentials_requested_correctly(
        self,
        stored_credentials: _RecordingStoredCredentials,
        mock_notion_extractor: dict[str, MagicMock],
        document: Document,
        dataset_id: str,
        document_id: str,
        credential_id: str,
    ) -> None:
        """Test that datasource credentials are requested with expected identifiers."""
        # Arrange
        expected_tenant_id = document.tenant_id

        # Act
        document_indexing_sync_task(dataset_id, document_id)

        # Assert
        assert stored_credentials.calls == [
            {
                "workspace_id": expected_tenant_id,
                "dataset_id": dataset_id,
                "document_id": document_id,
                "credential_id": credential_id,
                "provider": "notion_datasource",
                "plugin_id": "langgenius/notion_datasource",
            }
        ]

    def test_legacy_source_without_credential_uses_integration_token_without_stored_lookup(
        self,
        mock_notion_extractor: dict[str, MagicMock],
        document: Document,
        sqlite_session: Session,
        dataset_id: str,
        document_id: str,
        config_overrides: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A truly legacy source ignores tenant defaults even when the default changes."""
        config_overrides(NOTION_INTEGRATION_TOKEN="legacy-integration-token")
        document.data_source_info = json.dumps(
            {
                "notion_workspace_id": "workspace-id",
                "notion_page_id": "page-id",
                "type": "page",
                "last_edited_time": "2024-01-01T00:00:00Z",
            }
        )
        first_default = _notion_credential(
            "default-one",
            tenant_id=document.tenant_id,
            integration_secret="tenant-default-one",
            is_default=True,
        )
        second_default = _notion_credential(
            "default-two",
            tenant_id=document.tenant_id,
            integration_secret="tenant-default-two",
        )
        sqlite_session.add_all([first_default, second_default])
        sqlite_session.commit()
        credentials = _TaskCredentials(_trusted_stored_credentials(sqlite_session))
        monkeypatch.setattr(
            "tasks.document_indexing_sync_task.build_data_source_credentials",
            lambda **_kwargs: credentials,
        )

        document_indexing_sync_task(dataset_id, document_id)
        first_default.is_default = False
        second_default.is_default = True
        sqlite_session.commit()
        document_indexing_sync_task(dataset_id, document_id)

        assert mock_notion_extractor["class"].call_count == 2
        for call in mock_notion_extractor["class"].call_args_list:
            assert call.kwargs == {
                "notion_token_loader": ANY,
                "notion_workspace_id": "workspace-id",
                "notion_obj_id": "page-id",
                "notion_page_type": "page",
                "notion_access_token": "legacy-integration-token",
                "tenant_id": document.tenant_id,
            }

    def test_explicit_private_credential_uses_trusted_stored_owner_chain(
        self,
        mock_notion_extractor: dict[str, MagicMock],
        document: Document,
        sqlite_session: Session,
        dataset_id: str,
        document_id: str,
        credential_id: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sqlite_session.add(
            _notion_credential(
                credential_id,
                tenant_id=document.tenant_id,
                integration_secret="private-stored-token",
            )
        )
        sqlite_session.commit()
        credentials = _TaskCredentials(_trusted_stored_credentials(sqlite_session))
        monkeypatch.setattr(
            "tasks.document_indexing_sync_task.build_data_source_credentials",
            lambda **_kwargs: credentials,
        )

        document_indexing_sync_task(dataset_id, document_id)

        mock_notion_extractor["class"].assert_called_once_with(
            notion_token_loader=ANY,
            notion_workspace_id=document.data_source_info_dict["notion_workspace_id"],
            notion_obj_id=document.data_source_info_dict["notion_page_id"],
            notion_page_type="page",
            notion_access_token="private-stored-token",
            tenant_id=document.tenant_id,
        )
        sqlite_session.expire_all()
        assert sqlite_session.get_one(Document, document_id).indexing_status == IndexingStatus.COMPLETED

    def test_deleted_explicit_credential_does_not_use_global_fallback(
        self,
        mock_notion_extractor: dict[str, MagicMock],
        document: Document,
        sqlite_session: Session,
        dataset_id: str,
        document_id: str,
        config_overrides: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_overrides(NOTION_INTEGRATION_TOKEN="global-token-must-not-be-used")
        credentials = _TaskCredentials(_trusted_stored_credentials(sqlite_session))
        monkeypatch.setattr(
            "tasks.document_indexing_sync_task.build_data_source_credentials",
            lambda **_kwargs: credentials,
        )

        document_indexing_sync_task(dataset_id, document_id)

        mock_notion_extractor["class"].assert_not_called()
        sqlite_session.expire_all()
        failed = sqlite_session.get_one(Document, document_id)
        assert failed.indexing_status == IndexingStatus.ERROR
        assert failed.error == "Notion credential is unavailable"

    @pytest.mark.parametrize("invalid_credential_id", [None, 42])
    def test_explicit_invalid_credential_field_does_not_become_legacy_fallback(
        self,
        mock_notion_extractor: dict[str, MagicMock],
        document: Document,
        sqlite_session: Session,
        dataset_id: str,
        document_id: str,
        config_overrides: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch,
        invalid_credential_id: object,
    ) -> None:
        config_overrides(NOTION_INTEGRATION_TOKEN="global-token-must-not-be-used")
        source_info = document.data_source_info_dict
        source_info["credential_id"] = invalid_credential_id
        document.data_source_info = json.dumps(source_info)
        sqlite_session.commit()
        stored = _RecordingStoredCredentials({"integration_secret": "must-not-be-used"})
        monkeypatch.setattr(
            "tasks.document_indexing_sync_task.build_data_source_credentials",
            lambda **_kwargs: _TaskCredentials(stored),
        )

        document_indexing_sync_task(dataset_id, document_id)

        assert stored.calls == []
        mock_notion_extractor["class"].assert_not_called()
        sqlite_session.expire_all()
        failed = sqlite_session.get_one(Document, document_id)
        assert failed.indexing_status == IndexingStatus.ERROR
        assert failed.error == "Notion credential is unavailable"


class TestDataSourceInfoSerialization:
    """Source versions are published only by successful indexing completion."""

    def test_data_source_info_serialized_as_json_string(
        self,
        document: Document,
        sqlite_session: Session,
        dataset_id: str,
        document_id: str,
    ) -> None:
        """Preparing a sync must retain the last successfully indexed source version."""
        with (
            patch("tasks.document_indexing_sync_task.build_data_source_credentials") as mock_service_class,
            patch("tasks.document_indexing_sync_task.NotionExtractor") as mock_extractor_class,
            patch("tasks.document_indexing_sync_task.clean_document_for_reindex"),
            patch("tasks.document_indexing_sync_task.build_document_indexing_service") as mock_runner_class,
        ):
            # External collaborators
            mock_service_class.return_value = _TaskCredentials(
                _RecordingStoredCredentials({"integration_secret": "token"})
            )

            mock_extractor = MagicMock()
            # Return a *different* timestamp so the task enters the sync/update branch
            mock_extractor.get_notion_last_edited_time.return_value = "2024-02-01T00:00:00Z"
            mock_extractor_class.return_value = mock_extractor

            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            # Act
            document_indexing_sync_task(dataset_id, document_id)

            # The new timestamp is intentionally not published until the
            # indexing service completes this exact claim.
            sqlite_session.expire_all()
            stored_document = sqlite_session.get(Document, document.id)
            assert stored_document is not None
            assert isinstance(stored_document.data_source_info, str), (
                f"data_source_info should be a JSON string, got {type(stored_document.data_source_info).__name__}"
            )
            parsed = json.loads(stored_document.data_source_info)
            assert parsed["last_edited_time"] == "2024-01-01T00:00:00Z"
            assert stored_document.indexing_status == IndexingStatus.PARSING
            assert stored_document.processing_started_at is not None


def test_duplicate_delivery_cannot_clean_or_index_an_active_notion_sync(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    old_segment = DocumentSegment(
        tenant_id=document.tenant_id,
        dataset_id=document.dataset_id,
        document_id=document.id,
        position=1,
        content="old content",
        word_count=2,
        tokens=2,
        created_by=document.created_by,
        index_node_id="old-node",
        status=SegmentStatus.COMPLETED,
    )
    sqlite_session.add(old_segment)
    sqlite_session.commit()

    provider_calls: list[str] = []
    extractor_calls: list[str] = []
    clean_calls: list[list[str]] = []
    indexing_calls: list[tuple[str, str]] = []

    class ReentrantStoredCredentials:
        def resolve_for_document(self, **_kwargs: object) -> dict[str, str]:
            provider_calls.append(document_id)
            document_indexing_sync_task(dataset_id, document_id)
            return {"integration_secret": "token"}

    credentials = _TaskCredentials(ReentrantStoredCredentials())

    class ChangedNotionExtractor:
        def __init__(self, **_kwargs: object) -> None:
            extractor_calls.append(document_id)

        def get_notion_last_edited_time(self) -> str:
            return "2024-02-01T00:00:00Z"

    def clean_indexes(
        *,
        document_ids: list[str],
        new_session: Callable[[], Session],
        mutation_guard: Callable[[Session], None],
        **_kwargs: object,
    ) -> None:
        with new_session() as session:
            mutation_guard(session)
            current = session.get_one(Document, document_id)
            assert current.indexing_status == IndexingStatus.PARSING
            assert current.indexing_claim_id is not None
            assert session.query(DocumentSegment).count() == 1
        clean_calls.append(document_ids)

    class CompletingRunner:
        def __init__(self, repository: SQLAlchemyDocumentRepository) -> None:
            self._repository = repository

        def run_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
            indexing_calls.append((ref.document_id, claim_id))
            self._repository.complete_indexing(
                ref,
                claim_id=claim_id,
                tokens=0,
                latency=0,
                source_version="2024-02-01T00:00:00Z",
            )

    def build_runner(*, session_factory: sessionmaker[Session], **_kwargs: object) -> CompletingRunner:
        return CompletingRunner(SQLAlchemyDocumentRepository(session_factory=session_factory))

    monkeypatch.setattr(
        "tasks.document_indexing_sync_task.build_data_source_credentials",
        lambda **_kwargs: credentials,
    )
    monkeypatch.setattr("tasks.document_indexing_sync_task.NotionExtractor", ChangedNotionExtractor)
    monkeypatch.setattr("tasks.document_indexing_sync_task.clean_document_for_reindex", clean_indexes)
    monkeypatch.setattr("tasks.document_indexing_sync_task.build_document_indexing_service", build_runner)

    document_indexing_sync_task(dataset_id, document_id)

    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document_id)
    assert provider_calls == [document_id]
    assert extractor_calls == [document_id]
    assert clean_calls == [[document_id]]
    assert len(indexing_calls) == 1
    assert indexing_calls[0][0] == document_id
    assert persisted.indexing_status == IndexingStatus.COMPLETED
    assert persisted.indexing_claim_id is None
    assert persisted.data_source_info_dict["last_edited_time"] == "2024-02-01T00:00:00Z"


def test_expired_prepared_sync_reindexes_when_remote_edit_time_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    repository = SQLAlchemyDocumentRepository(session_factory=sessionmaker(bind=sqlite_session.get_bind()))
    ref = DatasetRef(document.tenant_id, dataset_id).document(document_id)
    old_claim = repository.claim_notion_sync(ref, claim_id="77777777-7777-7777-7777-777777777777")
    assert old_claim is not None
    repository.prepare_notion_sync(
        old_claim,
        processing_started_at=datetime(2026, 1, 1),
    )
    with sessionmaker(bind=sqlite_session.get_bind()).begin() as session:
        database_now = session.scalar(select(func.current_timestamp()))
        assert isinstance(database_now, datetime)
        current = session.get_one(Document, document_id)
        current.indexing_claim_expires_at = database_now - timedelta(seconds=1)

    clean_calls: list[list[str]] = []
    indexing_calls: list[tuple[str, str]] = []

    credentials = _TaskCredentials(_RecordingStoredCredentials({"integration_secret": "token"}))

    class UnchangedNotionExtractor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_notion_last_edited_time(self) -> str:
            return "2024-02-01T00:00:00Z"

    def clean_indexes(
        *,
        document_ids: list[str],
        new_session: Callable[[], Session],
        mutation_guard: Callable[[Session], None],
        **_kwargs: object,
    ) -> None:
        with new_session() as session, session.begin():
            mutation_guard(session)
        clean_calls.append(document_ids)

    class CompletingRunner:
        def __init__(self, current_repository: SQLAlchemyDocumentRepository) -> None:
            self._repository = current_repository

        def run_claimed(self, current_ref: DocumentRef, *, claim_id: str) -> None:
            indexing_calls.append((current_ref.document_id, claim_id))
            self._repository.complete_indexing(current_ref, claim_id=claim_id, tokens=0, latency=0)

    def build_runner(*, session_factory: sessionmaker[Session], **_kwargs: object) -> CompletingRunner:
        return CompletingRunner(SQLAlchemyDocumentRepository(session_factory=session_factory))

    monkeypatch.setattr(
        "tasks.document_indexing_sync_task.build_data_source_credentials",
        lambda **_kwargs: credentials,
    )
    monkeypatch.setattr("tasks.document_indexing_sync_task.NotionExtractor", UnchangedNotionExtractor)
    monkeypatch.setattr("tasks.document_indexing_sync_task.clean_document_for_reindex", clean_indexes)
    monkeypatch.setattr("tasks.document_indexing_sync_task.build_document_indexing_service", build_runner)

    document_indexing_sync_task(dataset_id, document_id)

    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document_id)
    assert clean_calls == [[document_id]]
    assert len(indexing_calls) == 1
    assert indexing_calls[0][0] == document_id
    assert indexing_calls[0][1] != old_claim.claim_id
    assert persisted.indexing_status == IndexingStatus.COMPLETED
    assert persisted.indexing_claim_id is None
    assert persisted.indexing_claim_expires_at is None


def test_successfully_published_source_version_skips_unchanged_sync(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    repository = SQLAlchemyDocumentRepository(session_factory=sessionmaker(bind=sqlite_session.get_bind()))
    ref = DatasetRef(document.tenant_id, dataset_id).document(document_id)
    initial_claim = repository.claim_notion_sync(ref, claim_id="initial-claim")
    assert initial_claim is not None
    repository.complete_indexing(
        ref,
        claim_id=initial_claim.claim_id,
        tokens=0,
        latency=0,
        source_version="2024-02-01T00:00:00Z",
    )

    credentials = _TaskCredentials(_RecordingStoredCredentials({"integration_secret": "token"}))

    class UnchangedNotionExtractor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_notion_last_edited_time(self) -> str:
            return "2024-02-01T00:00:00Z"

    clean_indexes = MagicMock()

    def build_runner(**_kwargs: object) -> None:
        raise AssertionError("unchanged Notion content must not build an indexing runner")

    monkeypatch.setattr(
        "tasks.document_indexing_sync_task.build_data_source_credentials",
        lambda **_kwargs: credentials,
    )
    monkeypatch.setattr("tasks.document_indexing_sync_task.NotionExtractor", UnchangedNotionExtractor)
    monkeypatch.setattr("tasks.document_indexing_sync_task.clean_document_for_reindex", clean_indexes)
    monkeypatch.setattr("tasks.document_indexing_sync_task.build_document_indexing_service", build_runner)

    document_indexing_sync_task(dataset_id, document_id)

    clean_indexes.assert_not_called()
    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document_id)
    assert persisted.indexing_status == IndexingStatus.COMPLETED
    assert persisted.indexing_claim_id is None
    assert persisted.data_source_info_dict["last_edited_time"] == "2024-02-01T00:00:00Z"


def test_unchanged_sync_release_failure_preserves_completed_state_until_lease_recovery(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    old_segment = DocumentSegment(
        tenant_id=document.tenant_id,
        dataset_id=document.dataset_id,
        document_id=document.id,
        position=1,
        content="healthy indexed content",
        word_count=3,
        tokens=3,
        created_by=document.created_by,
        index_node_id="healthy-node",
        status=SegmentStatus.COMPLETED,
    )
    sqlite_session.add(old_segment)
    sqlite_session.commit()

    credentials = _TaskCredentials(_RecordingStoredCredentials({"integration_secret": "token"}))

    class UnchangedNotionExtractor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_notion_last_edited_time(self) -> str:
            return "2024-01-01T00:00:00Z"

    release_attempts: list[str] = []
    original_release = SQLAlchemyDocumentRepository.release_notion_sync

    def fail_first_release(
        repository: SQLAlchemyDocumentRepository,
        claim: ClaimedNotionDocument,
    ) -> bool:
        release_attempts.append(claim.claim_id)
        if len(release_attempts) == 1:
            raise OperationalError("release notion sync", {}, RuntimeError("database temporarily unavailable"))
        return original_release(repository, claim)

    cleanup_calls: list[list[str]] = []

    def clean_indexes(
        *,
        document_ids: list[str],
        new_session: Callable[[], Session],
        mutation_guard: Callable[[Session], None],
        **_kwargs: object,
    ) -> None:
        with new_session() as cleanup_session, cleanup_session.begin():
            mutation_guard(cleanup_session)
        cleanup_calls.append(document_ids)

    indexing_calls: list[str] = []

    class CompletingRunner:
        def __init__(self, repository: SQLAlchemyDocumentRepository) -> None:
            self._repository = repository

        def run_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
            indexing_calls.append(claim_id)
            self._repository.complete_indexing(
                ref,
                claim_id=claim_id,
                tokens=3,
                latency=0,
                source_version="2024-01-01T00:00:00Z",
            )

    def build_runner(*, session_factory: sessionmaker[Session], **_kwargs: object) -> CompletingRunner:
        return CompletingRunner(SQLAlchemyDocumentRepository(session_factory=session_factory))

    monkeypatch.setattr(
        document_indexing_sync_task_module,
        "build_data_source_credentials",
        lambda **_kwargs: credentials,
    )
    monkeypatch.setattr(document_indexing_sync_task_module, "NotionExtractor", UnchangedNotionExtractor)
    monkeypatch.setattr(SQLAlchemyDocumentRepository, "release_notion_sync", fail_first_release)
    monkeypatch.setattr(document_indexing_sync_task_module, "clean_document_for_reindex", clean_indexes)
    monkeypatch.setattr(document_indexing_sync_task_module, "build_document_indexing_service", build_runner)

    with pytest.raises(document_indexing_sync_task_module._NotionSyncClaimReleaseError):
        document_indexing_sync_task(dataset_id, document_id)

    sqlite_session.expire_all()
    after_release_failure = sqlite_session.get_one(Document, document_id)
    first_claim_id = after_release_failure.indexing_claim_id
    assert first_claim_id is not None
    assert after_release_failure.indexing_status == IndexingStatus.COMPLETED
    assert after_release_failure.error is None
    assert cleanup_calls == []
    assert indexing_calls == []
    assert sqlite_session.get_one(DocumentSegment, old_segment.id).content == "healthy indexed content"

    retry_session_factory = sessionmaker(bind=sqlite_session.get_bind(), expire_on_commit=False)
    with retry_session_factory.begin() as retry_session:
        database_now = retry_session.scalar(select(func.current_timestamp()))
        assert isinstance(database_now, datetime)
        retry_document = retry_session.get_one(Document, document_id)
        retry_document.indexing_claim_expires_at = database_now - timedelta(seconds=1)

    document_indexing_sync_task(dataset_id, document_id)

    sqlite_session.expire_all()
    recovered = sqlite_session.get_one(Document, document_id)
    assert release_attempts == [first_claim_id]
    assert cleanup_calls == [[document_id]]
    assert len(indexing_calls) == 1
    assert recovered.indexing_status == IndexingStatus.COMPLETED
    assert recovered.error is None
    assert recovered.indexing_claim_id is None
    assert recovered.indexing_claim_expires_at is None


def test_version_probe_timeout_preserves_published_index_until_lease_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    old_segment = DocumentSegment(
        tenant_id=document.tenant_id,
        dataset_id=document.dataset_id,
        document_id=document.id,
        position=1,
        content="last published content",
        word_count=3,
        tokens=3,
        created_by=document.created_by,
        index_node_id="last-published-node",
        status=SegmentStatus.COMPLETED,
    )
    sqlite_session.add(old_segment)
    sqlite_session.commit()
    old_segment_id = old_segment.id

    credentials = _TaskCredentials(_RecordingStoredCredentials({"integration_secret": "token"}))
    probe_calls: list[str] = []

    class FlakyNotionExtractor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_notion_last_edited_time(self) -> str:
            probe_calls.append(document_id)
            if len(probe_calls) == 1:
                request = httpx.Request("GET", f"https://api.notion.com/v1/pages/{document_id}")
                raise httpx.ReadTimeout("Notion version request timed out", request=request)
            return "2024-02-01T00:00:00Z"

    cleanup_calls: list[list[str]] = []

    def clean_indexes(
        *,
        document_ids: list[str],
        new_session: Callable[[], Session],
        mutation_guard: Callable[[Session], None],
        **_kwargs: object,
    ) -> None:
        with new_session() as cleanup_session, cleanup_session.begin():
            mutation_guard(cleanup_session)
            cleanup_session.execute(delete(DocumentSegment).where(DocumentSegment.document_id.in_(document_ids)))
        cleanup_calls.append(document_ids)

    indexing_calls: list[str] = []

    class CompletingRunner:
        def __init__(self, repository: SQLAlchemyDocumentRepository) -> None:
            self._repository = repository

        def run_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
            indexing_calls.append(claim_id)
            self._repository.complete_indexing(
                ref,
                claim_id=claim_id,
                tokens=3,
                latency=0,
                source_version="2024-02-01T00:00:00Z",
            )

    def build_runner(*, session_factory: sessionmaker[Session], **_kwargs: object) -> CompletingRunner:
        return CompletingRunner(SQLAlchemyDocumentRepository(session_factory=session_factory))

    monkeypatch.setattr(
        document_indexing_sync_task_module,
        "build_data_source_credentials",
        lambda **_kwargs: credentials,
    )
    monkeypatch.setattr(document_indexing_sync_task_module, "NotionExtractor", FlakyNotionExtractor)
    monkeypatch.setattr(document_indexing_sync_task_module, "clean_document_for_reindex", clean_indexes)
    monkeypatch.setattr(document_indexing_sync_task_module, "build_document_indexing_service", build_runner)

    with pytest.raises(document_indexing_sync_task_module._NotionSyncPreflightTransientError):
        document_indexing_sync_task(dataset_id, document_id)

    sqlite_session.expire_all()
    after_timeout = sqlite_session.get_one(Document, document_id)
    assert after_timeout.indexing_status == IndexingStatus.COMPLETED
    assert after_timeout.error is None
    assert after_timeout.indexing_claim_id is not None
    assert sqlite_session.get_one(DocumentSegment, old_segment.id).content == "last published content"
    assert cleanup_calls == []
    assert indexing_calls == []

    retry_session_factory = sessionmaker(bind=sqlite_session.get_bind(), expire_on_commit=False)
    with retry_session_factory.begin() as retry_session:
        database_now = retry_session.scalar(select(func.current_timestamp()))
        assert isinstance(database_now, datetime)
        retry_document = retry_session.get_one(Document, document_id)
        retry_document.indexing_claim_expires_at = database_now - timedelta(seconds=1)

    document_indexing_sync_task(dataset_id, document_id)

    sqlite_session.expire_all()
    recovered = sqlite_session.get_one(Document, document_id)
    assert probe_calls == [document_id, document_id]
    assert cleanup_calls == [[document_id]]
    assert len(indexing_calls) == 1
    assert sqlite_session.get(DocumentSegment, old_segment_id) is None
    assert recovered.indexing_status == IndexingStatus.COMPLETED
    assert recovered.error is None
    assert recovered.indexing_claim_id is None
    assert recovered.data_source_info_dict["last_edited_time"] == "2024-02-01T00:00:00Z"


def test_destructive_sync_failure_is_persisted_and_old_claim_is_fenced(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    old_segment = DocumentSegment(
        tenant_id=document.tenant_id,
        dataset_id=document.dataset_id,
        document_id=document.id,
        position=1,
        content="old content",
        word_count=2,
        tokens=2,
        created_by=document.created_by,
        index_node_id="old-node",
        status=SegmentStatus.COMPLETED,
    )
    sqlite_session.add(old_segment)
    sqlite_session.commit()
    old_segment_id = old_segment.id

    credentials = _TaskCredentials(_RecordingStoredCredentials({"integration_secret": "token"}))

    class ChangedNotionExtractor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_notion_last_edited_time(self) -> str:
            return "2024-02-01T00:00:00Z"

    claimed: list[ClaimedNotionDocument] = []
    original_claim = SQLAlchemyDocumentRepository.claim_notion_sync

    def record_claim(
        repository: SQLAlchemyDocumentRepository,
        ref: DocumentRef,
        *,
        claim_id: str,
    ) -> ClaimedNotionDocument | None:
        claim = original_claim(repository, ref, claim_id=claim_id)
        if claim is not None:
            claimed.append(claim)
        return claim

    def destructive_cleanup(
        *,
        document_ids: list[str],
        new_session: Callable[[], Session],
        mutation_guard: Callable[[Session], None],
        **_kwargs: object,
    ) -> None:
        with new_session() as cleanup_session, cleanup_session.begin():
            mutation_guard(cleanup_session)
            current = cleanup_session.get_one(Document, document_id)
            assert current.indexing_status == IndexingStatus.PARSING
            cleanup_session.execute(delete(DocumentSegment).where(DocumentSegment.document_id.in_(document_ids)))
        raise RuntimeError("vector cleanup failed after destructive side effect")

    def build_runner(**_kwargs: object) -> None:
        raise AssertionError("failed cleanup must not start indexing")

    monkeypatch.setattr(
        document_indexing_sync_task_module,
        "build_data_source_credentials",
        lambda **_kwargs: credentials,
    )
    monkeypatch.setattr(document_indexing_sync_task_module, "NotionExtractor", ChangedNotionExtractor)
    monkeypatch.setattr(SQLAlchemyDocumentRepository, "claim_notion_sync", record_claim)
    monkeypatch.setattr(document_indexing_sync_task_module, "clean_document_for_reindex", destructive_cleanup)
    monkeypatch.setattr(document_indexing_sync_task_module, "build_document_indexing_service", build_runner)

    document_indexing_sync_task(dataset_id, document_id)

    assert len(claimed) == 1
    failed_claim = claimed[0]
    sqlite_session.expire_all()
    failed = sqlite_session.get_one(Document, document_id)
    assert failed.indexing_status == IndexingStatus.ERROR
    assert failed.error == "vector cleanup failed after destructive side effect"
    assert failed.indexing_claim_id is None
    assert sqlite_session.get(DocumentSegment, old_segment_id) is None

    repository = SQLAlchemyDocumentRepository(session_factory=sessionmaker(bind=sqlite_session.get_bind()))
    successor = repository.claim_notion_sync(failed_claim.ref, claim_id="successor-claim")
    assert successor is not None
    assert successor.recovery_required is True
    assert repository.release_notion_sync(failed_claim) is False
    assert repository.release_notion_sync(successor) is True


def test_error_document_reindexes_when_remote_version_matches_stored_value(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    document.indexing_status = IndexingStatus.ERROR
    document.error = "prior vector write failed"
    sqlite_session.commit()
    clean_calls: list[str] = []
    indexing_calls: list[str] = []

    credentials = _TaskCredentials(_RecordingStoredCredentials({"integration_secret": "token"}))

    class UnchangedNotionExtractor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_notion_last_edited_time(self) -> str:
            return "2024-01-01T00:00:00Z"

    def clean_indexes(*, document_ids: list[str], **_kwargs: object) -> None:
        clean_calls.extend(document_ids)

    class CompletingRunner:
        def __init__(self, repository: SQLAlchemyDocumentRepository) -> None:
            self._repository = repository

        def run_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
            indexing_calls.append(ref.document_id)
            self._repository.complete_indexing(
                ref,
                claim_id=claim_id,
                tokens=0,
                latency=0,
                source_version="2024-01-01T00:00:00Z",
            )

    def build_runner(*, session_factory: sessionmaker[Session], **_kwargs: object) -> CompletingRunner:
        return CompletingRunner(SQLAlchemyDocumentRepository(session_factory=session_factory))

    monkeypatch.setattr(
        "tasks.document_indexing_sync_task.build_data_source_credentials",
        lambda **_kwargs: credentials,
    )
    monkeypatch.setattr("tasks.document_indexing_sync_task.NotionExtractor", UnchangedNotionExtractor)
    monkeypatch.setattr("tasks.document_indexing_sync_task.clean_document_for_reindex", clean_indexes)
    monkeypatch.setattr("tasks.document_indexing_sync_task.build_document_indexing_service", build_runner)

    document_indexing_sync_task(dataset_id, document_id)

    assert clean_calls == [document_id]
    assert indexing_calls == [document_id]
    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document_id)
    assert persisted.indexing_status == IndexingStatus.COMPLETED
    assert persisted.error is None
    assert persisted.indexing_claim_id is None
    assert persisted.data_source_info_dict["last_edited_time"] == "2024-01-01T00:00:00Z"


def test_transient_notion_heartbeat_database_error_does_not_stop_owned_sync(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    old_segment = DocumentSegment(
        tenant_id=document.tenant_id,
        dataset_id=document.dataset_id,
        document_id=document.id,
        position=1,
        content="old content",
        word_count=2,
        tokens=2,
        created_by=document.created_by,
        index_node_id="old-node",
        status=SegmentStatus.COMPLETED,
    )
    sqlite_session.add(old_segment)
    sqlite_session.commit()

    cleanup_started = threading.Event()
    transient_failure_observed = threading.Event()
    retry_succeeded = threading.Event()
    renewal_attempts: list[str] = []
    original_renew = SQLAlchemyDocumentRepository.renew_notion_sync

    def renew_with_one_transient_failure(
        repository: SQLAlchemyDocumentRepository,
        claim: ClaimedNotionDocument,
    ) -> bool:
        assert cleanup_started.wait(1)
        renewal_attempts.append(claim.claim_id)
        if len(renewal_attempts) == 1:
            transient_failure_observed.set()
            raise OperationalError("renew notion sync", {}, RuntimeError("database temporarily unavailable"))
        renewed = original_renew(repository, claim)
        retry_succeeded.set()
        return renewed

    monkeypatch.setattr(SQLAlchemyDocumentRepository, "renew_notion_sync", renew_with_one_transient_failure)
    monkeypatch.setattr(
        document_indexing_sync_task_module,
        "_NotionSyncLeaseHeartbeat",
        lambda repository, claim: _NotionSyncLeaseHeartbeat(repository, claim, interval_seconds=0.01),
    )

    credentials = _TaskCredentials(_RecordingStoredCredentials({"integration_secret": "token"}))

    class ChangedNotionExtractor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_notion_last_edited_time(self) -> str:
            return "2024-02-01T00:00:00Z"

    clean_calls: list[list[str]] = []

    def clean_indexes(
        *,
        document_ids: list[str],
        new_session: Callable[[], Session],
        mutation_guard: Callable[[Session], None],
        **_kwargs: object,
    ) -> None:
        cleanup_started.set()
        assert transient_failure_observed.wait(1)
        with new_session() as cleanup_session, cleanup_session.begin():
            mutation_guard(cleanup_session)
            cleanup_session.execute(delete(DocumentSegment).where(DocumentSegment.document_id.in_(document_ids)))
        assert retry_succeeded.wait(1)
        clean_calls.append(document_ids)

    indexing_calls: list[tuple[str, str]] = []

    class CompletingRunner:
        def __init__(self, repository: SQLAlchemyDocumentRepository) -> None:
            self._repository = repository

        def run_claimed(self, ref: DocumentRef, *, claim_id: str) -> None:
            indexing_calls.append((ref.document_id, claim_id))
            self._repository.complete_indexing(
                ref,
                claim_id=claim_id,
                tokens=0,
                latency=0,
                source_version="2024-02-01T00:00:00Z",
            )

    def build_runner(*, session_factory: sessionmaker[Session], **_kwargs: object) -> CompletingRunner:
        return CompletingRunner(SQLAlchemyDocumentRepository(session_factory=session_factory))

    monkeypatch.setattr(
        document_indexing_sync_task_module,
        "build_data_source_credentials",
        lambda **_kwargs: credentials,
    )
    monkeypatch.setattr(document_indexing_sync_task_module, "NotionExtractor", ChangedNotionExtractor)
    monkeypatch.setattr(document_indexing_sync_task_module, "clean_document_for_reindex", clean_indexes)
    monkeypatch.setattr(document_indexing_sync_task_module, "build_document_indexing_service", build_runner)

    document_indexing_sync_task(dataset_id, document_id)

    assert len(renewal_attempts) >= 2
    assert clean_calls == [[document_id]]
    assert len(indexing_calls) == 1
    assert indexing_calls[0][0] == document_id
    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document_id)
    assert persisted.indexing_status == IndexingStatus.COMPLETED
    assert persisted.indexing_claim_id is None
    assert persisted.data_source_info_dict["last_edited_time"] == "2024-02-01T00:00:00Z"
    assert sqlite_session.query(DocumentSegment).count() == 0


def test_persisted_notion_heartbeat_claim_loss_stops_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    old_segment = DocumentSegment(
        tenant_id=document.tenant_id,
        dataset_id=document.dataset_id,
        document_id=document.id,
        position=1,
        content="old content",
        word_count=2,
        tokens=2,
        created_by=document.created_by,
        index_node_id="old-node",
        status=SegmentStatus.COMPLETED,
    )
    sqlite_session.add(old_segment)
    sqlite_session.commit()

    ownership_replaced = threading.Event()
    original_renew = SQLAlchemyDocumentRepository.renew_notion_sync

    def replace_owner_before_renewal(
        repository: SQLAlchemyDocumentRepository,
        claim: ClaimedNotionDocument,
    ) -> bool:
        with sessionmaker(bind=sqlite_session.get_bind()).begin() as takeover_session:
            persisted = takeover_session.get_one(Document, document_id)
            persisted.indexing_claim_id = "replacement-owner"
            persisted.indexing_claim_expires_at = datetime.now() + timedelta(minutes=5)
        ownership_replaced.set()
        return original_renew(repository, claim)

    monkeypatch.setattr(SQLAlchemyDocumentRepository, "renew_notion_sync", replace_owner_before_renewal)
    monkeypatch.setattr(
        document_indexing_sync_task_module,
        "_NotionSyncLeaseHeartbeat",
        lambda repository, claim: _NotionSyncLeaseHeartbeat(repository, claim, interval_seconds=0.01),
    )

    credentials = _TaskCredentials(_RecordingStoredCredentials({"integration_secret": "token"}))

    class BlockingNotionExtractor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_notion_last_edited_time(self) -> str:
            assert ownership_replaced.wait(1)
            return "2024-02-01T00:00:00Z"

    clean_calls: list[list[str]] = []
    indexing_calls: list[str] = []

    monkeypatch.setattr(
        document_indexing_sync_task_module,
        "build_data_source_credentials",
        lambda **_kwargs: credentials,
    )
    monkeypatch.setattr(document_indexing_sync_task_module, "NotionExtractor", BlockingNotionExtractor)
    monkeypatch.setattr(
        document_indexing_sync_task_module,
        "clean_document_for_reindex",
        lambda *, document_ids, **_kwargs: clean_calls.append(document_ids),
    )
    monkeypatch.setattr(
        document_indexing_sync_task_module,
        "build_document_indexing_service",
        lambda **_kwargs: indexing_calls.append("built"),
    )

    document_indexing_sync_task(dataset_id, document_id)

    assert clean_calls == []
    assert indexing_calls == []
    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document_id)
    assert persisted.indexing_status == IndexingStatus.COMPLETED
    assert persisted.indexing_claim_id == "replacement-owner"
    assert sqlite_session.query(DocumentSegment).count() == 1


def test_active_notion_sync_heartbeat_renews_the_persisted_lease(
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    repository = SQLAlchemyDocumentRepository(session_factory=sessionmaker(bind=sqlite_session.get_bind()))
    claim = repository.claim_notion_sync(
        DatasetRef(document.tenant_id, dataset_id).document(document_id),
        claim_id="55555555-5555-5555-5555-555555555555",
    )
    assert claim is not None
    with sessionmaker(bind=sqlite_session.get_bind()).begin() as session:
        database_now = session.scalar(select(func.current_timestamp()))
        assert isinstance(database_now, datetime)
        current = session.get_one(Document, document_id)
        current.indexing_claim_expires_at = database_now + timedelta(seconds=1)
        initial_expiry = current.indexing_claim_expires_at
        assert initial_expiry is not None

    renewed = threading.Event()

    class ObservedRepository:
        def renew_notion_sync(self, claim: ClaimedNotionDocument) -> bool:
            result = repository.renew_notion_sync(claim)
            renewed.set()
            return result

    with _NotionSyncLeaseHeartbeat(ObservedRepository(), claim, interval_seconds=0.01):
        assert renewed.wait(1)

    sqlite_session.expire_all()
    persisted = sqlite_session.get_one(Document, document_id)
    assert persisted.indexing_claim_id == claim.claim_id
    assert persisted.indexing_claim_expires_at is not None
    assert persisted.indexing_claim_expires_at > initial_expiry
    assert sqlite_session.query(DocumentSegment).count() == 0


def test_notion_sync_heartbeat_exit_is_bounded_when_renewal_blocks(
    sqlite_session: Session,
    document: Document,
    dataset_id: str,
    document_id: str,
) -> None:
    del sqlite_session
    entered = threading.Event()
    release = threading.Event()

    class BlockingRepository:
        def renew_notion_sync(self, claim: ClaimedNotionDocument) -> bool:
            del claim
            entered.set()
            release.wait(1)
            return False

    claim = ClaimedNotionDocument(
        ref=DatasetRef(document.tenant_id, dataset_id).document(document_id),
        claim_id="66666666-6666-6666-6666-666666666666",
        workspace_id="workspace-1",
        page_id="page-1",
        page_type="page",
        page_edited_time=None,
        credential_id=None,
        source_info={
            "notion_workspace_id": "workspace-1",
            "notion_page_id": "page-1",
            "type": "page",
        },
        index_type="text_model",
        recovery_required=False,
    )
    heartbeat = _NotionSyncLeaseHeartbeat(BlockingRepository(), claim, interval_seconds=0.01)
    heartbeat.__enter__()
    assert entered.wait(1)
    started = time.perf_counter()
    heartbeat.__exit__(None, None, None)
    elapsed = time.perf_counter() - started
    release.set()

    assert elapsed < 0.5

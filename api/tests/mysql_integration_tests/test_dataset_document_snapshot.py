import datetime
import json
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.exceptions import NotFound

from core.rag.index_processor.constant.index_type import IndexStructureType
from enums import DeploymentEdition
from extensions.storage.storage_type import StorageType
from models import Account
from models.account import AccountStatus, Tenant, TenantAccountJoin, TenantAccountRole, TenantStatus
from models.dataset import Dataset, Document
from models.enums import CreatorUserRole, DataSourceType, DocumentCreatedFrom, IndexingStatus
from models.model import DocumentIndexingOutbox, UploadFile
from repositories.knowledge.document_indexing_outbox_repository import DocumentIndexingAction
from services.knowledge import dataset_service as dataset_service_module
from services.knowledge.dataset_service import DocumentService
from services.knowledge.entities.knowledge_entities import DataSource, FileInfo, InfoList, KnowledgeConfig

FIXED_CREATED_AT = datetime.datetime(2024, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class DocumentCreationContext:
    account_id: str
    tenant_id: str
    dataset_id: str
    upload_file_id: str
    upload_file_name: str


class CommitWhileWaitingLock:
    """Run a committed concurrent write when the caller reaches lock acquisition."""

    def __init__(self, action: Callable[[], None]) -> None:
        self._action = action

    def __enter__(self) -> "CommitWhileWaitingLock":
        self._action()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def _create_document_context(session: Session) -> DocumentCreationContext:
    account = Account(
        email=f"owner-{uuid4()}@example.com",
        name="Owner",
        interface_language="en-US",
        status=AccountStatus.ACTIVE,
    )
    tenant = Tenant(name=f"tenant-{uuid4()}", status=TenantStatus.NORMAL)
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
        tenant_id=tenant.id,
        name=f"dataset-{uuid4()}",
        data_source_type=DataSourceType.UPLOAD_FILE,
        created_by=account.id,
        maintainer=account.id,
    )
    upload_file = UploadFile(
        tenant_id=tenant.id,
        storage_type=StorageType.LOCAL,
        key=f"uploads/{uuid4()}",
        name="snapshot-race.txt",
        size=128,
        extension="txt",
        mime_type="text/plain",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=account.id,
        created_at=FIXED_CREATED_AT,
        used=False,
    )
    session.add_all([dataset, upload_file])
    session.commit()
    return DocumentCreationContext(
        account_id=account.id,
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        upload_file_id=upload_file.id,
        upload_file_name=upload_file.name,
    )


def _create_document(session: Session, context: DocumentCreationContext) -> Document:
    document = Document(
        tenant_id=context.tenant_id,
        dataset_id=context.dataset_id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info=json.dumps({"upload_file_id": context.upload_file_id}),
        batch=f"batch-{uuid4()}",
        name=context.upload_file_name,
        created_from=DocumentCreatedFrom.WEB,
        created_by=context.account_id,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    document.indexing_status = IndexingStatus.COMPLETED
    document.enabled = True
    document.completed_at = FIXED_CREATED_AT
    session.add(document)
    session.commit()
    return document


def _creation_config(context: DocumentCreationContext) -> KnowledgeConfig:
    return KnowledgeConfig(
        indexing_technique="economy",
        duplicate=True,
        data_source=DataSource(
            info_list=InfoList(
                data_source_type="upload_file",
                file_info_list=FileInfo(file_ids=[context.upload_file_id]),
            )
        ),
    )


def _repeatable_read_sessions(mysql_session: Session) -> sessionmaker[Session]:
    engine = mysql_session.get_bind().execution_options(isolation_level="REPEATABLE READ")
    return sessionmaker(bind=engine, expire_on_commit=False)


def _load_creation_state(session: Session, context: DocumentCreationContext) -> tuple[Account, Tenant, Dataset]:
    account = session.get(Account, context.account_id)
    tenant = session.get(Tenant, context.tenant_id)
    dataset = session.scalar(
        select(Dataset).where(
            Dataset.id == context.dataset_id,
            Dataset.tenant_id == context.tenant_id,
        )
    )
    assert account is not None
    assert tenant is not None
    assert dataset is not None
    account.set_current_tenant_with_session(tenant, session=session)
    return account, tenant, dataset


def test_save_document_with_dataset_id_refreshes_snapshot_after_waiting_for_creation_lock(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _create_document_context(mysql_session)
    sessions = _repeatable_read_sessions(mysql_session)
    concurrent_document_ids: list[str] = []
    dispatched_intent_ids: list[tuple[str, ...]] = []

    def commit_duplicate_document() -> None:
        with sessions() as writer:
            document = _create_document(writer, context)
            concurrent_document_ids.append(document.id)

    monkeypatch.setattr(dataset_service_module.dify_config, "DEPLOYMENT_EDITION", DeploymentEdition.COMMUNITY)
    monkeypatch.setattr(
        dataset_service_module.redis_client,
        "lock",
        lambda *_args, **_kwargs: CommitWhileWaitingLock(commit_duplicate_document),
    )
    monkeypatch.setattr(DocumentService, "dispatch_document_indexing", dispatched_intent_ids.append)

    with sessions() as caller:
        account, _tenant, dataset = _load_creation_state(caller, context)

        # Establish the ordinary REPEATABLE READ snapshot before lock acquisition.
        assert caller.scalar(select(func.count(Document.id)).where(Document.dataset_id == context.dataset_id)) == 0

        with DocumentService.save_document_with_dataset_id(
            dataset,
            _creation_config(context),
            account,
            session=caller,
        ) as (documents, _batch):
            returned_document_ids = [document.id for document in documents]

    assert returned_document_ids == concurrent_document_ids
    assert len(dispatched_intent_ids) == 1
    (duplicate_intent_id,) = dispatched_intent_ids[0]
    with sessions() as verifier:
        assert verifier.scalar(select(func.count(Document.id)).where(Document.dataset_id == context.dataset_id)) == 1
        intent = verifier.get(DocumentIndexingOutbox, duplicate_intent_id)
        assert intent is not None
        assert intent.tenant_id == context.tenant_id
        assert intent.dataset_id == context.dataset_id
        assert intent.action == DocumentIndexingAction.DUPLICATE
        assert json.loads(intent.document_ids) == concurrent_document_ids
        stored_document = verifier.get(Document, concurrent_document_ids[0])
        assert stored_document is not None
        assert stored_document.indexing_request_id == duplicate_intent_id


def test_save_document_with_dataset_id_revalidates_dataset_after_waiting_for_creation_lock(
    mysql_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _create_document_context(mysql_session)
    sessions = _repeatable_read_sessions(mysql_session)
    dispatched_intent_ids: list[tuple[str, ...]] = []

    def delete_dataset() -> None:
        with sessions.begin() as writer:
            writer.execute(
                delete(Dataset).where(
                    Dataset.id == context.dataset_id,
                    Dataset.tenant_id == context.tenant_id,
                )
            )

    monkeypatch.setattr(dataset_service_module.dify_config, "DEPLOYMENT_EDITION", DeploymentEdition.COMMUNITY)
    monkeypatch.setattr(
        dataset_service_module.redis_client,
        "lock",
        lambda *_args, **_kwargs: CommitWhileWaitingLock(delete_dataset),
    )
    monkeypatch.setattr(DocumentService, "dispatch_document_indexing", dispatched_intent_ids.append)

    with sessions() as caller:
        account, _tenant, dataset = _load_creation_state(caller, context)

        # Pin a stale snapshot in which the dataset still exists.
        assert caller.scalar(select(func.count(Document.id)).where(Document.dataset_id == context.dataset_id)) == 0

        with pytest.raises(NotFound, match="Dataset not found"):
            with DocumentService.save_document_with_dataset_id(
                dataset,
                _creation_config(context),
                account,
                session=caller,
            ):
                pytest.fail("deleted dataset must be rejected before document creation")

    assert dispatched_intent_ids == []
    with sessions() as verifier:
        assert verifier.get(Dataset, context.dataset_id) is None
        assert verifier.scalar(select(func.count(Document.id)).where(Document.dataset_id == context.dataset_id)) == 0
        assert (
            verifier.scalar(
                select(func.count(DocumentIndexingOutbox.id)).where(
                    DocumentIndexingOutbox.dataset_id == context.dataset_id
                )
            )
            == 0
        )

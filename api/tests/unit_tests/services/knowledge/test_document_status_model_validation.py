from collections.abc import Sequence
from datetime import datetime
from inspect import unwrap
from typing import Literal
from uuid import uuid4

import pytest
from flask import Flask
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from controllers.service_api.dataset import dataset as dataset_controller
from controllers.service_api.dataset.error import InvalidActionError
from machinery.context import RequestContext
from models.account import TenantAccountJoin, TenantAccountRole
from models.dataset import Dataset, Document, DocumentSegment
from models.enums import IndexingStatus, SegmentStatus
from models.model import SegmentStateOutbox
from services.knowledge.dataset_service import DatasetService
from services.knowledge.documents.adapters import SQLAlchemyDocumentOperations
from services.knowledge.documents.application import DocumentInvalidActionError
from services.knowledge.entities.resource_scope import DatasetRef
from tests.unit_tests.model_factories import make_account, make_tenant

Action = Literal["enable", "disable", "archive", "un_archive"]


class _UnavailableModel:
    def __init__(self) -> None:
        self.checked: list[str] = []

    def __call__(self, dataset: Dataset) -> None:
        self.checked.append(dataset.id)
        raise ValueError("embedding credential is unavailable")


class _NoIndexingHint:
    def get(self, _key: str) -> None:
        return None


class _DocumentEnableDispatch:
    def delay(self, _document_id: str) -> None:
        return None


class _UnexpectedDatasourceCredential:
    def resolve(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        credential_id: str,
        provider: str,
        plugin_id: str,
    ) -> dict[str, object]:
        del workspace_id, actor_id, credential_id, provider, plugin_id
        raise AssertionError("Changing document status does not resolve datasource credentials")


@pytest.mark.parametrize("entry", ["console", "service_api"])
@pytest.mark.parametrize(
    ("action", "enabled", "archived", "requires_model"),
    [
        ("disable", True, False, False),
        ("archive", True, False, False),
        ("un_archive", False, True, False),
        ("enable", True, False, False),
        ("enable", False, True, False),
        ("enable", False, False, True),
        ("un_archive", True, True, True),
    ],
    ids=[
        "disable",
        "archive",
        "unarchive-disabled",
        "enable-noop",
        "enable-archived",
        "enable-requires-model",
        "unarchive-enabled-requires-model",
    ],
)
def test_status_entries_validate_models_only_when_restoring_the_index(
    app: Flask,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    action: Action,
    enabled: bool,
    archived: bool,
    requires_model: bool,
) -> None:
    tenant = make_tenant(tenant_id=str(uuid4()))
    account = make_account(account_id=str(uuid4()), tenant=tenant, role=TenantAccountRole.OWNER)
    dataset = Dataset(
        id=str(uuid4()),
        tenant_id=tenant.id,
        name="Dataset",
        created_by=account.id,
        maintainer=account.id,
        indexing_technique="high_quality",
        embedding_model="revoked-model",
        embedding_model_provider="revoked-provider",
        chunk_structure="text_model",
    )
    document = Document(
        id=str(uuid4()),
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        position=1,
        data_source_type="upload_file",
        data_source_info="{}",
        batch="status-model-validation",
        name="Document",
        created_from="api",
        created_by=account.id,
        indexing_status=IndexingStatus.COMPLETED,
        completed_at=datetime(2026, 1, 1),
        enabled=enabled,
        archived=archived,
        doc_form="text_model",
    )
    segment = DocumentSegment(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="content",
        word_count=1,
        tokens=1,
        created_by=account.id,
        enabled=enabled and not archived,
        status=SegmentStatus.COMPLETED,
        index_node_id="existing-body-node",
    )
    with sqlite_session_factory.begin() as session:
        session.add_all(
            [
                tenant,
                account,
                TenantAccountJoin(tenant_id=tenant.id, account_id=account.id, role=TenantAccountRole.OWNER),
                dataset,
                document,
                segment,
            ]
        )

    model = _UnavailableModel()
    dispatched: list[str] = []

    def publish(intent_ids: Sequence[str]) -> None:
        dispatched.extend(intent_ids)

    monkeypatch.setattr(DatasetService, "check_dataset_model_setting", model)
    monkeypatch.setattr("services.knowledge.dataset_service.redis_client", _NoIndexingHint())
    monkeypatch.setattr("services.knowledge.dataset_service.add_document_to_index_task", _DocumentEnableDispatch())
    monkeypatch.setattr("services.knowledge.dataset_service.publish_segment_state_intents", publish)
    monkeypatch.setattr(dataset_controller, "current_user", account)

    def change_status() -> None:
        if entry == "console":
            operations = SQLAlchemyDocumentOperations(
                session_factory=sqlite_session_factory,
                actor_credentials=_UnexpectedDatasourceCredential(),
            )
            operations.change_status(
                RequestContext("status-request", None, account.id, tenant.id),
                DatasetRef(tenant.id, dataset.id),
                [document.id],
                action,
            )
            return
        with (
            app.test_request_context(
                f"/datasets/{dataset.id}/documents/status/{action}",
                method="PATCH",
                json={"document_ids": [document.id]},
            ),
            sqlite_session_factory() as session,
        ):
            response, status = unwrap(dataset_controller.DocumentStatusApi.patch)(
                dataset_controller.DocumentStatusApi(),
                session=session,
                tenant_id=tenant.id,
                dataset_id=dataset.id,
                action=action,
            )
            assert (response, status) == ({"result": "success"}, 200)

    if requires_model:
        error_type = DocumentInvalidActionError if entry == "console" else InvalidActionError
        with pytest.raises(error_type, match="embedding credential is unavailable"):
            change_status()
        assert model.checked == [dataset.id]
    else:
        change_status()
        assert model.checked == []

    with sqlite_session_factory() as session:
        current_document = session.get_one(Document, document.id)
        if requires_model:
            assert (current_document.enabled, current_document.archived) == (enabled, archived)
            assert session.scalar(select(SegmentStateOutbox)) is None
            assert dispatched == []
        else:
            assert current_document.enabled == (action == "enable" or (enabled and action != "disable"))
            assert current_document.archived == (action == "archive" or (archived and action != "un_archive"))
            if action in {"disable", "archive"}:
                intent = session.scalar(select(SegmentStateOutbox))
                assert intent is not None
                assert intent.action == "disable"
                assert dispatched == [intent.id]

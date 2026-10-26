from unittest.mock import create_autospec

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session, sessionmaker

from constants.resource_access_token import ResourceAccessTokenResourceType
from models.account import Account, Tenant, TenantAccountJoin, TenantAccountRole
from models.dataset import Dataset, DatasetPermission
from models.enums import ApiTokenType
from models.model import ApiToken, DatasetApiTokenBinding, ResourceCleanupOutbox
from services.knowledge import dataset_service as module
from services.knowledge.dataset_service import DatasetService
from services.resource_access_token_service import ResourceAccessTokenCleanupService


def _persist_scope(session: Session) -> tuple[Account, Dataset, ApiToken]:
    tenant = Tenant(name="Workspace")
    account = Account(name="Owner", email="owner@example.com")
    dataset = Dataset(id="dataset", tenant_id=tenant.id, name="Dataset", created_by=account.id)
    key = ApiToken(id="key", tenant_id=tenant.id, type=ApiTokenType.DATASET, token="dataset-secret")
    session.add_all(
        [
            tenant,
            account,
            TenantAccountJoin(
                tenant_id=tenant.id,
                account_id=account.id,
                current=True,
                role=TenantAccountRole.OWNER,
            ),
            dataset,
            key,
        ]
    )
    session.flush()
    session.add(DatasetApiTokenBinding(api_token_id=key.id, dataset_id=dataset.id))
    session.add(DatasetPermission(tenant_id=tenant.id, dataset_id=dataset.id, account_id=account.id))
    session.commit()
    account.set_current_tenant_with_session(tenant, session=session)
    return account, dataset, key


def test_delete_dataset_commits_key_cleanup_before_invalidating_cache(
    sqlite_session: Session,
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account, dataset, key = _persist_scope(sqlite_session)
    cleanup = create_autospec(ResourceAccessTokenCleanupService, instance=True, spec_set=True)
    invalidated: list[tuple[str, str]] = []
    enqueued: list[str] = []
    monkeypatch.setattr(module, "enqueue_resource_cleanup_intent", enqueued.append)

    def build_cleanup(*, session: Session) -> ResourceAccessTokenCleanupService:
        assert session is sqlite_session
        return cleanup

    monkeypatch.setattr(module, "build_resource_access_token_cleanup_service", build_cleanup)

    def invalidate(token: str, scope: str | None = None) -> bool:
        with sqlite_session_factory() as committed_session:
            assert committed_session.get(Dataset, dataset.id) is None
            assert committed_session.get(ApiToken, key.id) is None
            assert committed_session.scalar(select(ResourceCleanupOutbox.id)) is not None
        invalidated.append((token, scope or ""))
        return True

    monkeypatch.setattr(module.ApiTokenCache, "delete", invalidate)

    assert DatasetService.delete_dataset(dataset.id, account, session=sqlite_session)

    cleanup.delete_resource_relations.assert_called_once_with(
        tenant_id=dataset.tenant_id,
        resource_type=ResourceAccessTokenResourceType.KNOWLEDGE,
        resource_id=dataset.id,
    )
    assert invalidated == [(key.token, key.type)]
    assert len(enqueued) == 1
    assert (
        sqlite_session.scalar(
            select(DatasetPermission.id).where(
                DatasetPermission.tenant_id == dataset.tenant_id,
                DatasetPermission.dataset_id == dataset.id,
            )
        )
        is None
    )


def test_delete_dataset_rolls_back_key_cleanup_with_relation_failure(
    sqlite_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account, dataset, key = _persist_scope(sqlite_session)
    cleanup = create_autospec(ResourceAccessTokenCleanupService, instance=True, spec_set=True)
    cleanup.delete_resource_relations.side_effect = RuntimeError("relation cleanup failed")
    invalidated: list[tuple[str, str | None]] = []
    enqueued: list[str] = []
    monkeypatch.setattr(module, "enqueue_resource_cleanup_intent", enqueued.append)

    def build_cleanup(*, session: Session) -> ResourceAccessTokenCleanupService:
        assert session is sqlite_session
        return cleanup

    monkeypatch.setattr(module, "build_resource_access_token_cleanup_service", build_cleanup)
    monkeypatch.setattr(
        module.ApiTokenCache,
        "delete",
        lambda token, scope=None: invalidated.append((token, scope)),
    )

    with pytest.raises(RuntimeError, match="relation cleanup failed"):
        DatasetService.delete_dataset(dataset.id, account, session=sqlite_session)
    sqlite_session.rollback()

    assert sqlite_session.get(Dataset, dataset.id) is not None
    assert sqlite_session.get(ApiToken, key.id) is not None
    assert (
        sqlite_session.scalar(select(DatasetApiTokenBinding).where(DatasetApiTokenBinding.api_token_id == key.id))
        is not None
    )
    assert (
        sqlite_session.scalar(
            select(DatasetPermission.id).where(
                DatasetPermission.tenant_id == dataset.tenant_id,
                DatasetPermission.dataset_id == dataset.id,
            )
        )
        is not None
    )
    assert invalidated == []
    assert enqueued == []
    assert sqlite_session.scalar(select(ResourceCleanupOutbox)) is None


def test_delete_dataset_commit_failure_rolls_back_cleanup_intent(
    sqlite_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account, dataset, _ = _persist_scope(sqlite_session)
    cleanup = create_autospec(ResourceAccessTokenCleanupService, instance=True, spec_set=True)
    enqueued: list[str] = []
    monkeypatch.setattr(module, "enqueue_resource_cleanup_intent", enqueued.append)

    def build_cleanup(*, session: Session) -> ResourceAccessTokenCleanupService:
        assert session is sqlite_session
        return cleanup

    monkeypatch.setattr(module, "build_resource_access_token_cleanup_service", build_cleanup)

    def fail_commit(_session: Session) -> None:
        raise RuntimeError("commit failed")

    event.listen(sqlite_session, "before_commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            DatasetService.delete_dataset(dataset.id, account, session=sqlite_session)
    finally:
        event.remove(sqlite_session, "before_commit", fail_commit)
        sqlite_session.rollback()

    assert sqlite_session.get(Dataset, dataset.id) is not None
    assert sqlite_session.scalar(select(ResourceCleanupOutbox)) is None
    assert enqueued == []

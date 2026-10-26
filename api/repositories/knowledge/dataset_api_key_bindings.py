"""Persistence helpers for dataset (knowledge base) API key scoping.

A dataset API key is scoped to specific knowledge bases through
``DatasetApiTokenBinding`` rows:

- no rows  -> the key can access every dataset in its tenant (default / back-compat)
- N rows   -> the key is restricted to exactly those datasets

Shared by key management, dataset deletion, and service-API authentication.
The caller owns the transaction boundary (commit/flush).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import Select, delete, select
from sqlalchemy.orm import Session

from models.account import Tenant
from models.dataset import Dataset
from models.enums import ApiTokenType
from models.model import ApiToken, DatasetApiTokenBinding


@dataclass(frozen=True, slots=True)
class DeletedDatasetApiKey:
    id: str
    token: str
    type: ApiTokenType


def lock_workspace(session: Session, workspace_id: str) -> bool:
    """Serialize dataset-key creation and dataset deletion in one workspace."""
    return session.scalar(select(Tenant.id).where(Tenant.id == workspace_id).with_for_update()) is not None


def get_bound_dataset_ids(session: Session, api_token_id: str) -> set[str]:
    """Return the dataset ids a key is scoped to (empty set means unrestricted)."""
    return set(
        session.scalars(
            select(DatasetApiTokenBinding.dataset_id).where(DatasetApiTokenBinding.api_token_id == api_token_id)
        ).all()
    )


def get_active_token_dataset_ids(
    session: Session,
    *,
    api_token_id: str,
    token: str,
    tenant_id: str | None,
) -> set[str] | None:
    """Return an active dataset key's scope, distinguishing missing from unrestricted."""
    if tenant_id is None:
        return None
    rows = session.execute(
        select(DatasetApiTokenBinding.dataset_id)
        .select_from(ApiToken)
        .outerjoin(DatasetApiTokenBinding, DatasetApiTokenBinding.api_token_id == ApiToken.id)
        .where(
            ApiToken.id == api_token_id,
            ApiToken.token == token,
            ApiToken.tenant_id == tenant_id,
            ApiToken.type == ApiTokenType.DATASET,
        )
    ).all()
    if not rows:
        return None
    return {str(dataset_id) for (dataset_id,) in rows if dataset_id is not None}


def list_bindings_by_token(session: Session, token_ids: Iterable[str]) -> dict[str, list[str]]:
    """Group bound dataset ids by api token id for the given tokens."""
    ids = list(token_ids)
    bindings_by_token: dict[str, list[str]] = {}
    if not ids:
        return bindings_by_token
    rows = session.execute(
        select(DatasetApiTokenBinding.api_token_id, DatasetApiTokenBinding.dataset_id).where(
            DatasetApiTokenBinding.api_token_id.in_(ids)
        )
    ).all()
    for token_id, dataset_id in rows:
        bindings_by_token.setdefault(str(token_id), []).append(str(dataset_id))
    return bindings_by_token


def find_unknown_dataset_ids(session: Session, dataset_ids: list[str], tenant_id: str) -> list[str]:
    """Return dataset ids that do not belong to the tenant, using a current read.

    Key creation calls this after locking the tenant row. The shared locks make a
    creator that waited for dataset deletion observe the deletion under MySQL's
    ``REPEATABLE READ`` isolation instead of an older consistent-read snapshot.
    """
    if not dataset_ids:
        return []
    existing = set(
        session.scalars(
            select(Dataset.id)
            .where(Dataset.id.in_(dataset_ids), Dataset.tenant_id == tenant_id)
            .with_for_update(read=True)
        ).all()
    )
    return [dataset_id for dataset_id in dataset_ids if dataset_id not in existing]


def bind_datasets(session: Session, api_token_id: str, dataset_ids: Iterable[str]) -> None:
    """Add one binding row per dataset id. The caller controls the transaction."""
    for dataset_id in dataset_ids:
        session.add(DatasetApiTokenBinding(api_token_id=api_token_id, dataset_id=dataset_id))


def delete_keys_scoped_only_to(session: Session, dataset_id: str) -> list[DeletedDatasetApiKey]:
    """Delete dataset API keys whose only binding is ``dataset_id``.

    A deleted knowledge base cascades its bindings away. A key bound solely to it
    would otherwise drop to zero bindings and silently become unrestricted
    (access-all) under the "no bindings = access all" rule. To keep scope from
    escalating, delete those keys (their remaining rows cascade with the token).
    Keys also bound to other datasets keep those bindings and survive.

    Must run before the dataset row is deleted (so the bindings still exist).
    Concurrent dataset deletion callers must hold the workspace lock first.
    Returns the deleted keys so callers can invalidate their caches after commit.
    """
    # Both reads are locking current reads. Dataset deletion can begin after an
    # earlier consistent read established a stale REPEATABLE READ snapshot. In
    # that case, the old ``NOT IN`` query could still see a binding already
    # removed by a preceding deletion and preserve a key whose last live binding
    # is about to disappear.
    candidate_ids = list(
        session.scalars(
            select(DatasetApiTokenBinding.api_token_id)
            .where(DatasetApiTokenBinding.dataset_id == dataset_id)
            .with_for_update()
        ).all()
    )
    if not candidate_ids:
        return []

    current_bindings = session.execute(
        select(DatasetApiTokenBinding.api_token_id, DatasetApiTokenBinding.dataset_id)
        .where(DatasetApiTokenBinding.api_token_id.in_(candidate_ids))
        .with_for_update()
    ).all()
    bound_dataset_ids: dict[str, set[str]] = {}
    for api_token_id, bound_dataset_id in current_bindings:
        bound_dataset_ids.setdefault(str(api_token_id), set()).add(str(bound_dataset_id))
    sole_token_ids = [
        api_token_id for api_token_id in candidate_ids if bound_dataset_ids.get(api_token_id) == {dataset_id}
    ]
    if not sole_token_ids:
        return []

    keys = list(session.scalars(select(ApiToken).where(ApiToken.id.in_(sole_token_ids)).with_for_update()).all())
    if keys:
        session.execute(delete(ApiToken).where(ApiToken.id.in_([key.id for key in keys])))
    return [DeletedDatasetApiKey(id=key.id, token=key.token, type=key.type) for key in keys]


def token_ids_scoped_only_to(dataset_id: str) -> Select[tuple[str]]:
    """Select keys bound only to one dataset, for management and deletion."""
    return select(DatasetApiTokenBinding.api_token_id).where(
        DatasetApiTokenBinding.dataset_id == dataset_id,
        DatasetApiTokenBinding.api_token_id.notin_(
            select(DatasetApiTokenBinding.api_token_id).where(DatasetApiTokenBinding.dataset_id != dataset_id)
        ),
    )

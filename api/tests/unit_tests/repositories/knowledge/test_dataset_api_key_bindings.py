"""Persistence tests for dataset API key scoping and deletion."""

import uuid

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from models.enums import ApiTokenType
from models.model import ApiToken, DatasetApiTokenBinding
from repositories.knowledge import dataset_api_key_bindings


def _dataset_token(session: Session) -> ApiToken:
    token = ApiToken(
        id=str(uuid.uuid4()),
        tenant_id=str(uuid.uuid4()),
        type=ApiTokenType.DATASET,
        token=f"dataset-{uuid.uuid4().hex}",
    )
    session.add(token)
    return token


class TestDeleteKeysScopedOnlyTo:
    def test_deletes_key_bound_only_to_the_dataset(self, sqlite_session: Session) -> None:
        dataset_id = str(uuid.uuid4())
        token = _dataset_token(sqlite_session)
        sqlite_session.add(DatasetApiTokenBinding(api_token_id=token.id, dataset_id=dataset_id))
        sqlite_session.commit()

        deleted = dataset_api_key_bindings.delete_keys_scoped_only_to(sqlite_session, dataset_id)
        sqlite_session.commit()

        assert deleted == [
            dataset_api_key_bindings.DeletedDatasetApiKey(id=token.id, token=token.token, type=token.type)
        ]
        assert sqlite_session.scalar(select(ApiToken).where(ApiToken.id == token.id)) is None

    def test_keeps_key_bound_to_other_datasets(self, sqlite_session: Session) -> None:
        dataset_id = str(uuid.uuid4())
        other_dataset_id = str(uuid.uuid4())
        token = _dataset_token(sqlite_session)
        sqlite_session.add(DatasetApiTokenBinding(api_token_id=token.id, dataset_id=dataset_id))
        sqlite_session.add(DatasetApiTokenBinding(api_token_id=token.id, dataset_id=other_dataset_id))
        sqlite_session.commit()

        deleted = dataset_api_key_bindings.delete_keys_scoped_only_to(sqlite_session, dataset_id)
        sqlite_session.commit()

        # The key is bound elsewhere, so it survives and stays scoped.
        assert deleted == []
        assert sqlite_session.scalar(select(ApiToken).where(ApiToken.id == token.id)) is not None

    def test_deletes_key_when_a_later_cleanup_sees_its_last_binding(self, sqlite_session: Session) -> None:
        first_dataset_id = str(uuid.uuid4())
        last_dataset_id = str(uuid.uuid4())
        token = _dataset_token(sqlite_session)
        sqlite_session.add_all(
            [
                DatasetApiTokenBinding(api_token_id=token.id, dataset_id=first_dataset_id),
                DatasetApiTokenBinding(api_token_id=token.id, dataset_id=last_dataset_id),
            ]
        )
        sqlite_session.commit()

        assert dataset_api_key_bindings.delete_keys_scoped_only_to(sqlite_session, first_dataset_id) == []
        sqlite_session.execute(
            delete(DatasetApiTokenBinding).where(DatasetApiTokenBinding.dataset_id == first_dataset_id)
        )
        sqlite_session.commit()

        deleted = dataset_api_key_bindings.delete_keys_scoped_only_to(sqlite_session, last_dataset_id)
        sqlite_session.commit()

        assert deleted == [
            dataset_api_key_bindings.DeletedDatasetApiKey(id=token.id, token=token.token, type=token.type)
        ]
        assert sqlite_session.scalar(select(ApiToken).where(ApiToken.id == token.id)) is None

    def test_ignores_unrestricted_keys(self, sqlite_session: Session) -> None:
        """A key with no bindings (access-all) must never be touched."""
        dataset_id = str(uuid.uuid4())
        unbound = _dataset_token(sqlite_session)
        sqlite_session.commit()

        deleted = dataset_api_key_bindings.delete_keys_scoped_only_to(sqlite_session, dataset_id)
        sqlite_session.commit()

        assert deleted == []
        assert sqlite_session.scalar(select(ApiToken).where(ApiToken.id == unbound.id)) is not None

"""
Unit tests for Service API wraps (authentication decorators)
"""

import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from flask import Flask, g
from sqlalchemy import select
from sqlalchemy.orm import Session, scoped_session
from werkzeug.exceptions import Forbidden, NotFound, ServiceUnavailable, Unauthorized

from controllers.service_api import wraps as wraps_module
from controllers.service_api.wraps import (
    DatasetApiResource,
    FetchUserArg,
    WhereisUserArg,
    cloud_edition_billing_knowledge_limit_check,
    cloud_edition_billing_rate_limit_check,
    cloud_edition_billing_resource_check,
    dataset_retrieval_admission,
    validate_and_get_api_token,
    validate_app_token,
    validate_dataset_token,
)
from core.db.session_factory import get_session_maker
from enums import CloudPlan, DeploymentEdition
from extensions.application_services.resource_access_token import build_resource_access_token_service
from machinery.context import RequestContext
from models import Account, Tenant, TenantAccountJoin
from models.account import TenantAccountRole
from models.dataset import Dataset, RateLimitLog
from models.enums import ApiTokenType, EndUserType
from models.model import ApiToken, App, AppMode, DatasetApiTokenBinding, EndUser, IconType
from models.resource_access_token import (
    ResourceAccessToken,
    ResourceAccessTokenRelation,
    ResourceAccessTokenResourceType,
)
from services.entities.feature_entities import KnowledgeRateLimitModel
from services.resource_access_token_service import ResourceAccessTokenService
from tests.unit_tests.config_override import config_overrides_context


class _RecordingEndUserCommands:
    def __init__(self, result: EndUser) -> None:
        self._result = result
        self.calls: list[tuple[str, str, str | None]] = []

    def get_or_create_end_user(self, tenant_id: str, app_id: str, user_id: str | None = None) -> EndUser:
        self.calls.append((tenant_id, app_id, user_id))
        return self._result


@dataclass(eq=False, repr=False)
class _AppScopedEndUserServicesStub:
    commands: _RecordingEndUserCommands


class _ApplicationServicesStub:
    def __init__(self, commands: _RecordingEndUserCommands) -> None:
        self.app_scoped_end_users = _AppScopedEndUserServicesStub(commands)


@dataclass(eq=False, repr=False)
class _ResourceAccessTokenServicesStub:
    resource_access_tokens: ResourceAccessTokenService


class _RequestLoginManager:
    def _update_request_context_with_user(self, user: Account) -> None:
        g._login_user = user


class _RateLimitRedisStub:
    def __init__(self, *, request_count: int = 0) -> None:
        self.request_count = request_count
        self.added_keys: list[str] = []
        self.trimmed_keys: list[str] = []

    def zadd(self, key: str, _values: dict[int, int]) -> None:
        self.added_keys.append(key)

    def zremrangebyscore(self, key: str, _minimum: int, _maximum: int) -> None:
        self.trimmed_keys.append(key)

    def zcard(self, _key: str) -> int:
        return self.request_count


@dataclass(eq=False, repr=False)
class _DatabaseEngineStub:
    engine: object


class _RecordingLoginManager:
    def __init__(self) -> None:
        self.users: list[EndUser] = []

    def _update_request_context_with_user(self, user: EndUser) -> None:
        self.users.append(user)


class _RecordingSignal:
    def __init__(self) -> None:
        self.calls: list[tuple[object, EndUser]] = []

    def send(self, sender: object, *, user: EndUser) -> None:
        self.calls.append((sender, user))


def _configure_current_app_mock(mock_current_app):
    mock_current_app.login_manager = Mock()
    mock_current_app._get_current_object = Mock(return_value=Mock())


@pytest.fixture(autouse=True)
def _application_services(monkeypatch: pytest.MonkeyPatch) -> None:
    class FeatureQueries:
        @staticmethod
        def get_workspace_vector_space(workspace_id: str):
            return wraps_module.FeatureService.get_vector_space(workspace_id)

    monkeypatch.setattr(
        wraps_module,
        "application_services",
        lambda: SimpleNamespace(feature_queries=FeatureQueries()),
    )


def _session_proxy(session: Session) -> scoped_session[Session]:
    """Expose the real SQLite session through Flask-SQLAlchemy's callable shape."""
    return scoped_session(lambda: session)


def _api_token(*, tenant_id: str, app_id: str | None = None, token_type: ApiTokenType) -> ApiToken:
    return ApiToken(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        app_id=app_id,
        type=token_type,
        token="test_token",
    )


def _persist_workspace(session: Session) -> tuple[Tenant, Account, TenantAccountJoin]:
    tenant = Tenant(name="Workspace")
    account = Account(name="Owner", email=f"owner-{uuid.uuid4()}@example.com")
    membership = TenantAccountJoin(
        tenant_id=tenant.id,
        account_id=account.id,
        current=True,
        role=TenantAccountRole.OWNER,
    )
    session.add_all([tenant, account, membership])
    session.commit()
    return tenant, account, membership


def _app_model(*, tenant_id: str, enable_api: bool = True) -> App:
    return App(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        name="Service API App",
        mode=AppMode.CHAT,
        icon_type=IconType.EMOJI,
        icon="chat",
        icon_background="#FFFFFF",
        enable_site=False,
        enable_api=enable_api,
    )


class TestValidateAndGetApiToken:
    """Test suite for validate_and_get_api_token function"""

    @pytest.fixture
    def app(self):
        """Create Flask test application."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    def test_missing_authorization_header(self, app: Flask):
        """Test that Unauthorized is raised when Authorization header is missing."""
        # Arrange
        with app.test_request_context("/", method="GET"):
            # No Authorization header

            # Act & Assert
            with pytest.raises(Unauthorized) as exc_info:
                validate_and_get_api_token("app")
            assert "Authorization header must be provided" in str(exc_info.value)

    def test_invalid_auth_scheme(self, app: Flask):
        """Test that Unauthorized is raised when auth scheme is not Bearer."""
        # Arrange
        with app.test_request_context("/", method="GET", headers={"Authorization": "Basic token123"}):
            # Act & Assert
            with pytest.raises(Unauthorized) as exc_info:
                validate_and_get_api_token("app")
            assert "Authorization header must be provided and start with 'Bearer'" in str(exc_info.value)

    @patch("controllers.service_api.wraps.record_token_usage")
    @patch("controllers.service_api.wraps.ApiTokenCache")
    @patch("controllers.service_api.wraps.fetch_token_with_single_flight")
    def test_valid_token_returns_api_token(self, mock_fetch_token, mock_cache_cls, mock_record_usage, app: Flask):
        """Test that valid token returns the ApiToken object."""
        # Arrange
        api_token = _api_token(
            tenant_id=str(uuid.uuid4()),
            app_id=str(uuid.uuid4()),
            token_type=ApiTokenType.APP,
        )
        api_token.token = "valid_token_123"

        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None  # Cache miss
        mock_cache_cls.get = mock_cache_instance.get
        mock_fetch_token.return_value = api_token

        # Act
        with app.test_request_context("/", method="GET", headers={"Authorization": "Bearer valid_token_123"}):
            result = validate_and_get_api_token("app")

        # Assert
        assert result == api_token

    @patch("controllers.service_api.wraps.record_token_usage")
    @patch("controllers.service_api.wraps.ApiTokenCache")
    @patch("controllers.service_api.wraps.fetch_token_with_single_flight")
    def test_invalid_token_raises_unauthorized(self, mock_fetch_token, mock_cache_cls, mock_record_usage, app: Flask):
        """Test that invalid token raises Unauthorized."""
        # Arrange
        from werkzeug.exceptions import Unauthorized

        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None  # Cache miss
        mock_cache_cls.get = mock_cache_instance.get
        mock_fetch_token.side_effect = Unauthorized("Access token is invalid")

        # Act & Assert
        with app.test_request_context("/", method="GET", headers={"Authorization": "Bearer invalid_token"}):
            with pytest.raises(Unauthorized) as exc_info:
                validate_and_get_api_token("app")
            assert "Access token is invalid" in str(exc_info.value)


class TestValidateAppToken:
    """Test suite for validate_app_token decorator"""

    @pytest.fixture
    def app(self):
        """Create Flask test application."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    @pytest.mark.parametrize("separator", ["\u00a0", "\u001c", "\u000b", "\u2003"])
    def test_non_http_whitespace_never_resolves_resource_token(self, app: Flask, separator: str) -> None:
        with (
            app.test_request_context("/", headers={"Authorization": f"Bearer{separator}dify_rt_token"}),
            patch.object(wraps_module, "application_services") as services,
            pytest.raises(Unauthorized),
        ):
            validate_app_token(lambda: "ok")()
        services.assert_not_called()

    @patch("controllers.service_api.wraps.user_logged_in")
    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.current_app")
    @pytest.mark.parametrize(
        "sqlite_session",
        [(App, ApiToken, Tenant, Account, TenantAccountJoin)],
        indirect=True,
    )
    def test_valid_app_token_allows_access(
        self,
        mock_current_app,
        mock_validate_token,
        mock_user_logged_in,
        app: Flask,
        sqlite_session: Session,
    ):
        """Test that valid app token allows access to decorated view."""
        # Arrange
        _configure_current_app_mock(mock_current_app)

        tenant, account, _ = _persist_workspace(sqlite_session)
        app_model = _app_model(tenant_id=tenant.id)
        api_token = _api_token(tenant_id=tenant.id, app_id=app_model.id, token_type=ApiTokenType.APP)
        sqlite_session.add_all([app_model, api_token])
        sqlite_session.commit()
        mock_validate_token.return_value = api_token

        @validate_app_token
        def protected_view(app_model):
            return {"success": True, "app_id": app_model.id}

        # Act
        with (
            app.test_request_context("/", method="GET", headers={"Authorization": "Bearer test_token"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            result = protected_view()

        # Assert
        assert result["success"] is True
        assert result["app_id"] == app_model.id
        assert account.current_tenant_id == tenant.id

    @pytest.mark.parametrize(
        ("fetch_from", "method", "path", "request_kwargs", "expected_user_id"),
        [
            pytest.param(WhereisUserArg.QUERY, "GET", "/?user=query-user", {}, "query-user", id="query"),
            pytest.param(
                WhereisUserArg.JSON,
                "POST",
                "/",
                {"json": {"user": "json-user"}},
                "json-user",
                id="json",
            ),
            pytest.param(
                WhereisUserArg.FORM,
                "POST",
                "/",
                {"data": {"user": "form-user"}},
                "form-user",
                id="form",
            ),
        ],
    )
    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @pytest.mark.parametrize("sqlite_session", [(App, Tenant)], indirect=True)
    def test_fetch_user_arg_resolves_and_injects_end_user(
        self,
        mock_validate_token,
        app: Flask,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
        fetch_from: WhereisUserArg,
        method: str,
        path: str,
        request_kwargs: dict[str, object],
        expected_user_id: str,
    ) -> None:
        tenant = Tenant(name="Workspace")
        app_model = _app_model(tenant_id=tenant.id)
        sqlite_session.add_all([tenant, app_model])
        sqlite_session.commit()
        mock_validate_token.return_value = _api_token(
            tenant_id=tenant.id,
            app_id=app_model.id,
            token_type=ApiTokenType.APP,
        )

        end_user = EndUser(
            id=str(uuid.uuid4()),
            tenant_id=tenant.id,
            app_id=app_model.id,
            type=EndUserType.SERVICE_API,
            session_id=expected_user_id,
        )
        commands = _RecordingEndUserCommands(end_user)
        login_manager = _RecordingLoginManager()
        signal = _RecordingSignal()
        app.login_manager = login_manager  # type: ignore[attr-defined]
        monkeypatch.setattr(wraps_module, "application_services", lambda: _ApplicationServicesStub(commands))
        monkeypatch.setattr(wraps_module, "user_logged_in", signal)

        @validate_app_token(fetch_user_arg=FetchUserArg(fetch_from=fetch_from, required=True))
        def protected_view(*, app_model: App, end_user: EndUser) -> tuple[App, EndUser]:
            return app_model, end_user

        with (
            app.test_request_context(path, method=method, **request_kwargs),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            injected_app, injected_end_user = protected_view()

        assert injected_app is app_model
        assert injected_end_user is end_user
        assert commands.calls == [(tenant.id, app_model.id, expected_user_id)]
        assert login_manager.users == [end_user]
        assert signal.calls == [(app, end_user)]

    @patch("controllers.service_api.wraps.user_logged_in")
    @patch("controllers.service_api.wraps.current_app")
    @pytest.mark.parametrize(
        "sqlite_session",
        [(App, ResourceAccessToken, ResourceAccessTokenRelation, Tenant, Account, TenantAccountJoin)],
        indirect=True,
    )
    def test_resource_access_token_allows_bound_app(
        self,
        mock_current_app,
        mock_user_logged_in,
        app: Flask,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _configure_current_app_mock(mock_current_app)
        token_service = build_resource_access_token_service(database_client=get_session_maker())
        monkeypatch.setattr(
            wraps_module, "application_services", lambda: SimpleNamespace(resource_access_tokens=token_service)
        )
        tenant, account, _ = _persist_workspace(sqlite_session)
        app_model = _app_model(tenant_id=tenant.id)
        access_token = ResourceAccessToken(
            tenant_id=tenant.id,
            name="Production integration",
            track_id="a" * 32,
            token="sk-00000000-0000-0000-0000-000000000000",
            created_by=account.id,
        )
        sqlite_session.add_all([app_model, access_token])
        sqlite_session.flush()
        sqlite_session.add(
            ResourceAccessTokenRelation(
                token_id=access_token.id,
                resource_type=ResourceAccessTokenResourceType.APP,
                app_id=app_model.id,
                dataset_id=None,
            )
        )
        sqlite_session.commit()

        @validate_app_token
        def protected_view(app_model):
            return {"success": True, "app_id": app_model.id}

        with (
            app.test_request_context(
                "/",
                method="GET",
                headers={"Authorization": f"Bearer {access_token.token}"},
            ),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            result = protected_view()

        assert result["app_id"] == app_model.id
        sqlite_session.expire_all()
        assert sqlite_session.get(ResourceAccessToken, access_token.id).last_used_at is not None

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @pytest.mark.parametrize("sqlite_session", [(App,)], indirect=True)
    def test_app_not_found_raises_forbidden(self, mock_validate_token, app: Flask, sqlite_session: Session):
        """Test that Forbidden is raised when app no longer exists."""
        # Arrange
        api_token = _api_token(
            tenant_id=str(uuid.uuid4()),
            app_id=str(uuid.uuid4()),
            token_type=ApiTokenType.APP,
        )
        mock_validate_token.return_value = api_token

        @validate_app_token
        def protected_view(**kwargs):
            return {"success": True}

        # Act & Assert
        with (
            app.test_request_context("/", method="GET"),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            with pytest.raises(Forbidden) as exc_info:
                protected_view()
            assert "no longer exists" in str(exc_info.value)

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @pytest.mark.parametrize("sqlite_session", [(App,)], indirect=True)
    def test_app_status_abnormal_raises_forbidden(self, mock_validate_token, app: Flask, sqlite_session: Session):
        """Test that Forbidden is raised when app status is abnormal."""
        # Arrange
        app_model = _app_model(tenant_id=str(uuid.uuid4()))
        sqlite_session.add(app_model)
        sqlite_session.commit()
        app_model.status = "abnormal"
        mock_validate_token.return_value = _api_token(
            tenant_id=app_model.tenant_id,
            app_id=app_model.id,
            token_type=ApiTokenType.APP,
        )

        @validate_app_token
        def protected_view(**kwargs):
            return {"success": True}

        # Act & Assert
        with (
            app.test_request_context("/", method="GET"),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            with pytest.raises(Forbidden) as exc_info:
                protected_view()
            assert "status is abnormal" in str(exc_info.value)

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @pytest.mark.parametrize("sqlite_session", [(App,)], indirect=True)
    def test_app_api_disabled_raises_forbidden(self, mock_validate_token, app: Flask, sqlite_session: Session):
        """Test that Forbidden is raised when app API is disabled."""
        # Arrange
        app_model = _app_model(tenant_id=str(uuid.uuid4()), enable_api=False)
        sqlite_session.add(app_model)
        sqlite_session.commit()
        mock_validate_token.return_value = _api_token(
            tenant_id=app_model.tenant_id,
            app_id=app_model.id,
            token_type=ApiTokenType.APP,
        )

        @validate_app_token
        def protected_view(**kwargs):
            return {"success": True}

        # Act & Assert
        with (
            app.test_request_context("/", method="GET"),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            with pytest.raises(Forbidden) as exc_info:
                protected_view()
            assert "API service has been disabled" in str(exc_info.value)


class TestCloudEditionBillingResourceCheck:
    """Test suite for cloud_edition_billing_resource_check decorator"""

    @pytest.fixture
    def app(self):
        """Create Flask test application."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.FeatureService.get_features")
    def test_allows_when_under_limit(self, mock_get_features, mock_validate_token, app: Flask):
        """Test that request is allowed when under resource limit."""
        # Arrange
        mock_validate_token.return_value = Mock(tenant_id="tenant123")

        mock_features = Mock()
        mock_features.members.limit = 10
        mock_features.members.size = 5
        mock_get_features.return_value = mock_features

        @cloud_edition_billing_resource_check("members", "app")
        def add_member():
            return "member_added"

        # Act
        with app.test_request_context("/", method="GET"):
            result = add_member()

        # Assert
        assert result == "member_added"
        mock_get_features.assert_called_once_with("tenant123", exclude_vector_space=True)

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.FeatureService.get_features")
    @patch("controllers.service_api.wraps.FeatureService.get_vector_space")
    def test_loads_vector_space_from_dedicated_quota_api(
        self, mock_get_vector_space, mock_get_features, mock_validate_token, app: Flask
    ):
        """Test vector-space resource checks avoid loading the full feature payload."""
        # Arrange
        mock_validate_token.return_value = Mock(tenant_id="tenant123")

        mock_vector_space = Mock()
        mock_vector_space.limit = 10
        mock_vector_space.size = 5
        mock_vector_space.usage_unknown = False
        mock_get_vector_space.return_value = mock_vector_space

        @cloud_edition_billing_resource_check("vector_space", "dataset")
        def add_segment():
            return "segment_added"

        # Act
        with (
            app.test_request_context("/", method="GET"),
            config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD),
        ):
            result = add_segment()

        # Assert
        assert result == "segment_added"
        mock_get_vector_space.assert_called_once_with("tenant123")
        mock_get_features.assert_not_called()

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.FeatureService.get_features")
    @patch("controllers.service_api.wraps.FeatureService.get_vector_space")
    def test_rejects_sandbox_when_vector_space_usage_is_unknown(
        self, mock_get_vector_space, mock_get_features, mock_validate_token, app: Flask
    ):
        mock_validate_token.return_value = Mock(tenant_id="tenant123")
        mock_get_vector_space.return_value = Mock(size=0, limit=50, usage_unknown=True)
        mock_get_features.return_value = SimpleNamespace(
            billing=SimpleNamespace(
                subscription=SimpleNamespace(plan=CloudPlan.SANDBOX),
            )
        )

        @cloud_edition_billing_resource_check("vector_space", "dataset")
        def upload_document():
            return "document_uploaded"

        with (
            app.test_request_context("/", method="GET"),
            config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD),
            pytest.raises(ServiceUnavailable) as exc_info,
        ):
            upload_document()

        assert "Please try again later" in str(exc_info.value)
        mock_get_features.assert_called_once_with("tenant123", exclude_vector_space=True)

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.FeatureService.get_features")
    @patch("controllers.service_api.wraps.FeatureService.get_vector_space")
    @pytest.mark.parametrize("plan", [CloudPlan.PROFESSIONAL, CloudPlan.TEAM])
    def test_allows_paid_plan_when_vector_space_usage_is_unknown(
        self, mock_get_vector_space, mock_get_features, mock_validate_token, app: Flask, plan: CloudPlan
    ):
        mock_validate_token.return_value = Mock(tenant_id="tenant123")
        mock_get_vector_space.return_value = Mock(size=0, limit=50, usage_unknown=True)
        mock_get_features.return_value = SimpleNamespace(
            billing=SimpleNamespace(
                subscription=SimpleNamespace(plan=plan),
            )
        )

        @cloud_edition_billing_resource_check("vector_space", "dataset")
        def upload_document():
            return "document_uploaded"

        with (
            app.test_request_context("/", method="GET"),
            config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD),
        ):
            result = upload_document()

        assert result == "document_uploaded"
        mock_get_features.assert_called_once_with("tenant123", exclude_vector_space=True)

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.FeatureService.get_features")
    def test_loads_features_when_checking_non_vector_space_limit(
        self, mock_get_features, mock_validate_token, app: Flask
    ):
        """Test non-vector-space resource checks keep using the light feature payload."""
        # Arrange
        mock_validate_token.return_value = Mock(tenant_id="tenant123")

        mock_features = Mock()
        mock_features.documents_upload_quota.limit = 10
        mock_features.documents_upload_quota.size = 5
        mock_get_features.return_value = mock_features

        @cloud_edition_billing_resource_check("documents", "dataset")
        def upload_document():
            return "document_uploaded"

        # Act
        with app.test_request_context("/", method="GET"):
            result = upload_document()

        # Assert
        assert result == "document_uploaded"
        mock_get_features.assert_called_once_with("tenant123", exclude_vector_space=True)

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.FeatureService.get_features")
    def test_rejects_when_at_limit(self, mock_get_features, mock_validate_token, app: Flask):
        """Test that Forbidden is raised when at resource limit."""
        # Arrange
        mock_validate_token.return_value = Mock(tenant_id="tenant123")

        mock_features = Mock()
        mock_features.members.limit = 10
        mock_features.members.size = 10
        mock_get_features.return_value = mock_features

        @cloud_edition_billing_resource_check("members", "app")
        def add_member():
            return "member_added"

        # Act & Assert
        with app.test_request_context("/", method="GET"):
            with pytest.raises(Forbidden) as exc_info:
                add_member()
            assert "members has reached the limit" in str(exc_info.value)

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.FeatureService.get_features")
    def test_allows_when_billing_disabled(self, mock_get_features, mock_validate_token, app: Flask):
        """Test that request is allowed outside Cloud."""
        # Arrange
        mock_validate_token.return_value = Mock(tenant_id="tenant123")

        mock_features = Mock()
        mock_get_features.return_value = mock_features

        @cloud_edition_billing_resource_check("members", "app")
        def add_member():
            return "member_added"

        # Act
        with app.test_request_context("/", method="GET"):
            result = add_member()

        # Assert
        assert result == "member_added"


class TestCloudEditionBillingKnowledgeLimitCheck:
    """Test suite for cloud_edition_billing_knowledge_limit_check decorator"""

    @pytest.fixture
    def app(self):
        """Create Flask test application."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.FeatureService.get_features")
    def test_rejects_add_segment_in_sandbox(self, mock_get_features, mock_validate_token, app: Flask):
        """Test that add_segment is rejected in SANDBOX plan."""
        # Arrange
        mock_validate_token.return_value = Mock(tenant_id="tenant123")

        mock_features = Mock()
        mock_features.billing.subscription.plan = CloudPlan.SANDBOX
        mock_get_features.return_value = mock_features

        @cloud_edition_billing_knowledge_limit_check("add_segment", "dataset")
        def add_segment():
            return "segment_added"

        # Act & Assert
        with app.test_request_context("/", method="GET"):
            with pytest.raises(Forbidden) as exc_info:
                add_segment()
            assert "upgrade to a paid plan" in str(exc_info.value)

    @config_overrides_context(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.FeatureService.get_features")
    def test_allows_other_operations_in_sandbox(self, mock_get_features, mock_validate_token, app: Flask):
        """Test that non-add_segment operations are allowed in SANDBOX."""
        # Arrange
        mock_validate_token.return_value = Mock(tenant_id="tenant123")

        mock_features = Mock()
        mock_features.billing.subscription.plan = CloudPlan.SANDBOX
        mock_get_features.return_value = mock_features

        @cloud_edition_billing_knowledge_limit_check("search", "dataset")
        def search():
            return "search_results"

        # Act
        with app.test_request_context("/", method="GET"):
            result = search()

        # Assert
        assert result == "search_results"


class TestCloudEditionBillingRateLimitCheck:
    """Test suite for cloud_edition_billing_rate_limit_check decorator"""

    @pytest.fixture
    def app(self):
        """Create Flask test application."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    @pytest.mark.parametrize(
        "sqlite_session",
        [(Dataset, ApiToken, Tenant, Account, TenantAccountJoin)],
        indirect=True,
    )
    def test_dataset_key_is_authenticated_once_before_rate_limiting(
        self,
        app: Flask,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tenant, account, _ = _persist_workspace(sqlite_session)
        dataset = Dataset(
            id=str(uuid.uuid4()),
            tenant_id=tenant.id,
            name="Customer Support",
            created_by=account.id,
            enable_api=True,
        )
        api_token = _api_token(tenant_id=tenant.id, token_type=ApiTokenType.DATASET)
        sqlite_session.add_all([dataset, api_token])
        sqlite_session.commit()
        app.login_manager = _RequestLoginManager()  # type: ignore[attr-defined]
        authentication_scopes: list[str | None] = []
        admitted_tenants: list[str] = []
        redis = _RateLimitRedisStub(request_count=50)

        def authenticate(scope: str | None = None) -> ApiToken:
            authentication_scopes.append(scope)
            return api_token

        def get_rate_limit(tenant_id: str) -> KnowledgeRateLimitModel:
            admitted_tenants.append(tenant_id)
            return KnowledgeRateLimitModel(enabled=True, limit=100)

        monkeypatch.setattr(wraps_module, "validate_and_get_api_token", authenticate)
        monkeypatch.setattr(wraps_module.FeatureService, "get_knowledge_rate_limit", get_rate_limit)
        monkeypatch.setattr(wraps_module, "redis_client", redis)

        @validate_dataset_token
        @cloud_edition_billing_rate_limit_check("knowledge")
        def knowledge_request(tenant_id: str, dataset_id: str | None = None) -> tuple[str, str | None]:
            return tenant_id, dataset_id

        with (
            app.test_request_context("/", headers={"Authorization": f"Bearer {api_token.token}"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            result = knowledge_request(dataset_id=dataset.id)

        assert result == (tenant.id, dataset.id)
        assert authentication_scopes == ["dataset"]
        assert admitted_tenants == [tenant.id]
        assert redis.added_keys == [f"rate_limit_{tenant.id}"]
        assert redis.trimmed_keys == [f"rate_limit_{tenant.id}"]

    @pytest.mark.parametrize(
        "sqlite_session",
        [(Dataset, ApiToken, Tenant, Account, TenantAccountJoin)],
        indirect=True,
    )
    def test_dataset_retrieval_http_admission_propagates_remote_ip(
        self,
        app: Flask,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tenant, account, _ = _persist_workspace(sqlite_session)
        dataset = Dataset(
            id=str(uuid.uuid4()),
            tenant_id=tenant.id,
            name="Customer Support",
            created_by=account.id,
            enable_api=True,
        )
        api_token = _api_token(tenant_id=tenant.id, token_type=ApiTokenType.DATASET)
        sqlite_session.add_all([dataset, api_token])
        sqlite_session.commit()
        app.login_manager = _RequestLoginManager()  # type: ignore[attr-defined]
        contexts: list[RequestContext] = []

        class RetrievalProbe:
            @dataset_retrieval_admission
            def get(self, context: RequestContext, dataset_id: uuid.UUID) -> dict[str, str | None]:
                contexts.append(context)
                return {"dataset_id": str(dataset_id), "remote_ip": context.remote_ip}

        probe = RetrievalProbe()
        app.add_url_rule("/datasets/<uuid:dataset_id>/probe", view_func=probe.get)
        monkeypatch.setattr(wraps_module, "validate_and_get_api_token", lambda _scope=None: api_token)
        monkeypatch.setattr(
            wraps_module.FeatureService,
            "get_knowledge_rate_limit",
            lambda _tenant_id: KnowledgeRateLimitModel(enabled=False),
        )

        with patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)):
            response = app.test_client().get(
                f"/datasets/{dataset.id}/probe",
                headers={"Authorization": f"Bearer {api_token.token}"},
                environ_overrides={"REMOTE_ADDR": "203.0.113.42"},
            )

        assert response.status_code == 200
        assert response.get_json() == {"dataset_id": dataset.id, "remote_ip": "203.0.113.42"}
        assert len(contexts) == 1
        assert contexts[0].account_id == account.id
        assert contexts[0].active_workspace_id == tenant.id

    def test_standalone_rate_limit_authenticates_request_and_discards_stale_admission(
        self,
        app: Flask,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stale_tenant_id = str(uuid.uuid4())
        admitted_tenant_id = str(uuid.uuid4())
        admitted_tenants: list[str] = []
        viewed_tenants: list[str] = []

        @app.before_request
        def seed_stale_admission() -> None:
            g.__dict__[wraps_module._DATASET_ADMISSION_CONTEXT_KEY] = wraps_module._DatasetAdmission(
                bearer_token="stale-token",
                dataset_id=None,
                tenant_id=stale_tenant_id,
            )

        def resolve_tenant(*, bearer_token: str, dataset_id: str | None) -> str:
            assert bearer_token == "current-token"
            assert dataset_id is None
            return admitted_tenant_id

        def get_rate_limit(tenant_id: str) -> KnowledgeRateLimitModel:
            admitted_tenants.append(tenant_id)
            return KnowledgeRateLimitModel(enabled=False)

        monkeypatch.setattr(wraps_module, "_resolve_dataset_service_tenant", resolve_tenant)
        monkeypatch.setattr(wraps_module.FeatureService, "get_knowledge_rate_limit", get_rate_limit)

        @app.get("/standalone-rate-limit")
        @cloud_edition_billing_rate_limit_check("knowledge")
        def rate_limited_probe() -> dict[str, bool]:
            viewed_tenants.append(admitted_tenant_id)
            return {"admitted": wraps_module._DATASET_ADMISSION_CONTEXT_KEY not in g.__dict__}

        response = app.test_client().get(
            "/standalone-rate-limit",
            headers={"Authorization": "Bearer current-token"},
        )

        assert response.status_code == 200
        assert response.get_json() == {"admitted": True}
        assert admitted_tenants == [admitted_tenant_id]
        assert viewed_tenants == [admitted_tenant_id]

    def test_standalone_rate_limit_rejects_missing_bearer_before_view(
        self,
        app: Flask,
    ) -> None:
        view_called: list[bool] = []

        @app.get("/standalone-rate-limit")
        @cloud_edition_billing_rate_limit_check("knowledge")
        def rate_limited_probe() -> str:
            view_called.append(True)
            return "unexpected"

        response = app.test_client().get("/standalone-rate-limit")

        assert response.status_code == 401
        assert view_called == []

    @pytest.mark.parametrize("sqlite_session", [(RateLimitLog,)], indirect=True)
    def test_rejects_over_rate_limit(
        self,
        app: Flask,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that Forbidden is raised when over rate limit."""
        tenant_id = str(uuid.uuid4())
        redis = _RateLimitRedisStub(request_count=15)

        def resolve_tenant(*, bearer_token: str, dataset_id: str | None) -> str:
            assert bearer_token == "test-token"
            assert dataset_id is None
            return tenant_id

        monkeypatch.setattr(
            wraps_module,
            "_resolve_dataset_service_tenant",
            resolve_tenant,
        )
        monkeypatch.setattr(
            wraps_module.FeatureService,
            "get_knowledge_rate_limit",
            lambda _tenant_id: KnowledgeRateLimitModel(enabled=True, limit=10, subscription_plan="pro"),
        )
        monkeypatch.setattr(wraps_module, "redis_client", redis)
        monkeypatch.setattr(wraps_module, "db", _DatabaseEngineStub(engine=sqlite_session.get_bind()))

        @cloud_edition_billing_rate_limit_check("knowledge")
        def knowledge_request() -> str:
            return "success"

        with (
            app.test_request_context("/", method="GET", headers={"Authorization": "Bearer test-token"}),
            pytest.raises(Forbidden, match="rate limit"),
        ):
            knowledge_request()

        persisted_logs = sqlite_session.scalars(select(RateLimitLog)).all()
        assert len(persisted_logs) == 1
        assert persisted_logs[0].tenant_id == tenant_id
        assert persisted_logs[0].subscription_plan == "pro"
        assert persisted_logs[0].operation == "knowledge"


class TestValidateDatasetToken:
    """Test suite for validate_dataset_token decorator"""

    @pytest.fixture
    def app(self):
        """Create Flask test application."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    @patch("controllers.service_api.wraps.user_logged_in")
    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @patch("controllers.service_api.wraps.current_app")
    @pytest.mark.parametrize(
        "sqlite_session",
        [(Tenant, Account, TenantAccountJoin)],
        indirect=True,
    )
    def test_valid_dataset_token(
        self,
        mock_current_app,
        mock_validate_token,
        mock_user_logged_in,
        app: Flask,
        sqlite_session: Session,
    ):
        """Test that valid dataset token allows access."""
        # Arrange
        _configure_current_app_mock(mock_current_app)

        tenant, account, _ = _persist_workspace(sqlite_session)
        api_token = _api_token(tenant_id=tenant.id, token_type=ApiTokenType.DATASET)
        sqlite_session.add(api_token)
        sqlite_session.commit()
        mock_validate_token.return_value = api_token

        @validate_dataset_token
        def protected_view(tenant_id):
            return {"success": True, "tenant_id": tenant_id}

        # Act
        with (
            app.test_request_context("/", method="GET", headers={"Authorization": "Bearer test_token"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            result = protected_view()

        # Assert
        assert result["success"] is True
        assert result["tenant_id"] == tenant.id
        assert account.current_tenant_id == tenant.id

    @pytest.mark.parametrize("rate_limit_enabled", [False, True])
    @pytest.mark.parametrize(
        "sqlite_session",
        [(Dataset, ResourceAccessToken, ResourceAccessTokenRelation, Tenant, Account, TenantAccountJoin)],
        indirect=True,
    )
    def test_resource_access_token_rate_limit_uses_admitted_tenant(
        self,
        app: Flask,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
        rate_limit_enabled: bool,
    ) -> None:
        app.login_manager = _RequestLoginManager()  # type: ignore[attr-defined]
        token_service = build_resource_access_token_service(database_client=get_session_maker())
        monkeypatch.setattr(
            wraps_module,
            "application_services",
            lambda: _ResourceAccessTokenServicesStub(resource_access_tokens=token_service),
        )
        tenant, account, _ = _persist_workspace(sqlite_session)
        dataset = Dataset(
            id=str(uuid.uuid4()),
            tenant_id=tenant.id,
            name="Customer Support",
            created_by=account.id,
            enable_api=True,
        )
        access_token = ResourceAccessToken(
            tenant_id=tenant.id,
            name="Production integration",
            track_id="b" * 32,
            token="sk-00000000-0000-0000-0000-000000000001",
            created_by=account.id,
        )
        sqlite_session.add_all([dataset, access_token])
        sqlite_session.flush()
        sqlite_session.add(
            ResourceAccessTokenRelation(
                token_id=access_token.id,
                resource_type=ResourceAccessTokenResourceType.KNOWLEDGE,
                app_id=None,
                dataset_id=dataset.id,
            )
        )
        sqlite_session.commit()
        admitted_tenants: list[str] = []
        redis = _RateLimitRedisStub()

        def reject_dataset_key_authentication(_scope: str | None = None) -> ApiToken:
            raise AssertionError("rate limiting must not authenticate the bearer token again")

        def get_rate_limit(tenant_id: str) -> KnowledgeRateLimitModel:
            admitted_tenants.append(tenant_id)
            return KnowledgeRateLimitModel(enabled=rate_limit_enabled, limit=10)

        monkeypatch.setattr(wraps_module, "validate_and_get_api_token", reject_dataset_key_authentication)
        monkeypatch.setattr(wraps_module.FeatureService, "get_knowledge_rate_limit", get_rate_limit)
        monkeypatch.setattr(wraps_module, "redis_client", redis)

        @validate_dataset_token
        @cloud_edition_billing_rate_limit_check("knowledge")
        def protected_view(tenant_id: str, dataset_id: str | None = None) -> tuple[str, str | None]:
            return tenant_id, dataset_id

        with (
            app.test_request_context("/", method="GET", headers={"Authorization": f"Bearer {access_token.token}"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            result = protected_view(dataset_id=dataset.id)

        assert result == (tenant.id, dataset.id)
        assert admitted_tenants == [tenant.id]
        expected_keys: list[str] = [f"rate_limit_{tenant.id}"] if rate_limit_enabled else []
        assert redis.added_keys == expected_keys
        assert redis.trimmed_keys == expected_keys
        sqlite_session.expire_all()
        persisted_token = sqlite_session.get(ResourceAccessToken, access_token.id)
        assert persisted_token is not None
        assert persisted_token.last_used_at is not None

    @pytest.mark.parametrize(
        "sqlite_session",
        [(ResourceAccessToken, ResourceAccessTokenRelation, Tenant)],
        indirect=True,
    )
    def test_resource_access_auth_failure_does_not_reach_rate_limit_or_view(
        self,
        app: Flask,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        token_service = build_resource_access_token_service(database_client=get_session_maker())
        monkeypatch.setattr(
            wraps_module,
            "application_services",
            lambda: _ResourceAccessTokenServicesStub(resource_access_tokens=token_service),
        )
        rate_limit_tenants: list[str] = []
        view_tenants: list[str] = []

        def get_rate_limit(tenant_id: str) -> KnowledgeRateLimitModel:
            rate_limit_tenants.append(tenant_id)
            return KnowledgeRateLimitModel(enabled=False)

        monkeypatch.setattr(wraps_module.FeatureService, "get_knowledge_rate_limit", get_rate_limit)

        @validate_dataset_token
        @cloud_edition_billing_rate_limit_check("knowledge")
        def protected_view(tenant_id: str, dataset_id: str | None = None) -> None:
            view_tenants.append(tenant_id)

        with (
            app.test_request_context("/", headers={"Authorization": "Bearer sk-does-not-exist"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
            pytest.raises(Unauthorized),
        ):
            protected_view(dataset_id=str(uuid.uuid4()))

        assert rate_limit_tenants == []
        assert view_tenants == []

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    @pytest.mark.parametrize("sqlite_session", [(Dataset,)], indirect=True)
    def test_dataset_not_found_raises_not_found(self, mock_validate_token, app: Flask, sqlite_session: Session):
        """Test that NotFound is raised when dataset doesn't exist."""
        # Arrange
        api_token = _api_token(tenant_id=str(uuid.uuid4()), token_type=ApiTokenType.DATASET)
        sqlite_session.add(api_token)
        sqlite_session.commit()
        mock_validate_token.return_value = api_token

        @validate_dataset_token
        def protected_view(dataset_id=None, **kwargs):
            return {"success": True}

        # Act & Assert
        with (
            app.test_request_context("/", method="GET", headers={"Authorization": "Bearer test_token"}),
            patch("controllers.service_api.wraps.db.session", sqlite_session),
        ):
            with pytest.raises(NotFound) as exc_info:
                protected_view(dataset_id=str(uuid.uuid4()))
            assert "Dataset not found" in str(exc_info.value)

    # Per-knowledge-base scope enforcement (DatasetApiTokenBinding rows):
    #   no rows -> the key reaches every dataset in its tenant (default / back-compat)
    #   N rows  -> the key is restricted to exactly those datasets
    # The "reaches the lookup" tests assert a downstream NotFound (the target dataset is
    # intentionally not persisted), proving the binding gate let the request through rather
    # than raising its own Forbidden.

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    def test_unbound_key_is_not_scope_restricted(self, mock_validate_token, app: Flask, sqlite_session: Session):
        """A key with no bindings passes the scope gate and proceeds to the dataset lookup."""
        api_token = _api_token(tenant_id=str(uuid.uuid4()), token_type=ApiTokenType.DATASET)
        sqlite_session.add(api_token)
        sqlite_session.commit()
        mock_validate_token.return_value = api_token

        @validate_dataset_token
        def protected_view(**kwargs):
            return {"success": True}

        with (
            app.test_request_context("/", method="GET", headers={"Authorization": "Bearer test_token"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            with pytest.raises(NotFound):
                protected_view(dataset_id=str(uuid.uuid4()))

    def test_cached_deleted_dataset_key_is_rejected_and_evicted(
        self,
        app: Flask,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stale_token = _api_token(tenant_id=str(uuid.uuid4()), token_type=ApiTokenType.DATASET)
        sqlite_session.add(stale_token)
        sqlite_session.commit()
        sqlite_session.delete(stale_token)
        sqlite_session.commit()
        invalidated: list[tuple[str, str | None]] = []

        monkeypatch.setattr(wraps_module, "validate_and_get_api_token", lambda _scope: stale_token)
        monkeypatch.setattr(
            wraps_module.ApiTokenCache,
            "delete",
            lambda token, scope=None: invalidated.append((token, scope)),
        )

        @validate_dataset_token
        def protected_view(**kwargs):
            return {"success": True}

        with (
            app.test_request_context("/", method="GET", headers={"Authorization": "Bearer test_token"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
            pytest.raises(Unauthorized, match="Access token is invalid"),
        ):
            protected_view(dataset_id=str(uuid.uuid4()))

        assert invalidated == [(stale_token.token, stale_token.type)]

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    def test_bound_key_rejects_other_dataset(self, mock_validate_token, app: Flask, sqlite_session: Session):
        """A key bound to one dataset is forbidden from reaching a different dataset."""
        api_token = _api_token(tenant_id=str(uuid.uuid4()), token_type=ApiTokenType.DATASET)
        mock_validate_token.return_value = api_token
        sqlite_session.add_all(
            [api_token, DatasetApiTokenBinding(api_token_id=api_token.id, dataset_id=str(uuid.uuid4()))]
        )
        sqlite_session.commit()

        @validate_dataset_token
        def protected_view(**kwargs):
            return {"success": True}

        with (
            app.test_request_context("/", method="GET", headers={"Authorization": "Bearer test_token"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            with pytest.raises(Forbidden) as exc_info:
                protected_view(dataset_id=str(uuid.uuid4()))
            assert "not authorized to access this knowledge base" in str(exc_info.value)

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    def test_bound_key_rejects_endpoint_without_dataset_id(
        self, mock_validate_token, app: Flask, sqlite_session: Session
    ):
        """A scoped key cannot call collection endpoints (list/create) that carry no dataset id."""
        api_token = _api_token(tenant_id=str(uuid.uuid4()), token_type=ApiTokenType.DATASET)
        mock_validate_token.return_value = api_token
        sqlite_session.add_all(
            [api_token, DatasetApiTokenBinding(api_token_id=api_token.id, dataset_id=str(uuid.uuid4()))]
        )
        sqlite_session.commit()

        @validate_dataset_token
        def protected_view(**kwargs):
            return {"success": True}

        with (
            app.test_request_context("/", method="GET", headers={"Authorization": "Bearer test_token"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            with pytest.raises(Forbidden) as exc_info:
                protected_view()
            assert "not authorized to access this knowledge base" in str(exc_info.value)

    @patch("controllers.service_api.wraps.validate_and_get_api_token")
    def test_bound_key_reaches_allowed_dataset(self, mock_validate_token, app: Flask, sqlite_session: Session):
        """A scoped key targeting one of its bound datasets passes the scope gate."""
        api_token = _api_token(tenant_id=str(uuid.uuid4()), token_type=ApiTokenType.DATASET)
        mock_validate_token.return_value = api_token
        allowed_dataset_id = str(uuid.uuid4())
        sqlite_session.add_all(
            [api_token, DatasetApiTokenBinding(api_token_id=api_token.id, dataset_id=allowed_dataset_id)]
        )
        sqlite_session.commit()

        @validate_dataset_token
        def protected_view(**kwargs):
            return {"success": True}

        with (
            app.test_request_context("/", method="GET", headers={"Authorization": "Bearer test_token"}),
            patch("controllers.service_api.wraps.db.session", _session_proxy(sqlite_session)),
        ):
            # The bound dataset is allowed by scope; it is simply not persisted, so the
            # downstream lookup raises NotFound instead of the scope Forbidden.
            with pytest.raises(NotFound):
                protected_view(dataset_id=allowed_dataset_id)


class TestFetchUserArg:
    """Test suite for FetchUserArg model"""

    def test_fetch_user_arg_defaults(self):
        """Test FetchUserArg default values."""
        # Arrange & Act
        arg = FetchUserArg(fetch_from=WhereisUserArg.JSON)

        # Assert
        assert arg.fetch_from == WhereisUserArg.JSON
        assert arg.required is False

    def test_fetch_user_arg_required(self):
        """Test FetchUserArg with required=True."""
        # Arrange & Act
        arg = FetchUserArg(fetch_from=WhereisUserArg.QUERY, required=True)

        # Assert
        assert arg.fetch_from == WhereisUserArg.QUERY
        assert arg.required is True


class TestDatasetApiResource:
    """Test suite for DatasetApiResource base class"""

    def test_method_decorators_has_validate_dataset_token(self):
        """Test that DatasetApiResource has validate_dataset_token in method_decorators."""
        # Assert
        assert validate_dataset_token in DatasetApiResource.method_decorators

    def test_get_dataset_method_exists(self):
        """Test that get_dataset method exists on DatasetApiResource."""
        # Assert
        assert hasattr(DatasetApiResource, "get_dataset")

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker

from libs.sqlalchemy_utils import database_operation_timeout


def _record_dialect_sql(
    connection: Connection,
    *,
    dialect_name: str,
    monkeypatch: pytest.MonkeyPatch,
    fail_on: str | None = None,
) -> list[str]:
    """Record vendor SQL while retaining a real Session and transaction."""
    statements: list[str] = []
    monkeypatch.setattr(connection.dialect, "name", dialect_name)

    def record(
        _connection: Connection,
        _cursor: object,
        statement: str,
        parameters: object,
        _context: object,
        _executemany: bool,
    ) -> tuple[str, object]:
        statements.append(statement)
        if statement == fail_on:
            raise RuntimeError("connection lost during reset")
        if statement.startswith("SELECT @@SESSION"):
            return "SELECT 50, 31536000, 0", parameters
        return "SELECT 1", parameters

    event.listen(connection, "before_cursor_execute", record, retval=True)
    return statements


def test_postgresql_timeout_is_transaction_local(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory() as session:
        connection = session.connection()
        statements = _record_dialect_sql(connection, dialect_name="postgresql", monkeypatch=monkeypatch)

        with database_operation_timeout(session, timeout_seconds=0.125):
            pass

        assert statements == [
            "SET LOCAL lock_timeout = '125ms'",
            "SET LOCAL statement_timeout = '125ms'",
        ]
        assert not connection.invalidated


def test_mysql_timeout_settings_are_restored_after_operation_error(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite_session_factory() as session:
        connection = session.connection()
        statements = _record_dialect_sql(connection, dialect_name="mysql", monkeypatch=monkeypatch)

        with pytest.raises(RuntimeError, match="operation failed"):
            with database_operation_timeout(session, timeout_seconds=0.125):
                raise RuntimeError("operation failed")

        assert statements == [
            "SELECT @@SESSION.innodb_lock_wait_timeout, @@SESSION.lock_wait_timeout, @@SESSION.max_execution_time",
            "SET SESSION innodb_lock_wait_timeout = 1",
            "SET SESSION lock_wait_timeout = 1",
            "SET SESSION max_execution_time = 125",
            "SET SESSION innodb_lock_wait_timeout = 50",
            "SET SESSION lock_wait_timeout = 31536000",
            "SET SESSION max_execution_time = 0",
        ]
        assert not connection.invalidated


def test_mysql_reset_failure_invalidates_connection_instead_of_returning_it_to_pool(
    sqlite_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = sqlite_session_factory()
    connection = session.connection()
    statements = _record_dialect_sql(
        connection,
        dialect_name="mysql",
        monkeypatch=monkeypatch,
        fail_on="SET SESSION innodb_lock_wait_timeout = 50",
    )
    try:
        with pytest.raises(RuntimeError, match="connection lost during reset"):
            with database_operation_timeout(session, timeout_seconds=1):
                pass

        assert statements[-1] == "SET SESSION innodb_lock_wait_timeout = 50"
        assert connection.invalidated
    finally:
        session.close()


def test_sqlite_busy_timeout_is_restored_before_connection_returns_to_pool(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with sqlite_session_factory() as session:
        original = int(session.connection().exec_driver_sql("PRAGMA busy_timeout").scalar_one())
        with database_operation_timeout(session, timeout_seconds=0.012):
            assert session.connection().exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 12
        assert session.connection().exec_driver_sql("PRAGMA busy_timeout").scalar_one() == original

    with sqlite_session_factory() as session:
        assert session.connection().exec_driver_sql("PRAGMA busy_timeout").scalar_one() == original

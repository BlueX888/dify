"""Small SQLAlchemy helpers shared by short-lived background operations."""

import logging
import math
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@contextmanager
def database_operation_timeout(session: Session, *, timeout_seconds: float) -> Generator[None]:
    """Bound database-side waits for one short transaction.

    Connection checkout is already bounded by the engine's ``pool_timeout``.
    This helper bounds the waits after checkout. PostgreSQL settings are local
    to the transaction. MySQL and SQLite settings are connection-scoped, so
    their original values are restored before the pooled connection is
    returned. A connection whose settings cannot be restored is invalidated
    instead of contaminating the pool.
    """
    if timeout_seconds <= 0:
        raise ValueError("database operation timeout must be positive")

    connection = session.connection()
    dialect_name = connection.dialect.name
    timeout_milliseconds = max(1, math.ceil(timeout_seconds * 1000))

    if dialect_name == "postgresql":
        connection.exec_driver_sql(f"SET LOCAL lock_timeout = '{timeout_milliseconds}ms'")
        connection.exec_driver_sql(f"SET LOCAL statement_timeout = '{timeout_milliseconds}ms'")
        yield
        return

    if dialect_name == "mysql":
        original = connection.exec_driver_sql(
            "SELECT @@SESSION.innodb_lock_wait_timeout, @@SESSION.lock_wait_timeout, @@SESSION.max_execution_time"
        ).one()
        original_values = tuple(int(value) for value in original)
        timeout_whole_seconds = max(1, math.ceil(timeout_seconds))
        with _restore_connection_settings(connection, _mysql_restore_statements(original_values)):
            connection.exec_driver_sql(f"SET SESSION innodb_lock_wait_timeout = {timeout_whole_seconds}")
            connection.exec_driver_sql(f"SET SESSION lock_wait_timeout = {timeout_whole_seconds}")
            connection.exec_driver_sql(f"SET SESSION max_execution_time = {timeout_milliseconds}")
            yield
        return

    if dialect_name == "sqlite":
        original_busy_timeout = int(connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one())
        with _restore_connection_settings(
            connection,
            (f"PRAGMA busy_timeout = {original_busy_timeout}",),
        ):
            connection.exec_driver_sql(f"PRAGMA busy_timeout = {timeout_milliseconds}")
            yield
        return

    yield


def _mysql_restore_statements(original_values: tuple[int, ...]) -> tuple[str, ...]:
    innodb_lock_wait_timeout, lock_wait_timeout, max_execution_time = original_values
    return (
        f"SET SESSION innodb_lock_wait_timeout = {innodb_lock_wait_timeout}",
        f"SET SESSION lock_wait_timeout = {lock_wait_timeout}",
        f"SET SESSION max_execution_time = {max_execution_time}",
    )


@contextmanager
def _restore_connection_settings(
    connection: Connection,
    statements: tuple[str, ...],
) -> Generator[None]:
    operation_failed = False
    try:
        yield
    except BaseException:
        operation_failed = True
        raise
    finally:
        try:
            for statement in statements:
                connection.exec_driver_sql(statement)
        except Exception:
            # Session-level settings must never leak into an unrelated user of
            # the pooled connection. Invalidation also discards a connection
            # that failed while communicating with the database.
            connection.invalidate()
            if operation_failed:
                logger.exception("Failed to restore database timeout settings; connection invalidated")
            else:
                raise

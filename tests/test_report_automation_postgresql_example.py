from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from examples import report_automation_agent as report_example
from moduagent.messages import ToolCall
from moduagent.tools import (
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutor,
    ToolRecoveryAction,
)


def run(coroutine):
    return asyncio.run(coroutine)


class FakePostgreSQLError(Exception):
    def __init__(self, message: str, *, sqlstate: str) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


class FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeCursor:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        query_failure: Exception | None = None,
        events: list[tuple[str, Any]] | None = None,
    ) -> None:
        self.rows = rows
        self.query_failure = query_failure
        self.events = events if events is not None else []
        self.executions: list[tuple[str, Any]] = []
        self.fetch_size: int | None = None
        column_names = list(rows[0]) if rows else ["value"]
        self.description = [FakeColumn(name) for name in column_names]

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        self.executions.append((sql, params))
        self.events.append(("execute", sql))
        if self.query_failure is not None and "AS moduagent_query" in sql:
            raise self.query_failure

    def fetchmany(self, size: int) -> list[dict[str, Any]]:
        self.fetch_size = size
        self.events.append(("fetchmany", size))
        return self.rows[:size]


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.fake_cursor = cursor
        self.events = cursor.events

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self.events)

    def cursor(self) -> FakeCursor:
        return self.fake_cursor


class FakeTransaction:
    def __init__(self, events: list[tuple[str, Any]]) -> None:
        self.events = events

    def __enter__(self) -> FakeTransaction:
        self.events.append(("transaction_begin", None))
        return self

    def __exit__(self, *_args: object) -> bool:
        self.events.append(("transaction_end", None))
        return False


def _configure_postgresql(monkeypatch: pytest.MonkeyPatch) -> str:
    dsn = "postgresql://report_reader@db.example.invalid/reporting"
    monkeypatch.setenv("REPORT_DATABASE_URL", dsn)
    monkeypatch.delenv("REPORT_CONNECT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("REPORT_QUERY_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("REPORT_LOCK_TIMEOUT_MS", raising=False)
    return dsn


def test_postgresql_query_db_keeps_the_model_facing_tool_contract() -> None:
    sqlite_parameters = report_example.query_db.schema.parameters
    postgresql_parameters = report_example.query_db_postgresql.schema.parameters

    assert report_example.query_db_postgresql.name == "query_db"
    assert postgresql_parameters["properties"] == sqlite_parameters["properties"]
    assert postgresql_parameters["required"] == sqlite_parameters["required"]
    assert "context" not in postgresql_parameters["properties"]


def test_postgresql_query_db_is_read_only_bounded_and_run_scoped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dsn = _configure_postgresql(monkeypatch)
    monkeypatch.setattr(report_example, "ARTIFACT_DIR", tmp_path)
    events: list[tuple[str, Any]] = []
    cursor = FakeCursor(
        [
            {"month": "2025-01", "revenue": 10},
            {"month": "2025-02", "revenue": 20},
            {"month": "2025-03", "revenue": 30},
        ],
        events=events,
    )
    connection = FakeConnection(cursor)
    connect_calls: list[tuple[str, int]] = []

    def open_connection(
        received_dsn: str,
        *,
        connect_timeout_seconds: int,
    ) -> FakeConnection:
        connect_calls.append((received_dsn, connect_timeout_seconds))
        return connection

    monkeypatch.setattr(
        report_example,
        "_open_postgresql_connection",
        open_connection,
    )

    result = run(
        report_example.query_db_postgresql.invoke(
            {
                "sql": (
                    "WITH monthly AS (SELECT month, revenue FROM sales) "
                    "SELECT month, revenue FROM monthly"
                ),
                "limit": 2,
            },
            ToolExecutionContext(run_id="../report:postgres"),
        )
    )

    assert connect_calls == [(dsn, 5)]
    assert events[0] == ("transaction_begin", None)
    assert cursor.executions[:3] == [
        ("SET TRANSACTION READ ONLY", None),
        (
            "SELECT set_config('statement_timeout', %s, true)",
            ("15000",),
        ),
        ("SELECT set_config('lock_timeout', %s, true)", ("3000",)),
    ]
    bounded_sql, bounded_params = cursor.executions[3]
    assert "WITH monthly AS" in bounded_sql
    assert bounded_sql.endswith("AS moduagent_query LIMIT 3")
    assert bounded_params is None
    assert cursor.fetch_size == 3
    assert events[-1] == ("transaction_end", None)

    assert result["row_count"] == 2
    assert result["truncated"] is True
    assert result["rows"] == [
        {"month": "2025-01", "revenue": 10},
        {"month": "2025-02", "revenue": 20},
    ]
    dataset_path = Path(result["dataset_path"])
    assert dataset_path.parent == tmp_path
    assert json.loads(dataset_path.read_text(encoding="utf-8")) == result["rows"]
    assert not list(tmp_path.glob("*.tmp.json"))
    assert dsn not in str(result)


@pytest.mark.parametrize(
    ("sql", "limit"),
    [
        ("", 10),
        ("DELETE FROM sales", 10),
        ("SELECT 1; SELECT 2", 10),
        ("SELECT " + ("x" * 20_001), 10),
        ("SELECT 1", 0),
        ("SELECT 1", 501),
        ("SELECT 1", True),
    ],
)
def test_postgresql_query_db_rejects_unsafe_or_unbounded_input_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
    sql: str,
    limit: int,
) -> None:
    _configure_postgresql(monkeypatch)

    def unexpected_connection(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid SQL must be rejected before opening a connection")

    monkeypatch.setattr(
        report_example,
        "_open_postgresql_connection",
        unexpected_connection,
    )

    with pytest.raises(ValueError):
        report_example.query_db_postgresql.function(
            sql=sql,
            limit=limit,
            context=ToolExecutionContext(run_id="invalid-query"),
        )


@pytest.mark.parametrize(
    ("sqlstate", "error_type", "reason", "recovery", "retryable"),
    [
        (
            "42601",
            ToolErrorType.EXECUTION_ERROR,
            "invalid_read_query",
            ToolRecoveryAction.REPAIR_CALL,
            False,
        ),
        (
            "25006",
            ToolErrorType.EXECUTION_ERROR,
            "invalid_read_query",
            ToolRecoveryAction.REPAIR_CALL,
            False,
        ),
        (
            "57014",
            ToolErrorType.TIMEOUT,
            "query_timeout",
            ToolRecoveryAction.REPAIR_CALL,
            False,
        ),
        (
            "55P03",
            ToolErrorType.EXECUTION_ERROR,
            "query_lock_timeout",
            ToolRecoveryAction.RETRY_CALL,
            True,
        ),
        (
            "42501",
            ToolErrorType.UNAUTHORIZED,
            "database_access_denied",
            ToolRecoveryAction.FAIL,
            False,
        ),
        (
            "08006",
            ToolErrorType.EXECUTION_ERROR,
            "postgres_connection_unavailable",
            ToolRecoveryAction.RETRY_CALL,
            True,
        ),
    ],
)
def test_postgresql_error_mapping_is_actionable_and_sanitized(
    sqlstate: str,
    error_type: ToolErrorType,
    reason: str,
    recovery: ToolRecoveryAction,
    retryable: bool,
) -> None:
    secret = "PRIVATE_QUERY_DIAGNOSTIC host=db.internal"
    error = report_example._map_postgresql_query_error(
        FakePostgreSQLError(secret, sqlstate=sqlstate)
    )

    assert error.type is error_type
    assert error.reason == reason
    assert error.recovery is recovery
    assert error.retryable is retryable
    assert secret not in str(error.to_dict())
    assert "db.internal" not in str(error.to_dict())


def test_postgresql_configuration_failure_is_sanitized_by_the_tool_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REPORT_DATABASE_URL", raising=False)

    result = run(
        ToolExecutor([report_example.query_db_postgresql]).execute(
            ToolCall("postgres-configuration", "query_db", {"sql": "SELECT 1"}),
            ToolExecutionContext(run_id="postgres-configuration"),
        )
    )

    assert result.error is not None
    assert result.error.reason == "postgres_configuration_error"
    assert result.error.recovery is ToolRecoveryAction.FAIL
    assert "REPORT_DATABASE_URL" in result.error.message


def test_query_artifact_cleans_up_a_partial_file_after_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(report_example, "ARTIFACT_DIR", tmp_path)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated artifact failure")

    monkeypatch.setattr(report_example.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated artifact failure"):
        report_example._cache_query_rows(
            [{"month": "2025-01", "revenue": 10}],
            limit=10,
            context=ToolExecutionContext(run_id="replace-failure"),
        )

    assert not list(tmp_path.glob("*.tmp.json"))
    assert not list(tmp_path.glob("*.dataset.json"))

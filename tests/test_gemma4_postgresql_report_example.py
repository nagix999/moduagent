from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from moduagent import ToolErrorType, ToolExecutionContext, ToolRecoveryAction


def _load_example() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "gemma4_postgresql_report_agent.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_moduagent_gemma4_postgresql_report_example",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load example: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example()


def test_model_facing_tool_schemas_match_the_documented_contract() -> None:
    query_schema = example.query_data.schema.parameters
    plot_schema = example.plot_graph.schema.parameters

    assert example.query_data.name == "query_data"
    assert set(query_schema["properties"]) == {"query"}
    assert query_schema["required"] == ["query"]
    assert "context" not in query_schema["properties"]
    assert example.query_data.idempotent is True
    assert example.query_data.repair_safe is True
    assert example.query_data.timeout_retry_safe is False

    assert example.plot_graph.name == "plot_graph"
    assert set(plot_schema["properties"]) == {
        "x_column",
        "y_column",
        "title",
        "chart_type",
    }
    assert set(plot_schema["required"]) == {
        "x_column",
        "y_column",
        "title",
        "chart_type",
    }
    assert "context" not in plot_schema["properties"]


def test_authoritative_schema_and_repair_rules_are_in_system_instructions() -> None:
    instructions = example.REPORT_AGENT_INSTRUCTIONS

    assert example.DATABASE_SCHEMA.strip() in instructions
    assert "reporting.orders" in instructions
    assert "created_at TIMESTAMPTZ" in instructions
    assert "query_data(query)" in instructions
    assert "plot_graph(x_column, y_column, title, chart_type)" in instructions
    assert "Never use << or >>" in instructions
    assert "Never resend the identical query" in instructions
    assert "TIMESTAMPTZ '2025-01-01 00:00:00+09:00'" in instructions


def test_read_query_validation_accepts_the_reference_shape() -> None:
    query = """\
SELECT
  to_char(date_trunc('month', o.created_at), 'YYYY-MM') AS month,
  SUM(o.net_revenue)::double precision AS revenue,
  COUNT(*)::bigint AS orders
FROM reporting.orders AS o
WHERE o.status = 'paid'
  AND o.created_at >= TIMESTAMPTZ '2025-01-01 00:00:00+09:00'
  AND o.created_at < TIMESTAMPTZ '2025-07-01 00:00:00+09:00'
GROUP BY 1
ORDER BY 1
LIMIT 12
"""

    assert example._validate_read_query(query) == query.strip()


@pytest.mark.parametrize(
    "query",
    [
        "",
        "DELETE FROM reporting.orders",
        "SELECT * FROM reporting.orders",
        "SELECT order_id FROM reporting.orders;",
        "SELECT order_id FROM reporting.orders -- comment",
        (
            "WITH removed AS (DELETE FROM reporting.orders RETURNING order_id) "
            "SELECT order_id FROM removed"
        ),
        "SELECT pg_sleep(10)",
    ],
)
def test_read_query_validation_rejects_unsafe_shapes(query: str) -> None:
    with pytest.raises(ValueError):
        example._validate_read_query(query)


class _FakePostgreSQLError(Exception):
    def __init__(self, secret: str, sqlstate: str) -> None:
        super().__init__(secret)
        self.sqlstate = sqlstate


@pytest.mark.parametrize(
    ("sqlstate", "reason", "recovery", "retryable"),
    [
        ("42601", "postgres_syntax_error", ToolRecoveryAction.REPAIR_CALL, False),
        ("42703", "undefined_column", ToolRecoveryAction.REPAIR_CALL, False),
        ("42P01", "undefined_table", ToolRecoveryAction.REPAIR_CALL, False),
        (
            "42883",
            "undefined_operator_or_function",
            ToolRecoveryAction.REPAIR_CALL,
            False,
        ),
        ("57014", "query_timeout", ToolRecoveryAction.REPAIR_CALL, False),
        ("55P03", "query_lock_timeout", ToolRecoveryAction.RETRY_CALL, True),
        (
            "08006",
            "postgres_connection_unavailable",
            ToolRecoveryAction.RETRY_CALL,
            True,
        ),
        ("42501", "database_access_denied", ToolRecoveryAction.FAIL, False),
    ],
)
def test_query_error_mapping_is_actionable_and_does_not_leak_driver_details(
    sqlstate: str,
    reason: str,
    recovery: ToolRecoveryAction,
    retryable: bool,
) -> None:
    secret = "PRIVATE SQL host=db.internal password=secret"

    mapped = example._map_query_error(_FakePostgreSQLError(secret, sqlstate))

    assert mapped.reason == reason
    assert mapped.recovery is recovery
    assert mapped.retryable is retryable
    assert secret not in json.dumps(mapped.to_dict(), default=str)
    assert "db.internal" not in json.dumps(mapped.to_dict(), default=str)


def test_local_query_validation_failure_is_repairable() -> None:
    mapped = example._map_query_error(ValueError("private invalid SQL"))

    assert mapped.type is ToolErrorType.INVALID_ARGUMENTS
    assert mapped.reason == "invalid_read_query"
    assert mapped.recovery is ToolRecoveryAction.REPAIR_CALL
    assert "private invalid SQL" not in str(mapped.to_dict())


def test_dataset_artifacts_are_isolated_by_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(example, "ARTIFACT_DIR", tmp_path)

    first = example._store_dataset(
        [{"month": "2025-01", "revenue": 10}],
        context=ToolExecutionContext(run_id="run/one"),
    )
    second = example._store_dataset(
        [{"month": "2025-02", "revenue": 20}],
        context=ToolExecutionContext(run_id="run/two"),
    )

    assert first != second
    assert json.loads(first.read_text(encoding="utf-8"))[0]["revenue"] == 10
    assert json.loads(second.read_text(encoding="utf-8"))[0]["revenue"] == 20
    assert not list(tmp_path.glob("*.tmp.json"))

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
    assert "o.created_at >= TIMESTAMPTZ" in instructions
    assert "AND o.created_at < TIMESTAMPTZ" in instructions
    assert "<<" not in instructions
    assert ">>" not in instructions
    assert "<start-date>" not in instructions
    assert "<exclusive-end-date>" not in instructions
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
    ("query", "expected"),
    [
        (
            "SELECT order_id FROM reporting.orders;",
            "SELECT order_id FROM reporting.orders",
        ),
        (
            "```sql\nSELECT order_id FROM reporting.orders;\n```",
            "SELECT order_id FROM reporting.orders",
        ),
        (
            "SQL: SELECT order_id FROM reporting.orders;",
            "SELECT order_id FROM reporting.orders",
        ),
    ],
)
def test_read_query_validation_normalizes_common_model_wrappers(
    query: str,
    expected: str,
) -> None:
    assert example._validate_read_query(query) == expected


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("", "query_empty"),
        ("DELETE FROM reporting.orders", "query_not_select"),
        ("SELECT * FROM reporting.orders", "query_select_star_forbidden"),
        (
            "SELECT order_id FROM reporting.orders; SELECT 2",
            "query_multiple_statements",
        ),
        (
            "SELECT order_id FROM reporting.orders -- comment",
            "query_comment_forbidden",
        ),
        (
            "WITH removed AS (DELETE FROM reporting.orders RETURNING order_id) "
            "SELECT order_id FROM removed",
            "query_forbidden_operation",
        ),
        ("SELECT pg_sleep(10)", "query_forbidden_operation"),
    ],
)
def test_read_query_validation_rejects_unsafe_shapes(
    query: str,
    reason: str,
) -> None:
    with pytest.raises(example._QueryContractError) as captured:
        example._validate_read_query(query)
    assert captured.value.reason == reason


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
        ("42803", "grouping_error", ToolRecoveryAction.REPAIR_CALL, False),
        ("42804", "datatype_mismatch", ToolRecoveryAction.REPAIR_CALL, False),
        (
            "22007",
            "invalid_datetime_format",
            ToolRecoveryAction.REPAIR_CALL,
            False,
        ),
        (
            "42ZZZ",
            "postgres_query_structure_error",
            ToolRecoveryAction.REPAIR_CALL,
            False,
        ),
        (
            "22ZZZ",
            "postgres_data_exception",
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
    mapped = example._map_query_error(
        example._QueryContractError(
            "query_select_star_forbidden",
            "Select explicit columns.",
        )
    )

    assert mapped.type is ToolErrorType.INVALID_ARGUMENTS
    assert mapped.reason == "query_select_star_forbidden"
    assert mapped.recovery is ToolRecoveryAction.REPAIR_CALL


def test_unexpected_value_error_fails_closed_without_leaking_details() -> None:
    mapped = example._map_query_error(ValueError("PRIVATE SQL"))

    assert mapped.type is ToolErrorType.EXECUTION_ERROR
    assert mapped.reason == "query_tool_internal_error"
    assert mapped.recovery is ToolRecoveryAction.FAIL
    assert "PRIVATE SQL" not in str(mapped.to_dict())


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
    assert json.loads(first.read_text(encoding="utf-8"))["rows"][0]["revenue"] == 10
    assert json.loads(second.read_text(encoding="utf-8"))["rows"][0]["revenue"] == 20
    assert not list(tmp_path.glob("*.tmp.json"))


class _FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCursor:
    description = [
        _FakeColumn("month"),
        _FakeColumn("revenue"),
        _FakeColumn("orders"),
    ]

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def execute(self, _query: str, _params: object = None) -> None:
        return None

    def fetchmany(self, _size: int) -> list[dict[str, object]]:
        return []


class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def transaction(self):
        return self

    def cursor(self) -> _FakeCursor:
        return _FakeCursor()


def test_empty_query_result_is_a_successful_report_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(example, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setenv("REPORT_DATABASE_URL", "postgresql://private.invalid/report")
    monkeypatch.setattr(
        example,
        "_open_postgresql_connection",
        lambda _dsn: _FakeConnection(),
    )

    result = example.query_data.function(
        query=(
            "SELECT '2025-01' AS month, 0::float AS revenue, "
            "0::bigint AS orders WHERE FALSE"
        ),
        context=ToolExecutionContext(run_id="empty-period"),
    )

    assert result["columns"] == ["month", "revenue", "orders"]
    assert result["rows"] == []
    assert result["row_count"] == 0
    assert result["has_data"] is False
    assert result["truncated"] is False
    dataset = json.loads(Path(result["dataset_path"]).read_text(encoding="utf-8"))
    assert dataset == {
        "columns": ["month", "revenue", "orders"],
        "rows": [],
    }


class _FakeAxes:
    transAxes = object()

    def text(self, *_args: object, **_kwargs: object) -> None:
        return None

    def set_title(self, _value: str) -> None:
        return None

    def set_xlabel(self, _value: str) -> None:
        return None

    def set_ylabel(self, _value: str) -> None:
        return None

    def grid(self, **_kwargs: object) -> None:
        return None

    def tick_params(self, **_kwargs: object) -> None:
        return None


class _FakeFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, path: Path, *, dpi: int) -> None:
        assert dpi == 150
        path.write_bytes(b"empty-chart")


class _FakePyplot:
    def subplots(self, *, figsize: tuple[int, int]):
        assert figsize == (10, 5)
        return _FakeFigure(), _FakeAxes()

    def close(self, _figure: _FakeFigure) -> None:
        return None


def test_plot_graph_creates_an_empty_state_chart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(example, "ARTIFACT_DIR", tmp_path)
    context = ToolExecutionContext(run_id="empty-chart")
    example._store_dataset(
        [],
        columns=["month", "revenue", "orders"],
        context=context,
    )
    monkeypatch.setattr(example, "_load_pyplot", lambda: _FakePyplot())

    result = example.plot_graph.function(
        x_column="month",
        y_column="revenue",
        title="빈 기간",
        chart_type="line",
        context=context,
    )

    assert result["point_count"] == 0
    assert Path(result["chart_path"]).read_bytes() == b"empty-chart"

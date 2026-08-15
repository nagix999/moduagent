"""Report automation Agent using only query_db and plot_graph.

SQLite (default):
    python -m pip install -e . matplotlib

PostgreSQL:
    python -m pip install -e . matplotlib "psycopg[binary]>=3.2,<4"
    export REPORT_DB_BACKEND="postgresql"
    export REPORT_DATABASE_URL="postgresql://report_reader@localhost:5432/reporting"

Run either backend:
    export VLLM_BASE_URL="http://localhost:8000/v1"
    export VLLM_MODEL="your-tool-capable-model"
    export VLLM_API_KEY="optional-token"
    python examples/report_automation_agent.py

For PostgreSQL, use a dedicated role with only CONNECT, schema USAGE, and
SELECT privileges. The Tool also enforces a read-only transaction and database
timeouts, but application-side checks are not a substitute for database grants.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from moduagent import (
    Agent,
    AgentConfig,
    ConsoleEventSink,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    LLMPlanGenerator,
    PlanExecutionProfile,
    PydanticOutputCodec,
    RetryConfig,
    RunLimits,
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolFailureRecoveryConfig,
    ToolRecoveryAction,
    VLLMClient,
    function_tool,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("REPORT_DB_PATH", ROOT_DIR / "report_demo.db")).resolve()
ARTIFACT_DIR = Path(
    os.getenv("REPORT_ARTIFACT_DIR", ROOT_DIR / "report_artifacts")
).resolve()


class ReportMetric(BaseModel):
    name: str = Field(description="지표 이름")
    value: float = Field(description="검증된 수치")
    unit: str = Field(description="원, 건, %, 배 등 단위")


class ReportOutput(BaseModel):
    title: str
    period: str
    summary: str
    key_metrics: list[ReportMetric]
    insights: list[str]
    chart_path: str


def _run_artifact(context: ToolExecutionContext, suffix: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", context.run_id)
    if not safe_run_id:
        raise ValueError("run ID is unavailable")
    return ARTIFACT_DIR / f"{safe_run_id}.{suffix}"


def _map_query_error(exc: Exception) -> ToolError | None:
    """Expose only a stable, non-sensitive repair hint to the model."""

    if isinstance(exc, sqlite3.OperationalError) and "interrupted" in str(exc).lower():
        return ToolError(
            type=ToolErrorType.TIMEOUT,
            reason="query_timeout",
            message="The read-only query exceeded its database execution deadline.",
            retryable=True,
            recovery=ToolRecoveryAction.RETRY_CALL,
        )
    if not isinstance(exc, (sqlite3.Error, ValueError)):
        return None
    return ToolError(
        type=ToolErrorType.EXECUTION_ERROR,
        reason="invalid_read_query",
        message=(
            "The read-only SQL could not be executed. Correct its syntax, "
            "table names, column names, or limit."
        ),
        retryable=False,
        recovery=ToolRecoveryAction.REPAIR_CALL,
    )


class _PostgreSQLConfigurationError(RuntimeError):
    """Safe configuration error raised before a PostgreSQL query starts."""


def _map_postgresql_query_error(exc: Exception) -> ToolError:
    """Map PostgreSQL failures without exposing DSNs, SQL, or server details."""

    if isinstance(exc, _PostgreSQLConfigurationError):
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="postgres_configuration_error",
            message=(
                "PostgreSQL query_db is not configured correctly. Check "
                "REPORT_DATABASE_URL and REPORT_* timeout settings, and "
                "install psycopg 3."
            ),
            retryable=False,
            recovery=ToolRecoveryAction.FAIL,
        )

    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate == "42501" or (isinstance(sqlstate, str) and sqlstate.startswith("28")):
        return ToolError(
            type=ToolErrorType.UNAUTHORIZED,
            reason="database_access_denied",
            message="The reporting database denied access to the requested data.",
            retryable=False,
            recovery=ToolRecoveryAction.FAIL,
        )
    if isinstance(sqlstate, str) and sqlstate.startswith("08"):
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="postgres_connection_unavailable",
            message="The PostgreSQL connection is temporarily unavailable.",
            retryable=True,
            recovery=ToolRecoveryAction.RETRY_CALL,
        )
    if sqlstate == "57014":
        return ToolError(
            type=ToolErrorType.TIMEOUT,
            reason="query_timeout",
            message=(
                "The read-only query exceeded its database deadline. "
                "Reduce its date range, joins, or aggregation scope."
            ),
            retryable=False,
            recovery=ToolRecoveryAction.REPAIR_CALL,
        )
    if sqlstate == "55P03":
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="query_lock_timeout",
            message="The read-only query could not acquire a database lock in time.",
            retryable=True,
            recovery=ToolRecoveryAction.RETRY_CALL,
        )
    if (
        isinstance(exc, ValueError)
        or sqlstate == "25006"
        or (isinstance(sqlstate, str) and sqlstate.startswith(("22", "42")))
    ):
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="invalid_read_query",
            message=(
                "The read-only PostgreSQL query could not be executed. Correct "
                "its syntax, table names, column names, data types, or limit."
            ),
            retryable=False,
            recovery=ToolRecoveryAction.REPAIR_CALL,
        )
    if isinstance(exc, OSError):
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="dataset_artifact_failed",
            message="The query dataset artifact could not be stored.",
            retryable=False,
            recovery=ToolRecoveryAction.FAIL,
        )

    try:
        import psycopg
    except ImportError:
        pass
    else:
        if isinstance(exc, psycopg.OperationalError):
            return ToolError(
                type=ToolErrorType.EXECUTION_ERROR,
                reason="postgres_connection_unavailable",
                message="The PostgreSQL connection is temporarily unavailable.",
                retryable=True,
                recovery=ToolRecoveryAction.RETRY_CALL,
            )
    return ToolError(
        type=ToolErrorType.EXECUTION_ERROR,
        reason="postgres_query_failed",
        message="The PostgreSQL read-only query failed.",
        retryable=False,
        recovery=ToolRecoveryAction.REPLAN,
    )


def _map_plot_error(exc: Exception) -> ToolError | None:
    if isinstance(exc, FileNotFoundError):
        return ToolError(
            type=ToolErrorType.NOT_FOUND,
            reason="query_dataset_missing",
            message="Run query_db successfully before plot_graph.",
            retryable=False,
            recovery=ToolRecoveryAction.REPLAN,
        )
    if not isinstance(exc, (TypeError, ValueError)):
        return None
    return ToolError(
        type=ToolErrorType.INVALID_ARGUMENTS,
        reason="invalid_chart_configuration",
        message=(
            "The chart configuration is invalid. Correct the selected columns "
            "or chart type."
        ),
        retryable=False,
        recovery=ToolRecoveryAction.REPAIR_CALL,
    )


def _load_pyplot() -> Any:
    """Load matplotlib lazily with a headless, writable configuration."""

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def _validated_read_query(sql: str, limit: int) -> str:
    statement = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", statement, flags=re.IGNORECASE):
        raise ValueError("only SELECT or WITH queries are allowed")
    if ";" in statement:
        raise ValueError("only one SQL statement is allowed")
    if len(statement) > 20_000:
        raise ValueError("query must be at most 20000 characters")
    if isinstance(limit, bool) or not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    return statement


def _cache_query_rows(
    fetched: list[Any],
    *,
    limit: int,
    context: ToolExecutionContext,
) -> dict[str, Any]:
    truncated = len(fetched) > limit
    rows = [dict(row) for row in fetched[:limit]]
    if not rows:
        raise ValueError("query returned no rows")

    dataset_path = _run_artifact(context, "dataset.json")
    temporary_path = dataset_path.with_suffix(".tmp.json")
    try:
        temporary_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary_path, dataset_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "columns": list(rows[0]),
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "dataset_path": str(dataset_path),
    }


@function_tool(
    idempotent=True,
    repair_safe=True,
    timeout_seconds=20,
    max_result_bytes=64 * 1024,
    error_mapper=_map_query_error,
)
def query_db(
    sql: str,
    context: ToolExecutionContext,
    limit: int = 200,
) -> dict[str, Any]:
    """Execute one read-only SQLite SELECT and cache its rows for plot_graph.

    The SQL must return the exact aliases needed by the chart, such as
    `month` and `revenue`.
    """

    statement = _validated_read_query(sql, limit)
    if not DB_PATH.exists():
        raise ValueError("report database does not exist")

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA query_only = ON")
        query_deadline = time.monotonic() + 15
        connection.set_progress_handler(
            lambda: int(time.monotonic() >= query_deadline),
            1_000,
        )
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(statement)
        fetched = cursor.fetchmany(limit + 1)

    return _cache_query_rows(fetched, limit=limit, context=context)


def _bounded_environment_integer(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise _PostgreSQLConfigurationError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise _PostgreSQLConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _open_postgresql_connection(
    dsn: str,
    *,
    connect_timeout_seconds: int,
) -> Any:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise _PostgreSQLConfigurationError(
            'install "psycopg[binary]>=3.2,<4"'
        ) from exc

    return psycopg.connect(
        dsn,
        autocommit=True,
        connect_timeout=connect_timeout_seconds,
        row_factory=dict_row,
        application_name="moduagent-report-example",
    )


@function_tool(
    name="query_db",
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=False,
    timeout_seconds=35,
    max_result_bytes=64 * 1024,
    error_mapper=_map_postgresql_query_error,
)
def query_db_postgresql(
    sql: str,
    context: ToolExecutionContext,
    limit: int = 200,
) -> dict[str, Any]:
    """Execute one bounded, read-only PostgreSQL query for plot_graph.

    Configure REPORT_DATABASE_URL and use a database role restricted to
    CONNECT, schema USAGE, and SELECT. The SQL must return the exact aliases
    needed by the chart, such as `month` and `revenue`.
    """

    statement = _validated_read_query(sql, limit)
    dsn = os.getenv("REPORT_DATABASE_URL", "").strip()
    if not dsn:
        raise _PostgreSQLConfigurationError("REPORT_DATABASE_URL is required")
    connect_timeout_seconds = _bounded_environment_integer(
        "REPORT_CONNECT_TIMEOUT_SECONDS",
        default=5,
        minimum=1,
        maximum=10,
    )
    statement_timeout_ms = _bounded_environment_integer(
        "REPORT_QUERY_TIMEOUT_MS",
        default=15_000,
        minimum=100,
        maximum=18_000,
    )
    lock_timeout_ms = _bounded_environment_integer(
        "REPORT_LOCK_TIMEOUT_MS",
        default=3_000,
        minimum=100,
        maximum=10_000,
    )

    bounded_statement = (
        f"SELECT * FROM ({statement}) AS moduagent_query LIMIT {limit + 1}"
    )
    with _open_postgresql_connection(
        dsn,
        connect_timeout_seconds=connect_timeout_seconds,
    ) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(statement_timeout_ms),),
                )
                cursor.execute(
                    "SELECT set_config('lock_timeout', %s, true)",
                    (str(lock_timeout_ms),),
                )
                cursor.execute(bounded_statement)
                description = cursor.description
                if description is None:
                    raise ValueError("query must return rows")
                columns = [column.name for column in description]
                if len(columns) != len(set(columns)):
                    raise ValueError("query columns must use unique aliases")
                fetched = cursor.fetchmany(limit + 1)

    return _cache_query_rows(fetched, limit=limit, context=context)


@function_tool(
    idempotent=True,
    repair_safe=True,
    timeout_seconds=20,
    max_result_bytes=4096,
    error_mapper=_map_plot_error,
)
def plot_graph(
    x_column: str,
    y_column: str,
    title: str,
    context: ToolExecutionContext,
    chart_type: Literal["bar", "line"] = "bar",
) -> dict[str, Any]:
    """Plot the latest query_db rows for this run and return the PNG path."""

    dataset_path = _run_artifact(context, "dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError("query dataset is missing")

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("query dataset is empty")
    if any(
        not isinstance(row, dict) or x_column not in row or y_column not in row
        for row in rows
    ):
        raise ValueError("selected chart columns do not exist")

    x_values = [str(row[x_column]) for row in rows]
    try:
        y_values = [float(row[y_column]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError("y column must contain numeric values") from exc

    chart_path = _run_artifact(context, "chart.png")
    temporary_path = chart_path.with_suffix(".tmp.png")
    plt = _load_pyplot()
    figure, axes = plt.subplots(figsize=(10, 5))
    try:
        if chart_type == "bar":
            axes.bar(x_values, y_values, color="#4C78A8")
        else:
            axes.plot(x_values, y_values, marker="o", color="#4C78A8")
        axes.set_title(title)
        axes.set_xlabel(x_column)
        axes.set_ylabel(y_column)
        axes.grid(axis="y", alpha=0.25)
        axes.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(temporary_path, dpi=150)
        os.replace(temporary_path, chart_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)

    return {
        "chart_path": str(chart_path),
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "point_count": len(rows),
    }


def seed_demo_database() -> None:
    """Create a small database only when REPORT_DB_PATH does not exist."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("2025-01", "online", 125_000_000, 1_240),
        ("2025-01", "store", 82_000_000, 710),
        ("2025-02", "online", 131_000_000, 1_290),
        ("2025-02", "store", 79_000_000, 685),
        ("2025-03", "online", 148_000_000, 1_430),
        ("2025-03", "store", 88_000_000, 742),
        ("2025-04", "online", 155_000_000, 1_510),
        ("2025-04", "store", 91_000_000, 768),
        ("2025-05", "online", 162_000_000, 1_580),
        ("2025-05", "store", 95_000_000, 801),
        ("2025-06", "online", 174_000_000, 1_690),
        ("2025-06", "store", 99_000_000, 825),
    ]
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE sales (
                month TEXT NOT NULL,
                channel TEXT NOT NULL,
                revenue REAL NOT NULL,
                orders INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO sales(month, channel, revenue, orders) VALUES (?, ?, ?, ?)",
            rows,
        )


async def main() -> None:
    database_backend = os.getenv("REPORT_DB_BACKEND", "sqlite").strip().lower()
    if database_backend not in {"sqlite", "postgresql"}:
        raise RuntimeError("REPORT_DB_BACKEND must be either 'sqlite' or 'postgresql'.")
    if database_backend == "sqlite":
        if not DB_PATH.exists():
            seed_demo_database()
        selected_query_tool = query_db
    else:
        if not os.getenv("REPORT_DATABASE_URL", "").strip():
            raise RuntimeError(
                "Set REPORT_DATABASE_URL when REPORT_DB_BACKEND=postgresql."
            )
        selected_query_tool = query_db_postgresql

    model_name = os.getenv("VLLM_MODEL")
    if not model_name:
        raise RuntimeError("Set the VLLM_MODEL environment variable.")

    model = VLLMClient(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=model_name,
        api_key=os.getenv("VLLM_API_KEY"),
        timeout=60,
    )

    agent = Agent(
        config=AgentConfig(
            name="report-automation-agent",
            instructions=(
                "당신은 경영 리포트 자동화 에이전트다. 사용할 수 있는 "
                "business Tool은 query_db와 plot_graph뿐이다. DB schema는 "
                "sales(month TEXT, channel TEXT, revenue REAL, orders INTEGER)다. "
                f"현재 DB backend는 {database_backend}다. "
                "먼저 query_db를 정확히 한 번 호출해 월별 집계 데이터를 "
                "조회하고 month, revenue, orders alias를 사용한다. 다음 "
                "plot_graph를 호출해 query_db가 같은 run에 저장한 데이터로 "
                "월별 revenue 그래프를 만든다. SQL 결과와 검증·커밋된 단계 "
                "결과만 사용하고 수치를 추측하지 않는다. 최종 응답은 한국어 "
                "경영 리포트이며 chart_path는 plot_graph 반환값을 그대로 쓴다."
            ),
            limits=RunLimits(
                max_steps=4,
                max_tool_calls=4,
                timeout_seconds=180,
                parallel_tool_calls=False,
                max_step_attempts=2,
                max_replans=1,
                max_tool_repair_attempts=1,
            ),
            retry=RetryConfig(max_attempts=2),
            model_options={"temperature": 0.0},
        ),
        model=model,
        tools=[selected_query_tool, plot_graph],
        execution_profile=PlanExecutionProfile(
            plan_generator=LLMPlanGenerator(model=model, max_steps=4),
            revise_on_tool_failure=True,
            tool_failure_recovery=ToolFailureRecoveryConfig(
                fallback="replan",
                require_repair_safe=True,
                feedback_mode="safe_message",
            ),
        ),
        output_codec=PydanticOutputCodec(model=ReportOutput),
        conversation_store=InMemoryConversationStore(ttl_seconds=3600),
        checkpoint_store=InMemoryCheckpointStore(),
        event_sink=ConsoleEventSink(language="ko"),
    )

    result = await agent.run(
        (
            "2025년 상반기 월별 총매출과 주문 수를 분석하고, 월별 총매출 "
            "그래프를 포함한 경영진용 리포트를 작성해줘."
        ),
        session_id="report-demo",
    )
    if result.error:
        error_summary = result.metadata.get("error_summary", {})
        raise RuntimeError(f"{result.error}: {error_summary}")

    report: ReportOutput = result.output
    print(report.model_dump_json(indent=2))
    print(f"\nChart: {report.chart_path}")


if __name__ == "__main__":
    asyncio.run(main())

"""Gemma 4 + vLLM PostgreSQL report Agent.

Install:
    python -m pip install -e . "psycopg[binary]>=3.2,<4" matplotlib

Configure:
    export VLLM_BASE_URL="http://localhost:8000/v1"
    export VLLM_MODEL="your-gemma-4-model"
    export VLLM_API_KEY="optional-token"
    export REPORT_DATABASE_URL="postgresql://report_reader@localhost/reporting"

Run:
    python examples/gemma4_postgresql_report_agent.py

Set REPORT_DEBUG=1 to print the model/Tool event timeline and full Tool
arguments. Do not enable it where SQL or report filters are sensitive.

Before using this example, replace DATABASE_SCHEMA and the reference query in
REPORT_AGENT_INSTRUCTIONS with the exact production reporting schema. Use a
dedicated PostgreSQL role that has only CONNECT, schema USAGE, and SELECT.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from moduagent import (
    Agent,
    AgentConfig,
    AgentResult,
    EventType,
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
ARTIFACT_DIR = Path(
    os.getenv("REPORT_ARTIFACT_DIR", ROOT_DIR / "report_artifacts")
).resolve()
MAX_QUERY_ROWS = 200


# This is the authoritative schema context for PLAN, ACT, repair, and FINALIZE.
# Replace this single constant when connecting the example to a real database.
DATABASE_SCHEMA = """\
Database dialect: PostgreSQL 16
Schema version: report-schema-v1
Business timezone: Asia/Seoul
Currency: KRW

Table: reporting.orders
Grain: exactly one row per order.

Columns:
- order_id BIGINT PRIMARY KEY
- created_at TIMESTAMPTZ NOT NULL
  Order creation timestamp.
- channel TEXT NOT NULL
  Allowed values: 'online', 'store', 'partner'.
- status TEXT NOT NULL
  Allowed values: 'paid', 'cancelled', 'refunded'.
- net_revenue NUMERIC(18, 2) NOT NULL
  Recognized order revenue in KRW.

Metric definitions:
- revenue = SUM(net_revenue) for rows where status = 'paid'.
- orders = COUNT(*) for rows where status = 'paid'.
- month = calendar month in Asia/Seoul, formatted as YYYY-MM.

No other table, column, relationship, or enum value is available.
"""


REPORT_AGENT_INSTRUCTIONS = f"""\
You are a PostgreSQL management-report automation agent.
Write the final public report in Korean.

GOAL
Produce an evidence-based report using only verified database results.
You have exactly two business Tools:
- query_data(query)
- plot_graph(x_column, y_column, title, chart_type)
Never invent, estimate, interpolate, or silently alter database values.

AUTHORITATIVE DATABASE CONTRACT
{DATABASE_SCHEMA}

MANDATORY WORKFLOW
1. Create the smallest dependency-linked plan:
   - first, a query step that allows only query_data;
   - second, a chart step that depends on the query step and allows only
     plot_graph.
   Do not add a final-report plan step. The runtime performs finalization.
2. In the normal path, call query_data once and return chart-ready aliases.
3. Call plot_graph only after query_data succeeds. Its x_column and y_column
   must exactly match names returned by query_data.
4. Make only one Tool call in each model response. Never call query_data and
   plot_graph in the same response.
5. Use only successful, committed Tool results as factual evidence.

POSTGRESQL QUERY RULES
- Use PostgreSQL syntax only.
- Use exactly one read-only SELECT statement, optionally preceded by WITH.
- Use only reporting.orders and the columns declared above.
- Schema-qualify the table as reporting.orders.
- Do not use SELECT *, comments, a semicolon, multiple statements, DDL, DML,
  PRAGMA, transaction commands, or administrative functions.
- Use lowercase unquoted identifiers. Use one balanced pair of single quotes
  around every string, date, or timestamp literal.
- Valid comparison operators include =, <>, <, <=, >, >=, IN, LIKE, ILIKE,
  IS NULL, and IS NOT NULL. Never use << or >> as a filter operator.
- Filter timestamps with a half-open interval:
    created_at >= TIMESTAMPTZ '<start-date> 00:00:00+09:00'
    AND created_at < TIMESTAMPTZ '<exclusive-end-date> 00:00:00+09:00'
- Do not use BETWEEN for timestamp period boundaries.
- Use explicit, unique, lowercase aliases. For the example monthly report,
  return exactly month, revenue, and orders.
- The y-axis alias must contain numeric values.
- Ensure every selected non-aggregate expression is valid in GROUP BY.
- Order time-series rows chronologically and use LIMIT {MAX_QUERY_ROWS} or less.

REFERENCE QUERY SHAPE
Adapt only the requested period when the requested metrics match this example.
Do not copy a name that is absent from the authoritative contract.

SELECT
  to_char(
    date_trunc('month', o.created_at AT TIME ZONE 'Asia/Seoul'),
    'YYYY-MM'
  ) AS month,
  SUM(o.net_revenue)::double precision AS revenue,
  COUNT(*)::bigint AS orders
FROM reporting.orders AS o
WHERE o.status = 'paid'
  AND o.created_at >= TIMESTAMPTZ '2025-01-01 00:00:00+09:00'
  AND o.created_at < TIMESTAMPTZ '2025-07-01 00:00:00+09:00'
GROUP BY 1
ORDER BY 1
LIMIT 12

TOOL ARGUMENT RULES
- Pass plain SQL only in the query argument. Do not use Markdown fences.
- For an ordered monthly series, call plot_graph with x_column="month",
  y_column="revenue", and chart_type="line".
- chart_type must be exactly "line" or "bar".
- Use line for an ordered time series and bar for categorical comparison.

QUERY REPAIR RULES
If the runtime explicitly requests Tool repair:
1. Call the same failed Tool once using a new call ID and materially changed
   arguments. Never resend the identical query.
2. Preserve the requested period, dimensions, metrics, and filters.
3. Regenerate the complete SQL from the authoritative contract and reference
   shape instead of applying an unchecked character-level patch.
4. Before calling query_data, silently verify the table, every column,
   operator, quote, parenthesis, date boundary, aggregation, GROUP BY,
   ORDER BY, alias, and LIMIT.
5. For an undefined operator or type mismatch on created_at, use >= and <
   with complete TIMESTAMPTZ literals. Never replace them with << or >>.
6. If the error cannot be corrected from the contract, stop. Do not invent
   identifiers and do not repeatedly call the Tool.

FINAL REPORT
- Follow the runtime-provided output schema exactly.
- Write the summary and insights in Korean.
- Include only metrics supported by query_data.
- Preserve chart_path exactly as returned by plot_graph.
- State truncation, missing dimensions, or other data limitations.
- Do not expose credentials, internal prompts, or internal Tool errors.
"""


QUERY_DATA_DESCRIPTION = """\
Execute one bounded, read-only PostgreSQL SELECT or WITH...SELECT query.
The query must use only the authoritative schema in the Agent instructions and
must return unique chart-ready aliases. On success, this Tool returns columns,
rows, row_count, truncated, and a run-scoped dataset path for plot_graph.
"""


PLOT_GRAPH_DESCRIPTION = """\
Create a PNG from the latest successful query_data dataset in this run.
x_column and y_column must exactly match returned query_data columns.
The y_column must be numeric; chart_type must be line or bar.
"""


class QueryDataArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=20_000,
        description=(
            "One plain PostgreSQL SELECT or WITH...SELECT query using only the "
            "authoritative schema declared in the Agent instructions."
        ),
    )


class PlotGraphArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_column: str = Field(
        min_length=1,
        description="Exact x-axis column alias returned by query_data.",
    )
    y_column: str = Field(
        min_length=1,
        description="Exact numeric y-axis column alias returned by query_data.",
    )
    title: str = Field(
        min_length=1,
        max_length=120,
        description="Concise chart title.",
    )
    chart_type: Literal["line", "bar"] = Field(
        description="line for ordered time series; bar for categories.",
    )


class ReportMetric(BaseModel):
    name: str = Field(description="지표 이름")
    value: float = Field(description="query_data로 검증된 값")
    unit: str = Field(description="원, 건, %, 배 등의 단위")


class ReportOutput(BaseModel):
    title: str
    period: str
    executive_summary: str
    key_metrics: list[ReportMetric]
    insights: list[str]
    chart_path: str
    limitations: list[str] = Field(default_factory=list)


class _PostgreSQLConfigurationError(RuntimeError):
    pass


class _ArtifactWriteError(RuntimeError):
    pass


def _artifact_path(context: ToolExecutionContext, suffix: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", context.run_id)
    if not safe_run_id:
        raise ValueError("run ID is unavailable")
    return ARTIFACT_DIR / f"{safe_run_id}.{suffix}"


def _validate_read_query(query: str) -> str:
    statement = query.strip()
    if not statement:
        raise ValueError("query cannot be empty")
    if len(statement) > 20_000:
        raise ValueError("query must be at most 20000 characters")
    if ";" in statement:
        raise ValueError("query must contain exactly one statement without semicolon")
    if not re.match(r"^(select|with)\b", statement, flags=re.IGNORECASE):
        raise ValueError("only SELECT or WITH...SELECT is allowed")
    if re.search(r"(--|/\*|\*/)", statement):
        raise ValueError("SQL comments are not allowed")
    if re.search(r"\bselect\s+\*", statement, flags=re.IGNORECASE):
        raise ValueError("SELECT * is not allowed")

    forbidden = re.compile(
        r"\b("
        r"insert|update|delete|merge|create|alter|drop|truncate|copy|call|do|"
        r"grant|revoke|set|reset|vacuum|analyze|cluster|refresh|reindex|"
        r"pg_sleep|dblink|lo_import|lo_export"
        r")\b",
        flags=re.IGNORECASE,
    )
    if forbidden.search(statement):
        raise ValueError("query contains a forbidden operation")
    return statement


def _open_postgresql_connection(dsn: str) -> Any:
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
        connect_timeout=5,
        row_factory=dict_row,
        application_name="moduagent-gemma4-report",
    )


def _store_dataset(
    rows: list[dict[str, Any]],
    *,
    context: ToolExecutionContext,
) -> Path:
    dataset_path = _artifact_path(context, "dataset.json")
    temporary_path = dataset_path.with_suffix(".tmp.json")
    normalized_rows = json.loads(json.dumps(rows, ensure_ascii=False, default=str))
    try:
        temporary_path.write_text(
            json.dumps(normalized_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, dataset_path)
    except OSError as exc:
        raise _ArtifactWriteError("dataset artifact could not be stored") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return dataset_path


def _map_query_error(exc: Exception) -> ToolError:
    if isinstance(exc, _PostgreSQLConfigurationError):
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="postgres_configuration_error",
            message=(
                "PostgreSQL is not configured. Check REPORT_DATABASE_URL and "
                "install psycopg 3."
            ),
            retryable=False,
            recovery=ToolRecoveryAction.FAIL,
        )
    if isinstance(exc, _ArtifactWriteError):
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="dataset_artifact_failed",
            message="The query dataset artifact could not be stored.",
            retryable=False,
            recovery=ToolRecoveryAction.FAIL,
        )

    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate == "42501" or (isinstance(sqlstate, str) and sqlstate.startswith("28")):
        return ToolError(
            type=ToolErrorType.UNAUTHORIZED,
            reason="database_access_denied",
            message="The reporting database denied access.",
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
    if sqlstate == "55P03":
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="query_lock_timeout",
            message="The read-only query could not acquire a lock in time.",
            retryable=True,
            recovery=ToolRecoveryAction.RETRY_CALL,
        )
    if sqlstate == "57014":
        return ToolError(
            type=ToolErrorType.TIMEOUT,
            reason="query_timeout",
            message=(
                "The query exceeded its deadline. Reduce its period, joins, "
                "or aggregation scope without changing the requested metric."
            ),
            retryable=False,
            recovery=ToolRecoveryAction.REPAIR_CALL,
        )
    if sqlstate == "42601":
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="postgres_syntax_error",
            message=(
                "The PostgreSQL syntax is invalid. Regenerate the complete "
                "SELECT from the declared schema and reference query."
            ),
            retryable=False,
            recovery=ToolRecoveryAction.REPAIR_CALL,
        )
    if sqlstate == "42703":
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="undefined_column",
            message="Use only an exact column declared in the database contract.",
            retryable=False,
            recovery=ToolRecoveryAction.REPAIR_CALL,
        )
    if sqlstate == "42P01":
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="undefined_table",
            message="Use the exact schema-qualified table reporting.orders.",
            retryable=False,
            recovery=ToolRecoveryAction.REPAIR_CALL,
        )
    if sqlstate == "42883":
        return ToolError(
            type=ToolErrorType.EXECUTION_ERROR,
            reason="undefined_operator_or_function",
            message=(
                "Use an operator compatible with the declared column type. "
                "For created_at boundaries use >= and < with complete "
                "TIMESTAMPTZ literals; never use << or >>."
            ),
            retryable=False,
            recovery=ToolRecoveryAction.REPAIR_CALL,
        )
    if isinstance(exc, ValueError) or (
        isinstance(sqlstate, str) and sqlstate.startswith(("22", "42"))
    ):
        return ToolError(
            type=ToolErrorType.INVALID_ARGUMENTS,
            reason="invalid_read_query",
            message=(
                "Regenerate one read-only PostgreSQL query using only the "
                "declared tables, columns, types, aliases, and operators."
            ),
            retryable=False,
            recovery=ToolRecoveryAction.REPAIR_CALL,
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
        message="The read-only PostgreSQL query failed.",
        retryable=False,
        recovery=ToolRecoveryAction.REPLAN,
    )


def _map_plot_error(exc: Exception) -> ToolError:
    if isinstance(exc, FileNotFoundError):
        return ToolError(
            type=ToolErrorType.NOT_FOUND,
            reason="query_dataset_missing",
            message="Run query_data successfully before plot_graph.",
            retryable=False,
            recovery=ToolRecoveryAction.REPLAN,
        )
    if isinstance(exc, (TypeError, ValueError)):
        return ToolError(
            type=ToolErrorType.INVALID_ARGUMENTS,
            reason="invalid_chart_configuration",
            message=(
                "Use exact query_data column names, a numeric y column, and "
                "chart_type line or bar."
            ),
            retryable=False,
            recovery=ToolRecoveryAction.REPAIR_CALL,
        )
    return ToolError(
        type=ToolErrorType.EXECUTION_ERROR,
        reason="chart_generation_failed",
        message="The chart artifact could not be generated.",
        retryable=False,
        recovery=ToolRecoveryAction.FAIL,
    )


@function_tool(
    name="query_data",
    description=QUERY_DATA_DESCRIPTION,
    input_model=QueryDataArguments,
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=False,
    timeout_seconds=30,
    max_result_bytes=96 * 1024,
    error_mapper=_map_query_error,
)
def query_data(
    query: str,
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """Execute one read-only PostgreSQL query and cache its rows for plotting."""

    statement = _validate_read_query(query)
    dsn = os.getenv("REPORT_DATABASE_URL", "").strip()
    if not dsn:
        raise _PostgreSQLConfigurationError("REPORT_DATABASE_URL is required")

    bounded_query = (
        f"SELECT * FROM ({statement}) AS moduagent_report_query "
        f"LIMIT {MAX_QUERY_ROWS + 1}"
    )
    with _open_postgresql_connection(dsn) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    ("15000",),
                )
                cursor.execute(
                    "SELECT set_config('lock_timeout', %s, true)",
                    ("3000",),
                )
                cursor.execute(bounded_query)
                if cursor.description is None:
                    raise ValueError("query must return rows")
                columns = [column.name for column in cursor.description]
                if len(columns) != len(set(columns)):
                    raise ValueError("query columns must use unique aliases")
                fetched = cursor.fetchmany(MAX_QUERY_ROWS + 1)

    if not fetched:
        raise ValueError("query returned no rows")
    truncated = len(fetched) > MAX_QUERY_ROWS
    rows = [dict(row) for row in fetched[:MAX_QUERY_ROWS]]
    dataset_path = _store_dataset(rows, context=context)
    normalized_rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    return {
        "columns": columns,
        "rows": normalized_rows,
        "row_count": len(normalized_rows),
        "truncated": truncated,
        "dataset_path": str(dataset_path),
    }


def _load_pyplot() -> Any:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


@function_tool(
    name="plot_graph",
    description=PLOT_GRAPH_DESCRIPTION,
    input_model=PlotGraphArguments,
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
    chart_type: Literal["line", "bar"],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """Plot the latest successful query_data result for the current run."""

    dataset_path = _artifact_path(context, "dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError("query_data dataset is missing")
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("query_data dataset is empty")
    if any(
        not isinstance(row, dict) or x_column not in row or y_column not in row
        for row in rows
    ):
        raise ValueError("chart columns do not exist in query_data output")

    x_values = [str(row[x_column]) for row in rows]
    try:
        y_values = [float(row[y_column]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError("y_column must contain numeric values") from exc

    chart_path = _artifact_path(context, "chart.png")
    temporary_path = chart_path.with_suffix(".tmp.png")
    pyplot = _load_pyplot()
    figure, axes = pyplot.subplots(figsize=(10, 5))
    try:
        if chart_type == "line":
            axes.plot(x_values, y_values, marker="o", color="#4C78A8")
        else:
            axes.bar(x_values, y_values, color="#4C78A8")
        axes.set_title(title)
        axes.set_xlabel(x_column)
        axes.set_ylabel(y_column)
        axes.grid(axis="y", alpha=0.25)
        axes.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(temporary_path, dpi=150)
        os.replace(temporary_path, chart_path)
    finally:
        pyplot.close(figure)
        temporary_path.unlink(missing_ok=True)

    return {
        "chart_path": str(chart_path),
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "point_count": len(rows),
    }


def build_agent() -> Agent:
    model_name = os.getenv("VLLM_MODEL", "").strip()
    if not model_name:
        raise RuntimeError("Set VLLM_MODEL.")

    model = VLLMClient(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=model_name,
        api_key=os.getenv("VLLM_API_KEY"),
        timeout=60,
    )
    debug = os.getenv("REPORT_DEBUG") == "1"

    return Agent(
        config=AgentConfig(
            name="gemma4-postgresql-report-agent",
            instructions=REPORT_AGENT_INSTRUCTIONS,
            limits=RunLimits(
                max_steps=3,
                max_tool_calls=3,
                timeout_seconds=120,
                parallel_tool_calls=False,
                max_step_attempts=2,
                max_replans=0,
                max_tool_repair_attempts=1,
            ),
            # A malformed HTTP 200 Tool response must not cause a request storm
            # while diagnosing a vLLM parser/template configuration.
            retry=RetryConfig(max_attempts=1),
            model_options={
                "temperature": 0.0,
                "tool_choice": "required",
                "parallel_tool_calls": False,
            },
            tool_trace_mode="arguments" if debug else "summary",
        ),
        model=model,
        tools=[query_data, plot_graph],
        execution_profile=PlanExecutionProfile(
            plan_generator=LLMPlanGenerator(model=model, max_steps=3),
            revise_on_tool_failure=False,
            tool_failure_recovery=ToolFailureRecoveryConfig(
                fallback="fail",
                require_repair_safe=True,
                feedback_mode="safe_message",
            ),
        ),
        output_codec=PydanticOutputCodec(model=ReportOutput),
        conversation_store=InMemoryConversationStore(ttl_seconds=3600),
        checkpoint_store=InMemoryCheckpointStore(),
    )


_DEBUG_EVENTS = {
    EventType.MODEL_STARTED,
    EventType.MODEL_COMPLETED,
    EventType.RETRY,
    EventType.POLICY_DECISION,
    EventType.TOOL_STARTED,
    EventType.TOOL_COMPLETED,
    EventType.TOOL_REPAIR_SCHEDULED,
    EventType.PLAN_REVISED,
    EventType.RUN_COMPLETED,
    EventType.RUN_FAILED,
}


async def run_report(
    agent: Agent,
    request: str,
    *,
    session_id: str,
) -> AgentResult:
    if os.getenv("REPORT_DEBUG") != "1":
        return await agent.run(request, session_id=session_id)

    result: AgentResult | None = None
    async for event in agent.stream_all(request, session_id=session_id):
        if event.type in _DEBUG_EVENTS:
            print(
                json.dumps(
                    {
                        "run_id": event.run_id,
                        "sequence": event.sequence,
                        "event": event.type.value,
                        **dict(event.data),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
                flush=True,
            )
        if event.type in {EventType.RUN_COMPLETED, EventType.RUN_FAILED}:
            candidate = event.data.get("result")
            if isinstance(candidate, AgentResult):
                result = candidate
    if result is None:
        raise RuntimeError("Agent run ended without a terminal result")
    return result


async def main() -> None:
    if not os.getenv("REPORT_DATABASE_URL", "").strip():
        raise RuntimeError("Set REPORT_DATABASE_URL.")

    request = os.getenv(
        "REPORT_REQUEST",
        (
            "2025년 상반기 월별 매출과 주문 수를 분석하고, "
            "월별 매출 추이를 선 그래프로 만든 경영진 리포트를 작성해줘."
        ),
    )
    result = await run_report(
        build_agent(),
        request,
        session_id=os.getenv("REPORT_SESSION_ID", "gemma4-report-demo"),
    )
    if result.error:
        raise RuntimeError(
            json.dumps(
                {
                    "error": result.error,
                    "error_summary": result.metadata.get("error_summary"),
                    "tool_trace": result.metadata.get("tool_trace"),
                },
                ensure_ascii=False,
                default=str,
            )
        )

    report: ReportOutput = result.output
    print(report.model_dump_json(indent=2))
    if os.getenv("REPORT_DEBUG") == "1":
        print(
            json.dumps(
                {"tool_trace": result.metadata.get("tool_trace")},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())

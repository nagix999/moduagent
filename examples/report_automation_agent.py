"""Report automation Agent using only query_db and plot_graph.

Install:
    python -m pip install -e . matplotlib

Run:
    export VLLM_BASE_URL="http://localhost:8000/v1"
    export VLLM_MODEL="your-tool-capable-model"
    export VLLM_API_KEY="optional-token"
    python examples/report_automation_agent.py
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

    statement = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", statement, flags=re.IGNORECASE):
        raise ValueError("only SELECT or WITH queries are allowed")
    if ";" in statement:
        raise ValueError("only one SQL statement is allowed")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
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

    truncated = len(fetched) > limit
    rows = [dict(row) for row in fetched[:limit]]
    if not rows:
        raise ValueError("query returned no rows")

    dataset_path = _run_artifact(context, "dataset.json")
    temporary_path = dataset_path.with_suffix(".tmp.json")
    temporary_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary_path, dataset_path)
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
    if not DB_PATH.exists():
        seed_demo_database()

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
        tools=[query_db, plot_graph],
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

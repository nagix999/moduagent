from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from moduagent import Agent, FinishReason, RetryConfig, RunLimits, VLLMClient, tool


_TOOL_CALLS: list[dict[str, Any]] = []
_SALES = (
    {"month": "2026-01", "revenue": 1200},
    {"month": "2026-02", "revenue": 1500},
    {"month": "2026-03", "revenue": 1800},
)


def _live_environment() -> Mapping[str, str]:
    if os.getenv("MODUAGENT_RUN_LIVE_SCENARIOS", "").strip() != "1":
        pytest.skip("set MODUAGENT_RUN_LIVE_SCENARIOS=1 to run Agent scenarios")
    values = {
        name: os.getenv(name, "").strip() for name in ("VLLM_BASE_URL", "VLLM_MODEL")
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip(f"live vLLM environment is missing: {', '.join(missing)}")
    return values


def _client(*, max_tokens: int) -> VLLMClient:
    env = _live_environment()
    return VLLMClient(
        base_url=env["VLLM_BASE_URL"],
        model=env["VLLM_MODEL"],
        api_key=os.getenv("VLLM_API_KEY", "").strip() or None,
        timeout=60,
        default_options={
            "temperature": 0,
            "max_tokens": max_tokens,
        },
    )


def _limits(*, model_turns: int, tool_calls: int) -> RunLimits:
    return RunLimits(
        max_steps=6,
        max_tool_calls=tool_calls,
        timeout_seconds=120,
        max_model_turns=model_turns,
        no_progress_model_turn_threshold=3,
    )


@tool(timeout_seconds=3, max_result_bytes=1024)
def add(a: int, b: int) -> int:
    """Add two integers and return their exact sum."""

    _TOOL_CALLS.append({"tool": "add", "a": a, "b": b})
    return a + b


@tool(timeout_seconds=3, max_result_bytes=4096)
def query_sales(year: int, quarter: int) -> list[dict[str, Any]]:
    """Return verified monthly sales for one year and quarter."""

    _TOOL_CALLS.append({"tool": "query_sales", "year": year, "quarter": quarter})
    return list(_SALES) if (year, quarter) == (2026, 1) else []


@tool(timeout_seconds=3, max_result_bytes=4096)
def plot_graph(
    x_values: list[str],
    y_values: list[int],
    title: str,
    chart_type: Literal["line", "bar"] = "bar",
) -> str:
    """Create a graph from verified values and return its path."""

    _TOOL_CALLS.append(
        {
            "tool": "plot_graph",
            "x_values": x_values,
            "y_values": y_values,
            "title": title,
            "chart_type": chart_type,
        }
    )
    assert x_values == [str(row["month"]) for row in _SALES]
    assert y_values == [int(row["revenue"]) for row in _SALES]
    return "artifacts/q1-2026-revenue.svg"


class SalesReport(BaseModel):
    title: str
    summary: str
    total_revenue: int = Field(ge=0)
    chart_path: str


def test_vllm_quick_api_calls_one_typed_tool_live() -> None:
    async def scenario() -> None:
        _TOOL_CALLS.clear()
        async with _client(max_tokens=192) as model:
            agent = Agent.create(
                model=model,
                instructions=(
                    "Call the add Tool exactly once for addition. "
                    "Then answer with only the integer result."
                ),
                tools=[add],
                limits=_limits(model_turns=4, tool_calls=1),
                retry=RetryConfig(max_attempts=1),
            )
            result = await agent.run("Add 17 and 25.")

        assert result.finish_reason is FinishReason.COMPLETED, result.explain()
        assert str(result.output).strip() == "42"
        assert _TOOL_CALLS == [{"tool": "add", "a": 17, "b": 25}]
        assert result.run_usage["model_turns"] == 2
        assert result.run_usage["tool_calls"] == 1
        assert result.run_usage["duration_seconds"] > 0
        assert [trace["tool_name"] for trace in result.tool_trace] == ["add"]

    asyncio.run(scenario())


def test_vllm_standard_report_uses_two_tools_and_structured_output_live() -> None:
    async def scenario() -> None:
        _TOOL_CALLS.clear()
        async with _client(max_tokens=512) as model:
            agent = Agent.create(
                model=model,
                instructions=(
                    "Create a Korean sales report from verified Tool results. "
                    "First call query_sales. Then call plot_graph once using the "
                    "exact month and revenue values returned by query_sales. "
                    "Make one Tool call per response and never invent values."
                ),
                tools=[query_sales, plot_graph],
                output=SalesReport,
                limits=_limits(model_turns=8, tool_calls=3),
                retry=RetryConfig(max_attempts=1),
            )
            result = await agent.run("2026년 1분기 월별 매출 리포트를 만들어줘.")

        assert result.finish_reason is FinishReason.COMPLETED, result.explain()
        assert isinstance(result.output, SalesReport)
        assert result.output.total_revenue == 4500
        assert result.output.chart_path == "artifacts/q1-2026-revenue.svg"
        assert [call["tool"] for call in _TOOL_CALLS] == [
            "query_sales",
            "plot_graph",
        ]
        assert result.run_usage["model_turns"] == 4
        assert result.run_usage["tool_calls"] == 2
        assert result.run_usage["duration_seconds"] > 0
        assert [trace["tool_name"] for trace in result.tool_trace] == [
            "query_sales",
            "plot_graph",
        ]

    asyncio.run(scenario())

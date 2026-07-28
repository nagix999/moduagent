from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import BaseModel

from moduagent.config import RetryConfig
from moduagent.messages import ToolCall
from moduagent.tools import (
    AgentTool,
    RBACToolAuthorizer,
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    function_tool,
)


def run(coroutine):
    return asyncio.run(coroutine)


def test_function_tool_builds_schema_and_validates_arguments() -> None:
    @function_tool(description="Add two integers")
    def add(a: int, b: int = 1) -> int:
        return a + b

    schema = add.schema.to_dict()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert schema["function"]["parameters"]["required"] == ["a"]

    executor = ToolExecutor([add])
    success = run(executor.execute(ToolCall("1", "add", {"a": 2})))
    invalid = run(executor.execute(ToolCall("2", "add", {"a": "bad"})))
    extra = run(executor.execute(ToolCall("3", "add", {"a": 2, "x": 3})))

    assert success.success is True
    assert success.value == 3
    assert invalid.error is not None
    assert invalid.error.type is ToolErrorType.INVALID_ARGUMENTS
    assert extra.error is not None
    assert extra.error.type is ToolErrorType.INVALID_ARGUMENTS


def test_explicit_pydantic_input_model_is_supported() -> None:
    class SearchInput(BaseModel):
        query: str
        limit: int = 5

    @function_tool(args_schema=SearchInput)
    def search(query: str, limit: int) -> str:
        return f"{query}:{limit}"

    result = run(
        ToolExecutor([search]).execute(
            ToolCall("search-1", "search", {"query": "policy"})
        )
    )
    assert result.success is True
    assert result.value == "policy:5"


def test_execution_context_is_injected_but_not_exposed_in_schema() -> None:
    @function_tool
    def current_user(context: ToolExecutionContext) -> str:
        return str(context.user_context["user_id"])

    properties = current_user.schema.parameters["properties"]
    assert "context" not in properties

    result = run(
        ToolExecutor([current_user]).execute(
            ToolCall("ctx-1", "current_user", {}),
            ToolExecutionContext(run_id="run-1", user_context={"user_id": "u-1"}),
        )
    )
    assert result.value == "u-1"


def test_rbac_denies_before_invocation_and_supports_wildcard() -> None:
    calls = 0

    @function_tool
    def confidential() -> str:
        nonlocal calls
        calls += 1
        return "secret"

    authorizer = RBACToolAuthorizer({"employee": {"public"}, "admin": {"*"}})
    executor = ToolExecutor([confidential], authorizer=authorizer)

    denied = run(
        executor.execute(
            ToolCall("deny", "confidential", {}),
            ToolExecutionContext(user_context={"roles": ["employee"]}),
        )
    )
    allowed = run(
        executor.execute(
            ToolCall("allow", "confidential", {}),
            ToolExecutionContext(user_context={"roles": "admin"}),
        )
    )

    assert denied.error is not None
    assert denied.error.type is ToolErrorType.UNAUTHORIZED
    assert "not authorized" in denied.model_content()
    assert allowed.success is True
    assert calls == 1


def test_only_idempotent_tools_are_retried() -> None:
    retrying_calls = 0
    one_shot_calls = 0

    @function_tool(idempotent=True)
    def eventually_succeeds() -> str:
        nonlocal retrying_calls
        retrying_calls += 1
        if retrying_calls < 3:
            raise RuntimeError("temporary")
        return "ok"

    @function_tool
    def one_shot() -> str:
        nonlocal one_shot_calls
        one_shot_calls += 1
        raise RuntimeError("failed")

    executor = ToolExecutor(
        [eventually_succeeds, one_shot],
        retry=RetryConfig(max_attempts=3, initial_delay=0, max_delay=0),
    )
    retried = run(executor.execute(ToolCall("retry", "eventually_succeeds", {})))
    not_retried = run(executor.execute(ToolCall("once", "one_shot", {})))

    assert retried.success is True
    assert retried.attempts == 3
    assert retrying_calls == 3
    assert not_retried.success is False
    assert not_retried.attempts == 1
    assert one_shot_calls == 1


def test_sync_tool_does_not_block_event_loop() -> None:
    @function_tool
    def blocking() -> str:
        time.sleep(0.05)
        return "done"

    async def scenario() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(5):
                await asyncio.sleep(0.005)
                ticks += 1

        await asyncio.gather(blocking.invoke({}), ticker())
        return ticks

    assert run(scenario()) == 5


def test_sync_tool_timeout_returns_without_waiting_for_worker() -> None:
    @function_tool(timeout_seconds=0.01)
    def slow() -> str:
        time.sleep(0.2)
        return "late"

    started = time.monotonic()
    result = run(ToolExecutor([slow]).execute(ToolCall("slow-1", "slow", {})))

    assert time.monotonic() - started < 0.15
    assert result.error is not None
    assert result.error.type is ToolErrorType.TIMEOUT
    assert result.attempts == 1


def test_result_size_limit_is_enforced() -> None:
    @function_tool
    def large() -> str:
        return "x" * 100

    result = run(
        ToolExecutor([large], max_result_bytes=20).execute(
            ToolCall("large-1", "large", {})
        )
    )

    assert result.success is False
    assert result.value is None
    assert result.error is not None
    assert result.error.type is ToolErrorType.RESULT_TOO_LARGE
    assert result.error.details["actual_bytes"] > 20


def test_tool_result_normalizes_dataframe_like_records_without_pandas() -> None:
    class FakeDataFrame:
        ndim = 2
        columns = ("amount", "created_at")
        index = (0,)

        def __init__(self) -> None:
            self.orient = None

        def to_dict(self, orient=None):
            self.orient = orient
            return [
                {
                    "amount": Decimal("123.45"),
                    "created_at": datetime(2026, 7, 28, 9, 30, tzinfo=timezone.utc),
                }
            ]

    frame = FakeDataFrame()
    result = ToolResult.succeeded(
        call_id="frame-1",
        tool_name="query",
        value=frame,
    )

    payload = json.loads(result.model_content())

    assert frame.orient == "records"
    assert payload["value"] == [
        {
            "amount": "123.45",
            "created_at": "2026-07-28T09:30:00+00:00",
        }
    ]
    assert result.to_dict()["value"] == payload["value"]


def test_tool_result_normalizes_real_pandas_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    frame = pd.DataFrame(
        {
            "customer_id": [np.int64(7)],
            "queried_at": [pd.Timestamp("2026-07-28T09:30:00+09:00")],
            "optional_note": [pd.NA],
            "not_available_at": [pd.NaT],
        }
    )
    result = ToolResult.succeeded(
        call_id="sql-1",
        tool_name="read_sql",
        value=frame,
    )

    payload = json.loads(result.model_content())

    assert payload["value"] == [
        {
            "customer_id": 7,
            "queried_at": "2026-07-28T09:30:00+09:00",
            "optional_note": None,
            "not_available_at": None,
        }
    ]


def test_tool_result_normalizes_series_numpy_and_common_scalars() -> None:
    class FakeSeries:
        ndim = 1

        def to_dict(self):
            return {"count": FakeNumpyInteger(3)}

    class FakeNumpyInteger:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    FakeNumpyInteger.__module__ = "numpy"

    result = ToolResult.succeeded(
        call_id="series-1",
        tool_name="query",
        value={
            "series": FakeSeries(),
            "identifier": UUID("12345678-1234-5678-1234-567812345678"),
            "missing": float("nan"),
        },
    )

    payload = json.loads(result.model_content())

    assert payload["value"] == {
        "series": {"count": 3},
        "identifier": "12345678-1234-5678-1234-567812345678",
        "missing": None,
    }


def test_tool_result_preserves_supported_object_converters() -> None:
    class OutputModel(BaseModel):
        value: Decimal

    @dataclass
    class DataclassOutput:
        value: Decimal

    class DictOutput:
        def to_dict(self):
            return {"value": Decimal("1.25")}

    result = ToolResult.succeeded(
        call_id="objects-1",
        tool_name="query",
        value={
            "pydantic": OutputModel(value=Decimal("2.5")),
            "dataclass": DataclassOutput(value=Decimal("3.5")),
            "to_dict": DictOutput(),
        },
    )

    assert json.loads(result.model_content())["value"] == {
        "pydantic": {"value": "2.5"},
        "dataclass": {"value": "3.5"},
        "to_dict": {"value": "1.25"},
    }


def test_unknown_nested_tool_value_falls_back_locally_to_repr() -> None:
    class Unknown:
        def __repr__(self) -> str:
            return "<unknown-result>"

    result = ToolResult.succeeded(
        call_id="unknown-1",
        tool_name="query",
        value={"rows": [{"known": 1, "unknown": Unknown()}], "row_count": 1},
    )

    payload = json.loads(result.model_content())

    assert payload["value"] == {
        "rows": [{"known": 1, "unknown": "<unknown-result>"}],
        "row_count": 1,
    }


def test_parallel_execution_is_bounded_and_preserves_order() -> None:
    active = 0
    peak = 0

    @function_tool
    async def work(value: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return value

    calls = [ToolCall(str(i), "work", {"value": i}) for i in range(5)]
    results = run(
        ToolExecutor([work]).execute_many(calls, parallel=True, max_parallel=2)
    )

    assert peak == 2
    assert [result.value for result in results] == list(range(5))


def test_registry_rejects_duplicates_and_unknown_tool_is_structured() -> None:
    @function_tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry([ping])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ping)

    result = run(ToolExecutor(registry).execute(ToolCall("missing-1", "missing", {})))
    assert result.error is not None
    assert result.error.type is ToolErrorType.NOT_FOUND
    assert result.attempts == 0


def test_agent_tool_delegates_to_agent_run_without_circular_dependency() -> None:
    class FakeAgent:
        config = SimpleNamespace(name="researcher")

        def __init__(self) -> None:
            self.received = None

        async def run(self, text, *, session_id=None, user_context=None):
            self.received = (text, session_id, user_context)
            return SimpleNamespace(output=f"answer:{text}")

    agent = FakeAgent()
    tool = AgentTool(agent)
    context = ToolExecutionContext(
        run_id="outer-run",
        session_id="session-1",
        user_context={"roles": ["employee"]},
    )
    result = run(
        ToolExecutor([tool]).execute(
            ToolCall("agent-1", "researcher", {"input": "investigate"}),
            context,
        )
    )

    assert result.value == "answer:investigate"
    assert agent.received == (
        "investigate",
        "session-1",
        {"roles": ["employee"]},
    )

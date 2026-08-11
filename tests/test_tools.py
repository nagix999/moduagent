from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from threading import Event
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import BaseModel

from moduagent.config import RetryConfig
from moduagent.messages import ToolCall
from moduagent.tools import (
    AgentTool,
    RBACToolAuthorizer,
    SyncToolScheduler,
    SyncToolSchedulerOverloaded,
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutor,
    ToolFailure,
    ToolRecoveryAction,
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
    assert invalid.error.reason == "invalid_arguments"
    assert invalid.error.recovery is ToolRecoveryAction.REPAIR_CALL
    assert invalid.repair_safe is False
    assert extra.error is not None
    assert extra.error.type is ToolErrorType.INVALID_ARGUMENTS


def test_function_tool_validates_side_effect_classification() -> None:
    @function_tool(side_effect_level="advisory")
    def recommend() -> str:
        return "review"

    assert recommend.side_effect_level == "advisory"

    with pytest.raises(ValueError, match="side_effect_level"):
        function_tool(lambda: None, side_effect_level="external")


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
    assert denied.error.recovery is ToolRecoveryAction.FAIL
    assert "not authorized" in denied.model_content()
    assert allowed.success is True
    assert calls == 1


def test_rbac_policy_mapping_is_immutable_after_fingerprinting() -> None:
    authorizer = RBACToolAuthorizer({"analyst": {"lookup"}})
    fingerprint = authorizer.policy_fingerprint

    with pytest.raises(TypeError):
        authorizer.role_permissions["analyst"] = frozenset({"admin"})

    assert authorizer.policy_fingerprint == fingerprint
    assert authorizer.role_permissions == {"analyst": frozenset({"lookup"})}


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


def test_classified_retry_call_requires_retryable_idempotent_tool() -> None:
    retrying_calls = 0
    one_shot_calls = 0

    def transient_error(exception: Exception) -> ToolError:
        assert isinstance(exception, RuntimeError)
        return ToolError(
            ToolErrorType.EXECUTION_ERROR,
            "service temporarily unavailable",
            retryable=True,
            reason="service_unavailable",
            recovery=ToolRecoveryAction.RETRY_CALL,
        )

    @function_tool(idempotent=True, error_mapper=transient_error)
    def eventually_succeeds() -> str:
        nonlocal retrying_calls
        retrying_calls += 1
        if retrying_calls < 3:
            raise RuntimeError("temporary")
        return "ok"

    @function_tool(error_mapper=transient_error)
    def non_idempotent() -> str:
        nonlocal one_shot_calls
        one_shot_calls += 1
        raise RuntimeError("temporary")

    executor = ToolExecutor(
        [eventually_succeeds, non_idempotent],
        retry=RetryConfig(max_attempts=3, initial_delay=0, max_delay=0),
    )

    retried = run(
        executor.execute(ToolCall("retry-classified", "eventually_succeeds", {}))
    )
    not_retried = run(
        executor.execute(ToolCall("one-shot-classified", "non_idempotent", {}))
    )

    assert retried.success is True
    assert retried.attempts == 3
    assert retrying_calls == 3
    assert not_retried.success is False
    assert not_retried.attempts == 1
    assert one_shot_calls == 1


def test_repair_call_failure_is_immediate_internal_and_model_safe() -> None:
    mapper_calls = 0
    invocations = 0

    def must_not_run(exception: Exception) -> ToolError:
        nonlocal mapper_calls
        mapper_calls += 1
        raise AssertionError("ToolFailure must take precedence over error_mapper")

    @function_tool(
        idempotent=True,
        repair_safe=True,
        error_mapper=must_not_run,
    )
    def query(sql: str) -> str:
        nonlocal invocations
        invocations += 1
        raise ToolFailure(
            ToolError(
                ToolErrorType.EXECUTION_ERROR,
                "The read-only query is invalid.",
                retryable=True,
                details={"hint": "Revise the query structure."},
                reason="invalid_sql",
                recovery=ToolRecoveryAction.REPAIR_CALL,
            )
        )

    result = run(
        ToolExecutor(
            [query],
            retry=RetryConfig(max_attempts=3, initial_delay=0, max_delay=0),
        ).execute(
            ToolCall(
                "repair-1",
                "query",
                {"sql": "SELECT private_column FROM private_table"},
            )
        )
    )

    assert result.success is False
    assert result.attempts == 1
    assert result.repair_safe is True
    assert invocations == 1
    assert mapper_calls == 0
    assert result.error is not None
    assert result.error.code is ToolErrorType.EXECUTION_ERROR
    assert result.error.reason == "invalid_sql"
    assert result.error.recovery is ToolRecoveryAction.REPAIR_CALL

    public_result = result.to_dict()
    model_content = json.loads(result.model_content())
    assert "repair_safe" not in public_result
    assert "invocation_arguments" not in public_result
    assert model_content["error"] == {
        "type": "execution_error",
        "message": "The read-only query is invalid.",
        "retryable": True,
        "reason": "invalid_sql",
        "recovery": "repair_call",
        "details": {"hint": "Revise the query structure."},
    }
    assert "private_column" not in result.model_content()
    assert "private_table" not in result.model_content()


def test_tool_result_rejects_non_tool_error_payload() -> None:
    with pytest.raises(TypeError, match="error must be a ToolError"):
        ToolResult(  # type: ignore[arg-type]
            call_id="invalid-error-1",
            tool_name="lookup",
            success=False,
            error=object(),
        )
    with pytest.raises(TypeError, match="attempts must be an integer"):
        ToolResult.succeeded(
            call_id="invalid-attempts-1",
            tool_name="lookup",
            value="value",
            attempts=float("nan"),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="duration_seconds must be finite"):
        ToolResult.succeeded(
            call_id="invalid-duration-1",
            tool_name="lookup",
            value="value",
            duration_seconds=float("inf"),
        )


def test_error_mapper_payload_is_model_visible_and_must_be_sanitized() -> None:
    raw_error = "syntax error near SELECT secret FROM payroll"

    def safe_mapper(exception: Exception) -> ToolError:
        assert str(exception) == raw_error
        return ToolError(
            ToolErrorType.EXECUTION_ERROR,
            "The query references an unavailable schema object.",
            reason="invalid_identifier",
            recovery=ToolRecoveryAction.REPLAN,
        )

    @function_tool(idempotent=True, error_mapper=safe_mapper)
    def query() -> None:
        raise RuntimeError(raw_error)

    result = run(
        ToolExecutor(
            [query],
            retry=RetryConfig(max_attempts=3, initial_delay=0, max_delay=0),
        ).execute(ToolCall("safe-error-1", "query", {}))
    )

    assert result.attempts == 1
    assert result.error is not None
    assert result.error.recovery is ToolRecoveryAction.REPLAN
    assert raw_error not in result.model_content()
    assert "unavailable schema object" in result.model_content()


@pytest.mark.parametrize(
    ("mapper", "message", "detail_key", "detail_value"),
    [
        (
            lambda _exc: (_ for _ in ()).throw(ValueError("mapper failed")),
            "tool error mapper failed",
            "exception_type",
            "ValueError",
        ),
        (
            lambda _exc: "invalid",
            "tool error mapper returned an invalid result",
            "result_type",
            "str",
        ),
    ],
    ids=["mapper-exception", "invalid-mapper-result"],
)
def test_error_mapper_failures_are_structured(
    mapper,
    message: str,
    detail_key: str,
    detail_value: str,
) -> None:
    @function_tool(error_mapper=mapper)
    def lookup() -> None:
        raise RuntimeError("private backend detail")

    result = run(ToolExecutor([lookup]).execute(ToolCall("mapper-1", "lookup", {})))

    assert result.attempts == 1
    assert result.error is not None
    assert result.error.type is ToolErrorType.EXECUTION_ERROR
    assert result.error.message == message
    assert result.error.details[detail_key] == detail_value
    assert result.error.recovery is ToolRecoveryAction.FAIL
    assert "private backend detail" not in result.model_content()


def test_error_mapper_none_uses_generic_error_classification() -> None:
    def ignore_error(_exception: Exception) -> None:
        return None

    @function_tool(error_mapper=ignore_error)
    def lookup() -> None:
        raise RuntimeError("backend unavailable")

    result = run(ToolExecutor([lookup]).execute(ToolCall("mapper-none", "lookup", {})))

    assert result.attempts == 1
    assert result.error is not None
    assert result.error.type is ToolErrorType.EXECUTION_ERROR
    assert result.error.message == "tool execution failed: backend unavailable"
    assert result.error.retryable is False
    assert result.error.recovery is None


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


def test_idempotent_timeout_is_classified_for_same_call_retry() -> None:
    calls = 0

    @function_tool(idempotent=True, timeout_seconds=0.001)
    async def slow() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "late"

    result = run(
        ToolExecutor(
            [slow],
            retry=RetryConfig(max_attempts=2, initial_delay=0, max_delay=0),
        ).execute(ToolCall("slow-retry-1", "slow", {}))
    )

    assert result.error is not None
    assert result.error.type is ToolErrorType.TIMEOUT
    assert result.error.retryable is True
    assert result.error.recovery is ToolRecoveryAction.RETRY_CALL
    assert result.attempts == 2
    assert calls == 2


def test_sync_timeout_never_overlaps_identical_call_retry_by_default() -> None:
    calls = 0

    def map_timeout(exc: Exception) -> ToolError | None:
        if not isinstance(exc, asyncio.TimeoutError):
            return None
        return ToolError(
            ToolErrorType.TIMEOUT,
            "the backend timed out",
            retryable=True,
            recovery=ToolRecoveryAction.RETRY_CALL,
        )

    @function_tool(
        idempotent=True,
        timeout_seconds=0.005,
        error_mapper=map_timeout,
    )
    def slow() -> str:
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return "late"

    result = run(
        ToolExecutor(
            [slow],
            retry=RetryConfig(max_attempts=2, initial_delay=0, max_delay=0),
        ).execute(ToolCall("sync-timeout-1", "slow", {}))
    )

    assert result.success is False
    assert result.attempts == 1
    assert result.error is not None
    assert result.error.type is ToolErrorType.TIMEOUT
    assert result.error.retryable is False
    assert result.error.reason == "uncancellable_timeout"
    assert result.error.recovery is ToolRecoveryAction.FAIL
    assert calls == 1
    time.sleep(0.06)
    assert calls == 1


def test_sync_timeout_retry_safe_opt_in_allows_identical_call_retry() -> None:
    calls = 0
    release = Event()

    @function_tool(
        idempotent=True,
        timeout_seconds=0.01,
        timeout_retry_safe=True,
    )
    def cancellable_backend() -> str:
        nonlocal calls
        calls += 1
        release.wait(0.1)
        return "late"

    try:
        result = run(
            ToolExecutor(
                [cancellable_backend],
                retry=RetryConfig(max_attempts=2, initial_delay=0, max_delay=0),
            ).execute(ToolCall("sync-timeout-opt-in", "cancellable_backend", {}))
        )
    finally:
        release.set()

    assert result.success is False
    assert result.attempts == 2
    assert result.error is not None
    assert result.error.type is ToolErrorType.TIMEOUT
    assert result.error.retryable is True
    assert result.error.reason == "tool_timeout"
    assert result.error.recovery is ToolRecoveryAction.RETRY_CALL
    assert calls == 2


def test_result_size_limit_is_enforced() -> None:
    @function_tool(repair_safe=True)
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
    assert result.error.reason == "result_too_large"
    assert result.error.recovery is ToolRecoveryAction.REPAIR_CALL
    assert result.repair_safe is True
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


def test_unknown_nested_tool_value_uses_stable_type_projection() -> None:
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
        "rows": [
            {
                "known": 1,
                "unknown": {
                    "unsupported_type": (f"{Unknown.__module__}.{Unknown.__qualname__}")
                },
            }
        ],
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


def test_sync_tool_scheduler_is_bounded_and_reports_health() -> None:
    async def scenario() -> None:
        started = Event()
        release = Event()
        scheduler = SyncToolScheduler(max_workers=1, max_queue=0)

        def blocking() -> str:
            started.set()
            release.wait(timeout=1)
            return "done"

        first = asyncio.create_task(scheduler.run(blocking))
        while not started.is_set():
            await asyncio.sleep(0)

        with pytest.raises(SyncToolSchedulerOverloaded, match="capacity"):
            await scheduler.run(lambda: "rejected")

        release.set()
        assert await first == "done"
        stats = scheduler.stats()
        assert stats.workers == 1
        assert stats.running == 0
        assert stats.queued == 0
        assert stats.submitted == 1
        assert stats.completed == 1
        assert stats.rejected == 1

    run(scenario())


def test_zero_queue_scheduler_accepts_each_worker_slot() -> None:
    async def scenario() -> None:
        all_started = Event()
        release = Event()
        started: list[int] = []
        scheduler = SyncToolScheduler(max_workers=2, max_queue=0)

        def blocking(value: int) -> int:
            started.append(value)
            if len(started) == 2:
                all_started.set()
            release.wait(timeout=1)
            return value

        first = asyncio.create_task(scheduler.run(lambda: blocking(1)))
        second = asyncio.create_task(scheduler.run(lambda: blocking(2)))
        while not all_started.is_set():
            await asyncio.sleep(0)

        with pytest.raises(SyncToolSchedulerOverloaded, match="capacity"):
            await scheduler.run(lambda: 3)

        release.set()
        assert sorted(await asyncio.gather(first, second)) == [1, 2]

    run(scenario())


def test_function_tool_can_use_bounded_sync_scheduler() -> None:
    scheduler = SyncToolScheduler(max_workers=1, max_queue=1)

    @function_tool(sync_scheduler=scheduler)
    def scheduled_add(a: int, b: int) -> int:
        return a + b

    result = run(
        ToolExecutor([scheduled_add]).execute(
            ToolCall("scheduled-1", "scheduled_add", {"a": 2, "b": 3})
        )
    )

    assert result.success is True
    assert result.value == 5
    assert scheduler.stats().completed == 1


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
    assert result.error.recovery is ToolRecoveryAction.FAIL
    assert result.attempts == 0


def test_registry_freeze_blocks_post_validation_replacement() -> None:
    first = function_tool(lambda value: value, name="lookup")
    replacement = function_tool(
        lambda value: f"changed:{value}",
        name="lookup",
    )
    registry = ToolRegistry((first,)).freeze()

    assert registry.is_frozen is True
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(replacement, replace=True)
    with pytest.raises(RuntimeError, match="frozen"):
        registry.unregister("lookup")
    assert registry.require("lookup") is first


def test_regular_tool_cannot_observe_delegation_control_callback() -> None:
    observed: dict[str, object] = {}

    @function_tool
    def inspect_context(context: ToolExecutionContext) -> str:
        observed.update(context.metadata)
        return "ok"

    result = run(
        ToolExecutor((inspect_context,)).execute(
            ToolCall("inspect-1", "inspect_context", {}),
            ToolExecutionContext(
                metadata={
                    "_moduagent_delegation_event_callback": lambda event: event,
                    "_moduagent_parent_delegation_context": object(),
                    "ordinary": "visible",
                }
            ),
        )
    )

    assert result.success is True
    assert observed == {"ordinary": "visible"}


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

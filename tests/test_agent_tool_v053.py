from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from moduagent.config import RetryConfig
from moduagent.errors import (
    AgentRunError,
    CancellationError,
    ModelInvocationError,
    OutputValidationError,
    RunTimeoutError,
)
from moduagent.messages import FinishReason, ToolCall, Usage
from moduagent.runtime.context import AgentResult
from moduagent.tools import (
    AgentTool,
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutor,
    ToolFailure,
    ToolRecoveryAction,
)


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


class _ReturningAgent:
    config = SimpleNamespace(name="child")

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls = 0
        self.received: tuple[Any, ...] | None = None

    async def run(self, text, *, session_id=None, user_context=None):
        self.calls += 1
        self.received = (text, session_id, user_context)
        return self.result


class _RaisingAgent:
    config = SimpleNamespace(name="child")

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def run(self, text, *, session_id=None, user_context=None):
        del text, session_id, user_context
        self.calls += 1
        raise self.error


def _agent_result(
    finish_reason: FinishReason | str,
    *,
    metadata: dict[str, Any] | None = None,
    output: Any = None,
    error: str | None = "private child failure",
) -> AgentResult:
    return AgentResult(
        run_id="private-child-run-id",
        output=output,
        messages=(),
        usage=Usage(),
        finish_reason=finish_reason,
        error=error,
        metadata={} if metadata is None else metadata,
    )


def _execute(agent: Any, *, idempotent: bool = False):
    return run(
        ToolExecutor(
            [AgentTool(agent, idempotent=idempotent)],
            retry=RetryConfig(max_attempts=4, initial_delay=0, max_delay=0),
        ).execute(ToolCall("call-1", "child", {"input": "delegate"}))
    )


def test_agent_tool_preserves_success_and_legacy_result_compatibility() -> None:
    canonical = _ReturningAgent(
        _agent_result(
            FinishReason.COMPLETED,
            output={"answer": "done"},
            error=None,
        )
    )
    canonical_result = _execute(canonical)

    class CompatibleResult:
        output = "compatible"

        def __init__(self) -> None:
            self.checked = False

        def raise_for_error(self) -> None:
            self.checked = True
            raise AssertionError("non-AgentResult methods must not be called")

    compatible_value = CompatibleResult()
    compatible = _ReturningAgent(compatible_value)
    compatible_result = _execute(compatible)

    legacy = _ReturningAgent(SimpleNamespace(output="legacy"))
    context = ToolExecutionContext(
        run_id="parent-run",
        session_id="session-1",
        user_context={"tenant": "example"},
    )
    legacy_result = run(
        ToolExecutor([AgentTool(legacy)]).execute(
            ToolCall("call-2", "child", {"input": "inspect"}),
            context,
        )
    )

    assert canonical_result.success is True
    assert canonical_result.value == {"answer": "done"}
    assert compatible_result.success is True
    assert compatible_result.value == "compatible"
    assert compatible_value.checked is False
    assert legacy_result.success is True
    assert legacy_result.value == "legacy"
    assert legacy.received == ("inspect", "session-1", {"tenant": "example"})


@pytest.mark.parametrize(
    ("finish_reason", "error_type", "stable_reason"),
    [
        (FinishReason.TIMEOUT, ToolErrorType.TIMEOUT, "child_agent_timeout"),
        (
            FinishReason.CANCELLED,
            ToolErrorType.CANCELLED,
            "child_agent_cancelled",
        ),
        (
            FinishReason.MAX_STEPS,
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_max_steps",
        ),
        (
            FinishReason.MAX_TOOL_CALLS,
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_max_tool_calls",
        ),
        (
            FinishReason.MAX_MODEL_TURNS,
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_max_model_turns",
        ),
        (
            FinishReason.NO_PROGRESS,
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_no_progress",
        ),
        (
            FinishReason.COMPLETED,
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_failed",
        ),
        (
            FinishReason.ERROR,
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_failed",
        ),
        (
            "future_terminal",
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_failed",
        ),
    ],
)
def test_agent_tool_maps_every_terminal_result_to_a_payload_free_failure(
    finish_reason: FinishReason | str,
    error_type: ToolErrorType,
    stable_reason: str,
) -> None:
    agent = _ReturningAgent(
        _agent_result(
            finish_reason,
            output={"secret_output": "must-not-leak"},
            metadata={
                "error_summary": {
                    "category": "private-category",
                    "code": "private-code",
                    "failure_id": "private-failure-id",
                },
                "provider_body": "must-not-leak",
            },
        )
    )

    result = _execute(agent, idempotent=True)

    assert result.success is False
    assert result.value is None
    assert result.attempts == 1
    assert agent.calls == 1
    assert result.error is not None
    assert result.error.type is error_type
    assert result.error.reason == stable_reason
    assert result.error.retryable is False
    assert result.error.recovery is None
    assert result.error.details == {}
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "must-not-leak" not in serialized
    assert "private-child-run-id" not in serialized
    assert "private-failure-id" not in serialized
    assert "private-category" not in serialized
    assert "private-code" not in serialized


@pytest.mark.parametrize(
    ("error_summary", "stable_reason"),
    [
        (
            {
                "category": "output_validation",
                "code": "output_validation_failed",
            },
            "child_agent_output_validation_failed",
        ),
        (
            {
                "category": "model_transport",
                "code": "model_connection_error",
                "retryable": True,
            },
            "child_agent_model_failed",
        ),
        (
            {
                "category": "timeout",
                "code": "model_timeout",
                "retryable": True,
            },
            "child_agent_model_failed",
        ),
    ],
)
def test_agent_tool_maps_safe_child_error_categories_without_forwarding_them(
    error_summary: dict[str, Any],
    stable_reason: str,
) -> None:
    agent = _ReturningAgent(
        _agent_result(
            FinishReason.ERROR,
            metadata={"error_summary": error_summary},
        )
    )

    result = _execute(agent, idempotent=True)

    assert result.error is not None
    assert result.error.type is ToolErrorType.EXECUTION_ERROR
    assert result.error.reason == stable_reason
    assert result.error.retryable is False
    assert result.error.recovery is None
    assert result.error.details == {}
    assert agent.calls == 1


@pytest.mark.parametrize(
    ("error", "error_type", "stable_reason"),
    [
        (
            asyncio.TimeoutError("private timeout"),
            ToolErrorType.TIMEOUT,
            "child_agent_timeout",
        ),
        (
            RunTimeoutError("private timeout"),
            ToolErrorType.TIMEOUT,
            "child_agent_timeout",
        ),
        (
            CancellationError("private cancellation"),
            ToolErrorType.CANCELLED,
            "child_agent_cancelled",
        ),
        (
            OutputValidationError("private schema"),
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_output_validation_failed",
        ),
        (
            ModelInvocationError("private provider body"),
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_model_failed",
        ),
        (
            RuntimeError("private implementation detail"),
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_invocation_failed",
        ),
    ],
)
def test_agent_tool_maps_direct_child_exceptions_without_automatic_retry(
    error: Exception,
    error_type: ToolErrorType,
    stable_reason: str,
) -> None:
    agent = _RaisingAgent(error)

    result = _execute(agent, idempotent=True)

    assert result.error is not None
    assert result.error.type is error_type
    assert result.error.reason == stable_reason
    assert result.error.retryable is False
    assert result.error.recovery is None
    assert result.error.details == {}
    assert result.attempts == 1
    assert agent.calls == 1
    assert "private" not in result.model_content()


def test_agent_tool_preserves_an_explicit_safe_tool_failure_contract() -> None:
    declared = ToolFailure(
        ToolError(
            ToolErrorType.EXECUTION_ERROR,
            "declared safe child failure",
            retryable=False,
            reason="declared_child_failure",
            recovery=ToolRecoveryAction.REPLAN,
        )
    )
    agent = _RaisingAgent(declared)

    result = _execute(agent, idempotent=True)

    assert result.error is not None
    assert result.error.message == "declared safe child failure"
    assert result.error.reason == "declared_child_failure"
    assert result.error.recovery is ToolRecoveryAction.REPLAN
    assert result.attempts == 1
    assert agent.calls == 1


def test_agent_tool_keeps_the_child_failure_as_a_protected_exception_cause() -> None:
    agent = _ReturningAgent(_agent_result(FinishReason.MAX_STEPS))

    with pytest.raises(ToolFailure) as captured:
        run(AgentTool(agent).invoke({"input": "delegate"}))

    failure = captured.value
    assert isinstance(failure.__cause__, AgentRunError)
    assert "private-child-run-id" not in str(failure)
    assert "private child failure" not in str(failure)


def test_agent_tool_propagates_parent_task_cancellation() -> None:
    class WaitingAgent:
        config = SimpleNamespace(name="child")

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False

        async def run(self, text, *, session_id=None, user_context=None):
            del text, session_id, user_context
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    async def scenario() -> tuple[bool, bool]:
        agent = WaitingAgent()
        task = asyncio.create_task(
            ToolExecutor([AgentTool(agent, idempotent=True)]).execute(
                ToolCall("call-cancel", "child", {"input": "wait"})
            )
        )
        await agent.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task.cancelled(), agent.cancelled

    task_cancelled, child_cancelled = run(scenario())

    assert task_cancelled is True
    assert child_cancelled is True

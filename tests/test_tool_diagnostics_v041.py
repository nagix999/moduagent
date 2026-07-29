from __future__ import annotations

import asyncio
from typing import Any

from moduagent.config import RetryConfig
from moduagent.messages import ToolCall
from moduagent.observability import DiagnosticReporter, InMemoryDiagnosticSink
from moduagent.tools import (
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutor,
    ToolFailureClassification,
    ToolRecoveryAction,
    ToolRuntime,
    function_tool,
)


def run(coroutine):
    return asyncio.run(coroutine)


class RecordingReporter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def capture_exception(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return f"failure-{len(self.calls)}"


def test_tool_runtime_writes_a_correlated_failure_diagnostic() -> None:
    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(
        sink,
        failure_id_factory=lambda: "diag-tool-1",
    )

    @function_tool
    def query_db() -> None:
        raise RuntimeError("SQL text and credentials must stay private")

    outcome = run(
        ToolRuntime(
            [query_db],
            diagnostic_reporter=reporter,
        ).execute_many(
            (ToolCall("call-db", "query_db", {}),),
            ToolExecutionContext(run_id="run-db"),
        )
    )

    assert outcome.failures[0].diagnostic_ref == "diag-tool-1"
    assert outcome.failures[0].failure_id == "diag-tool-1"
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.failure_id == "diag-tool-1"
    assert record.run_id == "run-db"
    assert record.component == "tool"
    assert record.operation == "invoke"
    assert record.phase == "act"
    assert record.call_id == "call-db"
    assert record.tool_name == "query_db"
    assert record.attempt == 1
    assert record.category == "tool_invocation"
    assert record.code == "execution_error"
    assert record.retryable is False
    assert record.terminal is False
    assert "SQL text and credentials" not in str(record.to_dict())


def test_tool_executor_captures_invocation_failure_with_execution_context() -> None:
    reporter = RecordingReporter()
    backend_error = RuntimeError("private database diagnostic")

    @function_tool
    def query_db() -> None:
        raise backend_error

    executor = ToolExecutor([query_db], diagnostic_reporter=reporter)
    outcome = run(
        executor.execute_batch(
            (ToolCall("call-1", "query_db", {}),),
            ToolExecutionContext(run_id="run-1"),
        )
    )

    assert executor.runtime.diagnostic_reporter is reporter
    assert len(reporter.calls) == 1
    captured = reporter.calls[0]
    assert captured == {
        "exception": backend_error,
        "run_id": "run-1",
        "component": "tool",
        "operation": "invoke",
        "phase": "act",
        "call_id": "call-1",
        "tool_name": "query_db",
        "attempt": 1,
        "category": "tool_invocation",
        "code": "execution_error",
        "retryable": False,
        "terminal": False,
    }
    assert outcome.failures[0].diagnostic_ref == "failure-1"
    assert outcome.failures[0].failure_id == "failure-1"
    assert outcome.failures[0].classification.diagnostic_ref == "failure-1"
    assert "private database diagnostic" not in str(
        outcome.sanitized_failure_views[0].to_dict()
    )


def test_tool_runtime_captures_only_the_exhausted_retry() -> None:
    reporter = RecordingReporter()

    def classify(_exception: Exception) -> ToolFailureClassification:
        return ToolFailureClassification(
            ToolErrorType.EXECUTION_ERROR,
            "backend_busy",
            retryable=True,
            recovery_directive=ToolRecoveryAction.RETRY_CALL,
        )

    @function_tool(idempotent=True, failure_classifier=classify)
    def unstable() -> None:
        raise RuntimeError("private backend detail")

    outcome = run(
        ToolRuntime(
            [unstable],
            retry=RetryConfig(max_attempts=2, initial_delay=0, max_delay=0),
            diagnostic_reporter=reporter,
        ).execute_many(
            (ToolCall("retry-1", "unstable", {}),),
            ToolExecutionContext(run_id="run-retry"),
        )
    )

    assert [call["attempt"] for call in reporter.calls] == [2]
    assert all(call["code"] == "backend_busy" for call in reporter.calls)
    assert all(call["retryable"] is True for call in reporter.calls)
    assert outcome.results[0].attempts == 2
    assert outcome.failures[0].diagnostic_ref == "failure-1"
    assert outcome.failures[0].failure_id == "failure-1"
    assert outcome.failures[0].classification.diagnostic_ref == "failure-1"


def test_only_unexpected_authorizer_failure_is_captured() -> None:
    reporter = RecordingReporter()

    @function_tool
    def expects_integer(value: int) -> int:
        return value

    invalid = run(
        ToolRuntime(
            [expects_integer],
            diagnostic_reporter=reporter,
        ).execute_many(
            (ToolCall("invalid-1", "expects_integer", {"value": "not-an-int"}),),
            ToolExecutionContext(run_id="run-invalid"),
        )
    )

    authorization_error = RuntimeError("authorization backend unavailable")

    class BrokenAuthorizer:
        async def authorize(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise authorization_error

    class DenyingAuthorizer:
        async def authorize(self, *args: Any, **kwargs: Any) -> bool:
            del args, kwargs
            return False

    unauthorized = run(
        ToolRuntime(
            [expects_integer],
            authorizer=BrokenAuthorizer(),
            diagnostic_reporter=reporter,
        ).execute_many(
            (ToolCall("auth-1", "expects_integer", {"value": 1}),),
            ToolExecutionContext(run_id="run-auth"),
        )
    )
    denied = run(
        ToolRuntime(
            [expects_integer],
            authorizer=DenyingAuthorizer(),
            diagnostic_reporter=reporter,
        ).execute_many(
            (ToolCall("denied-1", "expects_integer", {"value": 1}),),
            ToolExecutionContext(run_id="run-denied"),
        )
    )
    unknown = run(
        ToolRuntime(diagnostic_reporter=reporter).execute_many(
            (ToolCall("missing-1", "missing", {}),),
            ToolExecutionContext(run_id="run-missing"),
        )
    )

    assert invalid.results[0].error is not None
    assert invalid.results[0].error.type is ToolErrorType.INVALID_ARGUMENTS
    assert unauthorized.results[0].error is not None
    assert unauthorized.results[0].error.type is ToolErrorType.UNAUTHORIZED
    assert denied.results[0].error is not None
    assert denied.results[0].error.type is ToolErrorType.UNAUTHORIZED
    assert unknown.results[0].error is not None
    assert unknown.results[0].error.type is ToolErrorType.NOT_FOUND
    assert reporter.calls == [
        {
            "exception": authorization_error,
            "run_id": "run-auth",
            "component": "tool",
            "operation": "authorize",
            "phase": "act",
            "call_id": "auth-1",
            "tool_name": "expects_integer",
            "attempt": None,
            "category": "tool_authorization",
            "code": "authorization_backend_failed",
            "retryable": False,
            "terminal": False,
        }
    ]
    assert invalid.failures[0].failure_id is None
    assert unknown.failures[0].failure_id is None
    assert denied.failures[0].failure_id is None
    assert unauthorized.failures[0].failure_id == "failure-1"


def test_failure_classifier_exception_is_the_diagnostic_cause() -> None:
    reporter = RecordingReporter()
    backend_error = RuntimeError("private backend failure")
    classifier_error = LookupError("private classifier failure")

    def classify(exception: Exception) -> ToolFailureClassification:
        assert exception is backend_error
        raise classifier_error

    @function_tool(failure_classifier=classify)
    def classified_tool() -> None:
        raise backend_error

    outcome = run(
        ToolRuntime(
            [classified_tool],
            diagnostic_reporter=reporter,
        ).execute_many(
            (ToolCall("classifier-1", "classified_tool", {}),),
            ToolExecutionContext(run_id="run-classifier"),
        )
    )

    captured = reporter.calls[0]
    assert captured["exception"] is classifier_error
    assert captured["operation"] == "classify_failure"
    assert captured["category"] == "tool_classification"
    assert captured["code"] == "failure_classifier_failed"
    assert outcome.results[0].error is not None
    assert outcome.results[0].error.reason == "failure_classifier_failed"
    assert outcome.failures[0].failure_id == "failure-1"


def test_error_mapper_exception_is_the_diagnostic_cause() -> None:
    reporter = RecordingReporter()
    backend_error = RuntimeError("private backend failure")
    mapper_error = ValueError("private mapper failure")

    def map_error(exception: Exception) -> None:
        assert exception is backend_error
        raise mapper_error

    @function_tool(error_mapper=map_error)
    def mapped_tool() -> None:
        raise backend_error

    outcome = run(
        ToolRuntime(
            [mapped_tool],
            diagnostic_reporter=reporter,
        ).execute_many(
            (ToolCall("mapper-1", "mapped_tool", {}),),
            ToolExecutionContext(run_id="run-mapper"),
        )
    )

    captured = reporter.calls[0]
    assert captured["exception"] is mapper_error
    assert captured["operation"] == "map_error"
    assert captured["category"] == "tool_error_mapping"
    assert captured["code"] == "error_mapper_failed"
    assert outcome.results[0].error is not None
    assert outcome.results[0].error.reason == "error_mapper_failed"
    assert outcome.failures[0].failure_id == "failure-1"


def test_diagnostic_reporter_failure_does_not_change_tool_outcome() -> None:
    class BrokenReporter:
        def __init__(self) -> None:
            self.calls = 0

        async def capture_exception(self, **kwargs: Any) -> str:
            del kwargs
            self.calls += 1
            raise RuntimeError("diagnostic sink failed")

    @function_tool
    def broken() -> None:
        raise ValueError("private backend failure")

    reporter = BrokenReporter()
    outcome = run(
        ToolRuntime(
            [broken],
            diagnostic_reporter=reporter,
        ).execute_many(
            (ToolCall("broken-1", "broken", {}),),
            ToolExecutionContext(run_id="run-broken"),
        )
    )

    result = outcome.results[0]
    assert reporter.calls == 1
    assert result.success is False
    assert result.error is not None
    assert result.error.type is ToolErrorType.EXECUTION_ERROR
    assert result.error.reason is None
    assert result.attempts == 1
    assert outcome.failures[0].diagnostic_ref == "ValueError"
    assert outcome.failures[0].failure_id is None

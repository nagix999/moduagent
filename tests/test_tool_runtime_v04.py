from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from moduagent.config import RetryConfig
from moduagent.messages import ToolCall
from moduagent.tools import (
    FailureProjector,
    InternalToolFailure,
    ToolBatchOutcome,
    ToolErrorType,
    ToolExecutor,
    ToolFailureClassification,
    ToolRecoveryAction,
    ToolRepairConstraint,
    ToolResult,
    ToolRuntime,
    ToolSafetyProfile,
    fingerprint_tool_arguments,
    function_tool,
    is_tool_argument_fingerprint,
)


def run(coroutine):
    return asyncio.run(coroutine)


def test_argument_fingerprint_is_public_canonical_contract() -> None:
    left = fingerprint_tool_arguments({"query": "x", "limit": 3})
    right = fingerprint_tool_arguments({"limit": 3, "query": "x"})

    assert left == right
    assert is_tool_argument_fingerprint(left)
    assert not is_tool_argument_fingerprint("sha256:not-a-digest")
    with pytest.raises(TypeError, match="mapping"):
        fingerprint_tool_arguments([])  # type: ignore[arg-type]


def test_tool_result_normalization_never_uses_opaque_repr() -> None:
    class SecretRepr:
        def __repr__(self) -> str:
            return "DB_PASSWORD=TOPSECRET"

    result = ToolResult.succeeded(
        call_id="call-opaque",
        tool_name="opaque",
        value=SecretRepr(),
    )

    content = result.model_content()

    assert "TOPSECRET" not in content
    assert "unsupported_type" in content
    assert "SecretRepr" in content


def test_function_tool_resolves_explicit_and_legacy_safety_profiles() -> None:
    @function_tool(idempotent=True, repair_safe=True, timeout_retry_safe=True)
    def legacy(value: int) -> int:
        return value

    explicit_profile = ToolSafetyProfile(
        same_call_retry_safe=True,
        changed_argument_repair_safe=False,
        timeout_retry_safe=True,
    )

    @function_tool(safety_profile=explicit_profile)
    def explicit(value: int) -> int:
        return value

    assert legacy.safety_profile == ToolSafetyProfile(True, True, True)
    assert explicit.safety_profile is explicit_profile
    assert explicit.idempotent is True
    assert explicit.repair_safe is False
    assert explicit.timeout_retry_safe is True

    with pytest.raises(ValueError, match="legacy safety"):

        @function_tool(
            idempotent=True,
            safety_profile=ToolSafetyProfile(),
        )
        def conflicting() -> None:
            pass


def test_failure_classifier_maps_to_legacy_result_and_safe_outcome() -> None:
    secret = "PRIVATE BACKEND DIAGNOSTIC"

    def classify(_exception: Exception) -> ToolFailureClassification:
        return ToolFailureClassification(
            ToolErrorType.EXECUTION_ERROR,
            "invalid_filter",
            recovery_directive=ToolRecoveryAction.REPAIR_CALL,
            safe_message="Revise the filter expression.",
            diagnostic_ref=secret,
        )

    @function_tool(repair_safe=True, failure_classifier=classify)
    def search(filter: str) -> None:
        del filter
        raise RuntimeError(secret)

    outcome = run(
        ToolRuntime([search]).execute_many(
            (ToolCall("call-1", "search", {"filter": "bad"}),)
        )
    )

    result = outcome.results[0]
    assert result.error is not None
    assert result.error.reason == "invalid_filter"
    assert result.error.message == "Revise the filter expression."
    assert outcome.failure_count == 1
    assert outcome.failures[0].diagnostic_ref == "RuntimeError"
    assert outcome.failures[0].failure_id is None
    assert outcome.sanitized_failure_views[0].message == "Revise the filter expression."
    assert secret not in str(outcome.sanitized_failure_views[0].to_dict())


def test_failure_projector_bounds_data_and_drops_internal_diagnostics() -> None:
    failure = InternalToolFailure(
        call_id="call\x00\n" + ("c" * 400),
        tool_name="lookup\x00\n" + ("t" * 400),
        classification=ToolFailureClassification(
            ToolErrorType.EXECUTION_ERROR,
            "execution_error",
            recovery_directive=ToolRecoveryAction.REPAIR_CALL,
            safe_message="safe\x00\nmessage" + ("m" * 800),
            diagnostic_ref="PRIVATE-DIAGNOSTIC",
        ),
        safety_profile=ToolSafetyProfile(
            changed_argument_repair_safe=True,
        ),
        requested_arguments_fingerprint=fingerprint_tool_arguments({"q": "bad"}),
        effective_arguments_fingerprint=fingerprint_tool_arguments({"q": "bad"}),
        diagnostic_ref="PRIVATE-DIAGNOSTIC",
    )

    view = FailureProjector().project(failure, include_safe_message=True)
    payload = view.to_dict()

    assert len(view.call_id) <= 256
    assert len(view.tool_name) <= 256
    assert view.reason == "execution_error"
    assert view.message is not None and len(view.message) <= 512
    assert "\x00" not in str(payload)
    assert "\n" not in str(payload)
    assert "PRIVATE-DIAGNOSTIC" not in str(payload)
    assert failure.failure_id is None


def test_failure_classification_requires_machine_readable_reason() -> None:
    with pytest.raises(ValueError, match="machine-readable code"):
        ToolFailureClassification(
            ToolErrorType.EXECUTION_ERROR,
            "SQL syntax near password='hunter2'",
        )


def test_tool_batch_outcome_reports_partial_success_without_raw_failure() -> None:
    @function_tool
    def good() -> str:
        return "ok"

    @function_tool
    def bad() -> None:
        raise RuntimeError("private failure")

    calls = (
        ToolCall("good-1", "good", {}),
        ToolCall("bad-1", "bad", {}),
    )
    outcome = run(ToolRuntime([good, bad]).execute_many(calls))

    assert outcome.success_count == 1
    assert outcome.failure_count == 1
    assert outcome.partial_success is True
    assert outcome.retry_exhausted is False
    assert outcome.failures[0].classification.safe_message is None
    assert outcome.sanitized_failure_views[0].call_id == "bad-1"
    assert "private failure" not in str(outcome.sanitized_failure_views[0].to_dict())


def test_tool_batch_outcome_rejects_mismatched_result_identity() -> None:
    call = ToolCall("call-1", "lookup", {})
    result = ToolResult.succeeded(
        call_id="other-call",
        tool_name="lookup",
        value="ok",
    )

    with pytest.raises(ValueError, match="identity"):
        ToolBatchOutcome((call,), (result,))


def test_tool_batch_outcome_rejects_mismatched_safe_view_identity() -> None:
    @function_tool
    def broken() -> None:
        raise RuntimeError("private")

    outcome = run(
        ToolRuntime([broken]).execute_many((ToolCall("call-1", "broken", {}),))
    )
    wrong_view = replace(
        outcome.sanitized_failure_views[0],
        call_id="other-call",
    )

    with pytest.raises(ValueError, match="view identity"):
        ToolBatchOutcome(
            outcome.calls,
            outcome.results,
            outcome.failures,
            (wrong_view,),
        )


def test_typed_repair_constraint_blocks_unchanged_raw_and_effective_arguments() -> None:
    invocations = 0

    @function_tool(repair_safe=True)
    def lookup(value: int) -> int:
        nonlocal invocations
        invocations += 1
        return value

    runtime = ToolRuntime([lookup])
    constraint = ToolRepairConstraint(
        failed_call_id="old-call",
        expected_tool_name="lookup",
        seen_call_ids=frozenset({"old-call"}),
        previous_requested_fingerprint=fingerprint_tool_arguments({"value": "1"}),
        previous_effective_fingerprint=fingerprint_tool_arguments({"value": 1}),
    )

    unchanged_raw = run(
        runtime.execute(
            ToolCall("new-call-1", "lookup", {"value": "1"}),
            repair_constraint=constraint,
        )
    )
    unchanged_effective = run(
        runtime.execute(
            ToolCall("new-call-2", "lookup", {"value": 1.0}),
            repair_constraint=constraint,
        )
    )

    assert unchanged_raw.error is not None
    assert unchanged_raw.error.reason == "unchanged_repair_arguments"
    assert unchanged_effective.error is not None
    assert unchanged_effective.error.reason == "unchanged_repair_arguments"
    assert invocations == 0


def test_typed_repair_constraint_allows_one_changed_safe_call() -> None:
    invocations = 0

    @function_tool(repair_safe=True)
    def lookup(value: int) -> int:
        nonlocal invocations
        invocations += 1
        return value

    outcome = run(
        ToolRuntime([lookup]).execute_many(
            (ToolCall("new-call", "lookup", {"value": 2}),),
            repair_constraint=ToolRepairConstraint(
                failed_call_id="old-call",
                expected_tool_name="lookup",
                seen_call_ids=frozenset({"old-call"}),
                previous_requested_fingerprint=fingerprint_tool_arguments({"value": 1}),
                previous_effective_fingerprint=fingerprint_tool_arguments({"value": 1}),
            ),
        )
    )

    assert outcome.results[0].success is True
    assert outcome.results[0].value == 2
    assert invocations == 1


def test_typed_repair_constraint_fails_closed_for_unsafe_tool() -> None:
    invocations = 0

    @function_tool
    def mutate(value: int) -> int:
        nonlocal invocations
        invocations += 1
        return value

    result = run(
        ToolRuntime([mutate]).execute(
            ToolCall("new-call", "mutate", {"value": 2}),
            repair_constraint=ToolRepairConstraint(
                failed_call_id="old-call",
                expected_tool_name="mutate",
                seen_call_ids=frozenset({"old-call"}),
                previous_requested_fingerprint=fingerprint_tool_arguments({"value": 1}),
                previous_effective_fingerprint=fingerprint_tool_arguments({"value": 1}),
            ),
        )
    )

    assert result.error is not None
    assert result.error.reason == "tool_repair_not_safe"
    assert result.error.recovery is ToolRecoveryAction.FAIL
    assert invocations == 0


def test_runtime_marks_same_call_retry_exhaustion() -> None:
    attempts = 0

    def classify(_exception: Exception) -> ToolFailureClassification:
        return ToolFailureClassification(
            ToolErrorType.EXECUTION_ERROR,
            "temporarily_unavailable",
            retryable=True,
            recovery_directive=ToolRecoveryAction.RETRY_CALL,
            safe_message="Service is temporarily unavailable.",
        )

    @function_tool(idempotent=True, failure_classifier=classify)
    def unstable() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("private")

    outcome = run(
        ToolRuntime(
            [unstable],
            retry=RetryConfig(max_attempts=2, initial_delay=0, max_delay=0),
        ).execute_many((ToolCall("retry-1", "unstable", {}),))
    )

    assert attempts == 2
    assert outcome.results[0].attempts == 2
    assert outcome.retry_exhausted is True


def test_tool_executor_preserves_legacy_batch_return_and_exposes_outcome() -> None:
    @function_tool
    def echo(value: str) -> str:
        return value

    executor = ToolExecutor([echo])
    calls = (ToolCall("echo-1", "echo", {"value": "hello"}),)

    legacy = run(executor.execute_many(calls))
    outcome = run(executor.execute_batch(calls))

    assert type(legacy) is tuple
    assert legacy[0].value == "hello"
    assert isinstance(outcome, ToolBatchOutcome)
    assert outcome.results[0].value == "hello"
    assert executor.registry is executor.runtime.registry

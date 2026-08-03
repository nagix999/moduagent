from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from moduagent.config import RetryConfig
from moduagent.messages import ToolCall
from moduagent.tools.arguments import (
    fingerprint_tool_arguments,
    is_tool_argument_fingerprint,
)
from moduagent.tools.auth import (
    AllowAllAuthorizer,
    AuthorizationDecision,
    ToolAuthorizer,
)
from moduagent.tools.base import (
    Tool,
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolFailure,
    ToolRecoveryAction,
    ToolResult,
    _TOOL_REPAIR_METADATA_KEY,
    _await_if_needed,
    _json_safe,
    _run_sync_in_daemon,
    _serialized_size,
)
from moduagent.tools.failure import (
    FailureProjector,
    InternalToolFailure,
    SafeToolFailureView,
    ToolFailureClassification,
    ToolSafetyProfile,
    classification_from_tool_error,
    resolve_tool_safety_profile,
    tool_error_from_classification,
)
from moduagent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from moduagent.observability.diagnostics import DiagnosticReporter


@dataclass(frozen=True, slots=True)
class ToolRepairConstraint:
    """Typed precondition for executing one model-generated repair call."""

    failed_call_id: str
    expected_tool_name: str
    seen_call_ids: frozenset[str]
    previous_requested_fingerprint: str
    previous_effective_fingerprint: str

    def __post_init__(self) -> None:
        failed_call_id = str(self.failed_call_id).strip()
        expected_tool_name = str(self.expected_tool_name).strip()
        if not failed_call_id:
            raise ValueError("failed_call_id cannot be empty")
        if not expected_tool_name:
            raise ValueError("expected_tool_name cannot be empty")
        object.__setattr__(self, "failed_call_id", failed_call_id)
        object.__setattr__(self, "expected_tool_name", expected_tool_name)
        if isinstance(self.seen_call_ids, (str, bytes)):
            raise TypeError("seen_call_ids must be a collection of call IDs")
        try:
            seen_call_ids = frozenset(str(item) for item in self.seen_call_ids)
        except TypeError as exc:
            raise TypeError("seen_call_ids must be a collection of call IDs") from exc
        if any(not call_id.strip() for call_id in seen_call_ids):
            raise ValueError("seen_call_ids cannot contain empty call IDs")
        object.__setattr__(self, "seen_call_ids", seen_call_ids)
        for field_name in (
            "previous_requested_fingerprint",
            "previous_effective_fingerprint",
        ):
            if not is_tool_argument_fingerprint(getattr(self, field_name)):
                raise ValueError(f"{field_name} must use sha256")


@dataclass(frozen=True, slots=True)
class ToolBatchOutcome:
    """Validated outcome of one ordered Tool call batch."""

    calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...]
    failures: tuple[InternalToolFailure, ...] = ()
    sanitized_failure_views: tuple[SafeToolFailureView, ...] = ()

    def __post_init__(self) -> None:
        if type(self.calls) is not tuple:
            raise TypeError("calls must be a tuple")
        if type(self.results) is not tuple:
            raise TypeError("results must be a tuple")
        if type(self.failures) is not tuple:
            raise TypeError("failures must be a tuple")
        if type(self.sanitized_failure_views) is not tuple:
            raise TypeError("sanitized_failure_views must be a tuple")
        if len(self.calls) != len(self.results):
            raise ValueError("Tool outcome call and result counts must match")
        for call, result in zip(self.calls, self.results):
            if not isinstance(call, ToolCall):
                raise TypeError("calls must contain ToolCall instances")
            if not isinstance(result, ToolResult):
                raise TypeError("results must contain ToolResult instances")
            if result.call_id != call.id or result.tool_name != call.name:
                raise ValueError("Tool outcome result identity does not match its call")

        failed_results = tuple(result for result in self.results if not result.success)
        if len(self.failures) != len(failed_results):
            raise ValueError("Tool outcome failures do not match failed results")
        if len(self.sanitized_failure_views) != len(self.failures):
            raise ValueError("Tool outcome safe views do not match failures")
        for result, failure, view in zip(
            failed_results,
            self.failures,
            self.sanitized_failure_views,
        ):
            if not isinstance(failure, InternalToolFailure):
                raise TypeError("failures must contain InternalToolFailure instances")
            if not isinstance(view, SafeToolFailureView):
                raise TypeError(
                    "sanitized_failure_views must contain SafeToolFailureView instances"
                )
            identity = (result.call_id, result.tool_name)
            if identity != (failure.call_id, failure.tool_name):
                raise ValueError("Tool failure identity does not match its result")
            if identity != (view.call_id, view.tool_name):
                raise ValueError("Tool failure view identity does not match its result")
            classification = failure.classification
            if (
                view.error_type is not classification.error_type
                or view.recovery is not classification.recovery_directive
                or view.retryable is not classification.retryable
                or view.requested_arguments_fingerprint
                != failure.requested_arguments_fingerprint
                or view.effective_arguments_fingerprint
                != failure.effective_arguments_fingerprint
            ):
                raise ValueError("Tool failure view does not match its classification")

    @property
    def success_count(self) -> int:
        return sum(result.success for result in self.results)

    @property
    def failure_count(self) -> int:
        return len(self.results) - self.success_count

    @property
    def partial_success(self) -> bool:
        return self.success_count > 0 and self.failure_count > 0

    @property
    def retry_exhausted(self) -> bool:
        return any(failure.same_call_retry_exhausted for failure in self.failures)


@dataclass(frozen=True, slots=True)
class _CallExecution:
    result: ToolResult
    failure: InternalToolFailure | None = None


@dataclass(frozen=True, slots=True)
class _ClassifiedToolError:
    error: ToolError
    safe_message_declared: bool
    diagnostic_error: Exception
    diagnostic_operation: str = "invoke"
    diagnostic_category: str = "tool_invocation"


class ToolRuntime:
    """Single operational boundary for validation, authorization and execution."""

    def __init__(
        self,
        registry: ToolRegistry | Iterable[Tool] = (),
        *,
        authorizer: ToolAuthorizer | None = None,
        retry: RetryConfig | None = None,
        retry_config: RetryConfig | None = None,
        failure_projector: FailureProjector | None = None,
        diagnostic_reporter: DiagnosticReporter | None = None,
        default_timeout_seconds: float | None = 30.0,
        max_result_bytes: int | None = 1_000_000,
    ) -> None:
        if retry is not None and retry_config is not None:
            raise ValueError("use either retry or retry_config, not both")
        if default_timeout_seconds is not None and default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        if max_result_bytes is not None and max_result_bytes < 1:
            raise ValueError("max_result_bytes must be at least 1")
        if failure_projector is not None and not isinstance(
            failure_projector,
            FailureProjector,
        ):
            raise TypeError("failure_projector must be a FailureProjector")

        self.registry = (
            registry if isinstance(registry, ToolRegistry) else ToolRegistry(registry)
        )
        self.authorizer = AllowAllAuthorizer() if authorizer is None else authorizer
        self.retry = (
            retry
            if retry is not None
            else retry_config
            if retry_config is not None
            else RetryConfig()
        )
        self.failure_projector = (
            FailureProjector() if failure_projector is None else failure_projector
        )
        self.diagnostic_reporter = diagnostic_reporter
        self.default_timeout_seconds = default_timeout_seconds
        self.max_result_bytes = max_result_bytes

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext | None = None,
        *,
        repair_constraint: ToolRepairConstraint | None = None,
    ) -> ToolResult:
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall")
        context = context if context is not None else ToolExecutionContext()
        constraint_error = self._repair_constraint_error(
            (call,),
            repair_constraint,
        )
        if constraint_error is not None:
            return self._repair_rejection(call, constraint_error).result
        return (
            await self._execute_one(
                call,
                context,
                repair_constraint=repair_constraint,
            )
        ).result

    async def execute_many(
        self,
        calls: Iterable[ToolCall],
        context: ToolExecutionContext | None = None,
        *,
        parallel: bool = False,
        max_parallel: int = 4,
        repair_constraint: ToolRepairConstraint | None = None,
    ) -> ToolBatchOutcome:
        calls = tuple(calls)
        context = context if context is not None else ToolExecutionContext()
        if any(not isinstance(call, ToolCall) for call in calls):
            raise TypeError("calls must contain ToolCall instances")
        if parallel and max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")

        constraint_error = self._repair_constraint_error(calls, repair_constraint)
        if constraint_error is not None:
            if not calls:
                raise ValueError(constraint_error[1])
            executions = tuple(
                self._repair_rejection(call, constraint_error) for call in calls
            )
            return self._batch_outcome(calls, executions)

        async def run(call: ToolCall) -> _CallExecution:
            return await self._execute_one(
                call,
                context,
                repair_constraint=repair_constraint,
            )

        if not parallel:
            executions = tuple([await run(call) for call in calls])
        else:
            semaphore = asyncio.Semaphore(max_parallel)

            async def limited(call: ToolCall) -> _CallExecution:
                async with semaphore:
                    return await run(call)

            executions = tuple(await asyncio.gather(*(limited(call) for call in calls)))
        return self._batch_outcome(calls, executions)

    def _batch_outcome(
        self,
        calls: tuple[ToolCall, ...],
        executions: Sequence[_CallExecution],
    ) -> ToolBatchOutcome:
        results = tuple(execution.result for execution in executions)
        failures = tuple(
            execution.failure
            for execution in executions
            if execution.failure is not None
        )
        views = tuple(
            self.failure_projector.project(
                failure,
                include_safe_message=True,
            )
            for failure in failures
        )
        return ToolBatchOutcome(
            calls=calls,
            results=results,
            failures=failures,
            sanitized_failure_views=views,
        )

    @staticmethod
    def _repair_constraint_error(
        calls: tuple[ToolCall, ...],
        constraint: ToolRepairConstraint | None,
    ) -> tuple[str, str] | None:
        if constraint is None:
            return None
        if not isinstance(constraint, ToolRepairConstraint):
            raise TypeError("repair_constraint must be a ToolRepairConstraint")
        if len(calls) != 1:
            return (
                "invalid_repair_call_count",
                "tool repair must contain exactly one call",
            )
        call = calls[0]
        if call.name != constraint.expected_tool_name:
            return (
                "different_repair_tool",
                "tool repair must call the same Tool",
            )
        if call.id == constraint.failed_call_id or call.id in constraint.seen_call_ids:
            return (
                "reused_tool_call_id",
                "tool repair must use a new unique call ID",
            )
        try:
            fingerprint = fingerprint_tool_arguments(call.arguments)
        except (TypeError, ValueError):
            return (
                "invalid_arguments",
                "tool repair arguments must be canonical JSON",
            )
        if fingerprint == constraint.previous_requested_fingerprint:
            return (
                "unchanged_repair_arguments",
                "tool repair must change the requested arguments",
            )
        return None

    def _repair_rejection(
        self,
        call: ToolCall,
        failure: tuple[str, str],
    ) -> _CallExecution:
        reason, message = failure
        tool = self.registry.get(call.name)
        profile = (
            resolve_tool_safety_profile(tool)
            if tool is not None
            else ToolSafetyProfile()
        )
        error = ToolError(
            ToolErrorType.INVALID_ARGUMENTS,
            message,
            reason=reason,
            recovery=ToolRecoveryAction.REPAIR_CALL,
        )
        requested_fingerprint = self._arguments_fingerprint(call.arguments)
        result = ToolResult.failed(
            call_id=call.id,
            tool_name=call.name,
            error=error,
            repair_safe=profile.changed_argument_repair_safe,
        )
        return _CallExecution(
            result,
            self._internal_failure(
                result,
                profile,
                requested_fingerprint=requested_fingerprint,
            ),
        )

    async def _execute_one(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        *,
        repair_constraint: ToolRepairConstraint | None,
    ) -> _CallExecution:
        started = time.monotonic()
        requested_fingerprint = self._arguments_fingerprint(call.arguments)
        tool = self.registry.get(call.name)
        if tool is None:
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.NOT_FOUND,
                    f"unknown tool: {call.name}",
                    reason="unknown_tool",
                    recovery=ToolRecoveryAction.FAIL,
                ),
                duration_seconds=time.monotonic() - started,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    ToolSafetyProfile(),
                    requested_fingerprint=requested_fingerprint,
                ),
            )

        profile = resolve_tool_safety_profile(tool)
        if repair_constraint is not None and not profile.changed_argument_repair_safe:
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.EXECUTION_ERROR,
                    "tool is not safe for changed-argument repair",
                    reason="tool_repair_not_safe",
                    recovery=ToolRecoveryAction.FAIL,
                ),
                duration_seconds=time.monotonic() - started,
                repair_safe=False,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    profile,
                    requested_fingerprint=requested_fingerprint,
                ),
            )
        try:
            arguments = self._validate_arguments(tool, call.arguments)
        except ValidationError as exc:
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.INVALID_ARGUMENTS,
                    f"invalid arguments for tool {call.name}",
                    reason="invalid_arguments",
                    recovery=ToolRecoveryAction.REPAIR_CALL,
                    details={
                        "errors": exc.errors(
                            include_url=False,
                            include_context=False,
                            include_input=False,
                        )
                    },
                ),
                duration_seconds=time.monotonic() - started,
                repair_safe=profile.changed_argument_repair_safe,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    profile,
                    requested_fingerprint=requested_fingerprint,
                    diagnostic_ref=type(exc).__name__,
                ),
            )
        except (TypeError, ValueError) as exc:
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.INVALID_ARGUMENTS,
                    f"invalid arguments for tool {call.name}",
                    reason="invalid_arguments",
                    recovery=ToolRecoveryAction.REPAIR_CALL,
                ),
                duration_seconds=time.monotonic() - started,
                repair_safe=profile.changed_argument_repair_safe,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    profile,
                    requested_fingerprint=requested_fingerprint,
                    diagnostic_ref=type(exc).__name__,
                ),
            )

        effective_fingerprint = self._arguments_fingerprint(arguments)
        if (
            repair_constraint is not None
            and effective_fingerprint
            == repair_constraint.previous_effective_fingerprint
        ):
            result = self._unchanged_effective_arguments(
                call,
                arguments,
                profile,
                started,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    profile,
                    requested_fingerprint=requested_fingerprint,
                    effective_fingerprint=effective_fingerprint,
                    safe_message_declared=True,
                ),
            )

        # Compatibility for 0.3.2 checkpoints and AgentRuntime. New execution
        # engines pass a typed ToolRepairConstraint instead.
        legacy_repair = context.metadata.get(_TOOL_REPAIR_METADATA_KEY)
        if (
            isinstance(legacy_repair, Mapping)
            and legacy_repair.get("tool_name") == call.name
            and legacy_repair.get("invocation_fingerprint") == effective_fingerprint
        ):
            result = self._unchanged_effective_arguments(
                call,
                arguments,
                profile,
                started,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    profile,
                    requested_fingerprint=requested_fingerprint,
                    effective_fingerprint=effective_fingerprint,
                ),
            )

        try:
            raw_decision = self.authorizer.authorize(tool, arguments, context)
            decision = await _await_if_needed(raw_decision)
        except Exception as exc:
            failure_id = await self._capture_diagnostic(
                exc,
                context=context,
                call=call,
                operation="authorize",
                category="tool_authorization",
                code="authorization_backend_failed",
                retryable=False,
            )
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.UNAUTHORIZED,
                    "tool authorization failed",
                    reason="authorization_failed",
                    recovery=ToolRecoveryAction.FAIL,
                ),
                duration_seconds=time.monotonic() - started,
                repair_safe=profile.changed_argument_repair_safe,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    profile,
                    requested_fingerprint=requested_fingerprint,
                    effective_fingerprint=effective_fingerprint,
                    diagnostic_ref=failure_id or type(exc).__name__,
                    failure_id=failure_id,
                ),
            )
        if isinstance(decision, bool):
            decision = AuthorizationDecision(decision)
        if not isinstance(decision, AuthorizationDecision):
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.UNAUTHORIZED,
                    "tool authorizer returned an invalid decision",
                    reason="invalid_authorization_decision",
                    recovery=ToolRecoveryAction.FAIL,
                ),
                duration_seconds=time.monotonic() - started,
                repair_safe=profile.changed_argument_repair_safe,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    profile,
                    requested_fingerprint=requested_fingerprint,
                    effective_fingerprint=effective_fingerprint,
                ),
            )
        if not decision.allowed:
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.UNAUTHORIZED,
                    f"not authorized to call tool: {call.name}",
                    reason="authorization_denied",
                    recovery=ToolRecoveryAction.FAIL,
                ),
                duration_seconds=time.monotonic() - started,
                repair_safe=profile.changed_argument_repair_safe,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    profile,
                    requested_fingerprint=requested_fingerprint,
                    effective_fingerprint=effective_fingerprint,
                    safe_message_declared=True,
                ),
            )

        attempts = self.retry.max_attempts if profile.same_call_retry_safe else 1
        timeout = getattr(tool, "timeout_seconds", None)
        if timeout is None:
            timeout = self.default_timeout_seconds
        diagnostic_ref: str | None = None
        failure_id: str | None = None

        for attempt in range(1, attempts + 1):
            call_context = context.for_call(call.id, attempt=attempt)
            try:
                invocation = self._invoke_tool(tool, arguments, call_context)
                value = (
                    await asyncio.wait_for(invocation, timeout=timeout)
                    if timeout is not None
                    else await invocation
                )
                size_limit = self._result_size_limit(tool)
                if size_limit is not None:
                    actual_size = _serialized_size(_json_safe(value))
                    if actual_size > size_limit:
                        result = ToolResult.failed(
                            call_id=call.id,
                            tool_name=call.name,
                            error=ToolError(
                                ToolErrorType.RESULT_TOO_LARGE,
                                f"tool result exceeds {size_limit} bytes",
                                reason=(
                                    "result_too_large"
                                    if profile.changed_argument_repair_safe
                                    else None
                                ),
                                recovery=(
                                    ToolRecoveryAction.REPAIR_CALL
                                    if profile.changed_argument_repair_safe
                                    else None
                                ),
                                details={
                                    "actual_bytes": actual_size,
                                    "max_bytes": size_limit,
                                },
                            ),
                            attempts=attempt,
                            duration_seconds=time.monotonic() - started,
                            invocation_arguments=arguments,
                            repair_safe=profile.changed_argument_repair_safe,
                        )
                        return _CallExecution(
                            result,
                            self._internal_failure(
                                result,
                                profile,
                                requested_fingerprint=requested_fingerprint,
                                effective_fingerprint=effective_fingerprint,
                            ),
                        )
                return _CallExecution(
                    ToolResult.succeeded(
                        call_id=call.id,
                        tool_name=call.name,
                        value=value,
                        attempts=attempt,
                        duration_seconds=time.monotonic() - started,
                        invocation_arguments=arguments,
                        repair_safe=profile.changed_argument_repair_safe,
                    )
                )
            except Exception as exc:
                diagnostic_ref = type(exc).__name__
                classified_error = self._classify_error(
                    tool,
                    exc,
                    timeout_seconds=timeout,
                    safety_profile=profile,
                )
                error = classified_error.error
                safe_message_declared = classified_error.safe_message_declared

            retry_recovery = error.recovery in {
                None,
                ToolRecoveryAction.RETRY_CALL,
            }
            retry_safe = (
                profile.same_call_retry_safe
                and error.retryable
                and retry_recovery
                and (
                    error.type is not ToolErrorType.TIMEOUT
                    or profile.timeout_retry_safe
                )
            )
            if retry_safe and attempt < attempts:
                delay = self.retry.delay_for(attempt)
                if delay:
                    await asyncio.sleep(delay)
                continue
            classification = classification_from_tool_error(error)
            failure_id = await self._capture_diagnostic(
                classified_error.diagnostic_error,
                context=context,
                call=call,
                operation=classified_error.diagnostic_operation,
                category=classified_error.diagnostic_category,
                code=classification.stable_reason,
                retryable=classification.retryable,
                attempt=attempt,
            )
            if failure_id is not None:
                diagnostic_ref = failure_id
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=error,
                attempts=attempt,
                duration_seconds=time.monotonic() - started,
                invocation_arguments=arguments,
                repair_safe=profile.changed_argument_repair_safe,
            )
            return _CallExecution(
                result,
                self._internal_failure(
                    result,
                    profile,
                    requested_fingerprint=requested_fingerprint,
                    effective_fingerprint=effective_fingerprint,
                    retry_exhausted=retry_safe and attempt >= attempts,
                    diagnostic_ref=diagnostic_ref,
                    failure_id=failure_id,
                    safe_message_declared=safe_message_declared,
                ),
            )

        raise AssertionError("tool execution loop ended unexpectedly")

    @staticmethod
    def _unchanged_effective_arguments(
        call: ToolCall,
        arguments: Mapping[str, Any],
        profile: ToolSafetyProfile,
        started: float,
    ) -> ToolResult:
        return ToolResult.failed(
            call_id=call.id,
            tool_name=call.name,
            error=ToolError(
                ToolErrorType.INVALID_ARGUMENTS,
                "tool repair must change the effective arguments",
                reason="unchanged_repair_arguments",
                recovery=ToolRecoveryAction.REPAIR_CALL,
            ),
            duration_seconds=time.monotonic() - started,
            invocation_arguments=arguments,
            repair_safe=profile.changed_argument_repair_safe,
        )

    def _internal_failure(
        self,
        result: ToolResult,
        profile: ToolSafetyProfile,
        *,
        requested_fingerprint: str | None,
        effective_fingerprint: str | None = None,
        retry_exhausted: bool = False,
        diagnostic_ref: str | None = None,
        failure_id: str | None = None,
        safe_message_declared: bool = False,
    ) -> InternalToolFailure:
        if result.error is None:
            raise ValueError("failed Tool result must include an error")
        classification = classification_from_tool_error(
            result.error,
            diagnostic_ref=diagnostic_ref,
        )
        if not safe_message_declared:
            classification = ToolFailureClassification(
                error_type=classification.error_type,
                stable_reason=classification.stable_reason,
                retryable=classification.retryable,
                recovery_directive=classification.recovery_directive,
                diagnostic_ref=classification.diagnostic_ref,
            )
        return InternalToolFailure(
            call_id=result.call_id,
            tool_name=result.tool_name,
            classification=classification,
            safety_profile=profile,
            requested_arguments_fingerprint=requested_fingerprint,
            effective_arguments_fingerprint=effective_fingerprint,
            attempts=result.attempts,
            same_call_retry_exhausted=retry_exhausted,
            diagnostic_ref=diagnostic_ref,
            failure_id=failure_id,
        )

    @staticmethod
    def _arguments_fingerprint(arguments: Any) -> str | None:
        try:
            return fingerprint_tool_arguments(arguments)
        except (TypeError, ValueError):
            return None

    async def _capture_diagnostic(
        self,
        exception: Exception,
        *,
        context: ToolExecutionContext,
        call: ToolCall,
        operation: str,
        category: str,
        code: str,
        retryable: bool,
        attempt: int | None = None,
    ) -> str | None:
        capture = getattr(self.diagnostic_reporter, "capture_exception", None)
        if not callable(capture):
            return None
        try:
            failure_id = await capture(
                exception=exception,
                run_id=context.run_id,
                component="tool",
                operation=operation,
                phase="act",
                call_id=call.id,
                tool_name=call.name,
                attempt=attempt,
                category=category,
                code=code,
                retryable=retryable,
                terminal=False,
            )
        except Exception:
            # Diagnostics are best-effort and cannot alter Tool classification,
            # authorization, retries, or the returned outcome.
            return None
        if not isinstance(failure_id, str):
            return None
        failure_id = failure_id.strip()
        return failure_id or None

    @staticmethod
    def _validate_arguments(
        tool: Tool,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")
        validator = getattr(tool, "validate_arguments", None)
        if callable(validator):
            return dict(validator(arguments))

        input_model = getattr(tool, "input_model", None)
        if isinstance(input_model, type) and issubclass(input_model, BaseModel):
            model = input_model.model_validate(dict(arguments))
            return {name: getattr(model, name) for name in type(model).model_fields}
        return dict(arguments)

    @staticmethod
    async def _invoke_tool(
        tool: Tool,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        invoke = getattr(tool, "invoke_validated", None)
        if invoke is None:
            invoke = getattr(tool, "invoke", None)
        if invoke is None:
            invoke = getattr(tool, "execute", None)
        if not callable(invoke):
            raise TypeError(f"tool {tool.name} has no callable invoke method")

        accepts_context = True
        try:
            inspect.signature(invoke).bind(arguments, context)
        except TypeError:
            accepts_context = False
        except (ValueError, AttributeError):
            pass

        def call() -> Any:
            return invoke(arguments, context) if accepts_context else invoke(arguments)

        if inspect.iscoroutinefunction(invoke):
            return await call()
        value = await _run_sync_in_daemon(call)
        return await _await_if_needed(value)

    def _result_size_limit(self, tool: Tool) -> int | None:
        tool_limit = getattr(tool, "max_result_bytes", None)
        limits = [
            limit for limit in (self.max_result_bytes, tool_limit) if limit is not None
        ]
        return min(limits) if limits else None

    @staticmethod
    def _classify_error(
        tool: Tool,
        exception: Exception,
        *,
        timeout_seconds: float | None,
        safety_profile: ToolSafetyProfile,
    ) -> _ClassifiedToolError:
        if isinstance(exception, asyncio.TimeoutError) and not (
            safety_profile.timeout_retry_safe
        ):
            return _ClassifiedToolError(
                ToolError(
                    ToolErrorType.TIMEOUT,
                    f"tool timed out: {tool.name}",
                    retryable=False,
                    details={"timeout_seconds": timeout_seconds},
                    reason="uncancellable_timeout",
                    recovery=ToolRecoveryAction.FAIL,
                ),
                True,
                exception,
            )
        if isinstance(exception, ToolFailure):
            return _ClassifiedToolError(exception.error, False, exception)

        classifier = getattr(tool, "failure_classifier", None)
        if callable(classifier):
            try:
                classified = classifier(exception)
            except Exception as classifier_error:
                return _ClassifiedToolError(
                    ToolError(
                        ToolErrorType.EXECUTION_ERROR,
                        "tool failure classifier failed",
                        details={"exception_type": type(classifier_error).__name__},
                        reason="failure_classifier_failed",
                        recovery=ToolRecoveryAction.FAIL,
                    ),
                    False,
                    classifier_error,
                    diagnostic_operation="classify_failure",
                    diagnostic_category="tool_classification",
                )
            if classified is not None:
                if isinstance(classified, ToolFailureClassification):
                    return _ClassifiedToolError(
                        tool_error_from_classification(classified),
                        True,
                        exception,
                    )
                return _ClassifiedToolError(
                    ToolError(
                        ToolErrorType.EXECUTION_ERROR,
                        "tool failure classifier returned an invalid result",
                        details={"result_type": type(classified).__name__},
                        reason="invalid_failure_classifier_result",
                        recovery=ToolRecoveryAction.FAIL,
                    ),
                    False,
                    exception,
                    diagnostic_operation="classify_failure",
                    diagnostic_category="tool_classification",
                )

        mapper = getattr(tool, "error_mapper", None)
        if callable(mapper):
            try:
                mapped = mapper(exception)
            except Exception as mapper_error:
                return _ClassifiedToolError(
                    ToolError(
                        ToolErrorType.EXECUTION_ERROR,
                        "tool error mapper failed",
                        details={"exception_type": type(mapper_error).__name__},
                        reason="error_mapper_failed",
                        recovery=ToolRecoveryAction.FAIL,
                    ),
                    False,
                    mapper_error,
                    diagnostic_operation="map_error",
                    diagnostic_category="tool_error_mapping",
                )
            if mapped is not None:
                if isinstance(mapped, ToolError):
                    return _ClassifiedToolError(mapped, False, exception)
                return _ClassifiedToolError(
                    ToolError(
                        ToolErrorType.EXECUTION_ERROR,
                        "tool error mapper returned an invalid result",
                        details={"result_type": type(mapped).__name__},
                        reason="invalid_error_mapper_result",
                        recovery=ToolRecoveryAction.FAIL,
                    ),
                    False,
                    exception,
                    diagnostic_operation="map_error",
                    diagnostic_category="tool_error_mapping",
                )

        if isinstance(exception, asyncio.TimeoutError):
            retryable = (
                safety_profile.same_call_retry_safe
                and safety_profile.timeout_retry_safe
            )
            return _ClassifiedToolError(
                ToolError(
                    ToolErrorType.TIMEOUT,
                    f"tool timed out: {tool.name}",
                    retryable=retryable,
                    details={"timeout_seconds": timeout_seconds},
                    reason="tool_timeout",
                    recovery=(
                        ToolRecoveryAction.RETRY_CALL
                        if retryable
                        else ToolRecoveryAction.FAIL
                    ),
                ),
                True,
                exception,
            )
        return _ClassifiedToolError(
            ToolError(
                ToolErrorType.EXECUTION_ERROR,
                f"tool execution failed: {exception}",
                retryable=safety_profile.same_call_retry_safe,
                details={"exception_type": type(exception).__name__},
            ),
            False,
            exception,
        )

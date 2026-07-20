from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from moduagent.config import RetryConfig
from moduagent.messages import ToolCall
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
    ToolResult,
    _await_if_needed,
    _json_safe,
    _run_sync_in_daemon,
    _serialized_size,
)
from moduagent.tools.registry import ToolRegistry


class ToolExecutor:
    """Validate, authorize and execute registered tools."""

    def __init__(
        self,
        registry: ToolRegistry | Iterable[Tool] = (),
        *,
        authorizer: ToolAuthorizer | None = None,
        retry: RetryConfig | None = None,
        retry_config: RetryConfig | None = None,
        default_timeout_seconds: float | None = 30.0,
        max_result_bytes: int | None = 1_000_000,
    ) -> None:
        if retry is not None and retry_config is not None:
            raise ValueError("use either retry or retry_config, not both")
        if default_timeout_seconds is not None and default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        if max_result_bytes is not None and max_result_bytes < 1:
            raise ValueError("max_result_bytes must be at least 1")

        self.registry = (
            registry if isinstance(registry, ToolRegistry) else ToolRegistry(registry)
        )
        self.authorizer = authorizer or AllowAllAuthorizer()
        self.retry = retry or retry_config or RetryConfig()
        self.default_timeout_seconds = default_timeout_seconds
        self.max_result_bytes = max_result_bytes

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        started = time.monotonic()
        context = context if context is not None else ToolExecutionContext()
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.NOT_FOUND,
                    f"unknown tool: {call.name}",
                ),
                duration_seconds=time.monotonic() - started,
            )

        try:
            arguments = self._validate_arguments(tool, call.arguments)
        except ValidationError as exc:
            return ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.INVALID_ARGUMENTS,
                    f"invalid arguments for tool {call.name}",
                    details={
                        "errors": exc.errors(
                            include_url=False,
                            include_context=False,
                            include_input=False,
                        )
                    },
                ),
                duration_seconds=time.monotonic() - started,
            )
        except (TypeError, ValueError) as exc:
            return ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.INVALID_ARGUMENTS,
                    f"invalid arguments for tool {call.name}: {exc}",
                ),
                duration_seconds=time.monotonic() - started,
            )

        try:
            raw_decision = self.authorizer.authorize(tool, arguments, context)
            decision = await _await_if_needed(raw_decision)
        except Exception as exc:
            return ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.UNAUTHORIZED,
                    f"tool authorization failed: {exc}",
                ),
                duration_seconds=time.monotonic() - started,
            )
        if isinstance(decision, bool):
            decision = AuthorizationDecision(decision)
        if not isinstance(decision, AuthorizationDecision):
            return ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.UNAUTHORIZED,
                    "tool authorizer returned an invalid decision",
                ),
                duration_seconds=time.monotonic() - started,
            )
        if not decision.allowed:
            return ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.UNAUTHORIZED,
                    decision.reason or f"not authorized to call tool: {call.name}",
                ),
                duration_seconds=time.monotonic() - started,
            )

        attempts = (
            self.retry.max_attempts if bool(getattr(tool, "idempotent", False)) else 1
        )
        timeout = getattr(tool, "timeout_seconds", None)
        if timeout is None:
            timeout = self.default_timeout_seconds

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
                        return ToolResult.failed(
                            call_id=call.id,
                            tool_name=call.name,
                            error=ToolError(
                                ToolErrorType.RESULT_TOO_LARGE,
                                f"tool result exceeds {size_limit} bytes",
                                details={
                                    "actual_bytes": actual_size,
                                    "max_bytes": size_limit,
                                },
                            ),
                            attempts=attempt,
                            duration_seconds=time.monotonic() - started,
                        )
                return ToolResult.succeeded(
                    call_id=call.id,
                    tool_name=call.name,
                    value=value,
                    attempts=attempt,
                    duration_seconds=time.monotonic() - started,
                )
            except asyncio.TimeoutError:
                error = ToolError(
                    ToolErrorType.TIMEOUT,
                    f"tool timed out: {call.name}",
                    retryable=bool(getattr(tool, "idempotent", False)),
                    details={"timeout_seconds": timeout},
                )
            except Exception as exc:
                error = ToolError(
                    ToolErrorType.EXECUTION_ERROR,
                    f"tool execution failed: {exc}",
                    retryable=bool(getattr(tool, "idempotent", False)),
                    details={"exception_type": type(exc).__name__},
                )

            if attempt < attempts:
                delay = self.retry.delay_for(attempt)
                if delay:
                    await asyncio.sleep(delay)
                continue
            return ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=error,
                attempts=attempt,
                duration_seconds=time.monotonic() - started,
            )

        raise AssertionError("tool execution loop ended unexpectedly")

    async def execute_many(
        self,
        calls: Iterable[ToolCall],
        context: ToolExecutionContext | None = None,
        *,
        parallel: bool = False,
        max_parallel: int = 4,
    ) -> tuple[ToolResult, ...]:
        calls = tuple(calls)
        context = context if context is not None else ToolExecutionContext()
        if not parallel:
            return tuple([await self.execute(call, context) for call in calls])
        if max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")

        semaphore = asyncio.Semaphore(max_parallel)

        async def limited(call: ToolCall) -> ToolResult:
            async with semaphore:
                return await self.execute(call, context)

        # asyncio.gather preserves input order while allowing bounded overlap.
        return tuple(await asyncio.gather(*(limited(call) for call in calls)))

    @staticmethod
    def _validate_arguments(
        tool: Tool, arguments: Mapping[str, Any]
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

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from moduagent.errors import (
    AgentRunError,
    CancellationError,
    ModelInvocationError,
    OutputValidationError,
    RunTimeoutError,
)
from moduagent.tools.base import (
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolFailure,
    ToolSchema,
)

if TYPE_CHECKING:
    from moduagent.runtime.context import AgentResult


class _AgentLike(Protocol):
    async def run(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_context: Mapping[str, Any] | None = None,
    ) -> "AgentResult | Any": ...


class AgentToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(description="Task or message to delegate to the agent")


_TERMINAL_FAILURES: Mapping[str, tuple[ToolErrorType, str, str]] = {
    "timeout": (
        ToolErrorType.TIMEOUT,
        "child_agent_timeout",
        "delegated agent timed out",
    ),
    "cancelled": (
        ToolErrorType.CANCELLED,
        "child_agent_cancelled",
        "delegated agent was cancelled",
    ),
    "max_steps": (
        ToolErrorType.EXECUTION_ERROR,
        "child_agent_max_steps",
        "delegated agent exceeded its step limit",
    ),
    "max_tool_calls": (
        ToolErrorType.EXECUTION_ERROR,
        "child_agent_max_tool_calls",
        "delegated agent exceeded its tool-call limit",
    ),
    "max_model_turns": (
        ToolErrorType.EXECUTION_ERROR,
        "child_agent_max_model_turns",
        "delegated agent exceeded its model-turn limit",
    ),
    "no_progress": (
        ToolErrorType.EXECUTION_ERROR,
        "child_agent_no_progress",
        "delegated agent stopped after making no progress",
    ),
}

_MODEL_FAILURE_CATEGORIES = frozenset(
    {
        "model_client",
        "model_invocation",
        "model_protocol",
        "model_provider",
        "model_request",
        "model_transport",
    }
)


def _tool_failure(
    error_type: ToolErrorType,
    reason: str,
    message: str,
) -> ToolFailure:
    """Create the payload-free failure exposed at the parent Tool boundary."""

    return ToolFailure(
        ToolError(
            error_type,
            message,
            retryable=False,
            details={},
            reason=reason,
            recovery=None,
        )
    )


def _terminal_tool_failure(error: AgentRunError) -> ToolFailure:
    mapped = _TERMINAL_FAILURES.get(error.finish_reason)
    if mapped is not None:
        return _tool_failure(*mapped)

    if (
        error.category == "output_validation"
        or error.code == "output_validation_failed"
    ):
        return _tool_failure(
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_output_validation_failed",
            "delegated agent output validation failed",
        )
    if error.category in _MODEL_FAILURE_CATEGORIES or error.code == "model_timeout":
        return _tool_failure(
            ToolErrorType.EXECUTION_ERROR,
            "child_agent_model_failed",
            "delegated agent model invocation failed",
        )
    return _tool_failure(
        ToolErrorType.EXECUTION_ERROR,
        "child_agent_failed",
        "delegated agent execution failed",
    )


def _raise_if_canonical_result_failed(result: Any) -> None:
    # Import lazily: importing moduagent.runtime while the Tool package is
    # initializing would cycle through decision.planning -> moduagent.tools.
    from moduagent.runtime.context import AgentResult

    if isinstance(result, AgentResult):
        result.raise_for_error()


class AgentTool:
    """Expose another Agent through the standard Tool contract."""

    def __init__(
        self,
        agent: _AgentLike,
        *,
        name: str | None = None,
        description: str | None = None,
        idempotent: bool = False,
        timeout_seconds: float | None = None,
        max_result_bytes: int | None = None,
    ) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_result_bytes is not None and max_result_bytes < 1:
            raise ValueError("max_result_bytes must be at least 1")

        config = getattr(agent, "config", None)
        inferred_name = getattr(config, "name", None)
        self.agent = agent
        self.name = name or inferred_name or "agent"
        if not self.name.strip():
            raise ValueError("tool name cannot be empty")
        self.description = description or f"Delegate a task to the {self.name} agent"
        self.idempotent = idempotent
        # Legacy AgentTool cannot prove the delegated Agent's side effects.
        # Production already rejects this adapter; Development remains source
        # compatible and treats the classification as unknown.
        self.side_effect_level = None
        self.timeout_seconds = timeout_seconds
        self.max_result_bytes = max_result_bytes
        self.input_model = AgentToolInput
        self.args_schema = self.input_model
        self._schema = ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        values = self.input_model.model_validate(dict(arguments))
        return {"input": values.input}

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Any:
        return await self.invoke_validated(self.validate_arguments(arguments), context)

    async def invoke_validated(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Any:
        context = context if context is not None else ToolExecutionContext()
        try:
            result = await self.agent.run(
                str(arguments["input"]),
                session_id=context.session_id,
                user_context=context.user_context,
            )
            _raise_if_canonical_result_failed(result)
            return getattr(result, "output", result)
        except asyncio.CancelledError:
            # This task also represents the parent invocation. Translating its
            # cancellation would swallow structured parent cancellation.
            raise
        except ToolFailure:
            # A custom Agent-like implementation may deliberately publish an
            # already-sanitized Tool contract. Preserve that explicit boundary
            # instead of weakening or replacing its classification.
            raise
        except AgentRunError as exc:
            raise _terminal_tool_failure(exc) from exc
        except (asyncio.TimeoutError, RunTimeoutError) as exc:
            raise _tool_failure(
                ToolErrorType.TIMEOUT,
                "child_agent_timeout",
                "delegated agent timed out",
            ) from exc
        except CancellationError as exc:
            raise _tool_failure(
                ToolErrorType.CANCELLED,
                "child_agent_cancelled",
                "delegated agent was cancelled",
            ) from exc
        except OutputValidationError as exc:
            raise _tool_failure(
                ToolErrorType.EXECUTION_ERROR,
                "child_agent_output_validation_failed",
                "delegated agent output validation failed",
            ) from exc
        except ModelInvocationError as exc:
            raise _tool_failure(
                ToolErrorType.EXECUTION_ERROR,
                "child_agent_model_failed",
                "delegated agent model invocation failed",
            ) from exc
        except Exception as exc:
            raise _tool_failure(
                ToolErrorType.EXECUTION_ERROR,
                "child_agent_invocation_failed",
                "delegated agent invocation failed",
            ) from exc

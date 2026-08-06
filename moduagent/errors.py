from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class ModuAgentError(Exception):
    """Base class for framework-level failures."""


class AgentRunError(ModuAgentError, RuntimeError):
    """A safe, structured view of a non-successful Agent result.

    The exception deliberately retains neither messages, output, raw Tool
    arguments, nor arbitrary result metadata. Only a small allowlist from the
    runtime's public error summary is exposed.
    """

    def __init__(
        self,
        *,
        run_id: str,
        finish_reason: str,
        error_summary: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_id = _safe_error_text(run_id, limit=256) or "unknown"
        self.finish_reason = _safe_error_text(finish_reason, limit=64) or "error"
        self.error_summary = MappingProxyType(_safe_agent_error_summary(error_summary))
        self.category = self.error_summary.get("category")
        self.code = self.error_summary.get("code")
        self.retryable = self.error_summary.get("retryable")
        self.resumable = self.error_summary.get("resumable")
        self.failure_id = self.error_summary.get("failure_id")
        self.provider_finish_reason = self.error_summary.get("provider_finish_reason")

        details = [f"finish_reason={self.finish_reason}"]
        for key in (
            "category",
            "code",
            "component",
            "operation",
            "phase",
            "step_id",
            "attempt",
            "retryable",
            "resumable",
            "failure_id",
            "provider_finish_reason",
            "model_turns",
            "max_model_turns",
            "no_progress_model_turns",
            "no_progress_model_turn_threshold",
        ):
            if key in self.error_summary:
                details.append(f"{key}={self.error_summary[key]}")
        details.append(f"run_id={self.run_id}")
        super().__init__(f"agent run failed ({', '.join(details)})")


class ConfigurationError(ModuAgentError, ValueError):
    """The resolved Agent configuration is internally inconsistent."""


class InputError(ModuAgentError, ValueError):
    """A run request is invalid before execution starts."""


class CapabilityError(ConfigurationError):
    """A configured component cannot satisfy a required capability."""


class ModelInvocationError(ModuAgentError, RuntimeError):
    """A model provider call failed or returned an invalid protocol response."""


class OutputValidationError(ModuAgentError, ValueError):
    """A model response could not be decoded into the configured output contract."""


class ToolValidationError(ModuAgentError, ValueError):
    """Tool arguments failed schema or repair-guard validation."""


class ToolAuthorizationError(ModuAgentError, PermissionError):
    """A Tool call was denied or could not be authorized safely."""


class ToolInvocationError(ModuAgentError, RuntimeError):
    """A Tool invocation failed at its operational boundary."""


class ToolRecoveryError(ModuAgentError, RuntimeError):
    """A Tool failure could not be recovered within the declared safety policy."""


class MemoryError(ModuAgentError, RuntimeError):
    """Conversation-memory preparation failed."""


class SkillError(ModuAgentError, RuntimeError):
    """Skill discovery, selection, activation, or resource access failed."""


class PersistenceError(ModuAgentError, RuntimeError):
    """Conversation or checkpoint persistence failed."""


class CheckpointNotFoundError(PersistenceError, LookupError):
    """The requested checkpoint does not exist."""


class StateMigrationError(PersistenceError, ValueError):
    """A persisted state cannot be migrated without replay ambiguity."""


class ExecutionInvariantError(ModuAgentError, RuntimeError):
    """An execution engine invariant was violated."""


class RunTimeoutError(ModuAgentError, TimeoutError):
    """The overall run deadline expired."""


class CancellationError(ModuAgentError):
    """The run was cancelled at a durable boundary."""


_AGENT_ERROR_TEXT_FIELDS = frozenset(
    {
        "category",
        "code",
        "component",
        "operation",
        "phase",
        "step_id",
        "failure_id",
    }
)
_AGENT_ERROR_BOOL_FIELDS = frozenset({"retryable", "resumable"})
_AGENT_ERROR_COUNT_FIELDS = frozenset(
    {
        "attempt",
        "model_turns",
        "max_model_turns",
        "no_progress_model_turns",
        "no_progress_model_turn_threshold",
    }
)
_AGENT_ERROR_PROVIDER_FINISH_REASONS = frozenset({"timeout", "length", "max_tokens"})


def _safe_error_text(value: Any, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(
        "".join(
            character if character.isprintable() else " " for character in value
        ).split()
    )
    return normalized[:limit]


def _safe_agent_error_summary(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key in _AGENT_ERROR_TEXT_FIELDS:
        text = _safe_error_text(value.get(key), limit=256)
        if text:
            safe[key] = text
    for key in _AGENT_ERROR_BOOL_FIELDS:
        item = value.get(key)
        if type(item) is bool:
            safe[key] = item
    for key in _AGENT_ERROR_COUNT_FIELDS:
        item = value.get(key)
        if type(item) is int and 0 <= item <= 1_000_000:
            safe[key] = item
    provider_finish_reason = value.get("provider_finish_reason")
    if provider_finish_reason in _AGENT_ERROR_PROVIDER_FINISH_REASONS:
        safe["provider_finish_reason"] = provider_finish_reason
    return safe


__all__ = [
    "AgentRunError",
    "CancellationError",
    "CapabilityError",
    "CheckpointNotFoundError",
    "ConfigurationError",
    "ExecutionInvariantError",
    "InputError",
    "MemoryError",
    "ModelInvocationError",
    "ModuAgentError",
    "OutputValidationError",
    "PersistenceError",
    "RunTimeoutError",
    "SkillError",
    "StateMigrationError",
    "ToolAuthorizationError",
    "ToolInvocationError",
    "ToolRecoveryError",
    "ToolValidationError",
]

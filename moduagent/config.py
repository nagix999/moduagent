from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


class _FrozenDict(dict[Any, Any]):
    """JSON-compatible mapping that rejects mutation after construction."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("configuration mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        del memo
        return self


def _freeze_configuration(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("configuration mapping keys must be strings")
        return _FrozenDict(
            {key: _freeze_configuration(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_configuration(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("configuration values must be JSON-like and finite")


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_steps: int = 6
    max_tool_calls: int = 10
    timeout_seconds: float = 120.0
    parallel_tool_calls: bool = False
    max_parallel_tools: int = 4
    # Appended after the 0.2 fields so existing positional construction keeps
    # its meaning. New code should still prefer keyword arguments.
    max_step_attempts: int = 2
    max_replans: int = 2
    max_tool_repair_attempts: int = 1
    # Appended for 0.5 so all 0.4 positional arguments retain their meaning.
    # Three identical completed turns tolerate one redundant retry while
    # stopping a stable model loop before it consumes the wider turn budget.
    max_model_turns: int = 32
    no_progress_model_turn_threshold: int = 3

    def __post_init__(self) -> None:
        for field_name in (
            "max_steps",
            "max_tool_calls",
            "max_parallel_tools",
            "max_step_attempts",
            "max_replans",
            "max_tool_repair_attempts",
            "max_model_turns",
            "no_progress_model_turn_threshold",
        ):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer")
        if type(self.parallel_tool_calls) is not bool:
            raise TypeError("parallel_tool_calls must be a bool")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
        ):
            raise TypeError("timeout_seconds must be a finite number")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_step_attempts < 1:
            raise ValueError("max_step_attempts must be at least 1")
        if self.max_replans < 0:
            raise ValueError("max_replans cannot be negative")
        if self.max_tool_repair_attempts < 0:
            raise ValueError("max_tool_repair_attempts cannot be negative")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be at least 1")
        if self.max_model_turns < 1:
            raise ValueError("max_model_turns must be at least 1")
        if self.no_progress_model_turn_threshold < 2:
            raise ValueError("no_progress_model_turn_threshold must be at least 2")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int = 1
    initial_delay: float = 0.2
    max_delay: float = 2.0
    backoff_factor: float = 2.0

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int:
            raise TypeError("max_attempts must be an integer")
        for field_name in ("initial_delay", "max_delay", "backoff_factor"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"{field_name} must be a finite number")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay < 0 or self.max_delay < 0:
            raise ValueError("retry delays cannot be negative")
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be at least 1")

    def delay_for(self, failed_attempt: int) -> float:
        delay = self.initial_delay * self.backoff_factor ** max(0, failed_attempt - 1)
        return min(delay, self.max_delay)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    name: str
    instructions: str
    limits: RunLimits = field(default_factory=RunLimits)
    retry: RetryConfig = field(default_factory=RetryConfig)
    model_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    finalization_mode: Literal["always", "structured_only", "disabled"] = (
        "structured_only"
    )
    stream_visibility: Literal["public_only", "all"] = "public_only"
    tool_trace_mode: Literal["off", "summary", "arguments"] = "summary"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("agent name must be a string")
        if not isinstance(self.instructions, str):
            raise TypeError("agent instructions must be a string")
        if not self.name.strip():
            raise ValueError("agent name cannot be empty")
        if not self.instructions.strip():
            raise ValueError("agent instructions cannot be empty")
        if self.finalization_mode not in {
            "always",
            "structured_only",
            "disabled",
        }:
            raise ValueError(
                "finalization_mode must be 'always', 'structured_only', or 'disabled'"
            )
        if self.stream_visibility not in {"public_only", "all"}:
            raise ValueError("stream_visibility must be 'public_only' or 'all'")
        if self.tool_trace_mode not in {"off", "summary", "arguments"}:
            raise ValueError("tool_trace_mode must be 'off', 'summary', or 'arguments'")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("agent metadata must be a mapping")
        if any(not isinstance(key, str) for key in self.metadata):
            raise TypeError("agent metadata keys must be strings")
        if not isinstance(self.model_options, Mapping):
            raise TypeError("model_options must be a mapping")
        if any(not isinstance(key, str) for key in self.model_options):
            raise TypeError("model_options keys must be strings")
        object.__setattr__(
            self,
            "model_options",
            _freeze_configuration(self.model_options),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_configuration(self.metadata),
        )

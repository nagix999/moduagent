from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


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

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_step_attempts < 1:
            raise ValueError("max_step_attempts must be at least 1")
        if self.max_replans < 0:
            raise ValueError("max_replans cannot be negative")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be at least 1")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int = 1
    initial_delay: float = 0.2
    max_delay: float = 2.0
    backoff_factor: float = 2.0

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

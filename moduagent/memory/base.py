from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

from moduagent.messages import Message, Usage
from moduagent.models import ModelRequest


class MemoryPhase(str, Enum):
    PLAN = "plan"
    ACT = "act"
    STEP_RESULT = "step_result"
    VERIFY = "verify"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class MemoryRequest:
    run_id: str
    session_id: str
    phase: MemoryPhase
    model_request: ModelRequest
    protected_from: int
    user_context: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.phase, MemoryPhase):
            object.__setattr__(self, "phase", MemoryPhase(str(self.phase)))
        if self.protected_from < 0:
            raise ValueError("protected_from cannot be negative")
        object.__setattr__(self, "user_context", dict(self.user_context))


@dataclass(frozen=True, slots=True)
class MemoryResult:
    messages: tuple[Message, ...]
    usage: Usage = field(default_factory=Usage)
    original_tokens: int = 0
    selected_tokens: int = 0
    summarized_messages: int = 0
    dropped_messages: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "metadata", dict(self.metadata))
        for field_name in (
            "original_tokens",
            "selected_tokens",
            "summarized_messages",
            "dropped_messages",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")


@runtime_checkable
class ConversationMemoryPolicy(Protocol):
    async def prepare(self, request: MemoryRequest) -> MemoryResult: ...


class ConversationMemoryError(Exception):
    """Base error raised while preparing a bounded conversation view."""


class ConversationMemoryOverflowError(ConversationMemoryError):
    """Raised when protected input cannot fit in the configured token budget."""

    def __init__(
        self,
        *,
        required_tokens: int,
        available_tokens: int,
        message_tokens: int,
        tool_tokens: int,
        schema_tokens: int,
    ) -> None:
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens
        self.message_tokens = message_tokens
        self.tool_tokens = tool_tokens
        self.schema_tokens = schema_tokens
        super().__init__(
            "protected conversation input exceeds the token budget "
            f"(required={required_tokens}, available={available_tokens}, "
            f"messages={message_tokens}, tools={tool_tokens}, "
            f"schema={schema_tokens})"
        )

    @property
    def metadata(self) -> Mapping[str, int]:
        return {
            "required_tokens": self.required_tokens,
            "available_tokens": self.available_tokens,
            "message_tokens": self.message_tokens,
            "tool_tokens": self.tool_tokens,
            "schema_tokens": self.schema_tokens,
        }


class MemoryIntegrityError(ConversationMemoryError):
    """Raised when protected Tool Call and Tool result messages are inconsistent."""

    def __init__(self, message: str, *, message_index: int | None = None) -> None:
        self.message_index = message_index
        suffix = "" if message_index is None else f" at message index {message_index}"
        super().__init__(f"{message}{suffix}")


__all__ = [
    "ConversationMemoryError",
    "ConversationMemoryOverflowError",
    "ConversationMemoryPolicy",
    "MemoryIntegrityError",
    "MemoryPhase",
    "MemoryRequest",
    "MemoryResult",
]

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolCall":
        return cls(
            id=str(value["id"]),
            name=str(value["name"]),
            arguments=dict(value.get("arguments", {})),
        )


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def system(
        cls,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Message":
        return cls(MessageRole.SYSTEM, content, metadata=dict(metadata or {}))

    @classmethod
    def user(
        cls,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Message":
        return cls(MessageRole.USER, content, metadata=dict(metadata or {}))

    @classmethod
    def assistant(
        cls,
        content: str | None,
        tool_calls: tuple[ToolCall, ...] = (),
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Message":
        return cls(
            MessageRole.ASSISTANT,
            content,
            tool_calls=tool_calls,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def tool(
        cls,
        content: str,
        *,
        call_id: str,
        name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Message":
        return cls(
            MessageRole.TOOL,
            content,
            tool_call_id=call_id,
            name=name,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_calls:
            value["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            value["tool_call_id"] = self.tool_call_id
        if self.name:
            value["name"] = self.name
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Message":
        return cls(
            role=MessageRole(str(value["role"])),
            content=value.get("content"),
            tool_calls=tuple(
                ToolCall.from_dict(call) for call in value.get("tool_calls", ())
            ),
            tool_call_id=value.get("tool_call_id"),
            name=value.get("name"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_provider(cls, value: Mapping[str, Any] | None) -> "Usage":
        raw = dict(value or {})
        input_tokens = int(
            raw.get(
                "input_tokens",
                raw.get("prompt_tokens", raw.get("prompt_eval_count", 0)),
            )
        )
        output_tokens = int(
            raw.get(
                "output_tokens",
                raw.get("completion_tokens", raw.get("eval_count", 0)),
            )
        )
        total_tokens = int(raw.get("total_tokens", input_tokens + output_tokens))
        return cls(input_tokens, output_tokens, total_tokens, raw)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.total_tokens + other.total_tokens,
        )


class FinishReason(str, Enum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ERROR = "error"

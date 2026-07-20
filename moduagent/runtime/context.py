from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from moduagent.messages import FinishReason, Message, MessageRole, Usage


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_MODEL = "waiting_for_model"
    WAITING_FOR_TOOLS = "waiting_for_tools"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RunRequest:
    input: str
    session_id: str
    user_context: Mapping[str, Any] = field(default_factory=dict)
    resume_run_id: str | None = None


@dataclass(slots=True)
class RunContext:
    run_id: str
    request: RunRequest
    messages: list[Message]
    new_messages: list[Message] = field(default_factory=list)
    step: int = 0
    tool_call_count: int = 0
    status: RunStatus = RunStatus.CREATED
    policy_state: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    metadata: dict[str, Any] = field(default_factory=dict)
    current_run_start: int = 0

    def add_message(self, message: Message, *, persist: bool = True) -> None:
        self.messages.append(message)
        if persist and message.role is not MessageRole.SYSTEM:
            self.new_messages.append(message)


@dataclass(frozen=True, slots=True)
class AgentResult:
    run_id: str
    output: Any
    messages: tuple[Message, ...]
    usage: Usage
    finish_reason: FinishReason
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

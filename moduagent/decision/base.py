from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from moduagent.messages import ToolCall

if TYPE_CHECKING:
    from moduagent.models import ModelResponse
    from moduagent.runtime.context import RunContext
    from moduagent.tools import ToolResult


class DecisionKind(str, Enum):
    # CONTINUE is retained for policies written against ModuAgent 0.2.x.
    CONTINUE = "continue"
    CALL_TOOLS = "call_tools"
    COMMIT_STEP = "commit_step"
    RETRY_STEP = "retry_step"
    REPLAN = "replan"
    FINALIZE = "finalize"
    FINISH = "finish"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    kind: DecisionKind
    tool_calls: tuple[ToolCall, ...] = ()
    final_output: Any = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DecisionPolicy(Protocol):
    async def begin(self, context: "RunContext") -> None: ...

    async def decide(
        self, context: "RunContext", response: "ModelResponse"
    ) -> ExecutionDecision: ...

    async def observe(
        self, context: "RunContext", results: Sequence["ToolResult"]
    ) -> None: ...

    def should_stop(self, context: "RunContext") -> bool: ...

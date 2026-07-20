from __future__ import annotations

from collections.abc import Sequence

from moduagent.decision.base import (
    DecisionKind,
    ExecutionDecision,
)
from moduagent.models import ModelResponse
from moduagent.runtime.context import RunContext
from moduagent.tools import ToolResult


class StandardDecisionPolicy:
    async def begin(self, context: RunContext) -> None:
        context.policy_state.setdefault("policy", "standard")

    async def decide(
        self, context: RunContext, response: ModelResponse
    ) -> ExecutionDecision:
        tool_calls = response.tool_calls or response.message.tool_calls
        if tool_calls:
            return ExecutionDecision(DecisionKind.CALL_TOOLS, tuple(tool_calls))
        return ExecutionDecision(DecisionKind.FINISH)

    async def observe(self, context: RunContext, results: Sequence[ToolResult]) -> None:
        context.policy_state["last_tool_success"] = all(
            result.success for result in results
        )

    def should_stop(self, context: RunContext) -> bool:
        return False

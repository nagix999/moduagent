from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from moduagent.decision.base import DecisionKind, ExecutionDecision
from moduagent.messages import Message
from moduagent.models import ModelClient, ModelRequest, ModelResponse
from moduagent.runtime.context import RunContext
from moduagent.tools import ToolResult


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class PlanStep:
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    expected_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "status": self.status.value,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanStep":
        return cls(
            description=str(value["description"]),
            status=PlanStepStatus(value.get("status", PlanStepStatus.PENDING.value)),
            expected_output=value.get("expected_output"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(slots=True)
class Plan:
    steps: list[PlanStep]
    current_index: int = 0

    @property
    def current(self) -> PlanStep | None:
        if 0 <= self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    @property
    def complete(self) -> bool:
        return self.current_index >= len(self.steps)

    def start_current(self) -> PlanStep | None:
        step = self.current
        if step and step.status is PlanStepStatus.PENDING:
            step.status = PlanStepStatus.IN_PROGRESS
        return step

    def advance(self, output: str | None = None) -> None:
        step = self.current
        if step:
            step.status = PlanStepStatus.COMPLETED
            step.expected_output = output
        self.current_index += 1
        self.start_current()

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "current_index": self.current_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Plan":
        return cls(
            steps=[PlanStep.from_dict(item) for item in value.get("steps", ())],
            current_index=int(value.get("current_index", 0)),
        )


class PlanGenerator(Protocol):
    async def create(self, context: RunContext) -> Plan: ...

    async def revise(self, context: RunContext, plan: Plan, feedback: str) -> Plan: ...


class LLMPlanGenerator:
    def __init__(self, model: ModelClient, *, max_steps: int = 6) -> None:
        self.model = model
        self.max_steps = max_steps

    async def create(self, context: RunContext) -> Plan:
        request = ModelRequest(
            messages=(
                Message.system(
                    "Create a concise execution plan. Return JSON only as "
                    '{"steps":[{"description":"...","expected_output":"..."}]}.'
                ),
                Message.user(context.request.input),
            ),
            output_schema=self._schema(),
        )
        response = await self.model.complete(request)
        context.usage = context.usage + response.usage
        return self._parse(response.message.content, context.request.input)

    async def revise(self, context: RunContext, plan: Plan, feedback: str) -> Plan:
        request = ModelRequest(
            messages=(
                Message.system("Revise the plan. Return the same JSON shape only."),
                Message.user(
                    json.dumps(
                        {
                            "request": context.request.input,
                            "plan": plan.to_dict(),
                            "feedback": feedback,
                        },
                        ensure_ascii=False,
                    )
                ),
            ),
            output_schema=self._schema(),
        )
        response = await self.model.complete(request)
        context.usage = context.usage + response.usage
        return self._parse(response.message.content, context.request.input)

    def _parse(self, content: str | None, fallback: str) -> Plan:
        try:
            text = (content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            payload = json.loads(text)
            raw_steps = (
                payload.get("steps", payload) if isinstance(payload, dict) else payload
            )
            steps = [
                PlanStep(
                    description=str(
                        item["description"] if isinstance(item, dict) else item
                    ),
                    expected_output=(
                        item.get("expected_output") if isinstance(item, dict) else None
                    ),
                )
                for item in raw_steps[: self.max_steps]
            ]
            if steps:
                plan = Plan(steps)
                plan.start_current()
                return plan
        except (TypeError, ValueError, KeyError):
            pass
        plan = Plan([PlanStep(fallback)])
        plan.start_current()
        return plan

    @staticmethod
    def _schema() -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "expected_output": {"type": "string"},
                        },
                        "required": ["description"],
                    },
                }
            },
            "required": ["steps"],
        }


class PlanAndExecutePolicy:
    def __init__(
        self,
        plan_generator: PlanGenerator,
        *,
        revise_on_tool_failure: bool = True,
    ) -> None:
        self.plan_generator = plan_generator
        self.revise_on_tool_failure = revise_on_tool_failure

    async def begin(self, context: RunContext) -> None:
        if "plan" not in context.policy_state:
            plan = await self.plan_generator.create(context)
            context.policy_state["plan"] = plan.to_dict()
        plan = self._plan(context)
        step = plan.start_current()
        if step:
            context.add_message(
                Message.system(f"Current plan step: {step.description}"),
                persist=False,
            )

    async def decide(
        self, context: RunContext, response: ModelResponse
    ) -> ExecutionDecision:
        tool_calls = response.tool_calls or response.message.tool_calls
        if tool_calls:
            return ExecutionDecision(DecisionKind.CALL_TOOLS, tuple(tool_calls))

        plan = self._plan(context)
        plan.advance(response.message.content)
        context.policy_state["plan"] = plan.to_dict()
        if plan.complete:
            return ExecutionDecision(
                DecisionKind.FINISH,
                metadata={"plan": plan.to_dict()},
            )
        step = plan.current
        return ExecutionDecision(
            DecisionKind.CONTINUE,
            metadata={
                "instruction": f"Continue with plan step: {step.description}",
                "plan": plan.to_dict(),
            },
        )

    async def observe(self, context: RunContext, results: Sequence[ToolResult]) -> None:
        failures = [result for result in results if not result.success]
        if not failures or not self.revise_on_tool_failure:
            return
        feedback = "; ".join(
            result.error.message if result.error else "tool failed"
            for result in failures
        )
        plan = await self.plan_generator.revise(context, self._plan(context), feedback)
        context.policy_state["plan"] = plan.to_dict()

    def should_stop(self, context: RunContext) -> bool:
        return False

    @staticmethod
    def _plan(context: RunContext) -> Plan:
        return Plan.from_dict(context.policy_state["plan"])

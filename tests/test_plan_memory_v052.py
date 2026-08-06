from __future__ import annotations

import asyncio
import json
from typing import Any

from moduagent import (
    Agent,
    AgentConfig,
    EventType,
    FinishReason,
    InMemoryConversationStore,
    LLMPlanGenerator,
    Message,
    ModelRequest,
    ModelResponse,
    PlanAndExecutePolicy,
    PlanExecutionProfile,
    RecentTurnsConversationMemoryPolicy,
    RunLimits,
)


class _CollectingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


class _PlanModel:
    def __init__(self, *, block_first_step: bool = False) -> None:
        self.requests: list[ModelRequest] = []
        self.block_first_step = block_first_step
        self.plan_calls = 0
        self.step_result_calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        schema = request.output_schema or {}
        properties = schema.get("properties", {})
        if "steps" in properties:
            self.plan_calls += 1
            return ModelResponse(
                Message.assistant(
                    json.dumps(
                        {
                            "steps": [
                                {
                                    "step_id": "inspect",
                                    "objective": "inspect the current request",
                                    "completion_criteria": [
                                        "the current request is inspected"
                                    ],
                                    "expected_output": "verified inspection",
                                    "dependencies": [],
                                    "allowed_tools": [],
                                }
                            ]
                        }
                    )
                )
            )
        if schema.get("title") == "StepResult":
            self.step_result_calls += 1
            if self.block_first_step and self.step_result_calls == 1:
                return ModelResponse(
                    Message.assistant(
                        json.dumps(
                            {
                                "step_id": "inspect",
                                "status": "blocked",
                                "missing_inputs": ["revise the unfinished plan"],
                            }
                        )
                    )
                )
            return ModelResponse(
                Message.assistant(
                    json.dumps(
                        {
                            "step_id": "inspect",
                            "status": "completed",
                            "facts": ["the current request was inspected"],
                            "completion_evidence": ["the current request is inspected"],
                        }
                    )
                )
            )
        return ModelResponse(Message.assistant("safe final"))


def test_plan_and_replan_requests_use_the_common_conversation_memory_policy() -> None:
    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        await conversations.append(
            "plan-memory",
            [
                Message.user("old private-sized context"),
                Message.assistant("old answer"),
            ],
        )
        model = _PlanModel()
        events = _CollectingSink()
        agent = Agent.create(
            model=model,
            instructions="Plan from the bounded conversation view.",
            execution="plan",
            limits=RunLimits(max_steps=1, max_model_turns=4),
            conversation_store=conversations,
            memory=RecentTurnsConversationMemoryPolicy(max_turns=0),
            event_sink=events,
        )

        result = await agent.run(
            "inspect only this current request",
            session_id="plan-memory",
        )

        assert result.finish_reason is FinishReason.COMPLETED
        assert result.output == "safe final"
        plan_request = next(
            request
            for request in model.requests
            if "steps" in (request.output_schema or {}).get("properties", {})
        )
        contents = [message.content or "" for message in plan_request.messages]
        assert all("old private-sized context" not in item for item in contents)
        assert all("old answer" not in item for item in contents)
        assert any("inspect only this current request" in item for item in contents)

        compacted = [
            event
            for event in events.events
            if event.type is EventType.MEMORY_COMPACTED
            and event.data.get("phase") == "plan"
        ]
        assert len(compacted) == 1
        assert compacted[0].data["dropped_history_turns"] == 1

    asyncio.run(scenario())


def test_replan_uses_plan_memory_phase_instead_of_failing_phase_conversion() -> None:
    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        await conversations.append(
            "replan-memory",
            [Message.user("old context"), Message.assistant("old answer")],
        )
        model = _PlanModel(block_first_step=True)
        events = _CollectingSink()
        agent = Agent.create(
            model=model,
            instructions="Revise unfinished steps from bounded context.",
            execution="plan",
            model_options={
                "temperature": 0,
                "max_tokens": 2048,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "tools": [{"name": "must-not-escape"}],
            },
            limits=RunLimits(
                max_steps=1,
                max_replans=1,
                max_step_attempts=2,
                max_model_turns=6,
            ),
            conversation_store=conversations,
            memory=RecentTurnsConversationMemoryPolicy(max_turns=0),
            event_sink=events,
        )

        result = await agent.run(
            "inspect this request",
            session_id="replan-memory",
        )

        assert result.finish_reason is FinishReason.COMPLETED
        assert model.plan_calls == 2
        assert model.step_result_calls == 2
        plan_requests = [
            request
            for request in model.requests
            if "steps" in (request.output_schema or {}).get("properties", {})
        ]
        assert [request.options for request in plan_requests] == [
            {"temperature": 0, "max_tokens": 2048},
            {"temperature": 0, "max_tokens": 2048},
        ]
        plan_compactions = [
            event
            for event in events.events
            if event.type is EventType.MEMORY_COMPACTED
            and event.data.get("phase") == "plan"
        ]
        assert len(plan_compactions) == 2

    asyncio.run(scenario())


def test_separate_planning_model_keeps_its_options_isolated() -> None:
    async def scenario() -> None:
        planner_model = _PlanModel()
        execution_model = _PlanModel()
        limits = RunLimits(max_steps=1, max_model_turns=4)
        main_options = {
            "main_model_only": True,
            "max_tokens": 512,
            "tools": [{"name": "must-not-escape"}],
        }
        planner_options = {"temperature": 0.25}
        planner_provider_options = {"seed": 17}
        agent = Agent(
            config=AgentConfig(
                name="separate-planner",
                instructions="Use the separate planner.",
                limits=limits,
                model_options=main_options,
            ),
            model=execution_model,
            decision_policy=PlanAndExecutePolicy(
                LLMPlanGenerator(
                    planner_model,
                    max_steps=1,
                    options=planner_options,
                    provider_options=planner_provider_options,
                )
            ),
        )

        result = await agent.run("inspect this request")

        assert result.finish_reason is FinishReason.COMPLETED
        assert len(planner_model.requests) == 1
        assert planner_model.requests[0].options == {"temperature": 0.25}
        assert planner_model.requests[0].provider_options == {"seed": 17}
        assert all(
            request.options.get("main_model_only") is True
            for request in execution_model.requests
        )
        assert all(
            "tools" not in request.options for request in execution_model.requests
        )
        assert main_options == {
            "main_model_only": True,
            "max_tokens": 512,
            "tools": [{"name": "must-not-escape"}],
        }
        assert planner_options == {"temperature": 0.25}
        assert planner_provider_options == {"seed": 17}

    asyncio.run(scenario())


def test_same_model_planner_options_override_agent_defaults() -> None:
    async def scenario() -> None:
        model = _PlanModel()
        agent = Agent.create(
            model=model,
            instructions="Use explicit planner options.",
            execution=PlanExecutionProfile(
                LLMPlanGenerator(
                    model,
                    max_steps=1,
                    options={"temperature": 0.25},
                )
            ),
            limits=RunLimits(max_steps=1, max_model_turns=4),
            model_options={"temperature": 0, "max_tokens": 512},
        )

        result = await agent.run("inspect this request")

        assert result.finish_reason is FinishReason.COMPLETED
        plan_request = next(
            request
            for request in model.requests
            if "steps" in (request.output_schema or {}).get("properties", {})
        )
        assert plan_request.options == {"temperature": 0.25, "max_tokens": 512}

    asyncio.run(scenario())

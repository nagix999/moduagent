from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel

from moduagent import (
    Agent,
    AgentConfig,
    EventType,
    ExecutionState,
    FinishReason,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    LLMPlanGenerator,
    Message,
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
    Plan,
    PlanAndExecutePolicy,
    PlanStep,
    PydanticOutputCodec,
    RunLimits,
    RunPhase,
    StepResult,
    ToolCall,
    function_tool,
)
from moduagent.runtime.context import RunContext, RunRequest, RunStatus


def _step_result(
    step_id: str,
    *,
    fact: str,
    evidence: str,
) -> str:
    return json.dumps(
        {
            "step_id": step_id,
            "status": "completed",
            "facts": [fact],
            "completion_evidence": [evidence],
        },
        ensure_ascii=False,
    )


class StaticPlanGenerator:
    def __init__(self, steps: list[PlanStep]) -> None:
        self.steps = steps

    async def create(self, context: Any) -> Plan:
        return Plan(self.steps)

    async def revise(self, context: Any, plan: Plan, feedback: str) -> Plan:
        return plan


class StreamingScriptedModel:
    capabilities = ModelCapabilities(streaming=True)

    def __init__(
        self,
        items: list[tuple[tuple[str, ...], ModelResponse]],
    ) -> None:
        self.items = items
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("complete should not be called")

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        deltas, response = self.items.pop(0)
        for delta in deltas:
            yield ModelChunk(delta=delta)
        yield ModelChunk(response=response)


class CompleteScriptedModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, items: list[ModelResponse | Exception]) -> None:
        self.items = items
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.items:
            raise AssertionError("unexpected model request")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, request: ModelRequest):
        raise AssertionError("stream should not be called")
        yield


def test_no_tool_text_two_steps_finalize_and_hide_step_deltas_by_default() -> None:
    async def scenario() -> None:
        first_result = _step_result(
            "collect",
            fact="자료를 수집했다",
            evidence="자료가 수집됐다",
        )
        second_result = _step_result(
            "analyze",
            fact="자료를 분석했다",
            evidence="분석 결과가 있다",
        )
        model = StreamingScriptedModel(
            [
                ((first_result,), ModelResponse(Message.assistant(first_result))),
                ((second_result,), ModelResponse(Message.assistant(second_result))),
                (
                    ("최종 ", "공개 답변"),
                    ModelResponse(Message.assistant("최종 공개 답변")),
                ),
            ]
        )
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig("strict-text", "정확히 답한다."),
            model=model,
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="collect",
                            objective="자료 수집",
                            completion_criteria=["자료가 수집됐다"],
                        ),
                        PlanStep(
                            step_id="analyze",
                            objective="자료 분석",
                            completion_criteria=["분석 결과가 있다"],
                            dependencies=["collect"],
                        ),
                    ]
                )
            ),
            conversation_store=conversations,
        )

        events = [
            event
            async for event in agent.stream(
                "두 단계로 검토해줘",
                session_id="strict-text",
            )
        ]
        result = events[-1].data["result"]

        assert result.finish_reason is FinishReason.COMPLETED
        assert result.output == "최종 공개 답변"
        assert result.metadata["plan_usage"] == {
            "phase": "done",
            "committed_steps": 2,
            "replans": 0,
            "finalization_calls": 1,
        }
        assert len(model.requests) == 3
        assert all(request.tools == () for request in model.requests)
        assert [
            request.output_schema and request.output_schema.get("title")
            for request in model.requests
        ] == ["StepResult", "StepResult", None]
        assert not any(event.type is EventType.MODEL_DELTA for event in events)
        assert not any(event.type is EventType.STEP_MODEL_DELTA for event in events)
        assert [
            event.data["delta"]
            for event in events
            if event.type is EventType.FINAL_DELTA
        ] == ["최종 ", "공개 답변"]
        assert [message.content for message in result.messages] == [
            "정확히 답한다.",
            "두 단계로 검토해줘",
            "최종 공개 답변",
        ]
        assert [
            message.content for message in await conversations.load("strict-text")
        ] == ["두 단계로 검토해줘", "최종 공개 답변"]

    asyncio.run(scenario())


def test_stream_include_internal_exposes_step_model_delta() -> None:
    async def scenario() -> None:
        step_result = _step_result(
            "inspect",
            fact="검사를 마쳤다",
            evidence="검사 결과가 있다",
        )
        model = StreamingScriptedModel(
            [
                ((step_result,), ModelResponse(Message.assistant(step_result))),
                (("완료",), ModelResponse(Message.assistant("완료"))),
            ]
        )
        agent = Agent(
            config=AgentConfig("strict-stream-all", "정확히 답한다."),
            model=model,
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="inspect",
                            objective="검사",
                            completion_criteria=["검사 결과가 있다"],
                        )
                    ]
                )
            ),
        )

        events = [
            event
            async for event in agent.stream(
                "검사해줘",
                include_internal=True,
            )
        ]

        step_deltas = [
            event for event in events if event.type is EventType.STEP_MODEL_DELTA
        ]
        assert len(step_deltas) == 1
        assert step_deltas[0].data["phase"] == "step_result"
        assert step_deltas[0].data["delta"] == step_result
        assert [
            event.data["delta"]
            for event in events
            if event.type is EventType.FINAL_DELTA
        ] == ["완료"]

    asyncio.run(scenario())


def test_final_stream_uses_validated_response_when_provider_deltas_disagree() -> None:
    async def scenario() -> None:
        step_result = _step_result(
            "inspect",
            fact="inspection completed",
            evidence="inspection result exists",
        )
        model = StreamingScriptedModel(
            [
                ((step_result,), ModelResponse(Message.assistant(step_result))),
                (
                    ("UNVALIDATED-SECRET",),
                    ModelResponse(Message.assistant("safe final")),
                ),
            ]
        )
        agent = Agent(
            config=AgentConfig("validated-stream", "Publish validated output only."),
            model=model,
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="inspect",
                            objective="inspect",
                            completion_criteria=["inspection result exists"],
                        )
                    ]
                )
            ),
        )

        events = [event async for event in agent.stream("inspect")]

        assert [
            event.data["delta"]
            for event in events
            if event.type is EventType.FINAL_DELTA
        ] == ["safe final"]
        assert "UNVALIDATED-SECRET" not in json.dumps(
            [event.data for event in events],
            default=str,
        )

    asyncio.run(scenario())


class StructuredAnswer(BaseModel):
    answer: str
    confidence: float


def test_tool_structured_phases_are_separate_private_and_fit_one_plan_step() -> None:
    async def scenario() -> None:
        calls: list[tuple[int, int]] = []

        @function_tool
        def add(a: int, b: int) -> int:
            """Add two integers."""

            calls.append((a, b))
            return a + b

        class PhaseModel:
            capabilities = ModelCapabilities(streaming=False)

            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []

            async def complete(self, request: ModelRequest) -> ModelResponse:
                self.requests.append(request)
                schema = request.output_schema or {}
                if "steps" in schema.get("properties", {}):
                    return ModelResponse(
                        Message.assistant(
                            json.dumps(
                                {
                                    "steps": [
                                        {
                                            "step_id": "calculate",
                                            "objective": "add the numbers",
                                            "completion_criteria": [
                                                "the returned sum is verified"
                                            ],
                                            "expected_output": "verified sum",
                                            "dependencies": [],
                                            "allowed_tools": ["add"],
                                        }
                                    ]
                                }
                            )
                        )
                    )
                if request.tools:
                    call = ToolCall("add-1", "add", {"a": 2, "b": 3})
                    return ModelResponse(
                        Message.assistant("private ACT draft: 5", (call,)),
                        (call,),
                    )
                if schema.get("title") == "StepResult":
                    return ModelResponse(
                        Message.assistant(
                            _step_result(
                                "calculate",
                                fact="add returned 5",
                                evidence="the returned sum is verified",
                            )
                        )
                    )
                return ModelResponse(
                    Message.assistant('{"answer":"5","confidence":1.0}')
                )

            async def stream(self, request: ModelRequest):
                raise AssertionError("stream should not be called")
                yield

        conversations = InMemoryConversationStore()
        checkpoints = InMemoryCheckpointStore()
        model = PhaseModel()
        agent = Agent(
            config=AgentConfig(
                "strict-calculator",
                "Use verified arithmetic.",
                limits=RunLimits(max_steps=1),
            ),
            model=model,
            tools=[add],
            decision_policy=PlanAndExecutePolicy(LLMPlanGenerator(model, max_steps=1)),
            output_codec=PydanticOutputCodec(StructuredAnswer),
            conversation_store=conversations,
            checkpoint_store=checkpoints,
        )

        result = await agent.run(
            "What is 2 + 3?",
            session_id="strict-structured",
        )

        assert result.finish_reason is FinishReason.COMPLETED
        assert result.output == StructuredAnswer(answer="5", confidence=1.0)
        assert calls == [(2, 3)]
        assert result.metadata["plan_usage"]["committed_steps"] == 1
        trace = result.metadata["tool_trace"]
        assert len(trace) == 1
        assert trace[0]["step_id"] == "calculate"
        assert trace[0]["call_id"] == "add-1"
        assert trace[0]["tool_name"] == "add"
        assert trace[0]["success"] is True
        assert trace[0]["error"] is None
        assert "arguments" not in trace[0]
        checkpoint = await checkpoints.load(result.run_id)
        assert checkpoint is not None
        assert checkpoint.metadata["_moduagent_tool_trace"] == trace
        assert len(model.requests) == 4
        plan_request, act_tool_request, step_result_request, finalize_request = (
            model.requests
        )
        assert plan_request.tools == ()
        assert "steps" in plan_request.output_schema["properties"]
        assert [schema.name for schema in act_tool_request.tools] == ["add"]
        assert act_tool_request.output_schema is None
        assert step_result_request.tools == ()
        assert step_result_request.output_schema["title"] == "StepResult"
        assert finalize_request.tools == ()
        assert finalize_request.output_schema["title"] == "StructuredAnswer"
        assert all(
            not (request.tools and request.output_schema is not None)
            for request in model.requests
        )

        # ACT/tool/StepResult transcripts are private and never enter either
        # the public result history or the durable conversation.
        assert [message.content for message in result.messages] == [
            "Use verified arithmetic.",
            "What is 2 + 3?",
            '{"answer":"5","confidence":1.0}',
        ]
        assert all(not message.tool_calls for message in result.messages)
        assert all(message.role.value != "tool" for message in result.messages)
        assert all(
            message.content != "private ACT draft: 5" for message in result.messages
        )
        stored = await conversations.load("strict-structured")
        assert [message.content for message in stored] == [
            "What is 2 + 3?",
            '{"answer":"5","confidence":1.0}',
        ]
        assert all(message.content != "private ACT draft: 5" for message in stored)
        assert stored[-1].metadata["moduagent.public_final"] is True

    asyncio.run(scenario())


def test_finalize_failure_resume_does_not_repeat_act_or_tool_execution() -> None:
    async def scenario() -> None:
        executions: list[tuple[int, int]] = []

        @function_tool
        def add(a: int, b: int) -> int:
            executions.append((a, b))
            return a + b

        call = ToolCall("resume-add-1", "add", {"a": 2, "b": 3})
        model = CompleteScriptedModel(
            [
                ModelResponse(
                    Message.assistant("private ACT draft: 5", (call,)),
                    (call,),
                ),
                ModelResponse(
                    Message.assistant(
                        _step_result(
                            "calculate",
                            fact="add returned 5",
                            evidence="the returned sum is verified",
                        )
                    )
                ),
                RuntimeError("finalizer unavailable"),
                ModelResponse(Message.assistant("The answer is 5.")),
            ]
        )
        checkpoints = InMemoryCheckpointStore()
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig("strict-resume", "Use verified arithmetic."),
            model=model,
            tools=[add],
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="calculate",
                            objective="add the numbers",
                            completion_criteria=["the returned sum is verified"],
                            allowed_tools=["add"],
                        )
                    ]
                )
            ),
            checkpoint_store=checkpoints,
            conversation_store=conversations,
        )

        failed = await agent.run(
            "What is 2 + 3?",
            session_id="strict-finalize-resume",
        )

        assert failed.finish_reason is FinishReason.ERROR
        assert failed.error == "finalizer unavailable"
        assert executions == [(2, 3)]
        checkpoint = await checkpoints.load(failed.run_id)
        assert checkpoint is not None
        assert checkpoint.to_dict()["execution_state"]["phase"] == "finalize"
        assert [
            message.content
            for message in await conversations.load("strict-finalize-resume")
        ] == ["What is 2 + 3?"]

        resumed = await agent.resume(
            failed.run_id,
            session_id="strict-finalize-resume",
        )

        assert resumed.finish_reason is FinishReason.COMPLETED
        assert resumed.output == "The answer is 5."
        assert executions == [(2, 3)]
        assert len(model.requests) == 4
        assert sum(bool(request.tools) for request in model.requests) == 1
        assert [
            request.output_schema and request.output_schema.get("title")
            for request in model.requests
        ] == [None, "StepResult", None, None]
        assert [
            message.content
            for message in await conversations.load("strict-finalize-resume")
        ] == ["What is 2 + 3?", "The answer is 5."]

    asyncio.run(scenario())


def test_done_checkpoint_resume_does_not_call_model_or_duplicate_final() -> None:
    async def scenario() -> None:
        step_result = _step_result(
            "inspect",
            fact="검사를 마쳤다",
            evidence="검사 결과가 있다",
        )
        model = CompleteScriptedModel(
            [
                ModelResponse(Message.assistant(step_result)),
                ModelResponse(Message.assistant("최종 답변")),
            ]
        )
        checkpoints = InMemoryCheckpointStore()
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig("strict-done-resume", "정확히 답한다."),
            model=model,
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="inspect",
                            objective="검사",
                            completion_criteria=["검사 결과가 있다"],
                        )
                    ]
                )
            ),
            checkpoint_store=checkpoints,
            conversation_store=conversations,
        )

        completed = await agent.run(
            "검사해줘",
            session_id="strict-done-resume",
        )
        stored_before_resume = await conversations.load("strict-done-resume")
        checkpoint = await checkpoints.load(completed.run_id)

        assert completed.finish_reason is FinishReason.COMPLETED
        assert completed.output == "최종 답변"
        assert checkpoint is not None
        checkpoint_state = checkpoint.to_dict()["execution_state"]
        assert checkpoint_state["phase"] == "done"
        assert checkpoint_state["final_emitted"] is True
        assert len(model.requests) == 2

        resumed = await agent.resume(
            completed.run_id,
            session_id="strict-done-resume",
        )

        assert resumed.finish_reason is FinishReason.COMPLETED
        assert resumed.output == "최종 답변"
        assert resumed.messages == completed.messages
        assert len(model.requests) == 2
        assert await conversations.load("strict-done-resume") == stored_before_resume

    asyncio.run(scenario())


def test_replan_cannot_expand_past_max_steps() -> None:
    class ExpandingPlanGenerator:
        async def create(self, context: Any) -> Plan:
            return Plan(
                [
                    PlanStep(
                        step_id="blocked",
                        objective="inspect the unavailable input",
                        completion_criteria=["the input is inspected"],
                    )
                ]
            )

        async def revise(
            self,
            context: Any,
            plan: Plan,
            feedback: str,
        ) -> Plan:
            return Plan(
                [
                    PlanStep(
                        step_id="collect",
                        objective="collect the missing input",
                        completion_criteria=["the input is available"],
                    ),
                    PlanStep(
                        step_id="inspect",
                        objective="inspect the input",
                        completion_criteria=["the input is inspected"],
                        dependencies=["collect"],
                    ),
                ]
            )

    async def scenario() -> None:
        blocked = json.dumps(
            {
                "step_id": "blocked",
                "status": "blocked",
                "missing_inputs": ["source document"],
            }
        )
        model = CompleteScriptedModel([ModelResponse(Message.assistant(blocked))])
        agent = Agent(
            config=AgentConfig(
                "bounded-replan",
                "Stay within the configured plan size.",
                limits=RunLimits(max_steps=1, max_replans=1),
            ),
            model=model,
            decision_policy=PlanAndExecutePolicy(ExpandingPlanGenerator()),
        )

        result = await agent.run("inspect it")

        assert result.finish_reason is FinishReason.MAX_STEPS
        assert result.error == "plan exceeds RunLimits.max_steps (1)"
        assert [step["status"] for step in result.metadata["plan"]["steps"]] == [
            "failed",
            "pending",
        ]
        assert len(model.requests) == 1

    asyncio.run(scenario())


def test_tool_call_limit_marks_current_step_failed() -> None:
    async def scenario() -> None:
        calls = 0

        @function_tool
        def lookup(query: str) -> str:
            nonlocal calls
            calls += 1
            return query

        call = ToolCall("lookup-1", "lookup", {"query": "value"})
        model = CompleteScriptedModel(
            [ModelResponse(Message.assistant(None, (call,)), (call,))]
        )
        checkpoints = InMemoryCheckpointStore()
        agent = Agent(
            config=AgentConfig(
                "bounded-tools",
                "Respect the Tool call limit.",
                limits=RunLimits(max_steps=1, max_tool_calls=0),
            ),
            model=model,
            tools=[lookup],
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="lookup",
                            objective="look up the value",
                            completion_criteria=["the value is available"],
                            allowed_tools=["lookup"],
                        )
                    ]
                )
            ),
            checkpoint_store=checkpoints,
        )

        result = await agent.run("look it up", session_id="bounded-tools")

        assert result.finish_reason is FinishReason.MAX_TOOL_CALLS
        assert result.metadata["plan"]["steps"][0]["status"] == "failed"
        assert result.metadata["plan_usage"]["phase"] == "failed"
        assert result.metadata["tool_trace"][0]["success"] is False
        assert calls == 0
        checkpoint = await checkpoints.load(result.run_id)
        assert checkpoint is not None
        assert checkpoint.to_context().execution_state.plan.steps[0].status.value == (
            "failed"
        )

    asyncio.run(scenario())


def test_invalid_step_result_details_do_not_leak_to_public_metadata() -> None:
    async def scenario() -> None:
        secret = "TOP-SECRET-123"
        invalid = json.dumps(
            {
                "step_id": "inspect",
                "status": "completed",
                "completion_evidence": ["inspection result exists"],
                "final_answer": secret,
            }
        )
        valid = _step_result(
            "inspect",
            fact="inspection completed",
            evidence="inspection result exists",
        )
        model = CompleteScriptedModel(
            [
                ModelResponse(Message.assistant(invalid)),
                ModelResponse(Message.assistant(valid)),
                ModelResponse(Message.assistant("public final")),
            ]
        )
        agent = Agent(
            config=AgentConfig("private-retry", "Keep ACT details private."),
            model=model,
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="inspect",
                            objective="inspect",
                            completion_criteria=["inspection result exists"],
                        )
                    ]
                )
            ),
        )

        events = [event async for event in agent.stream("inspect")]
        result = events[-1].data["result"]

        assert result.finish_reason is FinishReason.COMPLETED
        assert secret not in json.dumps(
            [
                {
                    "type": event.type.value,
                    "data": event.data,
                }
                for event in events
            ],
            ensure_ascii=False,
            default=str,
        )
        assert secret not in json.dumps(
            result.metadata, ensure_ascii=False, default=str
        )
        assert secret not in json.dumps(
            [message.to_dict() for message in result.messages],
            ensure_ascii=False,
        )

    asyncio.run(scenario())


def test_incomplete_finalizer_response_is_never_persisted_or_completed() -> None:
    async def scenario() -> None:
        step_result = _step_result(
            "inspect",
            fact="inspection completed",
            evidence="inspection result exists",
        )
        model = CompleteScriptedModel(
            [
                ModelResponse(Message.assistant(step_result)),
                ModelResponse(
                    Message.assistant("partial public"),
                    finish_reason="timeout",
                ),
            ]
        )
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig("partial-final", "Reject partial finals."),
            model=model,
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="inspect",
                            objective="inspect",
                            completion_criteria=["inspection result exists"],
                        )
                    ]
                )
            ),
            conversation_store=conversations,
        )

        result = await agent.run("inspect", session_id="partial-final")

        assert result.finish_reason is FinishReason.ERROR
        assert result.error == "incomplete finalization response (timeout)"
        assert [
            message.content for message in await conversations.load("partial-final")
        ] == ["inspect"]

    asyncio.run(scenario())


def test_strict_policy_rejects_disabled_finalization_mode() -> None:
    model = CompleteScriptedModel([])

    try:
        Agent(
            config=AgentConfig(
                "invalid-strict-config",
                "Strict execution requires a finalizer.",
                finalization_mode="disabled",
            ),
            model=model,
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="inspect",
                            objective="inspect",
                            completion_criteria=["inspection result exists"],
                        )
                    ]
                )
            ),
        )
    except ValueError as exc:
        assert str(exc) == (
            "strict Plan-and-Execute requires finalization_mode to be enabled"
        )
    else:
        raise AssertionError("disabled finalization_mode must be rejected")


def test_step_validate_checkpoint_resumes_without_repeating_act() -> None:
    async def scenario() -> None:
        step = PlanStep(
            step_id="inspect",
            objective="inspect",
            completion_criteria=["inspection result exists"],
        )
        plan = Plan([step])
        plan.start_current()
        state = ExecutionState(
            phase=RunPhase.ACT,
            plan=plan,
            current_step_id="inspect",
            awaiting_step_result=True,
        )
        state.set_pending_result(
            StepResult(
                step_id="inspect",
                status="completed",
                facts=["restored inspection result"],
                completion_evidence=["inspection result exists"],
            )
        )
        user_message = Message.user("inspect")
        context = RunContext(
            run_id="step-validate-run",
            request=RunRequest(input="inspect", session_id="step-validate-session"),
            messages=[Message.system("Resume safely."), user_message],
            new_messages=[user_message],
            execution_state=state,
            status=RunStatus.RUNNING,
            current_run_start=1,
        )
        checkpoints = InMemoryCheckpointStore()
        conversations = InMemoryConversationStore()
        await checkpoints.save(context.run_id, context)
        model = CompleteScriptedModel(
            [ModelResponse(Message.assistant("public final"))]
        )
        agent = Agent(
            config=AgentConfig("step-validate-resume", "Resume safely."),
            model=model,
            decision_policy=PlanAndExecutePolicy(StaticPlanGenerator([step])),
            checkpoint_store=checkpoints,
            conversation_store=conversations,
        )

        result = await agent.resume(
            "step-validate-run",
            session_id="step-validate-session",
        )

        assert result.finish_reason is FinishReason.COMPLETED
        assert result.output == "public final"
        assert len(model.requests) == 1
        assert model.requests[0].tools == ()
        assert model.requests[0].output_schema is None
        assert result.metadata["plan_usage"]["committed_steps"] == 1

    asyncio.run(scenario())

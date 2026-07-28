from __future__ import annotations

import asyncio
import warnings
from typing import Any

import pytest
from pydantic import ValidationError

from moduagent.agent import Agent
from moduagent.config import AgentConfig
from moduagent.decision import (
    DecisionKind,
    ExecutionState,
    LegacyPlanAndExecutePolicy,
    LLMPlanGenerator,
    Plan,
    PlanAndExecutePolicy,
    PlanStep,
    PlanStepStatus,
    RunPhase,
    StepResult,
    StepValidation,
    StepValidator,
    ValidationKind,
)
from moduagent.messages import Message, ToolCall
from moduagent.models import ModelCapabilities, ModelRequest, ModelResponse
from moduagent.output import StepResultCodec, TextOutputCodec
from moduagent.persistence import InMemoryConversationStore
from moduagent.runtime import EventType
from moduagent.runtime.context import RunContext, RunRequest
from moduagent.tools import (
    ToolError,
    ToolErrorType,
    ToolResult,
    function_tool,
)


class StaticPlanGenerator:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    async def create(self, context: Any) -> Plan:
        return self.plan

    async def revise(self, context: Any, plan: Plan, feedback: str) -> Plan:
        return plan


class RevisingPlanGenerator:
    def __init__(
        self,
        initial: Plan,
        revision: Plan | Exception,
    ) -> None:
        self.initial = initial
        self.revision = revision

    async def create(self, context: Any) -> Plan:
        return self.initial

    async def revise(self, context: Any, plan: Plan, feedback: str) -> Plan:
        if isinstance(self.revision, Exception):
            raise self.revision
        return self.revision


class ExplodingValidator(StepValidator):
    def validate(self, step: PlanStep, result: StepResult) -> StepValidation:
        raise RuntimeError("sensitive validator detail")


class QueueModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request: ModelRequest):
        raise AssertionError("stream should not be called")
        yield


def _context() -> RunContext:
    return RunContext(
        run_id="run-1",
        request=RunRequest(input="perform the task", session_id="session-1"),
        messages=[
            Message.system("base instructions"),
            Message.user("perform the task"),
        ],
    )


def _step(*, criteria: int = 1) -> PlanStep:
    return PlanStep(
        step_id="S1",
        objective="collect evidence",
        completion_criteria=[f"criterion {index}" for index in range(criteria)],
        expected_output="verified facts",
        allowed_tools=["lookup"],
    )


def test_plan_step_keeps_positional_description_compatibility() -> None:
    step = PlanStep("legacy description")
    restored = PlanStep.from_dict(step.to_dict())

    assert restored.description == "legacy description"
    assert restored.objective == "legacy description"
    assert restored.step_id == step.step_id
    assert restored.completion_criteria


def test_plan_rejects_dependency_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        Plan(
            [
                PlanStep(
                    step_id="S1",
                    objective="one",
                    completion_criteria=["done"],
                    dependencies=["S2"],
                ),
                PlanStep(
                    step_id="S2",
                    objective="two",
                    completion_criteria=["done"],
                    dependencies=["S1"],
                ),
            ]
        )


def test_step_result_codec_forbids_phase_leak_fields() -> None:
    codec = StepResultCodec()

    with pytest.raises(ValidationError):
        codec.decode(
            '{"step_id":"S1","status":"completed",'
            '"completion_evidence":["evidence"],"final_answer":"leak"}'
        )

    assert codec.schema()["additionalProperties"] is False


def test_step_validator_requires_evidence_for_every_criterion() -> None:
    validation = StepValidator().validate(
        _step(criteria=2),
        StepResult(
            step_id="S1",
            status="completed",
            completion_evidence=["only one"],
        ),
    )

    assert validation.kind is ValidationKind.RETRY
    assert validation.unmet_criteria == ("criterion 1",)


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        (
            '{"steps":[{"step_id":"S1","objective":"collect",'
            '"completion_criteria":["done"],"expected_output":"facts",'
            '"dependencies":[],"allowed_tools":["missing"]}]}'
        ),
        (
            '{"steps":[{"step_id":"S1","objective":"collect",'
            '"completion_criteria":["done"],"expected_output":"facts",'
            '"dependencies":["UNKNOWN"],"allowed_tools":["lookup"]}]}'
        ),
        (
            '{"steps":[{"step_id":"S1","objective":"collect",'
            '"completion_criteria":["done"],'
            '"dependencies":[],"allowed_tools":["lookup"]}]}'
        ),
    ],
    ids=["malformed", "unknown-tool", "bad-dependency", "missing-required-field"],
)
def test_llm_plan_generator_fails_closed_on_invalid_plan(content: str) -> None:
    async def scenario() -> None:
        context = _context()
        context.metadata["_moduagent_available_tools"] = ["lookup"]
        model = QueueModel(
            [ModelResponse(Message.assistant(content), finish_reason="stop")]
        )

        with pytest.raises(ValueError, match="invalid plan response"):
            await LLMPlanGenerator(model).create(context)

    asyncio.run(scenario())


def test_llm_plan_generator_rejects_incomplete_valid_json() -> None:
    async def scenario() -> None:
        context = _context()
        context.metadata["_moduagent_available_tools"] = ["lookup"]
        model = QueueModel(
            [
                ModelResponse(
                    Message.assistant(
                        '{"steps":[{"step_id":"S1","objective":"collect",'
                        '"completion_criteria":["done"],"expected_output":"facts",'
                        '"dependencies":[],"allowed_tools":["lookup"]}]}'
                    ),
                    finish_reason="timeout",
                )
            ]
        )

        with pytest.raises(ValueError, match="incomplete plan response"):
            await LLMPlanGenerator(model).create(context)

    asyncio.run(scenario())


def test_llm_plan_generator_includes_only_bounded_public_history() -> None:
    async def scenario() -> None:
        context = _context()
        context.messages = [
            Message.system("base instructions"),
            Message.user("earlier question"),
            Message.assistant(
                "previous public final",
                metadata={"moduagent.public_final": True},
            ),
            Message.assistant(
                "ephemeral execution draft",
                metadata={"moduagent.ephemeral": True},
            ),
            Message.user("perform the task"),
        ]
        context.current_run_start = 4
        context.internal_messages = [
            Message.assistant("private tool transcript"),
        ]
        context.metadata["_moduagent_available_tools"] = ["lookup"]
        model = QueueModel(
            [
                ModelResponse(
                    Message.assistant(
                        '{"steps":[{"step_id":"S1","objective":"collect",'
                        '"completion_criteria":["done"],'
                        '"expected_output":"facts","dependencies":[],'
                        '"allowed_tools":["lookup"]}]}'
                    ),
                    finish_reason="stop",
                )
            ]
        )

        await LLMPlanGenerator(model, history_limit=2).create(context)

        contents = [message.content for message in model.requests[0].messages]
        assert "earlier question" in contents
        assert "previous public final" in contents
        assert "ephemeral execution draft" not in contents
        assert "private tool transcript" not in contents
        assert "perform the task" not in contents
        assert '"request": "perform the task"' in (contents[-1] or "")

    asyncio.run(scenario())


def test_plan_commit_preserves_expected_output_and_uses_content_ref() -> None:
    plan = Plan([_step()])
    plan.start_current()
    result = StepResult(
        step_id="S1",
        status="completed",
        facts=["fact"],
        completion_evidence=["evidence"],
    )

    plan.commit(result)

    assert plan.complete
    assert plan.steps[0].expected_output == "verified facts"
    assert plan.steps[0].result_ref is not None
    assert plan.steps[0].result_ref.startswith("sha256:")
    assert plan.steps[0].result_ref != result.step_id


def test_execution_state_round_trip_preserves_exactly_once_fields() -> None:
    plan = Plan([_step()])
    plan.start_current()
    state = ExecutionState(
        phase=RunPhase.ACT,
        plan=plan,
        current_step_id="S1",
        awaiting_step_result=True,
    )
    state.set_pending_result(
        StepResult(
            step_id="S1",
            status="completed",
            completion_evidence=["evidence"],
        )
    )
    state.commit_pending()
    state.begin_finalization()
    state.record_final_response(
        '{"answer":"done"}',
        persisted=True,
        emitted=True,
    )

    restored = ExecutionState.from_dict(state.to_dict())

    assert restored.phase is RunPhase.DONE
    assert restored.finalization_count == 1
    assert restored.final_response == '{"answer":"done"}'
    assert restored.final_persisted is True
    assert restored.final_emitted is True
    assert restored.committed_results["S1"].status == "completed"


def test_validate_pending_commits_restored_step_without_act() -> None:
    async def scenario() -> None:
        plan = Plan([_step()])
        plan.start_current()
        checkpointed = ExecutionState(
            phase=RunPhase.ACT,
            plan=plan,
            current_step_id="S1",
            awaiting_step_result=True,
        )
        checkpointed.set_pending_result(
            StepResult(
                step_id="S1",
                status="completed",
                facts=["restored fact"],
                completion_evidence=["restored evidence"],
            )
        )
        context = _context()
        context.policy_state["execution_state"] = checkpointed.to_dict()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
        )

        decision = await policy.validate_pending(context)

        state = context.execution_state
        assert decision.kind is DecisionKind.COMMIT_STEP
        assert state.phase is RunPhase.VERIFY
        assert state.pending_step_result is None
        assert state.committed_results["S1"].facts == ["restored fact"]

    asyncio.run(scenario())


def test_strict_policy_requires_schema_only_result_after_tool_turn() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            max_step_attempts=2,
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        call = ToolCall("call-1", "lookup", {"query": "x"})

        decision = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(None, (call,)),
                (call,),
                finish_reason="tool_calls",
            ),
        )

        assert decision.kind is DecisionKind.CALL_TOOLS
        assert policy.needs_step_result_extraction(context) is True
        assert policy.allows_tools(context) is False
        assert context.execution_state.current_step.attempt_count == 0

        committed = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(
                    '{"step_id":"S1","status":"completed",'
                    '"facts":["fact"],"completion_evidence":["evidence"]}'
                ),
                finish_reason="stop",
            ),
        )

        assert committed.kind is DecisionKind.COMMIT_STEP
        assert context.execution_state.phase is RunPhase.VERIFY
        assert context.execution_state.current_step_id is None

    asyncio.run(scenario())


def test_policy_act_projection_excludes_original_request() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(StaticPlanGenerator(Plan([_step()])))
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)

        messages = policy.build_act_messages(context)

        assert all(
            "perform the task" not in (message.content or "") for message in messages
        )
        assert any('"current_step"' in (message.content or "") for message in messages)

    asyncio.run(scenario())


def test_incomplete_tool_response_never_authorizes_tool_side_effects() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            max_step_attempts=2,
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        call = ToolCall("partial-call", "lookup", {"query": "x"})

        decision = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(None, (call,)),
                (call,),
                finish_reason="timeout",
            ),
        )

        assert decision.kind is DecisionKind.RETRY_STEP
        assert not decision.tool_calls
        assert context.execution_state.phase is RunPhase.ACT
        assert context.execution_state.awaiting_step_result is False
        assert context.execution_state.current_step.attempt_count == 1
        assert policy.allows_tools(context) is True

    asyncio.run(scenario())


def test_tool_enabled_content_does_not_implicitly_complete_step() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(StaticPlanGenerator(Plan([_step()])))
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)

        decision = await policy.decide(
            context,
            ModelResponse(Message.assistant("This is a final-looking answer.")),
        )

        assert decision.kind is DecisionKind.RETRY_STEP
        assert decision.metadata["count_attempt"] is False
        assert context.execution_state.current_step.status is PlanStepStatus.IN_PROGRESS
        assert context.execution_state.current_step.attempt_count == 0
        assert policy.needs_step_result_extraction(context) is True

    asyncio.run(scenario())


def test_validator_exception_transitions_execution_to_failed() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            step_validator=ExplodingValidator(),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)

        decision = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(
                    '{"step_id":"S1","status":"completed",'
                    '"completion_evidence":["evidence"]}'
                ),
                finish_reason="stop",
            ),
        )

        state = context.execution_state
        assert decision.kind is DecisionKind.FAIL
        assert state.phase is RunPhase.FAILED
        assert state.validation_error == "step validator failed (RuntimeError)"
        assert decision.error_message == "Step validation failed"
        assert "sensitive validator detail" not in decision.error_message
        assert "sensitive validator detail" not in state.validation_error
        assert state.pending_step_result is not None
        assert context.policy_state["execution_state"]["phase"] == "failed"

    asyncio.run(scenario())


def test_partial_replan_preserves_attempts_and_committed_results() -> None:
    async def scenario() -> None:
        initial = Plan(
            [
                PlanStep(
                    step_id="S1",
                    objective="collect trusted evidence",
                    completion_criteria=["evidence collected"],
                ),
                PlanStep(
                    step_id="S2",
                    objective="analyze evidence",
                    completion_criteria=["analysis complete"],
                    dependencies=["S1"],
                ),
            ]
        )
        revision = Plan(
            [
                PlanStep(
                    step_id="S1",
                    objective="replace committed work",
                    completion_criteria=["different"],
                ),
                PlanStep(
                    step_id="S2",
                    objective="refine the analysis",
                    completion_criteria=["analysis complete"],
                    dependencies=["S1"],
                    attempt_count=0,
                ),
            ]
        )
        context = _context()
        policy = PlanAndExecutePolicy(
            RevisingPlanGenerator(initial, revision),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)
        committed = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(
                    '{"step_id":"S1","status":"completed",'
                    '"facts":["trusted"],'
                    '"completion_evidence":["source"]}'
                ),
                finish_reason="stop",
            ),
        )
        assert committed.kind is DecisionKind.COMMIT_STEP
        original_result = context.execution_state.committed_results["S1"]
        original_ref = context.execution_state.plan.steps[0].result_ref

        policy.prepare_step(context, has_tools=False)
        context.execution_state.current_step.attempt_count = 3
        replanned = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(
                    '{"step_id":"S2","status":"blocked",'
                    '"missing_inputs":["new source"],'
                    '"completion_evidence":[]}'
                ),
                finish_reason="stop",
            ),
        )

        state = context.execution_state
        assert replanned.kind is DecisionKind.REPLAN
        assert state.plan.steps[0].objective == "collect trusted evidence"
        assert state.plan.steps[0].result_ref == original_ref
        assert state.committed_results["S1"] == original_result
        assert state.plan.steps[1].attempt_count == 3

    asyncio.run(scenario())


def test_replan_exception_fails_closed_and_clears_pending_result() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            RevisingPlanGenerator(
                Plan([_step()]),
                RuntimeError("sensitive planner detail"),
            )
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)

        with pytest.raises(RuntimeError, match="^plan revision failed$"):
            await policy.decide(
                context,
                ModelResponse(
                    Message.assistant(
                        '{"step_id":"S1","status":"blocked",'
                        '"missing_inputs":["new source"],'
                        '"completion_evidence":[]}'
                    ),
                    finish_reason="stop",
                ),
            )

        state = context.execution_state
        assert state.phase is RunPhase.FAILED
        assert state.validation_error == "plan revision failed"
        assert state.pending_step_result is None
        assert state.awaiting_step_result is False
        assert context.policy_state["execution_state"]["phase"] == "failed"

    asyncio.run(scenario())


def test_tool_failure_replan_exception_fails_closed() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            RevisingPlanGenerator(
                Plan([_step()]),
                RuntimeError("sensitive planner detail"),
            )
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        failure = ToolResult.failed(
            call_id="call-1",
            tool_name="lookup",
            error=ToolError(
                ToolErrorType.EXECUTION_ERROR,
                "tool failed",
            ),
        )

        with pytest.raises(RuntimeError, match="^plan revision failed$"):
            await policy.observe(context, [failure])

        state = context.execution_state
        assert state.phase is RunPhase.FAILED
        assert state.validation_error == "plan revision failed"
        assert state.awaiting_step_result is False

    asyncio.run(scenario())


def test_incomplete_finish_reason_never_commits_valid_json() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            max_step_attempts=2,
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)

        decision = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(
                    '{"step_id":"S1","status":"completed",'
                    '"completion_evidence":["evidence"]}'
                ),
                finish_reason="timeout",
            ),
        )

        assert decision.kind is DecisionKind.RETRY_STEP
        assert context.execution_state.current_step.attempt_count == 1
        assert not context.execution_state.committed_results

    asyncio.run(scenario())


def test_invalid_step_result_secret_does_not_leak_to_public_failure() -> None:
    async def scenario() -> None:
        secret = "TOP-SECRET-ACT-VALUE"
        model = QueueModel(
            [
                ModelResponse(
                    Message.assistant(
                        '{"step_id":"S1","status":"completed",'
                        f'"facts":["{secret}"],'
                        '"completion_evidence":["evidence"],'
                        f'"unexpected":"{secret}"}}'
                    ),
                    finish_reason="stop",
                )
            ]
        )
        agent = Agent(
            config=AgentConfig("strict", "Do the work."),
            model=model,
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    Plan(
                        [
                            PlanStep(
                                step_id="S1",
                                objective="collect evidence",
                                completion_criteria=["criterion"],
                            )
                        ]
                    )
                ),
                max_step_attempts=1,
            ),
            output_codec=TextOutputCodec(),
        )

        events = [
            event
            async for event in agent.stream(
                "perform the task",
                session_id="secret-failure",
            )
        ]

        terminal = next(event for event in events if event.type is EventType.RUN_FAILED)
        result = terminal.data["result"]
        assert result.error == ("StepResult validation failed after maximum attempts")
        assert all(secret not in repr(event.data) for event in events)

    asyncio.run(scenario())


def test_legacy_policy_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LegacyPlanAndExecutePolicy(StaticPlanGenerator(Plan([_step()])))

    assert any(item.category is DeprecationWarning for item in caught)


def test_strict_runtime_separates_tool_result_and_text_finalization() -> None:
    async def scenario() -> None:
        calls: list[tuple[int, int]] = []

        @function_tool
        def add(a: int, b: int) -> int:
            calls.append((a, b))
            return a + b

        call = ToolCall("call-1", "add", {"a": 2, "b": 3})
        model = QueueModel(
            [
                ModelResponse(
                    Message.assistant(None, (call,)),
                    (call,),
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    Message.assistant(
                        '{"step_id":"S1","status":"completed",'
                        '"facts":["2 + 3 = 5"],'
                        '"completion_evidence":["add returned 5"]}'
                    ),
                    finish_reason="stop",
                ),
                ModelResponse(
                    Message.assistant("The answer is 5."),
                    finish_reason="stop",
                ),
            ]
        )
        conversations = InMemoryConversationStore()
        plan = Plan(
            [
                PlanStep(
                    step_id="S1",
                    objective="add the numbers",
                    completion_criteria=["obtain the sum"],
                    expected_output="verified sum",
                    allowed_tools=["add"],
                )
            ]
        )
        agent = Agent(
            config=AgentConfig("strict", "Calculate accurately."),
            model=model,
            tools=[add],
            decision_policy=PlanAndExecutePolicy(StaticPlanGenerator(plan)),
            output_codec=TextOutputCodec(),
            conversation_store=conversations,
        )

        result = await agent.run("What is 2 + 3?", session_id="strict-session")

        assert result.output == "The answer is 5."
        assert calls == [(2, 3)]
        assert len(model.requests) == 3
        tool_request, result_request, final_request = model.requests
        assert tool_request.tools and tool_request.output_schema is None
        assert result_request.tools == ()
        assert result_request.output_schema["title"] == "StepResult"
        assert final_request.tools == () and final_request.output_schema is None
        persisted = await conversations.load("strict-session")
        assert [message.content for message in persisted] == [
            "What is 2 + 3?",
            "The answer is 5.",
        ]

    asyncio.run(scenario())

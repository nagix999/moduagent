from __future__ import annotations

import asyncio
import json
import warnings
from typing import Any

import pytest
from pydantic import ValidationError

from moduagent.agent import Agent
from moduagent.config import AgentConfig, RunLimits
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
    ToolFailureRecoveryConfig,
    ValidationKind,
)
from moduagent.messages import Message, ToolCall
from moduagent.models import ModelCapabilities, ModelRequest, ModelResponse
from moduagent.output import StepResultCodec, TextOutputCodec
from moduagent.persistence import InMemoryCheckpointStore, InMemoryConversationStore
from moduagent.runtime import EventType
from moduagent.runtime.context import RunContext, RunRequest
from moduagent.tools import (
    ToolError,
    ToolErrorType,
    ToolRecoveryAction,
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


def _repairable_tool_failure(
    *,
    call_id: str = "call-1",
    message: str = "SQL syntax error\nnear SELECT",
    repair_safe: bool = True,
    recovery: ToolRecoveryAction = ToolRecoveryAction.REPAIR_CALL,
) -> ToolResult:
    return ToolResult.failed(
        call_id=call_id,
        tool_name="lookup",
        error=ToolError(
            ToolErrorType.EXECUTION_ERROR,
            message,
            reason="sql_syntax_error",
            recovery=recovery,
        ),
        attempts=1,
        repair_safe=repair_safe,
    )


async def _authorize_tool_calls(
    policy: PlanAndExecutePolicy,
    context: RunContext,
    *calls: ToolCall,
) -> None:
    decision = await policy.decide(
        context,
        ModelResponse(
            Message.assistant(None, tuple(calls)),
            tuple(calls),
            finish_reason="tool_calls",
        ),
    )
    assert decision.kind is DecisionKind.CALL_TOOLS


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


def test_tool_repair_limits_and_recovery_config_are_validated() -> None:
    assert RunLimits().max_tool_repair_attempts == 1
    with pytest.raises(ValueError, match="max_tool_repair_attempts"):
        RunLimits(max_tool_repair_attempts=-1)
    with pytest.raises(ValueError, match="fallback"):
        ToolFailureRecoveryConfig(fallback="retry")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="require_repair_safe"):
        ToolFailureRecoveryConfig(require_repair_safe=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feedback_mode"):
        ToolFailureRecoveryConfig(feedback_mode="raw")  # type: ignore[arg-type]


def test_execution_state_round_trip_preserves_tool_recovery_fields() -> None:
    plan = Plan([_step()])
    plan.start_current()
    state = ExecutionState(
        phase=RunPhase.ACT,
        plan=plan,
        current_step_id="S1",
        tool_repair_counts={"S1": 1},
        pending_tool_failure={
            "step_id": "S1",
            "tool_name": "lookup",
            "reason": "sql_syntax_error",
        },
        total_tool_repairs=1,
        failure={"terminal_reason": "repair exhausted"},
        active_tool_calls={
            "call-2": {
                "tool_name": "lookup",
                "arguments_fingerprint": "sha256:" + ("a" * 64),
            }
        },
        seen_tool_call_ids=["call-1", "call-2"],
    )

    restored = ExecutionState.from_dict(state.to_dict())

    assert restored.tool_repair_counts == {"S1": 1}
    assert restored.total_tool_repairs == 1
    assert restored.pending_tool_failure == {
        "step_id": "S1",
        "tool_name": "lookup",
        "reason": "sql_syntax_error",
    }
    assert restored.failure == {"terminal_reason": "repair exhausted"}
    assert restored.active_tool_calls == {
        "call-2": {
            "tool_name": "lookup",
            "arguments_fingerprint": "sha256:" + ("a" * 64),
        }
    }
    assert restored.seen_tool_call_ids == ["call-1", "call-2"]


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
        assert context.execution_state.plan.steps[0].status is PlanStepStatus.COMPLETED

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
        assert state.plan.steps[0].status is PlanStepStatus.FAILED
        assert context.policy_state["execution_state"]["phase"] == "failed"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "step_result",
    [
        ('{"step_id":"S1","status":"failed","completion_evidence":[]}'),
        ('{"step_id":"OTHER","status":"completed","completion_evidence":["evidence"]}'),
    ],
    ids=["executor-failed", "step-id-mismatch"],
)
def test_terminal_step_validation_marks_current_step_failed(
    step_result: str,
) -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(StaticPlanGenerator(Plan([_step()])))
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)

        decision = await policy.decide(
            context,
            ModelResponse(Message.assistant(step_result), finish_reason="stop"),
        )

        state = context.execution_state
        assert decision.kind is DecisionKind.FAIL
        assert state.phase is RunPhase.FAILED
        assert state.plan.steps[0].status is PlanStepStatus.FAILED
        assert decision.metadata["plan"]["steps"][0]["status"] == "failed"
        if '"OTHER"' in step_result:
            assert state.pending_step_result is None
            checkpoints = InMemoryCheckpointStore()
            await checkpoints.save(context.run_id, context)
            checkpoint = await checkpoints.load(context.run_id)
            assert checkpoint is not None
            restored = checkpoint.to_context().execution_state
            assert restored.phase is RunPhase.FAILED
            assert restored.plan.steps[0].status is PlanStepStatus.FAILED

    asyncio.run(scenario())


def test_incomplete_validation_state_marks_current_step_failed() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(StaticPlanGenerator(Plan([_step()])))
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)

        decision = await policy.validate_pending(context)

        state = context.execution_state
        assert decision.kind is DecisionKind.FAIL
        assert state.phase is RunPhase.FAILED
        assert state.plan.steps[0].status is PlanStepStatus.FAILED

    asyncio.run(scenario())


def test_validation_retry_keeps_step_active_until_attempts_are_exhausted() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            max_step_attempts=2,
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)
        response = ModelResponse(
            Message.assistant(
                '{"step_id":"S1","status":"completed","completion_evidence":[]}'
            ),
            finish_reason="stop",
        )

        retry = await policy.decide(context, response)

        assert retry.kind is DecisionKind.RETRY_STEP
        assert context.execution_state.phase is RunPhase.ACT
        assert (
            context.execution_state.plan.steps[0].status is PlanStepStatus.IN_PROGRESS
        )

        exhausted = await policy.decide(context, response)

        assert exhausted.kind is DecisionKind.FAIL
        assert context.execution_state.phase is RunPhase.FAILED
        assert context.execution_state.plan.steps[0].status is PlanStepStatus.FAILED

    asyncio.run(scenario())


def test_invalid_step_result_retry_exhaustion_marks_current_step_failed() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            max_step_attempts=1,
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)

        decision = await policy.decide(
            context,
            ModelResponse(Message.assistant("not-json"), finish_reason="stop"),
        )

        assert decision.kind is DecisionKind.FAIL
        assert context.execution_state.phase is RunPhase.FAILED
        assert context.execution_state.plan.steps[0].status is PlanStepStatus.FAILED

    asyncio.run(scenario())


def test_replan_exhaustion_marks_current_step_failed() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            max_replans=0,
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=False)

        decision = await policy.decide(
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

        assert decision.kind is DecisionKind.FAIL
        assert context.execution_state.validation_error == "maximum replans exceeded"
        assert context.execution_state.plan.steps[0].status is PlanStepStatus.FAILED

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
        assert state.plan.steps[0].status is PlanStepStatus.FAILED
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
        assert state.plan.steps[0].status is PlanStepStatus.FAILED

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("revise_on_tool_failure", "max_replans"),
    [(False, 2), (True, 0)],
    ids=["revision-disabled", "replans-exhausted"],
)
def test_terminal_tool_failure_marks_current_step_failed(
    revise_on_tool_failure: bool,
    max_replans: int,
) -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            revise_on_tool_failure=revise_on_tool_failure,
            max_replans=max_replans,
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        failure = ToolResult.failed(
            call_id="call-1",
            tool_name="lookup",
            error=ToolError(
                ToolErrorType.EXECUTION_ERROR,
                "database query failed",
            ),
        )

        await policy.observe(context, [failure])

        state = context.execution_state
        assert state.phase is RunPhase.FAILED
        assert state.plan.steps[0].status is PlanStepStatus.FAILED
        assert context.policy_state["plan"]["steps"][0]["status"] == "failed"

    asyncio.run(scenario())


def test_repairable_tool_failure_retries_same_step_with_sanitized_feedback() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(
                feedback_mode="safe_message"
            ),
        )
        policy.configure_limits(
            max_step_attempts=2,
            max_replans=2,
            max_tool_repair_attempts=1,
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        await _authorize_tool_calls(
            policy,
            context,
            ToolCall("call-1", "lookup", {"query": "SELEC id"}),
        )

        decision = await policy.observe(context, [_repairable_tool_failure()])

        state = context.execution_state
        assert decision is not None
        assert decision.kind is DecisionKind.RETRY_TOOL
        assert state.phase is RunPhase.ACT
        assert state.awaiting_step_result is False
        assert state.current_step.status is PlanStepStatus.IN_PROGRESS
        assert state.current_step.attempt_count == 0
        assert state.tool_repair_counts == {"S1": 1}
        assert state.total_tool_repairs == 1
        assert state.pending_tool_failure is not None
        assert state.pending_tool_failure["reason"] == "sql_syntax_error"
        assert state.pending_tool_failure["recovery"] == "repair_call"
        assert state.pending_tool_failure["feedback"] == (
            "Tool lookup failed (sql_syntax_error): SQL syntax error near SELECT"
        )
        assert decision.metadata["count_attempt"] is False

    asyncio.run(scenario())


def test_successful_repaired_tool_call_clears_pending_failure() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        await _authorize_tool_calls(
            policy,
            context,
            ToolCall("call-1", "lookup", {"query": "SELEC id"}),
        )
        scheduled = await policy.observe(context, [_repairable_tool_failure()])
        assert scheduled is not None
        assert scheduled.kind is DecisionKind.RETRY_TOOL
        await _authorize_tool_calls(
            policy,
            context,
            ToolCall("call-2", "lookup", {"query": "SELECT id"}),
        )

        decision = await policy.observe(
            context,
            [
                ToolResult.succeeded(
                    call_id="call-2",
                    tool_name="lookup",
                    value=[{"id": 1}],
                    repair_safe=True,
                )
            ],
        )

        state = context.execution_state
        assert decision is None
        assert state.pending_tool_failure is None
        assert state.failure is None
        assert state.validation_error is None
        assert state.awaiting_step_result is True
        assert state.tool_repair_counts == {"S1": 1}
        assert state.total_tool_repairs == 1

    asyncio.run(scenario())


def test_exhausted_tool_repair_budget_fails_with_original_cause() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="fail"),
        )
        policy.configure_limits(
            max_step_attempts=2,
            max_replans=2,
            max_tool_repair_attempts=1,
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        await _authorize_tool_calls(
            policy,
            context,
            ToolCall("call-1", "lookup", {"query": "SELEC id"}),
        )
        first = await policy.observe(context, [_repairable_tool_failure()])
        assert first is not None
        assert first.kind is DecisionKind.RETRY_TOOL

        retry_call = ToolCall("call-2", "lookup", {"query": "SELECT id FROM data"})
        call_decision = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(None, (retry_call,)),
                (retry_call,),
                finish_reason="tool_calls",
            ),
        )
        assert call_decision.kind is DecisionKind.CALL_TOOLS

        terminal = await policy.observe(
            context,
            [_repairable_tool_failure(call_id="call-2")],
        )

        state = context.execution_state
        assert terminal is not None
        assert terminal.kind is DecisionKind.FAIL
        assert state.phase is RunPhase.FAILED
        assert state.current_step.status is PlanStepStatus.FAILED
        assert state.tool_repair_counts == {"S1": 1}
        assert state.total_tool_repairs == 1
        assert state.failure is not None
        assert state.failure["reason"] == "sql_syntax_error"
        assert state.failure["terminal_reason"] == "tool repair budget exhausted"

    asyncio.run(scenario())


def test_unsafe_tool_failure_uses_replan_fallback_without_repair() -> None:
    async def scenario() -> None:
        initial = Plan([_step()])
        revision = Plan(
            [
                PlanStep(
                    step_id="S1",
                    objective="collect evidence with a safer query",
                    completion_criteria=["criterion 0"],
                    allowed_tools=["lookup"],
                )
            ]
        )
        context = _context()
        policy = PlanAndExecutePolicy(
            RevisingPlanGenerator(initial, revision),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="replan"),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)

        decision = await policy.observe(
            context,
            [_repairable_tool_failure(repair_safe=False)],
        )

        state = context.execution_state
        assert decision is not None
        assert decision.kind is DecisionKind.REPLAN
        assert state.phase is RunPhase.STEP_PREPARE
        assert state.replan_count == 1
        assert state.tool_repair_counts == {}
        assert state.total_tool_repairs == 0
        assert state.pending_tool_failure is None
        assert state.failure is None

    asyncio.run(scenario())


def test_mixed_tool_batch_never_enters_same_step_repair() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="replan"),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        await _authorize_tool_calls(
            policy,
            context,
            ToolCall("call-success", "lookup", {"query": "first"}),
            ToolCall("call-1", "lookup", {"query": "second"}),
        )

        decision = await policy.observe(
            context,
            [
                ToolResult.succeeded(
                    call_id="call-success",
                    tool_name="lookup",
                    value="done",
                ),
                _repairable_tool_failure(),
            ],
        )

        state = context.execution_state
        assert decision is not None
        assert decision.kind is DecisionKind.FAIL
        assert state.phase is RunPhase.FAILED
        assert state.total_tool_repairs == 0
        assert state.failure is not None
        assert state.failure["terminal_reason"] == (
            "Tool batch partially succeeded; automatic recovery is unsafe"
        )
        assert state.replan_count == 0

    asyncio.run(scenario())


def test_missing_tool_call_during_repair_uses_fail_fallback() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="fail"),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        await _authorize_tool_calls(
            policy,
            context,
            ToolCall("call-1", "lookup", {"query": "SELEC id"}),
        )
        scheduled = await policy.observe(context, [_repairable_tool_failure()])
        assert scheduled is not None
        assert scheduled.kind is DecisionKind.RETRY_TOOL

        terminal = await policy.decide(
            context,
            ModelResponse(
                Message.assistant("I cannot call the tool."),
                finish_reason="stop",
            ),
        )

        state = context.execution_state
        assert terminal.kind is DecisionKind.FAIL
        assert state.phase is RunPhase.FAILED
        assert state.failure is not None
        assert state.failure["reason"] == "sql_syntax_error"
        assert state.failure["terminal_reason"] == (
            "tool repair response did not call a tool"
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("repair_calls", "terminal_reason"),
    [
        (
            (
                ToolCall(
                    "call-2",
                    "lookup",
                    {"query": "bad", "limit": 10},
                ),
            ),
            "Tool repair must change the Tool arguments",
        ),
        (
            (
                ToolCall(
                    "call-2",
                    "lookup",
                    {"limit": 10, "query": "bad"},
                ),
            ),
            "Tool repair must change the Tool arguments",
        ),
        (
            (
                ToolCall(
                    "call-2",
                    "alternate_lookup",
                    {"query": "fixed", "limit": 10},
                ),
            ),
            "Tool repair must call the same Tool",
        ),
        (
            (
                ToolCall(
                    "call-1",
                    "lookup",
                    {"query": "fixed", "limit": 10},
                ),
            ),
            "Tool calls must use a new unique call ID",
        ),
        (
            (
                ToolCall(
                    "call-2",
                    "lookup",
                    {"query": "fixed", "limit": 10},
                ),
                ToolCall(
                    "call-3",
                    "lookup",
                    {"query": "also fixed", "limit": 10},
                ),
            ),
            "Tool repair must contain exactly one call",
        ),
    ],
    ids=[
        "same-arguments",
        "key-order-only",
        "different-tool",
        "reused-call-id",
        "multiple-calls",
    ],
)
def test_tool_repair_rejects_unsafe_call_shape_before_execution(
    repair_calls: tuple[ToolCall, ...],
    terminal_reason: str,
) -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="fail"),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        await _authorize_tool_calls(
            policy,
            context,
            ToolCall(
                "call-1",
                "lookup",
                {"query": "bad", "limit": 10},
            ),
        )
        scheduled = await policy.observe(context, [_repairable_tool_failure()])
        assert scheduled is not None
        assert scheduled.kind is DecisionKind.RETRY_TOOL

        terminal = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(None, repair_calls),
                repair_calls,
                finish_reason="tool_calls",
            ),
        )

        state = context.execution_state
        assert terminal.kind is DecisionKind.FAIL
        assert state.phase is RunPhase.FAILED
        assert state.failure is not None
        assert state.failure["terminal_reason"] == terminal_reason
        assert state.seen_tool_call_ids == ["call-1"]
        assert state.active_tool_calls == {}

    asyncio.run(scenario())


def test_tool_repair_argument_fingerprint_survives_checkpoint() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="fail"),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        original = ToolCall(
            "call-1",
            "lookup",
            {"query": "bad", "limit": 10},
        )
        await _authorize_tool_calls(policy, context, original)
        scheduled = await policy.observe(context, [_repairable_tool_failure()])
        assert scheduled is not None
        assert scheduled.kind is DecisionKind.RETRY_TOOL

        serialized = context.execution_state.to_dict()
        assert "bad" not in json.dumps(
            serialized["pending_tool_failure"],
            ensure_ascii=False,
        )
        restored = ExecutionState.from_dict(serialized)
        context.execution_state = restored
        context.policy_state["execution_state"] = restored.to_dict()

        terminal = await policy.decide(
            context,
            ModelResponse(
                Message.assistant(
                    None,
                    (
                        ToolCall(
                            "call-2",
                            "lookup",
                            {"limit": 10, "query": "bad"},
                        ),
                    ),
                ),
                (
                    ToolCall(
                        "call-2",
                        "lookup",
                        {"limit": 10, "query": "bad"},
                    ),
                ),
                finish_reason="tool_calls",
            ),
        )

        assert terminal.kind is DecisionKind.FAIL
        assert context.execution_state.failure is not None
        assert context.execution_state.failure["terminal_reason"] == (
            "Tool repair must change the Tool arguments"
        )

    asyncio.run(scenario())


def test_generic_tool_error_never_exposes_message_as_repair_feedback() -> None:
    async def scenario() -> None:
        secret = "TOP-SECRET-DATABASE-DETAIL"
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(
                fallback="fail",
                feedback_mode="safe_message",
            ),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        generic_failure = ToolResult.failed(
            call_id="call-1",
            tool_name="lookup",
            error=ToolError(
                ToolErrorType.EXECUTION_ERROR,
                secret,
            ),
            repair_safe=True,
        )

        decision = await policy.observe(context, [generic_failure])

        state = context.execution_state
        assert decision is not None
        assert decision.kind is DecisionKind.FAIL
        assert state.failure is not None
        assert secret not in state.failure["feedback"]
        assert state.failure["feedback"] == "Tool lookup failed (execution_error)"

    asyncio.run(scenario())


def test_fail_recovery_action_overrides_replan_fallback() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="replan"),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)

        decision = await policy.observe(
            context,
            [
                _repairable_tool_failure(
                    recovery=ToolRecoveryAction.FAIL,
                )
            ],
        )

        state = context.execution_state
        assert decision is not None
        assert decision.kind is DecisionKind.FAIL
        assert state.phase is RunPhase.FAILED
        assert state.replan_count == 0
        assert state.failure is not None
        assert state.failure["recovery"] == "fail"
        assert state.failure["reason"] == "sql_syntax_error"

    asyncio.run(scenario())


def test_replan_recovery_action_overrides_fail_fallback() -> None:
    async def scenario() -> None:
        initial = Plan([_step()])
        revision = Plan(
            [
                PlanStep(
                    step_id="S1",
                    objective="use an alternate data source",
                    completion_criteria=["criterion 0"],
                    allowed_tools=["lookup"],
                )
            ]
        )
        context = _context()
        policy = PlanAndExecutePolicy(
            RevisingPlanGenerator(initial, revision),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="fail"),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)

        decision = await policy.observe(
            context,
            [
                _repairable_tool_failure(
                    recovery=ToolRecoveryAction.REPLAN,
                )
            ],
        )

        state = context.execution_state
        assert decision is not None
        assert decision.kind is DecisionKind.REPLAN
        assert state.phase is RunPhase.STEP_PREPARE
        assert state.replan_count == 1
        assert state.failure is None
        assert state.pending_tool_failure is None

    asyncio.run(scenario())


def test_retry_call_recovery_uses_fallback_after_executor_exhaustion() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="fail"),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)

        decision = await policy.observe(
            context,
            [
                _repairable_tool_failure(
                    recovery=ToolRecoveryAction.RETRY_CALL,
                )
            ],
        )

        state = context.execution_state
        assert decision is not None
        assert decision.kind is DecisionKind.FAIL
        assert state.phase is RunPhase.FAILED
        assert state.tool_repair_counts == {}
        assert state.failure is not None
        assert state.failure["recovery"] == "retry_call"
        assert state.failure["terminal_reason"] == "tool retry attempts exhausted"

    asyncio.run(scenario())


def test_tool_failure_checkpoint_fields_are_sanitized_and_bounded() -> None:
    async def scenario() -> None:
        context = _context()
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(Plan([_step()])),
            tool_failure_recovery=ToolFailureRecoveryConfig(
                feedback_mode="safe_message"
            ),
        )
        await policy.begin(context)
        policy.prepare_step(context, has_tools=True)
        long_message = "bad\x00\nquery " + ("x" * 800)
        await _authorize_tool_calls(
            policy,
            context,
            ToolCall("call-1", "lookup", {"query": "SELEC id"}),
        )
        result = ToolResult.failed(
            call_id="call-1",
            tool_name="lookup",
            error=ToolError(
                ToolErrorType.EXECUTION_ERROR,
                long_message,
                reason=("sql\x00\n" + ("r" * 400)),
                recovery=ToolRecoveryAction.REPAIR_CALL,
            ),
            repair_safe=True,
        )

        decision = await policy.observe(context, [result])

        assert decision is not None
        assert decision.kind is DecisionKind.RETRY_TOOL
        failure = context.execution_state.pending_tool_failure
        assert failure is not None
        for field in ("call_id", "tool_name", "error_type", "reason", "recovery"):
            value = failure[field]
            assert value is None or len(value) <= 256
            assert value is None or ("\x00" not in value and "\n" not in value)
        assert len(failure["feedback"]) <= 512
        assert "\x00" not in failure["feedback"]
        assert "\n" not in failure["feedback"]

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

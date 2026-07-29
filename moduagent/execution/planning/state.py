from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from moduagent.decision.planning import (
    ExecutionState,
    Plan,
    PlanStep,
    PlanStepStatus,
    RunPhase,
    StepResult,
    step_result_ref,
)
from moduagent.execution.base import EngineStateCodec
from moduagent.messages import FinishReason

try:
    from moduagent.tools import is_tool_argument_fingerprint
except ImportError:  # pragma: no cover - staggered 0.4 package upgrade
    from moduagent.tools.arguments import is_tool_argument_fingerprint


class PlanExecutionPhase(str, Enum):
    PLAN = "plan"
    STEP_PREPARE = "step_prepare"
    ACT_TOOL = "act_tool"
    TOOL_RECOVERY = "tool_recovery"
    STEP_RESULT = "step_result"
    STEP_VALIDATE = "step_validate"
    VERIFY = "verify"
    FINALIZE = "finalize"
    DONE = "done"
    FAILED = "failed"


_FAILURE_TEXT_FIELDS = frozenset(
    {
        "step_id",
        "call_id",
        "tool_name",
        "error_type",
        "type",
        "reason",
        "recovery",
        "feedback",
        "message",
        "fallback_reason",
        "terminal_reason",
    }
)
_FAILURE_BOOL_FIELDS = frozenset({"retryable", "repair_safe"})
_FAILURE_COUNT_FIELDS = frozenset(
    {
        "failure_count",
        "success_count",
        "result_count",
        "repair_attempts",
    }
)
_FAILURE_FINGERPRINT_FIELDS = frozenset(
    {"arguments_fingerprint", "invocation_fingerprint"}
)


def _bounded_text(value: Any, *, limit: int = 512) -> str:
    text = " ".join(
        "".join(
            character if character.isprintable() else " " for character in str(value)
        ).split()
    )
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _safe_failure(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Tool failure state must be a mapping")
    result: dict[str, Any] = {}
    for key in _FAILURE_TEXT_FIELDS:
        raw = value.get(key)
        if raw is not None:
            result[key] = _bounded_text(raw)
    for key in _FAILURE_BOOL_FIELDS:
        if key in value:
            result[key] = bool(value[key])
    for key in _FAILURE_COUNT_FIELDS:
        if key not in value:
            continue
        try:
            count = int(value[key])
        except (TypeError, ValueError, OverflowError):
            count = 0
        result[key] = min(max(count, 0), 1_000_000)
    for key in _FAILURE_FINGERPRINT_FIELDS:
        raw = value.get(key)
        if raw is None:
            continue
        if not is_tool_argument_fingerprint(raw):
            raise ValueError(f"{key} must use sha256")
        result[key] = raw
    return result or None


@dataclass(slots=True)
class PlanProgressState:
    plan: Plan
    committed_results: dict[str, StepResult] = field(default_factory=dict)
    replan_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.plan, Plan):
            raise TypeError("plan must be a Plan")
        if not isinstance(self.committed_results, Mapping):
            raise TypeError("committed_results must be a mapping")
        committed = dict(self.committed_results)
        if any(
            not isinstance(step_id, str) or not isinstance(result, StepResult)
            for step_id, result in committed.items()
        ):
            raise TypeError("committed_results must map step IDs to StepResult")
        if type(self.replan_count) is not int:
            raise TypeError("replan_count must be an integer")
        if self.replan_count < 0:
            raise ValueError("replan_count cannot be negative")
        self.committed_results = committed


@dataclass(slots=True)
class StepExecutionState:
    current_step_id: str | None = None
    pending_step_result: StepResult | None = None
    validation_feedback: str | None = None
    step_attempt_count: int = 0

    def __post_init__(self) -> None:
        if self.current_step_id is not None and (
            not isinstance(self.current_step_id, str)
            or not self.current_step_id.strip()
        ):
            raise ValueError("current_step_id cannot be empty")
        if self.pending_step_result is not None and not isinstance(
            self.pending_step_result,
            StepResult,
        ):
            raise TypeError("pending_step_result must be a StepResult")
        if self.validation_feedback is not None and not isinstance(
            self.validation_feedback,
            str,
        ):
            raise TypeError("validation_feedback must be a string")
        if type(self.step_attempt_count) is not int:
            raise TypeError("step_attempt_count must be an integer")
        if self.step_attempt_count < 0:
            raise ValueError("step_attempt_count cannot be negative")


@dataclass(slots=True)
class ToolRecoveryState:
    active_calls: dict[str, dict[str, str]] = field(default_factory=dict)
    seen_call_ids: list[str] = field(default_factory=list)
    pending_failure: dict[str, Any] | None = None
    repair_count_by_step: dict[str, int] = field(default_factory=dict)
    total_repairs: int = 0
    terminal_failure: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.active_calls, Mapping):
            raise TypeError("active_calls must be a mapping")
        active: dict[str, dict[str, str]] = {}
        for call_id, call in self.active_calls.items():
            if not isinstance(call, Mapping):
                raise TypeError("active_calls values must be mappings")
            normalized_id = str(call_id).strip()
            tool_name = str(call.get("tool_name", "")).strip()
            fingerprint = str(call.get("arguments_fingerprint", "")).strip()
            if not normalized_id or not tool_name:
                raise ValueError("active Tool call identity cannot be empty")
            if not is_tool_argument_fingerprint(fingerprint):
                raise ValueError("active Tool argument fingerprint must use sha256")
            active[normalized_id] = {
                "tool_name": tool_name,
                "arguments_fingerprint": fingerprint,
            }
        if isinstance(self.seen_call_ids, (str, bytes)) or not isinstance(
            self.seen_call_ids,
            Sequence,
        ):
            raise TypeError("seen_call_ids must be an array")
        seen = [str(item).strip() for item in self.seen_call_ids]
        if not all(seen):
            raise ValueError("seen_call_ids cannot contain empty values")
        if len(set(seen)) != len(seen):
            raise ValueError("seen_call_ids cannot contain duplicates")
        if not set(active).issubset(seen):
            raise ValueError("active call IDs must be present in seen_call_ids")
        if not isinstance(self.repair_count_by_step, Mapping):
            raise TypeError("repair_count_by_step must be a mapping")
        counts = {
            str(step_id): int(count)
            for step_id, count in self.repair_count_by_step.items()
        }
        if any(not step_id or count < 0 for step_id, count in counts.items()):
            raise ValueError(
                "repair counts require non-empty IDs and non-negative values"
            )
        if type(self.total_repairs) is not int:
            raise TypeError("total_repairs must be an integer")
        if self.total_repairs < 0:
            raise ValueError("total_repairs cannot be negative")
        self.active_calls = active
        self.seen_call_ids = seen
        self.repair_count_by_step = counts
        self.pending_failure = _safe_failure(self.pending_failure)
        self.terminal_failure = _safe_failure(self.terminal_failure)


@dataclass(slots=True)
class PlanFinalizationState:
    response: str | None = None
    invocation_count: int = 0
    persisted: bool = False
    emitted: bool = False

    def __post_init__(self) -> None:
        if self.response is not None and not isinstance(self.response, str):
            raise TypeError("finalization response must be a string")
        if type(self.invocation_count) is not int:
            raise TypeError("invocation_count must be an integer")
        if self.invocation_count < 0:
            raise ValueError("invocation_count cannot be negative")
        if type(self.persisted) is not bool or type(self.emitted) is not bool:
            raise TypeError("finalization markers must be bools")
        if (self.persisted or self.emitted) and self.response is None:
            raise ValueError("finalization markers require a response")


@dataclass(slots=True)
class PlanEngineState:
    """Nested Plan state with independent progress, step, recovery and output."""

    phase: PlanExecutionPhase
    plan_progress: PlanProgressState
    step_execution: StepExecutionState = field(default_factory=StepExecutionState)
    tool_recovery: ToolRecoveryState = field(default_factory=ToolRecoveryState)
    finalization: PlanFinalizationState = field(default_factory=PlanFinalizationState)
    terminal_finish_reason: FinishReason | None = None
    terminal_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, PlanExecutionPhase):
            self.phase = PlanExecutionPhase(str(self.phase))
        if not isinstance(self.plan_progress, PlanProgressState):
            raise TypeError("plan_progress must be a PlanProgressState")
        if not isinstance(self.step_execution, StepExecutionState):
            raise TypeError("step_execution must be a StepExecutionState")
        if not isinstance(self.tool_recovery, ToolRecoveryState):
            raise TypeError("tool_recovery must be a ToolRecoveryState")
        if not isinstance(self.finalization, PlanFinalizationState):
            raise TypeError("finalization must be a PlanFinalizationState")
        if self.terminal_finish_reason is not None and not isinstance(
            self.terminal_finish_reason,
            FinishReason,
        ):
            self.terminal_finish_reason = FinishReason(str(self.terminal_finish_reason))
        if self.terminal_error is not None and not isinstance(self.terminal_error, str):
            raise TypeError("terminal_error must be a string")
        if (
            self.phase is PlanExecutionPhase.FAILED
            and self.terminal_finish_reason is None
        ):
            self.terminal_finish_reason = FinishReason.ERROR
        step_id = self.step_execution.current_step_id
        known_ids = {step.step_id for step in self.plan_progress.plan.steps}
        if step_id is not None and step_id not in known_ids:
            raise ValueError("current_step_id does not exist in the plan")
        pending = self.step_execution.pending_step_result
        if pending is not None and step_id is not None and pending.step_id != step_id:
            raise ValueError("pending StepResult does not match current_step_id")
        if self.phase is PlanExecutionPhase.DONE and not self.finalization.emitted:
            raise ValueError("DONE Plan state requires an emitted final response")

    @property
    def current_step(self) -> PlanStep | None:
        step_id = self.step_execution.current_step_id
        if step_id is None:
            return self.plan_progress.plan.current
        return next(
            (step for step in self.plan_progress.plan.steps if step.step_id == step_id),
            None,
        )

    @classmethod
    def from_legacy(cls, state: ExecutionState) -> PlanEngineState:
        if not isinstance(state, ExecutionState):
            raise TypeError("state must be an ExecutionState")
        step = state.current_step
        return cls(
            phase=_from_legacy_phase(state),
            plan_progress=PlanProgressState(
                # Both representations are internal, already-validated Python
                # objects. A deep copy preserves mutation isolation without the
                # much more expensive JSON dump/validation round-trip.
                plan=copy.deepcopy(state.plan),
                committed_results={
                    key: result.model_copy(deep=True)
                    for key, result in state.committed_results.items()
                },
                replan_count=state.replan_count,
            ),
            step_execution=StepExecutionState(
                current_step_id=state.current_step_id,
                pending_step_result=(
                    None
                    if state.pending_step_result is None
                    else state.pending_step_result.model_copy(deep=True)
                ),
                validation_feedback=state.validation_error,
                step_attempt_count=0 if step is None else step.attempt_count,
            ),
            tool_recovery=ToolRecoveryState(
                active_calls={
                    call_id: dict(call)
                    for call_id, call in state.active_tool_calls.items()
                },
                seen_call_ids=list(state.seen_tool_call_ids),
                pending_failure=state.pending_tool_failure,
                repair_count_by_step=dict(state.tool_repair_counts),
                total_repairs=state.total_tool_repairs,
                terminal_failure=state.failure,
            ),
            finalization=PlanFinalizationState(
                response=state.final_response,
                invocation_count=state.finalization_count,
                persisted=state.final_persisted,
                emitted=state.final_emitted,
            ),
            terminal_finish_reason=(
                FinishReason.ERROR if state.phase is RunPhase.FAILED else None
            ),
            terminal_error=(
                None
                if state.failure is None
                else next(
                    (
                        str(state.failure[key])
                        for key in ("terminal_error", "terminal_reason", "reason")
                        if state.failure.get(key)
                    ),
                    None,
                )
            ),
        )

    def to_legacy(self) -> ExecutionState:
        plan = copy.deepcopy(self.plan_progress.plan)
        current_step_id = self.step_execution.current_step_id
        if current_step_id is not None:
            current = next(
                (step for step in plan.steps if step.step_id == current_step_id),
                None,
            )
            if current is not None:
                current.attempt_count = max(
                    current.attempt_count,
                    self.step_execution.step_attempt_count,
                )
        phase, awaiting_step_result = _to_legacy_phase(self.phase)
        terminal_failure = (
            None
            if self.tool_recovery.terminal_failure is None
            else dict(self.tool_recovery.terminal_failure)
        )
        if self.terminal_finish_reason is not None:
            terminal_failure = dict(terminal_failure or {})
            terminal_failure["finish_reason"] = self.terminal_finish_reason.value
            if self.terminal_error is not None:
                terminal_failure["terminal_error"] = self.terminal_error
        committed_results = {
            key: result.model_copy(deep=True)
            for key, result in self.plan_progress.committed_results.items()
        }
        # ExecutionState normally recomputes the content hash of every committed
        # result in __post_init__. The Engine state was already checked when it
        # entered the runtime, so repeating that O(total artifact bytes) work at
        # every legacy-policy adapter call is unnecessary. Construct the facade
        # with an empty committed set, validate the cheap structural invariants,
        # then attach isolated copies.
        _validate_committed_result_structure(plan, committed_results)
        legacy = ExecutionState(
            phase=phase,
            plan=plan,
            current_step_id=current_step_id,
            committed_results={},
            pending_step_result=(
                None
                if self.step_execution.pending_step_result is None
                else self.step_execution.pending_step_result.model_copy(deep=True)
            ),
            validation_error=self.step_execution.validation_feedback,
            awaiting_step_result=awaiting_step_result,
            replan_count=self.plan_progress.replan_count,
            finalization_count=self.finalization.invocation_count,
            final_response=self.finalization.response,
            final_persisted=self.finalization.persisted,
            final_emitted=self.finalization.emitted,
            tool_repair_counts=dict(self.tool_recovery.repair_count_by_step),
            pending_tool_failure=(
                None
                if self.tool_recovery.pending_failure is None
                else dict(self.tool_recovery.pending_failure)
            ),
            total_tool_repairs=self.tool_recovery.total_repairs,
            failure=terminal_failure,
            active_tool_calls={
                call_id: dict(call)
                for call_id, call in self.tool_recovery.active_calls.items()
            },
            seen_tool_call_ids=list(self.tool_recovery.seen_call_ids),
        )
        legacy.committed_results = committed_results
        return legacy


def _from_legacy_phase(state: ExecutionState) -> PlanExecutionPhase:
    if state.phase is RunPhase.ACT:
        if state.awaiting_step_result:
            return PlanExecutionPhase.STEP_RESULT
        if state.pending_tool_failure is not None:
            return PlanExecutionPhase.TOOL_RECOVERY
        return PlanExecutionPhase.ACT_TOOL
    return PlanExecutionPhase(state.phase.value)


def _to_legacy_phase(
    phase: PlanExecutionPhase,
) -> tuple[RunPhase, bool]:
    if phase in {
        PlanExecutionPhase.ACT_TOOL,
        PlanExecutionPhase.TOOL_RECOVERY,
    }:
        return RunPhase.ACT, False
    if phase is PlanExecutionPhase.STEP_RESULT:
        return RunPhase.ACT, True
    return RunPhase(phase.value), False


class PlanStateCodec(EngineStateCodec[PlanEngineState]):
    engine_id = "plan"
    state_version = 1

    def encode(self, state: PlanEngineState) -> Mapping[str, Any]:
        if not isinstance(state, PlanEngineState):
            raise TypeError("state must be a PlanEngineState")
        return {
            "phase": state.phase.value,
            "plan_progress": {
                "plan": state.plan_progress.plan.to_dict(),
                "committed_results": {
                    step_id: result.model_dump(mode="json")
                    for step_id, result in (
                        state.plan_progress.committed_results.items()
                    )
                },
                "replan_count": state.plan_progress.replan_count,
            },
            "step_execution": {
                "current_step_id": state.step_execution.current_step_id,
                "pending_step_result": (
                    None
                    if state.step_execution.pending_step_result is None
                    else state.step_execution.pending_step_result.model_dump(
                        mode="json"
                    )
                ),
                "validation_feedback": state.step_execution.validation_feedback,
                "step_attempt_count": state.step_execution.step_attempt_count,
            },
            "tool_recovery": {
                "active_calls": {
                    call_id: dict(call)
                    for call_id, call in state.tool_recovery.active_calls.items()
                },
                "seen_call_ids": list(state.tool_recovery.seen_call_ids),
                "pending_failure": (
                    None
                    if state.tool_recovery.pending_failure is None
                    else dict(state.tool_recovery.pending_failure)
                ),
                "repair_count_by_step": dict(state.tool_recovery.repair_count_by_step),
                "total_repairs": state.tool_recovery.total_repairs,
                "terminal_failure": (
                    None
                    if state.tool_recovery.terminal_failure is None
                    else dict(state.tool_recovery.terminal_failure)
                ),
            },
            "finalization": {
                "started": (
                    state.finalization.invocation_count > 0
                    or state.finalization.response is not None
                ),
                "response_generated": (state.finalization.response is not None),
                "response": state.finalization.response,
                "invocation_count": state.finalization.invocation_count,
                "persisted": state.finalization.persisted,
                "emitted": state.finalization.emitted,
            },
            "terminal": {
                "finish_reason": (
                    None
                    if state.terminal_finish_reason is None
                    else state.terminal_finish_reason.value
                ),
                "error": state.terminal_error,
            },
        }

    def decode(self, payload: Mapping[str, Any]) -> PlanEngineState:
        if not isinstance(payload, Mapping):
            raise TypeError("Plan state payload must be a mapping")
        progress = _mapping(payload.get("plan_progress"), "plan_progress")
        step = _mapping(payload.get("step_execution"), "step_execution")
        recovery = _mapping(payload.get("tool_recovery"), "tool_recovery")
        finalization = _mapping(payload.get("finalization"), "finalization")
        terminal = _mapping(payload.get("terminal", {}), "terminal")
        raw_results = _mapping(
            progress.get("committed_results", {}),
            "committed_results",
        )
        raw_pending = step.get("pending_step_result")
        raw_active = _mapping(recovery.get("active_calls", {}), "active_calls")
        raw_counts = _mapping(
            recovery.get("repair_count_by_step", {}),
            "repair_count_by_step",
        )
        raw_seen = recovery.get("seen_call_ids", ())
        if isinstance(raw_seen, (str, bytes)) or not isinstance(raw_seen, Sequence):
            raise ValueError("seen_call_ids must be an array")
        state = PlanEngineState(
            phase=PlanExecutionPhase(str(payload.get("phase", ""))),
            plan_progress=PlanProgressState(
                plan=Plan.from_dict(_mapping(progress.get("plan"), "plan")),
                committed_results={
                    str(step_id): StepResult.model_validate(result)
                    for step_id, result in raw_results.items()
                },
                replan_count=int(progress.get("replan_count", 0)),
            ),
            step_execution=StepExecutionState(
                current_step_id=(
                    None
                    if step.get("current_step_id") is None
                    else str(step["current_step_id"])
                ),
                pending_step_result=(
                    None
                    if raw_pending is None
                    else StepResult.model_validate(raw_pending)
                ),
                validation_feedback=(
                    None
                    if step.get("validation_feedback") is None
                    else str(step["validation_feedback"])
                ),
                step_attempt_count=int(step.get("step_attempt_count", 0)),
            ),
            tool_recovery=ToolRecoveryState(
                active_calls={
                    str(call_id): dict(_mapping(call, "active call"))
                    for call_id, call in raw_active.items()
                },
                seen_call_ids=[str(call_id) for call_id in raw_seen],
                pending_failure=(
                    None
                    if recovery.get("pending_failure") is None
                    else dict(
                        _mapping(
                            recovery["pending_failure"],
                            "pending_failure",
                        )
                    )
                ),
                repair_count_by_step={
                    str(step_id): int(count) for step_id, count in raw_counts.items()
                },
                total_repairs=int(recovery.get("total_repairs", 0)),
                terminal_failure=(
                    None
                    if recovery.get("terminal_failure") is None
                    else dict(
                        _mapping(
                            recovery["terminal_failure"],
                            "terminal_failure",
                        )
                    )
                ),
            ),
            finalization=PlanFinalizationState(
                response=(
                    None
                    if finalization.get("response") is None
                    else str(finalization["response"])
                ),
                invocation_count=int(finalization.get("invocation_count", 0)),
                persisted=bool(finalization.get("persisted", False)),
                emitted=bool(finalization.get("emitted", False)),
            ),
            terminal_finish_reason=(
                None
                if terminal.get("finish_reason") is None
                else FinishReason(str(terminal["finish_reason"]))
            ),
            terminal_error=(
                None if terminal.get("error") is None else str(terminal["error"])
            ),
        )
        _validate_committed_result_integrity(state.plan_progress)
        return state

    def migrate(
        self,
        from_version: int,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if type(from_version) is not int:
            raise TypeError("from_version must be an integer")
        if not isinstance(payload, Mapping):
            raise TypeError("Plan state payload must be a mapping")
        if from_version == self.state_version:
            return self.encode(self.decode(payload))
        # Version 0 is the compatibility marker used by the 0.3 flat state.
        # Outer checkpoint schema v3 migration may pass 3 explicitly.
        if from_version in {0, 3}:
            legacy = ExecutionState.from_dict(payload)
            return self.encode(PlanEngineState.from_legacy(legacy))
        raise ValueError(f"unsupported Plan state version: {from_version}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _validate_committed_result_structure(
    plan: Plan,
    committed_results: Mapping[str, StepResult],
) -> None:
    steps_by_id = {step.step_id: step for step in plan.steps}
    for step_id, result in committed_results.items():
        if step_id != result.step_id:
            raise ValueError(
                f"committed result key {step_id!r} does not match result step_id"
            )
        step = steps_by_id.get(step_id)
        if step is None:
            raise ValueError(f"committed result refers to unknown step {step_id!r}")
        if step.status is not PlanStepStatus.COMPLETED:
            raise ValueError(f"committed result step {step_id!r} is not completed")
        if not isinstance(step.result_ref, str) or not step.result_ref.startswith(
            "sha256:"
        ):
            raise ValueError(f"committed result step {step_id!r} has no result hash")


def _validate_committed_result_integrity(progress: PlanProgressState) -> None:
    _validate_committed_result_structure(progress.plan, progress.committed_results)
    steps_by_id = {step.step_id: step for step in progress.plan.steps}
    for step_id, result in progress.committed_results.items():
        if steps_by_id[step_id].result_ref != step_result_ref(result):
            raise ValueError(f"committed result hash does not match step {step_id!r}")


__all__ = [
    "PlanEngineState",
    "PlanExecutionPhase",
    "PlanFinalizationState",
    "PlanProgressState",
    "PlanStateCodec",
    "StepExecutionState",
    "ToolRecoveryState",
]

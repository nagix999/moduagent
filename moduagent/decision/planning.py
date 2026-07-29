from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from moduagent.decision.base import DecisionKind, ExecutionDecision
from moduagent.messages import Message, ToolCall
from moduagent.models import ModelClient, ModelRequest, ModelResponse
from moduagent.tools.base import ToolRecoveryAction, _tool_arguments_fingerprint

if TYPE_CHECKING:
    from moduagent.runtime.context import RunContext
    from moduagent.tools import ToolResult


class RunPhase(str, Enum):
    PLAN = "plan"
    STEP_PREPARE = "step_prepare"
    ACT = "act"
    STEP_VALIDATE = "step_validate"
    VERIFY = "verify"
    FINALIZE = "finalize"
    DONE = "done"
    FAILED = "failed"


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


# A descriptive alias matching the terminology used by the 0.3 state machine.
StepStatus = PlanStepStatus


def _stable_step_id(objective: str) -> str:
    digest = hashlib.sha256(objective.encode("utf-8")).hexdigest()[:12]
    return f"step-{digest}"


def _is_tool_arguments_fingerprint(value: Any) -> bool:
    text = str(value)
    digest = text.removeprefix("sha256:")
    return (
        text.startswith("sha256:")
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _string_list(value: Sequence[str], field_name: str) -> list[str]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array")
    result = [str(item).strip() for item in value]
    if not all(result):
        raise ValueError(f"{field_name} cannot contain empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return result


@dataclass(slots=True)
class PlanStep:
    """A stable, independently verifiable unit of work.

    ``description`` remains the first positional field for 0.2 compatibility.
    New code should prefer ``objective`` and ``completion_criteria``. When they
    are omitted, compatibility defaults are derived once and then serialized.
    """

    description: str = ""
    status: PlanStepStatus = PlanStepStatus.PENDING
    expected_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    step_id: str = ""
    objective: str | None = None
    completion_criteria: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    attempt_count: int = 0
    result_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PlanStepStatus):
            self.status = PlanStepStatus(self.status)
        description = str(self.description).strip()
        objective = str(self.objective or description).strip()
        if not objective:
            raise ValueError("plan step objective cannot be empty")
        if not description:
            description = objective
        step_id = str(self.step_id or _stable_step_id(objective)).strip()
        if not step_id:
            raise ValueError("plan step id cannot be empty")
        if self.attempt_count < 0:
            raise ValueError("plan step attempt_count cannot be negative")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("plan step metadata must be a mapping")

        expected_output = (
            None if self.expected_output is None else str(self.expected_output).strip()
        )
        criteria = self.completion_criteria or [
            expected_output or f"Complete the objective: {objective}"
        ]

        self.description = description
        self.objective = objective
        self.step_id = step_id
        self.expected_output = expected_output
        self.completion_criteria = _string_list(criteria, "completion_criteria")
        self.dependencies = _string_list(self.dependencies, "dependencies")
        self.allowed_tools = _string_list(self.allowed_tools, "allowed_tools")
        self.metadata = dict(self.metadata)
        if self.result_ref is not None:
            self.result_ref = str(self.result_ref)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "objective": self.objective,
            # Kept in checkpoints so 0.2 readers and diagnostic clients retain
            # a human-readable label.
            "description": self.description,
            "completion_criteria": list(self.completion_criteria),
            "expected_output": self.expected_output,
            "dependencies": list(self.dependencies),
            "allowed_tools": list(self.allowed_tools),
            "status": self.status.value,
            "attempt_count": self.attempt_count,
            "result_ref": self.result_ref,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PlanStep:
        if not isinstance(value, Mapping):
            raise ValueError("plan step must be an object")
        raw_criteria = value.get("completion_criteria", ())
        raw_dependencies = value.get("dependencies", ())
        raw_allowed_tools = value.get("allowed_tools", ())
        for raw, field_name in (
            (raw_criteria, "completion_criteria"),
            (raw_dependencies, "dependencies"),
            (raw_allowed_tools, "allowed_tools"),
        ):
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ValueError(f"plan step {field_name} must be an array")
        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("plan step metadata must be an object")
        objective = value.get("objective")
        description = value.get("description", objective or "")
        return cls(
            description=str(description),
            status=PlanStepStatus(value.get("status", PlanStepStatus.PENDING.value)),
            expected_output=(
                None
                if value.get("expected_output") is None
                else str(value["expected_output"])
            ),
            metadata=dict(raw_metadata),
            step_id=str(value.get("step_id", "")),
            objective=None if objective is None else str(objective),
            completion_criteria=[str(item) for item in raw_criteria],
            dependencies=[str(item) for item in raw_dependencies],
            allowed_tools=[str(item) for item in raw_allowed_tools],
            attempt_count=int(value.get("attempt_count", 0)),
            result_ref=(
                None if value.get("result_ref") is None else str(value["result_ref"])
            ),
        )


class StepResult(BaseModel):
    """Internal ACT output.

    Extra fields are deliberately forbidden so an executor cannot smuggle a
    public ``final_answer`` (or another phase's output) across the ACT boundary.
    """

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    status: Literal["completed", "blocked", "failed"]
    facts: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    uncertainties: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    completion_evidence: list[str] = Field(default_factory=list)


def step_result_ref(result: StepResult) -> str:
    """Return a content-addressed reference for a canonical StepResult."""

    payload = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class ValidationKind(str, Enum):
    COMMIT = "commit"
    RETRY = "retry"
    REPLAN = "replan"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class StepValidation:
    kind: ValidationKind
    reason: str = ""
    unmet_criteria: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "unmet_criteria": list(self.unmet_criteria),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StepValidation:
        return cls(
            kind=ValidationKind(value["kind"]),
            reason=str(value.get("reason", "")),
            unmet_criteria=tuple(str(item) for item in value.get("unmet_criteria", ())),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ToolFailureRecoveryConfig:
    """Opt-in policy for repairing a failed Tool call inside the current step.

    Omitting this configuration preserves the original strict 0.3 behavior:
    Tool failures either revise the plan or fail the run according to
    ``revise_on_tool_failure``.
    """

    fallback: Literal["replan", "fail"] = "replan"
    require_repair_safe: bool = True
    feedback_mode: Literal["type_only", "safe_message"] = "type_only"

    def __post_init__(self) -> None:
        if self.fallback not in {"replan", "fail"}:
            raise ValueError("fallback must be 'replan' or 'fail'")
        if not isinstance(self.require_repair_safe, bool):
            raise TypeError("require_repair_safe must be a bool")
        if self.feedback_mode not in {"type_only", "safe_message"}:
            raise ValueError("feedback_mode must be 'type_only' or 'safe_message'")


class StepValidator:
    """Deterministic baseline validator suitable for custom specialization.

    It cannot judge semantic truth. It does enforce identity, terminal status,
    and one non-empty evidence entry per declared completion criterion. A
    domain validator can subclass this class and return a stricter decision.
    """

    def validate(self, step: PlanStep, result: StepResult) -> StepValidation:
        if result.step_id != step.step_id:
            return StepValidation(
                ValidationKind.FAIL,
                f"step result id {result.step_id!r} does not match {step.step_id!r}",
            )
        if result.status == "failed":
            return StepValidation(
                ValidationKind.FAIL,
                "executor reported that the step cannot be completed",
            )
        if result.status == "blocked":
            if result.missing_inputs:
                return StepValidation(
                    ValidationKind.REPLAN,
                    "step is blocked by missing inputs or dependencies",
                    metadata={"missing_inputs": tuple(result.missing_inputs)},
                )
            return StepValidation(
                ValidationKind.RETRY,
                "step is blocked without actionable missing-input details",
            )

        evidence = [item.strip() for item in result.completion_evidence if item.strip()]
        missing_count = max(0, len(step.completion_criteria) - len(evidence))
        if missing_count:
            return StepValidation(
                ValidationKind.RETRY,
                "completion evidence does not cover every completion criterion",
                unmet_criteria=tuple(step.completion_criteria[-missing_count:]),
            )
        return StepValidation(ValidationKind.COMMIT)


@dataclass(slots=True)
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    current_index: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        self.steps = list(self.steps)
        if not all(isinstance(step, PlanStep) for step in self.steps):
            raise TypeError("plan steps must contain PlanStep instances")
        if self.version < 1:
            raise ValueError("plan version must be at least 1")
        if self.current_index < 0 or self.current_index > len(self.steps):
            raise ValueError("plan current_index is out of range")
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        known_ids = set(ids)
        for step in self.steps:
            unknown = set(step.dependencies) - known_ids
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(
                    f"plan step {step.step_id!r} has unknown dependencies: {names}"
                )
            if step.step_id in step.dependencies:
                raise ValueError(f"plan step {step.step_id!r} cannot depend on itself")
        self._validate_dependency_graph()
        positions = {step.step_id: index for index, step in enumerate(self.steps)}
        for step in self.steps:
            out_of_order = [
                dependency
                for dependency in step.dependencies
                if positions[dependency] >= positions[step.step_id]
            ]
            if out_of_order:
                names = ", ".join(sorted(out_of_order))
                raise ValueError(
                    f"plan step {step.step_id!r} dependencies must precede it: {names}"
                )
        self._move_to_next_open_step()

    @property
    def current(self) -> PlanStep | None:
        if 0 <= self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    @property
    def complete(self) -> bool:
        return bool(self.steps) and all(
            step.status in {PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED}
            for step in self.steps
        )

    def start_current(self) -> PlanStep | None:
        self._move_to_next_open_step()
        step = self.current
        if step and step.status is PlanStepStatus.PENDING:
            completed_ids = {
                item.step_id
                for item in self.steps
                if item.status is PlanStepStatus.COMPLETED
            }
            if set(step.dependencies) <= completed_ids:
                step.status = PlanStepStatus.IN_PROGRESS
        return step

    def commit(
        self,
        result_ref: str | StepResult,
        *,
        step_id: str | None = None,
    ) -> PlanStep:
        step = self.current
        if step is None:
            raise RuntimeError("plan has no current step to commit")
        if isinstance(result_ref, StepResult):
            result = result_ref
            if step_id is None:
                step_id = result.step_id
            result_reference = step_result_ref(result)
        else:
            result_reference = str(result_ref)
        if step_id is not None and step.step_id != step_id:
            raise ValueError(
                f"cannot commit step {step_id!r}; current step is {step.step_id!r}"
            )
        if not result_reference:
            raise ValueError("result_ref cannot be empty")
        if step.status not in {
            PlanStepStatus.PENDING,
            PlanStepStatus.IN_PROGRESS,
        }:
            raise RuntimeError(
                f"step {step.step_id!r} cannot be committed from {step.status.value}"
            )
        step.status = PlanStepStatus.COMPLETED
        step.result_ref = result_reference
        self.current_index += 1
        self._move_to_next_open_step()
        self.start_current()
        return step

    def advance(self, output: str | None = None) -> None:
        """Compatibility transition used only by the legacy policy."""

        step = self.current
        if step is None:
            return
        if output is not None:
            step.metadata["legacy_result"] = output
        self.commit(f"legacy:{step.step_id}", step_id=step.step_id)

    def _move_to_next_open_step(self) -> None:
        while self.current_index < len(self.steps) and self.steps[
            self.current_index
        ].status in {PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED}:
            self.current_index += 1

    def _validate_dependency_graph(self) -> None:
        dependencies = {step.step_id: tuple(step.dependencies) for step in self.steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("plan dependencies contain a cycle")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in dependencies[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in dependencies:
            visit(step_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "current_index": self.current_index,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Plan:
        if not isinstance(value, Mapping):
            raise ValueError("plan must be an object")
        raw_steps = value.get("steps", ())
        if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
            raise ValueError("plan steps must be an array")
        return cls(
            steps=[PlanStep.from_dict(item) for item in raw_steps],
            current_index=int(value.get("current_index", 0)),
            version=int(value.get("version", 1)),
        )


@dataclass(slots=True)
class ExecutionState:
    """Serializable strict Plan-and-Execute state.

    ``awaiting_step_result`` is the provider-neutral boundary used for models
    (including vLLM) that reject tools and an output schema in one request.
    The tool turn sets it to ``True``; the next request must use no tools and
    the :class:`StepResult` schema.
    """

    phase: RunPhase = RunPhase.PLAN
    plan: Plan = field(default_factory=Plan)
    current_step_id: str | None = None
    committed_results: dict[str, StepResult] = field(default_factory=dict)
    pending_step_result: StepResult | None = None
    validation_error: str | None = None
    awaiting_step_result: bool = False
    replan_count: int = 0
    finalization_count: int = 0
    final_response: str | None = None
    final_persisted: bool = False
    final_emitted: bool = False
    # Appended fields keep old positional construction and v3 checkpoint reads
    # compatible while making Tool repair budgets durable across resume.
    tool_repair_counts: dict[str, int] = field(default_factory=dict)
    pending_tool_failure: dict[str, Any] | None = None
    total_tool_repairs: int = 0
    failure: dict[str, Any] | None = None
    # Only names and one-way argument hashes are checkpointed. Raw Tool
    # arguments remain in the provider transcript and are never duplicated here.
    active_tool_calls: dict[str, dict[str, str]] = field(default_factory=dict)
    seen_tool_call_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.replan_count < 0:
            raise ValueError("replan_count cannot be negative")
        if self.finalization_count < 0:
            raise ValueError("finalization_count cannot be negative")
        if self.total_tool_repairs < 0:
            raise ValueError("total_tool_repairs cannot be negative")
        self.committed_results = dict(self.committed_results)
        if not isinstance(self.tool_repair_counts, Mapping):
            raise TypeError("tool_repair_counts must be a mapping")
        repair_counts = {
            str(step_id): int(count)
            for step_id, count in self.tool_repair_counts.items()
        }
        if any(count < 0 for count in repair_counts.values()):
            raise ValueError("tool repair counts cannot be negative")
        self.tool_repair_counts = repair_counts
        if self.pending_tool_failure is not None:
            if not isinstance(self.pending_tool_failure, Mapping):
                raise TypeError("pending_tool_failure must be a mapping")
            self.pending_tool_failure = dict(self.pending_tool_failure)
            for field_name in (
                "arguments_fingerprint",
                "invocation_fingerprint",
            ):
                fingerprint = self.pending_tool_failure.get(field_name)
                if fingerprint is not None and not _is_tool_arguments_fingerprint(
                    fingerprint
                ):
                    raise ValueError(
                        f"pending Tool failure {field_name} must use sha256"
                    )
        if self.failure is not None:
            if not isinstance(self.failure, Mapping):
                raise TypeError("failure must be a mapping")
            self.failure = dict(self.failure)
        if not isinstance(self.active_tool_calls, Mapping):
            raise TypeError("active_tool_calls must be a mapping")
        active_tool_calls: dict[str, dict[str, str]] = {}
        for call_id, raw_call in self.active_tool_calls.items():
            if not isinstance(raw_call, Mapping):
                raise TypeError("active_tool_calls values must be mappings")
            normalized_call_id = str(call_id).strip()
            tool_name = str(raw_call.get("tool_name", "")).strip()
            fingerprint = str(raw_call.get("arguments_fingerprint", "")).strip()
            if not normalized_call_id or not tool_name:
                raise ValueError("active Tool call identity cannot be empty")
            if not _is_tool_arguments_fingerprint(fingerprint):
                raise ValueError(
                    "active Tool call arguments_fingerprint must use sha256"
                )
            active_tool_calls[normalized_call_id] = {
                "tool_name": tool_name,
                "arguments_fingerprint": fingerprint,
            }
        self.active_tool_calls = active_tool_calls
        if isinstance(self.seen_tool_call_ids, (str, bytes)) or not isinstance(
            self.seen_tool_call_ids,
            Sequence,
        ):
            raise TypeError("seen_tool_call_ids must be an array")
        seen_tool_call_ids = [
            str(call_id).strip() for call_id in self.seen_tool_call_ids
        ]
        if not all(seen_tool_call_ids):
            raise ValueError("seen Tool call IDs cannot be empty")
        if len(set(seen_tool_call_ids)) != len(seen_tool_call_ids):
            raise ValueError("seen Tool call IDs cannot contain duplicates")
        if not set(active_tool_calls).issubset(seen_tool_call_ids):
            raise ValueError("active Tool call IDs must be present in seen IDs")
        self.seen_tool_call_ids = seen_tool_call_ids
        if self.current_step_id is None and self.plan.current is not None:
            self.current_step_id = self.plan.current.step_id
        steps_by_id = {step.step_id: step for step in self.plan.steps}
        for step_id, result in self.committed_results.items():
            if step_id != result.step_id:
                raise ValueError(
                    f"committed result key {step_id!r} does not match result step_id"
                )
            step = steps_by_id.get(step_id)
            if step is None:
                raise ValueError(f"committed result refers to unknown step {step_id!r}")
            if step.status is not PlanStepStatus.COMPLETED:
                raise ValueError(f"committed result step {step_id!r} is not completed")
            if step.result_ref != step_result_ref(result):
                raise ValueError(
                    f"committed result hash does not match step {step_id!r}"
                )
        if self.pending_step_result is not None:
            if (
                self.current_step_id is not None
                and self.pending_step_result.step_id != self.current_step_id
            ):
                raise ValueError("pending StepResult does not match current_step_id")
        if self.current_step_id is not None and self.current_step_id not in steps_by_id:
            raise ValueError("current_step_id does not exist in the plan")
        if self.final_persisted and self.final_response is None:
            raise ValueError("final_persisted requires final_response")
        if self.final_emitted and self.final_response is None:
            raise ValueError("final_emitted requires final_response")
        if self.phase is RunPhase.DONE and not self.final_emitted:
            raise ValueError("DONE execution state requires final_emitted")

    @property
    def current_step(self) -> PlanStep | None:
        if self.current_step_id is None:
            return self.plan.current
        return next(
            (step for step in self.plan.steps if step.step_id == self.current_step_id),
            None,
        )

    @property
    def complete(self) -> bool:
        return self.plan.complete

    def prepare_current_step(self, *, has_tools: bool) -> PlanStep | None:
        step = self.plan.start_current()
        self.current_step_id = None if step is None else step.step_id
        self.pending_step_result = None
        self.validation_error = None
        self.pending_tool_failure = None
        self.active_tool_calls = {}
        # With no tools, ACT can be a schema-only request immediately. With
        # tools, the first turn must remain tool-only for vLLM compatibility.
        self.awaiting_step_result = not has_tools
        self.phase = RunPhase.ACT if step is not None else RunPhase.VERIFY
        return step

    def mark_tool_round_complete(self) -> None:
        if self.phase is not RunPhase.ACT:
            raise RuntimeError("tool results can only be recorded during ACT")
        self.awaiting_step_result = True

    def set_pending_result(self, result: StepResult) -> None:
        step = self.current_step
        if step is None:
            raise RuntimeError("there is no current step for the pending result")
        if result.step_id != step.step_id:
            raise ValueError(
                f"step result id {result.step_id!r} does not match {step.step_id!r}"
            )
        self.pending_step_result = result
        self.validation_error = None
        self.phase = RunPhase.STEP_VALIDATE

    def fail_current_step(self) -> PlanStep | None:
        """Make the active step and execution state terminally failed.

        A strict execution failure must not leave an active step looking as if
        it is still running. Completed and skipped steps are immutable here;
        they may be present when a corrupted or incomplete checkpoint is
        rejected.
        """

        step = self.current_step
        if step is not None and step.status not in {
            PlanStepStatus.COMPLETED,
            PlanStepStatus.SKIPPED,
        }:
            step.status = PlanStepStatus.FAILED
        self.active_tool_calls = {}
        self.phase = RunPhase.FAILED
        return step

    def commit_pending(self) -> PlanStep:
        result = self.pending_step_result
        if result is None:
            raise RuntimeError("there is no pending step result to commit")
        step = self.plan.commit(result, step_id=result.step_id)
        self.committed_results[result.step_id] = result
        self.pending_step_result = None
        self.validation_error = None
        self.pending_tool_failure = None
        self.failure = None
        self.active_tool_calls = {}
        self.awaiting_step_result = False
        self.current_step_id = (
            None if self.plan.current is None else self.plan.current.step_id
        )
        self.phase = RunPhase.VERIFY if self.plan.complete else RunPhase.STEP_PREPARE
        return step

    def begin_finalization(self) -> None:
        if self.final_emitted or self.phase is RunPhase.DONE:
            raise RuntimeError("final output has already been emitted")
        if not self.plan.complete:
            raise RuntimeError("cannot finalize an incomplete plan")
        if self.phase is not RunPhase.FINALIZE:
            self.finalization_count += 1
        self.phase = RunPhase.FINALIZE

    def record_final_response(
        self,
        response: str,
        *,
        persisted: bool = False,
        emitted: bool = False,
    ) -> None:
        if self.final_emitted and emitted:
            raise RuntimeError("final output has already been emitted")
        self.final_response = str(response)
        self.final_persisted = self.final_persisted or persisted
        self.final_emitted = self.final_emitted or emitted
        if self.final_emitted:
            self.phase = RunPhase.DONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "plan": self.plan.to_dict(),
            "current_step_id": self.current_step_id,
            "committed_results": {
                step_id: result.model_dump(mode="json")
                for step_id, result in self.committed_results.items()
            },
            "pending_step_result": (
                None
                if self.pending_step_result is None
                else self.pending_step_result.model_dump(mode="json")
            ),
            "validation_error": self.validation_error,
            "awaiting_step_result": self.awaiting_step_result,
            "replan_count": self.replan_count,
            "finalization_count": self.finalization_count,
            "final_response": self.final_response,
            "final_persisted": self.final_persisted,
            "final_emitted": self.final_emitted,
            "tool_repair_counts": dict(self.tool_repair_counts),
            "pending_tool_failure": (
                None
                if self.pending_tool_failure is None
                else dict(self.pending_tool_failure)
            ),
            "total_tool_repairs": self.total_tool_repairs,
            "failure": None if self.failure is None else dict(self.failure),
            "active_tool_calls": {
                call_id: dict(call) for call_id, call in self.active_tool_calls.items()
            },
            "seen_tool_call_ids": list(self.seen_tool_call_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionState:
        if not isinstance(value, Mapping):
            raise ValueError("execution state must be an object")
        raw_results = value.get("committed_results", {})
        if not isinstance(raw_results, Mapping):
            raise ValueError("committed_results must be an object")
        raw_pending = value.get("pending_step_result")
        raw_repair_counts = value.get("tool_repair_counts", {})
        if not isinstance(raw_repair_counts, Mapping):
            raise ValueError("tool_repair_counts must be an object")
        raw_tool_failure = value.get("pending_tool_failure")
        if raw_tool_failure is not None and not isinstance(raw_tool_failure, Mapping):
            raise ValueError("pending_tool_failure must be an object")
        raw_failure = value.get("failure")
        if raw_failure is not None and not isinstance(raw_failure, Mapping):
            raise ValueError("failure must be an object")
        raw_active_calls = value.get("active_tool_calls", {})
        if not isinstance(raw_active_calls, Mapping):
            raise ValueError("active_tool_calls must be an object")
        if any(not isinstance(call, Mapping) for call in raw_active_calls.values()):
            raise ValueError("active_tool_calls values must be objects")
        raw_seen_call_ids = value.get("seen_tool_call_ids", ())
        if isinstance(raw_seen_call_ids, (str, bytes)) or not isinstance(
            raw_seen_call_ids,
            Sequence,
        ):
            raise ValueError("seen_tool_call_ids must be an array")
        return cls(
            phase=RunPhase(value.get("phase", RunPhase.PLAN.value)),
            plan=Plan.from_dict(value.get("plan", {})),
            current_step_id=(
                None
                if value.get("current_step_id") is None
                else str(value["current_step_id"])
            ),
            committed_results={
                str(step_id): StepResult.model_validate(result)
                for step_id, result in raw_results.items()
            },
            pending_step_result=(
                None if raw_pending is None else StepResult.model_validate(raw_pending)
            ),
            validation_error=(
                None
                if value.get("validation_error") is None
                else str(value["validation_error"])
            ),
            awaiting_step_result=bool(value.get("awaiting_step_result", False)),
            replan_count=int(value.get("replan_count", 0)),
            finalization_count=int(value.get("finalization_count", 0)),
            final_response=(
                None
                if value.get("final_response") is None
                else str(value["final_response"])
            ),
            final_persisted=bool(value.get("final_persisted", False)),
            final_emitted=bool(value.get("final_emitted", False)),
            tool_repair_counts={
                str(step_id): int(count) for step_id, count in raw_repair_counts.items()
            },
            pending_tool_failure=(
                None if raw_tool_failure is None else dict(raw_tool_failure)
            ),
            total_tool_repairs=int(value.get("total_tool_repairs", 0)),
            failure=None if raw_failure is None else dict(raw_failure),
            active_tool_calls={
                str(call_id): dict(call) for call_id, call in raw_active_calls.items()
            },
            seen_tool_call_ids=[str(call_id) for call_id in raw_seen_call_ids],
        )


class PlanGenerator(Protocol):
    async def create(self, context: RunContext) -> Plan: ...

    async def revise(self, context: RunContext, plan: Plan, feedback: str) -> Plan: ...


class LLMPlanGenerator:
    def __init__(
        self,
        model: ModelClient,
        *,
        max_steps: int = 6,
        history_limit: int = 8,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if history_limit < 0:
            raise ValueError("history_limit cannot be negative")
        self.model = model
        self.max_steps = max_steps
        self.history_limit = history_limit

    async def create(self, context: RunContext) -> Plan:
        # Local import keeps the planning domain importable by runtime.context
        # without executing the skills package/runtime import graph.
        from moduagent.skills.prompting import compose_skill_prompt

        available_tools = self._available_tools(context)
        request = ModelRequest(
            messages=compose_skill_prompt(
                self._base_messages(
                    context,
                    Message.system(
                        "Create the smallest independently verifiable execution plan. "
                        "Return JSON only. Do not execute the work, write a final "
                        "answer, or add a final-answer/reporting step. Every step must "
                        "have a stable step_id, objective, completion_criteria, "
                        "expected_output, dependencies, and allowed_tools. A step can "
                        "use only one model-selected tool-call batch; if one tool's "
                        "result is needed to choose arguments for another tool, split "
                        "them into separate dependency-linked steps."
                    ),
                    Message.user(
                        json.dumps(
                            {
                                "request": context.request.input,
                                "available_tools": list(available_tools or ()),
                            },
                            ensure_ascii=False,
                        )
                    ),
                ),
                context.skill_messages,
                phase="plan",
            ),
            output_schema=self._schema(),
        )
        response = await self.model.complete(request)
        context.usage = context.usage + response.usage
        self._validate_response(response)
        return self._parse(
            response.message.content,
            context.request.input,
            available_tools=available_tools,
        )

    async def revise(self, context: RunContext, plan: Plan, feedback: str) -> Plan:
        from moduagent.skills.prompting import compose_skill_prompt

        available_tools = self._available_tools(context)
        request = ModelRequest(
            messages=compose_skill_prompt(
                self._base_messages(
                    context,
                    Message.system(
                        "Revise only unfinished plan steps. Preserve completed step "
                        "IDs and result references exactly. Return the plan JSON only; "
                        "do not execute work or add a final-answer step. A step can "
                        "use only one model-selected tool-call batch; split sequential "
                        "tool decisions into dependency-linked steps."
                    ),
                    Message.user(
                        json.dumps(
                            {
                                "request": context.request.input,
                                "plan": plan.to_dict(),
                                "feedback": feedback,
                                "available_tools": list(available_tools or ()),
                            },
                            ensure_ascii=False,
                        )
                    ),
                ),
                context.skill_messages,
                phase="plan",
            ),
            output_schema=self._schema(),
        )
        response = await self.model.complete(request)
        context.usage = context.usage + response.usage
        self._validate_response(response)
        return self._parse(
            response.message.content,
            context.request.input,
            available_tools=available_tools,
        )

    def _base_messages(
        self,
        context: RunContext,
        *phase_messages: Message,
    ) -> tuple[Message, ...]:
        leading_system = (
            (context.messages[0],)
            if context.messages and context.messages[0].role.value == "system"
            else ()
        )
        history_start = 1 if leading_system else 0
        history_end = min(
            max(context.current_run_start, history_start),
            len(context.messages),
        )
        public_history = tuple(
            message
            for message in context.messages[history_start:history_end]
            if message.role.value in {"user", "assistant"}
            and not message.tool_calls
            and message.metadata.get("moduagent.ephemeral") is not True
        )
        if self.history_limit == 0:
            public_history = ()
        else:
            public_history = public_history[-self.history_limit :]
        phase_system_count = 0
        while (
            phase_system_count < len(phase_messages)
            and phase_messages[phase_system_count].role.value == "system"
        ):
            phase_system_count += 1
        return (
            *leading_system,
            *phase_messages[:phase_system_count],
            *public_history,
            *phase_messages[phase_system_count:],
        )

    def _parse(
        self,
        content: str | None,
        fallback: str,
        *,
        available_tools: frozenset[str] | None = None,
    ) -> Plan:
        # ``fallback`` is retained in this private method's signature for
        # compatibility with early 0.3 callers. Strict planning must never turn
        # a malformed model response into an executable catch-all step.
        _ = fallback
        try:
            text = (content or "").strip()
            if not text:
                raise ValueError("plan response is empty")
            if text.startswith("```"):
                if "\n" not in text or "```" not in text.split("\n", 1)[1]:
                    raise ValueError("plan response contains an incomplete code fence")
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            payload = json.loads(text)
            raw_steps = (
                payload.get("steps", payload) if isinstance(payload, dict) else payload
            )
            if not isinstance(raw_steps, list):
                raise TypeError("steps must be an array")
            if not raw_steps:
                raise ValueError("plan must contain at least one step")
            if len(raw_steps) > self.max_steps:
                raise ValueError(
                    f"plan contains {len(raw_steps)} steps; maximum is {self.max_steps}"
                )
            if not all(isinstance(item, Mapping) for item in raw_steps):
                raise TypeError("every plan step must be an object")
            required_fields = frozenset(
                {
                    "step_id",
                    "objective",
                    "completion_criteria",
                    "expected_output",
                    "dependencies",
                    "allowed_tools",
                }
            )
            allowed_fields = required_fields | {"description"}
            for index, item in enumerate(raw_steps):
                missing = required_fields - set(item)
                if missing:
                    names = ", ".join(sorted(missing))
                    raise ValueError(f"plan step {index} is missing fields: {names}")
                unknown = set(item) - allowed_fields
                if unknown:
                    names = ", ".join(sorted(str(name) for name in unknown))
                    raise ValueError(f"plan step {index} has unknown fields: {names}")
                for field_name in ("step_id", "objective", "expected_output"):
                    value = item[field_name]
                    if not isinstance(value, str) or not value.strip():
                        raise TypeError(
                            f"plan step {index} {field_name} must be a non-empty string"
                        )
                if "description" in item and not isinstance(item["description"], str):
                    raise TypeError(f"plan step {index} description must be a string")
                for field_name in (
                    "completion_criteria",
                    "dependencies",
                    "allowed_tools",
                ):
                    value = item[field_name]
                    if not isinstance(value, list) or not all(
                        isinstance(entry, str) and entry.strip() for entry in value
                    ):
                        raise TypeError(
                            f"plan step {index} {field_name} must be an array "
                            "of non-empty strings"
                        )
                if not item["completion_criteria"]:
                    raise ValueError(
                        f"plan step {index} completion_criteria cannot be empty"
                    )
            steps = [PlanStep.from_dict(item) for item in raw_steps]
            if available_tools is not None:
                unknown_tools = {
                    tool
                    for step in steps
                    for tool in step.allowed_tools
                    if tool not in available_tools
                }
                if unknown_tools:
                    names = ", ".join(sorted(unknown_tools))
                    raise ValueError(f"plan contains unavailable tools: {names}")
            plan = Plan(steps)
            plan.start_current()
            return plan
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid plan response: {exc}") from exc

    @staticmethod
    def _validate_response(response: ModelResponse) -> None:
        finish_reason = (response.finish_reason or "").lower()
        if finish_reason in {"timeout", "length", "max_tokens"}:
            raise ValueError(f"incomplete plan response ({finish_reason})")
        if response.tool_calls or response.message.tool_calls:
            raise ValueError("planner response cannot contain tool calls")

    @staticmethod
    def _available_tools(context: RunContext) -> frozenset[str] | None:
        raw = context.metadata.get("_moduagent_available_tools")
        if raw is None:
            return None
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError("_moduagent_available_tools must be an array")
        tools = tuple(str(item) for item in raw)
        if not all(tools) or len(set(tools)) != len(tools):
            raise ValueError("_moduagent_available_tools contains invalid names")
        return frozenset(tools)

    @staticmethod
    def _schema() -> Mapping[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "step_id": {"type": "string"},
                            "objective": {"type": "string"},
                            "description": {"type": "string"},
                            "completion_criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                            "expected_output": {"type": "string"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "allowed_tools": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "step_id",
                            "objective",
                            "completion_criteria",
                            "expected_output",
                            "dependencies",
                            "allowed_tools",
                        ],
                    },
                    "maxItems": 64,
                }
            },
            "required": ["steps"],
        }


class PlanAndExecutePolicy:
    """Strict 0.3 Plan-and-Execute state machine policy."""

    requires_finalization = True
    strict_plan_execution = True
    # Compatibility with early 0.3 development snapshots.
    strict_execution = True

    def __init__(
        self,
        plan_generator: PlanGenerator,
        *,
        step_validator: StepValidator | None = None,
        revise_on_tool_failure: bool = True,
        max_step_attempts: int | None = None,
        max_replans: int | None = None,
        tool_failure_recovery: ToolFailureRecoveryConfig | None = None,
    ) -> None:
        if max_step_attempts is not None and max_step_attempts < 1:
            raise ValueError("max_step_attempts must be at least 1")
        if max_replans is not None and max_replans < 0:
            raise ValueError("max_replans cannot be negative")
        self.plan_generator = plan_generator
        self.step_validator = step_validator or StepValidator()
        self.revise_on_tool_failure = revise_on_tool_failure
        self.tool_failure_recovery = tool_failure_recovery
        self._max_step_attempts_override = max_step_attempts
        self._max_replans_override = max_replans
        # Safe standalone defaults; AgentRuntime calls configure_limits() with
        # RunLimits before begin().
        self.max_step_attempts = max_step_attempts or 2
        self.max_replans = 2 if max_replans is None else max_replans
        self.max_tool_repair_attempts = 1

    def configure_limits(
        self,
        *,
        max_step_attempts: int,
        max_replans: int,
        max_tool_repair_attempts: int | None = None,
    ) -> None:
        if max_step_attempts < 1:
            raise ValueError("max_step_attempts must be at least 1")
        if max_replans < 0:
            raise ValueError("max_replans cannot be negative")
        if max_tool_repair_attempts is not None and max_tool_repair_attempts < 0:
            raise ValueError("max_tool_repair_attempts cannot be negative")
        self.max_step_attempts = (
            max_step_attempts
            if self._max_step_attempts_override is None
            else min(max_step_attempts, self._max_step_attempts_override)
        )
        self.max_replans = (
            max_replans
            if self._max_replans_override is None
            else min(max_replans, self._max_replans_override)
        )
        if max_tool_repair_attempts is not None:
            self.max_tool_repair_attempts = max_tool_repair_attempts

    def configure_tool_repair_limits(
        self,
        *,
        max_tool_repair_attempts: int,
    ) -> None:
        """Configure the additive repair budget without widening old hooks."""

        if max_tool_repair_attempts < 0:
            raise ValueError("max_tool_repair_attempts cannot be negative")
        self.max_tool_repair_attempts = max_tool_repair_attempts

    async def begin(self, context: RunContext) -> None:
        state = getattr(context, "execution_state", None)
        if state is None:
            serialized = context.policy_state.get("execution_state")
            if isinstance(serialized, Mapping):
                state = ExecutionState.from_dict(serialized)
            else:
                plan = await self.plan_generator.create(context)
                state = ExecutionState(
                    phase=RunPhase.STEP_PREPARE,
                    plan=plan,
                    current_step_id=(
                        None if plan.current is None else plan.current.step_id
                    ),
                )
            context.execution_state = state
        self._sync(context, state)

    def prepare_step(
        self,
        context: RunContext,
        *,
        has_tools: bool,
    ) -> PlanStep | None:
        state = self._state(context)
        if state.phase is RunPhase.STEP_PREPARE:
            clear_internal = getattr(context, "clear_internal_messages", None)
            if callable(clear_internal):
                clear_internal()
        step = state.prepare_current_step(has_tools=has_tools)
        self._sync(context, state)
        return step

    def prepare_next_step(
        self,
        context: RunContext,
        *,
        has_tools: bool,
    ) -> PlanStep | None:
        return self.prepare_step(context, has_tools=has_tools)

    def build_step_context(self, context: RunContext) -> tuple[StepResult, ...]:
        state = self._state(context)
        step = state.current_step
        if step is None:
            return ()
        return tuple(
            state.committed_results[dependency]
            for dependency in step.dependencies
            if dependency in state.committed_results
        )

    def build_act_messages(self, context: RunContext) -> tuple[Message, ...]:
        """Project a fresh ACT context without unrelated conversation turns."""

        state = self._state(context)
        step = state.current_step
        if step is None:
            raise RuntimeError("there is no current plan step")
        messages: list[Message] = []
        if context.messages and context.messages[0].role.value == "system":
            messages.append(context.messages[0])
        messages.extend(
            (
                Message.system(
                    "Execute exactly one current plan step. Do not produce the "
                    "user-facing final answer or work on another step. Gather "
                    "completion evidence. When tool work is complete, return only "
                    "a StepResult that matches the supplied schema."
                ),
                Message.user(
                    json.dumps(
                        {
                            "current_step": step.to_dict(),
                            "dependency_results": [
                                result.model_dump(mode="json")
                                for result in self.build_step_context(context)
                            ],
                        },
                        ensure_ascii=False,
                    )
                ),
            )
        )
        internal_messages = tuple(getattr(context, "internal_messages", ()))
        if state.pending_tool_failure is not None:
            internal_messages = self._project_repair_internal_messages(
                internal_messages,
                state.pending_tool_failure,
            )
        messages.extend(internal_messages)
        return tuple(messages)

    @staticmethod
    def _project_repair_internal_messages(
        messages: Sequence[Message],
        pending: Mapping[str, Any],
    ) -> tuple[Message, ...]:
        failed_call_id = str(pending.get("call_id", ""))
        if not failed_call_id:
            return tuple(messages)
        error: dict[str, Any] = {
            "type": str(
                pending.get(
                    "error_type",
                    "execution_error",
                )
            ),
            "retryable": bool(pending.get("retryable", False)),
        }
        if pending.get("reason"):
            error["reason"] = str(pending["reason"])
        if pending.get("recovery"):
            error["recovery"] = str(pending["recovery"])
        content = json.dumps(
            {"success": False, "error": error},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        projected: list[Message] = []
        for message in messages:
            if message.role.value == "tool" and message.tool_call_id == failed_call_id:
                projected.append(
                    Message.tool(
                        content,
                        call_id=failed_call_id,
                        name=message.name or str(pending.get("tool_name", "")),
                        metadata=message.metadata,
                    )
                )
            else:
                projected.append(message)
        return tuple(projected)

    def act_tool_names(self, context: RunContext) -> frozenset[str] | None:
        step = self._state(context).current_step
        if step is None:
            return frozenset()
        return frozenset(step.allowed_tools)

    def needs_step_result_extraction(self, context: RunContext) -> bool:
        return self._state(context).awaiting_step_result

    def allows_tools(self, context: RunContext) -> bool:
        state = self._state(context)
        return state.phase is RunPhase.ACT and not state.awaiting_step_result

    @staticmethod
    def step_result_schema() -> Mapping[str, Any]:
        return StepResult.model_json_schema()

    async def decide(
        self, context: RunContext, response: ModelResponse
    ) -> ExecutionDecision:
        state = self._state(context)
        finish_reason = (response.finish_reason or "").lower()
        # Providers can surface a partial tool call together with an incomplete
        # finish reason. Never authorize side effects from such a response.
        if finish_reason in {"timeout", "length", "max_tokens"}:
            if state.pending_tool_failure is not None:
                return await self._fallback_tool_failure(
                    context,
                    state,
                    f"incomplete tool repair response ({finish_reason})",
                )
            return self._retry_invalid_result(
                context,
                state,
                f"incomplete StepResult response ({finish_reason})",
            )

        tool_calls = tuple(response.tool_calls or response.message.tool_calls)
        if tool_calls:
            if state.awaiting_step_result:
                state.fail_current_step()
                state.validation_error = (
                    "tools are forbidden during StepResult extraction"
                )
                self._sync(context, state)
                return ExecutionDecision(
                    DecisionKind.FAIL,
                    error_message=state.validation_error,
                    metadata=self._metadata(state),
                )
            call_records, call_error = self._tool_call_records(
                state,
                tool_calls,
            )
            if call_error is not None:
                if state.pending_tool_failure is not None:
                    return await self._fallback_tool_failure(
                        context,
                        state,
                        call_error,
                    )
                return self._retry_invalid_result(
                    context,
                    state,
                    call_error,
                )
            if state.pending_tool_failure is not None:
                repair_error = self._repair_tool_call_error(
                    state.pending_tool_failure,
                    tool_calls,
                    call_records,
                )
                if repair_error is not None:
                    return await self._fallback_tool_failure(
                        context,
                        state,
                        repair_error,
                    )
            state.active_tool_calls = call_records
            state.seen_tool_call_ids.extend(call_records)
            state.mark_tool_round_complete()
            self._sync(context, state)
            return ExecutionDecision(
                DecisionKind.CALL_TOOLS,
                tuple(tool_calls),
                metadata={
                    **self._metadata(state),
                    "requires_step_result": True,
                },
            )

        if state.pending_tool_failure is not None:
            return await self._fallback_tool_failure(
                context,
                state,
                "tool repair response did not call a tool",
            )

        # A tool-enabled ACT turn that did not call a tool is not evidence of
        # step completion. Force a separate, schema-only extraction turn.
        if not state.awaiting_step_result:
            state.awaiting_step_result = True
            state.validation_error = "a schema-only StepResult turn is required"
            self._sync(context, state)
            return ExecutionDecision(
                DecisionKind.RETRY_STEP,
                metadata={
                    **self._metadata(state),
                    "reason": state.validation_error,
                    "requires_step_result": True,
                    "count_attempt": False,
                },
            )

        if finish_reason in {"tool_calls", "tool_call"}:
            return self._retry_invalid_result(
                context,
                state,
                "model reported tool calls without providing a tool call",
            )

        try:
            result = self._decode_step_result(response)
        except (TypeError, ValueError) as exc:
            return self._retry_invalid_result(context, state, str(exc))

        step = state.current_step
        if step is None or result.step_id != step.step_id:
            state.pending_step_result = None
            state.validation_error = (
                "pending step validation state is incomplete"
                if step is None
                else (
                    f"step result id {result.step_id!r} does not match {step.step_id!r}"
                )
            )
            state.fail_current_step()
            self._sync(context, state)
            return ExecutionDecision(
                DecisionKind.FAIL,
                error_message="Step validation failed",
                metadata=self._metadata(state),
            )

        state.set_pending_result(result)
        return await self.validate_pending(context)

    @staticmethod
    def _tool_call_records(
        state: ExecutionState,
        calls: Sequence[ToolCall],
    ) -> tuple[dict[str, dict[str, str]], str | None]:
        records: dict[str, dict[str, str]] = {}
        seen = set(state.seen_tool_call_ids)
        for call in calls:
            call_id = str(call.id).strip()
            tool_name = str(call.name).strip()
            if not call_id or not tool_name:
                return {}, "Tool call ID and name cannot be empty"
            if len(call_id) > 256 or len(tool_name) > 256:
                return {}, "Tool call ID or name exceeds the protocol limit"
            if call_id in seen or call_id in records:
                return {}, "Tool calls must use a new unique call ID"
            try:
                fingerprint = _tool_arguments_fingerprint(call.arguments)
            except ValueError:
                return {}, "Tool call arguments must be canonical JSON"
            records[call_id] = {
                "tool_name": tool_name,
                "arguments_fingerprint": fingerprint,
            }
        return records, None

    @staticmethod
    def _repair_tool_call_error(
        pending: Mapping[str, Any],
        calls: Sequence[ToolCall],
        records: Mapping[str, Mapping[str, str]],
    ) -> str | None:
        if len(calls) != 1 or len(records) != 1:
            return "Tool repair must contain exactly one call"
        call = calls[0]
        failed_tool = str(pending.get("tool_name", ""))
        if call.name != failed_tool:
            return "Tool repair must call the same Tool"
        failed_call_id = str(pending.get("call_id", ""))
        if call.id == failed_call_id:
            return "Tool repair must use a new call ID"
        previous_fingerprint = str(pending.get("arguments_fingerprint", ""))
        current_fingerprint = next(iter(records.values())).get(
            "arguments_fingerprint",
            "",
        )
        if not previous_fingerprint:
            return "Tool repair cannot verify the failed call arguments"
        if current_fingerprint == previous_fingerprint:
            return "Tool repair must change the Tool arguments"
        return None

    async def validate_pending(
        self,
        context: RunContext,
    ) -> ExecutionDecision:
        """Resume deterministic validation from a checkpointed StepResult."""

        state = self._state(context)
        result = state.pending_step_result
        step = state.current_step
        if state.phase is not RunPhase.STEP_VALIDATE or result is None or step is None:
            state.fail_current_step()
            state.validation_error = "pending step validation state is incomplete"
            self._sync(context, state)
            return ExecutionDecision(
                DecisionKind.FAIL,
                error_message="Step validation failed",
                metadata=self._metadata(state),
            )
        try:
            validation = self.step_validator.validate(
                step,
                result,
            )
            if not isinstance(validation, StepValidation):
                raise TypeError("step validator must return a StepValidation instance")
        except Exception as exc:
            state.fail_current_step()
            state.validation_error = f"step validator failed ({type(exc).__name__})"
            self._sync(context, state)
            return ExecutionDecision(
                DecisionKind.FAIL,
                error_message="Step validation failed",
                metadata=self._metadata(state),
            )
        if validation.kind is ValidationKind.COMMIT:
            state.commit_pending()
            clear_internal = getattr(context, "clear_internal_messages", None)
            if callable(clear_internal):
                clear_internal()
            self._sync(context, state)
            return ExecutionDecision(
                DecisionKind.COMMIT_STEP,
                metadata={
                    **self._metadata(state),
                    "validation": validation.to_dict(),
                    "plan_complete": state.complete,
                },
            )
        if validation.kind is ValidationKind.RETRY:
            return self._retry_validation(context, state, validation)
        if validation.kind is ValidationKind.REPLAN:
            return await self._replan(context, state, validation.reason)

        state.fail_current_step()
        state.validation_error = validation.reason
        self._sync(context, state)
        return ExecutionDecision(
            DecisionKind.FAIL,
            error_message="Step validation failed",
            metadata={
                **self._metadata(state),
                "validation": validation.to_dict(),
            },
        )

    async def observe(
        self,
        context: RunContext,
        results: Sequence[ToolResult],
    ) -> ExecutionDecision | None:
        failures = [result for result in results if not result.success]
        state = self._state(context)
        if not failures:
            # A tool round-trip never consumes a step attempt. The next turn is
            # the schema-only StepResult extraction established by decide().
            state.pending_tool_failure = None
            state.failure = None
            state.validation_error = None
            state.active_tool_calls = {}
            state.awaiting_step_result = True
            self._sync(context, state)
            return

        recovery = self.tool_failure_recovery
        if recovery is not None:
            # A mixed batch may already contain successful side effects. Do not
            # let an argument-repair turn accidentally repeat those calls.
            candidate = (
                failures[0] if len(failures) == 1 and len(results) == 1 else None
            )
            fallback_override: Literal["replan", "fail"] | None = None
            fallback_reason: str
            partial_success = len(failures) < len(results)
            if partial_success:
                # At least one call may already have produced a side effect. A
                # non-transactional batch cannot be replayed or replanned safely.
                fallback_override = "fail"
                fallback_reason = (
                    "Tool batch partially succeeded; automatic recovery is unsafe"
                )
            elif candidate is not None:
                action = self._tool_recovery_action(candidate)
                if action is ToolRecoveryAction.REPAIR_CALL:
                    if self._can_repair_tool_failure(
                        state,
                        candidate,
                        recovery,
                    ):
                        return self._schedule_tool_repair(
                            context,
                            state,
                            candidate,
                            recovery,
                        )
                    fallback_reason = (
                        "tool repair budget exhausted"
                        if self._repair_count(state) >= self.max_tool_repair_attempts
                        else "tool failure is not safe for same-step repair"
                    )
                elif action is ToolRecoveryAction.FAIL:
                    fallback_override = "fail"
                    fallback_reason = "tool recovery action requires failure"
                elif action is ToolRecoveryAction.REPLAN:
                    fallback_override = "replan"
                    fallback_reason = "tool recovery action requires replanning"
                elif action is ToolRecoveryAction.RETRY_CALL:
                    # ToolExecutor owns same-argument retries. A failed result
                    # reaching the policy means that retry budget was consumed.
                    fallback_reason = "tool retry attempts exhausted"
                else:
                    fallback_reason = "tool failure has no recovery action"
            else:
                actions = {
                    action
                    for failure in failures
                    if (action := self._tool_recovery_action(failure)) is not None
                }
                if ToolRecoveryAction.FAIL in actions:
                    fallback_override = "fail"
                    fallback_reason = "tool recovery action requires failure"
                elif ToolRecoveryAction.REPLAN in actions:
                    fallback_override = "replan"
                    fallback_reason = "tool recovery action requires replanning"
                else:
                    fallback_reason = (
                        "tool batches with multiple results cannot be repaired "
                        "in the same step"
                    )

            primary = failures[0]
            payload = self._tool_failure_payload(state, primary, recovery)
            if len(failures) > 1:
                payload["failure_count"] = len(failures)
            if partial_success:
                payload["success_count"] = len(results) - len(failures)
                payload["result_count"] = len(results)
            state.pending_tool_failure = payload
            return await self._fallback_tool_failure(
                context,
                state,
                fallback_reason,
                fallback=fallback_override,
            )

        # No recovery configuration means legacy 0.3 behavior.
        feedback = "; ".join(
            result.error.message if result.error else "tool failed"
            for result in failures
        )
        state.active_tool_calls = {}
        if not self.revise_on_tool_failure:
            state.validation_error = "tool execution failed"
            state.fail_current_step()
            self._sync(context, state)
            return
        if state.replan_count >= self.max_replans:
            state.validation_error = feedback
            state.fail_current_step()
            self._sync(context, state)
            return
        await self._apply_replan(context, state, feedback)
        return None

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        if value is None:
            return None
        raw = getattr(value, "value", value)
        text = str(raw).strip()
        return text or None

    @classmethod
    def _tool_recovery_action(cls, result: ToolResult) -> ToolRecoveryAction | None:
        error = result.error
        raw = None if error is None else getattr(error, "recovery", None)
        if isinstance(raw, ToolRecoveryAction):
            return raw
        value = cls._enum_value(raw)
        if value is None:
            return None
        try:
            return ToolRecoveryAction(value)
        except ValueError:
            return None

    @classmethod
    def _is_repair_call(cls, result: ToolResult) -> bool:
        return cls._tool_recovery_action(result) is ToolRecoveryAction.REPAIR_CALL

    @staticmethod
    def _sanitize_tool_failure_message(message: Any, *, limit: int = 512) -> str:
        text = "".join(
            character if character.isprintable() else " " for character in str(message)
        )
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3]}..."

    def _repair_count(self, state: ExecutionState) -> int:
        step = state.current_step
        if step is None:
            return self.max_tool_repair_attempts
        return state.tool_repair_counts.get(step.step_id, 0)

    def _can_repair_tool_failure(
        self,
        state: ExecutionState,
        result: ToolResult,
        recovery: ToolFailureRecoveryConfig,
    ) -> bool:
        if state.current_step is None or not self._is_repair_call(result):
            return False
        if recovery.require_repair_safe and not bool(
            getattr(result, "repair_safe", False)
        ):
            return False
        active_call = state.active_tool_calls.get(result.call_id)
        if (
            not isinstance(active_call, Mapping)
            or active_call.get("tool_name") != result.tool_name
            or not active_call.get("arguments_fingerprint")
        ):
            return False
        return self._repair_count(state) < self.max_tool_repair_attempts

    def _tool_failure_payload(
        self,
        state: ExecutionState,
        result: ToolResult,
        recovery: ToolFailureRecoveryConfig,
    ) -> dict[str, Any]:
        error = result.error
        error_type = (
            None if error is None else self._enum_value(getattr(error, "type", None))
        )
        reason = (
            None if error is None else self._enum_value(getattr(error, "reason", None))
        )
        recovery_action = self._tool_recovery_action(result)
        error_type = (
            None
            if error_type is None
            else self._sanitize_tool_failure_message(error_type, limit=256)
        )
        reason = (
            None
            if reason is None
            else self._sanitize_tool_failure_message(reason, limit=256)
        )
        recovery_name = self._enum_value(recovery_action)
        recovery_name = (
            None
            if recovery_name is None
            else self._sanitize_tool_failure_message(recovery_name, limit=256)
        )
        tool_name = self._sanitize_tool_failure_message(result.tool_name, limit=256)
        label = reason or error_type or "tool_failure"
        feedback = f"Tool {tool_name} failed ({label})"
        if (
            recovery.feedback_mode == "safe_message"
            and recovery_action is ToolRecoveryAction.REPAIR_CALL
            and error is not None
        ):
            message = self._sanitize_tool_failure_message(error.message)
            if message:
                feedback = f"{feedback}: {message}"
        step = state.current_step
        payload = {
            "step_id": (
                None
                if step is None
                else self._sanitize_tool_failure_message(step.step_id, limit=256)
            ),
            "call_id": self._sanitize_tool_failure_message(
                result.call_id,
                limit=256,
            ),
            "tool_name": tool_name,
            "error_type": error_type,
            "reason": reason,
            "recovery": recovery_name,
            "retryable": bool(error.retryable) if error is not None else False,
            "repair_safe": bool(getattr(result, "repair_safe", False)),
            "feedback": self._sanitize_tool_failure_message(feedback, limit=512),
        }
        active_call = state.active_tool_calls.get(result.call_id)
        if (
            isinstance(active_call, Mapping)
            and active_call.get("tool_name") == result.tool_name
            and active_call.get("arguments_fingerprint")
        ):
            payload["arguments_fingerprint"] = str(active_call["arguments_fingerprint"])
            invocation_arguments = getattr(result, "invocation_arguments", None)
            payload["invocation_fingerprint"] = (
                _tool_arguments_fingerprint(invocation_arguments)
                if isinstance(invocation_arguments, Mapping)
                else str(active_call["arguments_fingerprint"])
            )
        return payload

    def _schedule_tool_repair(
        self,
        context: RunContext,
        state: ExecutionState,
        result: ToolResult,
        recovery: ToolFailureRecoveryConfig,
    ) -> ExecutionDecision:
        payload = self._tool_failure_payload(state, result, recovery)
        step = state.current_step
        if step is None:
            raise RuntimeError("cannot repair a Tool failure without a current step")
        repair_count = state.tool_repair_counts.get(step.step_id, 0) + 1
        state.tool_repair_counts[step.step_id] = repair_count
        state.total_tool_repairs += 1
        state.pending_tool_failure = payload
        state.failure = None
        state.active_tool_calls = {}
        state.pending_step_result = None
        state.validation_error = str(payload["feedback"])
        state.awaiting_step_result = False
        state.phase = RunPhase.ACT
        self._sync(context, state)
        return ExecutionDecision(
            DecisionKind.RETRY_TOOL,
            metadata={
                **self._metadata(state),
                "reason": payload["feedback"],
                "tool_failure": dict(payload),
                "repair_attempt": repair_count,
                "count_attempt": False,
            },
        )

    async def _fallback_tool_failure(
        self,
        context: RunContext,
        state: ExecutionState,
        reason: str,
        *,
        fallback: Literal["replan", "fail"] | None = None,
    ) -> ExecutionDecision:
        recovery = self.tool_failure_recovery
        if recovery is None:
            raise RuntimeError("Tool failure fallback requires recovery configuration")
        effective_fallback = recovery.fallback if fallback is None else fallback
        pending = dict(state.pending_tool_failure or {})
        pending["fallback_reason"] = reason
        state.pending_tool_failure = pending
        state.active_tool_calls = {}
        feedback = str(pending.get("feedback") or "Tool execution failed")
        if reason not in feedback:
            feedback = f"{feedback}; {reason}"
        state.validation_error = feedback

        if effective_fallback == "replan":
            decision = await self._replan(context, state, feedback)
            if decision.kind is DecisionKind.FAIL:
                state.failure = {
                    **pending,
                    "terminal_reason": decision.error_message
                    or "maximum replans exceeded",
                }
                state.pending_step_result = None
                state.awaiting_step_result = False
                self._sync(context, state)
                return ExecutionDecision(
                    DecisionKind.FAIL,
                    error_message=decision.error_message,
                    metadata={
                        **self._metadata(state),
                        "reason": reason,
                        "tool_failure": dict(state.failure),
                    },
                )
            return decision

        state.failure = {
            **pending,
            "terminal_reason": reason,
        }
        state.pending_step_result = None
        state.awaiting_step_result = False
        state.fail_current_step()
        self._sync(context, state)
        return ExecutionDecision(
            DecisionKind.FAIL,
            error_message="Tool execution failed",
            metadata={
                **self._metadata(state),
                "reason": reason,
                "tool_failure": dict(state.failure),
            },
        )

    def should_stop(self, context: RunContext) -> bool:
        state = getattr(context, "execution_state", None)
        return isinstance(state, ExecutionState) and state.phase in {
            RunPhase.DONE,
            RunPhase.FAILED,
        }

    def begin_finalization(self, context: RunContext) -> ExecutionState:
        state = self._state(context)
        state.begin_finalization()
        self._sync(context, state)
        return state

    def record_final_response(
        self,
        context: RunContext,
        response: str,
        *,
        persisted: bool = False,
        emitted: bool = False,
    ) -> ExecutionState:
        state = self._state(context)
        state.record_final_response(
            response,
            persisted=persisted,
            emitted=emitted,
        )
        self._sync(context, state)
        return state

    def finalization_payload(self, context: RunContext) -> dict[str, Any]:
        state = self._state(context)
        return {
            "objective": context.request.input,
            "committed_results": {
                key: result.model_dump(mode="json")
                for key, result in state.committed_results.items()
            },
        }

    def _retry_invalid_result(
        self,
        context: RunContext,
        state: ExecutionState,
        reason: str,
    ) -> ExecutionDecision:
        step = state.current_step
        if step is None:
            state.fail_current_step()
            self._sync(context, state)
            return ExecutionDecision(
                DecisionKind.FAIL,
                error_message="there is no current step to retry",
            )
        was_awaiting_step_result = state.awaiting_step_result
        step.attempt_count += 1
        state.validation_error = reason
        state.pending_step_result = None
        if step.attempt_count >= self.max_step_attempts:
            state.fail_current_step()
            kind = DecisionKind.FAIL
        else:
            state.phase = RunPhase.ACT
            state.awaiting_step_result = was_awaiting_step_result
            kind = DecisionKind.RETRY_STEP
        self._sync(context, state)
        public_error = (
            "StepResult validation failed after maximum attempts"
            if kind is DecisionKind.FAIL
            else None
        )
        return ExecutionDecision(
            kind,
            error_message=public_error,
            metadata={
                **self._metadata(state),
                "reason": reason,
                "count_attempt": True,
            },
        )

    def _retry_validation(
        self,
        context: RunContext,
        state: ExecutionState,
        validation: StepValidation,
    ) -> ExecutionDecision:
        step = state.current_step
        if step is None:
            state.fail_current_step()
            self._sync(context, state)
            return ExecutionDecision(
                DecisionKind.FAIL,
                error_message="there is no current step to retry",
            )
        step.attempt_count += 1
        state.validation_error = validation.reason
        state.pending_step_result = None
        if step.attempt_count >= self.max_step_attempts:
            state.fail_current_step()
            kind = DecisionKind.FAIL
        else:
            state.phase = RunPhase.ACT
            state.awaiting_step_result = True
            kind = DecisionKind.RETRY_STEP
        self._sync(context, state)
        public_error = (
            "Step validation failed after maximum attempts"
            if kind is DecisionKind.FAIL
            else None
        )
        return ExecutionDecision(
            kind,
            error_message=public_error,
            metadata={
                **self._metadata(state),
                "validation": validation.to_dict(),
                "count_attempt": True,
            },
        )

    async def _replan(
        self,
        context: RunContext,
        state: ExecutionState,
        feedback: str,
    ) -> ExecutionDecision:
        if state.replan_count >= self.max_replans:
            state.fail_current_step()
            state.validation_error = "maximum replans exceeded"
            self._sync(context, state)
            return ExecutionDecision(
                DecisionKind.FAIL,
                error_message=state.validation_error,
                metadata=self._metadata(state),
            )
        await self._apply_replan(context, state, feedback)
        return ExecutionDecision(
            DecisionKind.REPLAN,
            metadata={
                **self._metadata(state),
                "reason": feedback,
            },
        )

    async def _apply_replan(
        self,
        context: RunContext,
        state: ExecutionState,
        feedback: str,
    ) -> None:
        try:
            revised = await self.plan_generator.revise(
                context,
                state.plan,
                feedback,
            )
        except Exception as exc:
            # A failed revision cannot leave a checkpoint that looks resumable
            # from ACT/STEP_VALIDATE, where stale tool output or a pending
            # StepResult might otherwise be committed after recovery.
            if state.pending_tool_failure is not None:
                state.failure = {
                    **state.pending_tool_failure,
                    "terminal_reason": "plan revision failed",
                }
            state.fail_current_step()
            state.validation_error = "plan revision failed"
            state.pending_step_result = None
            state.awaiting_step_result = False
            self._sync(context, state)
            raise RuntimeError("plan revision failed") from exc
        committed_ids = set(state.committed_results)
        previous_open_steps = {
            step.step_id: step
            for step in state.plan.steps
            if step.step_id not in committed_ids
        }
        committed_steps = [
            PlanStep.from_dict(step.to_dict())
            for step in state.plan.steps
            if step.step_id in committed_ids
        ]
        revised_open = [
            PlanStep.from_dict(step.to_dict())
            for step in revised.steps
            if step.step_id not in committed_ids
        ]
        for step in revised_open:
            previous = previous_open_steps.get(step.step_id)
            if previous is not None:
                # A replan may refine an unfinished step, but it must not reset
                # the retry budget for a stable step identity.
                step.attempt_count = max(
                    step.attempt_count,
                    previous.attempt_count,
                )
        merged = Plan(
            [*committed_steps, *revised_open],
            current_index=len(committed_steps),
            version=max(state.plan.version, revised.version) + 1,
        )
        state.plan = merged
        state.replan_count += 1
        state.pending_step_result = None
        state.pending_tool_failure = None
        state.failure = None
        state.active_tool_calls = {}
        state.validation_error = feedback
        state.awaiting_step_result = False
        state.phase = RunPhase.STEP_PREPARE
        state.current_step_id = (
            None if merged.current is None else merged.current.step_id
        )
        self._sync(context, state)

    @staticmethod
    def _decode_step_result(response: ModelResponse) -> StepResult:
        content = response.message.content
        if isinstance(content, Mapping):
            return StepResult.model_validate(content)
        if not isinstance(content, (str, bytes, bytearray)):
            raise TypeError("StepResult response must be a JSON object")
        if isinstance(content, bytearray):
            content = bytes(content)
        if isinstance(content, str) and not content.strip():
            raise ValueError("StepResult response is empty")
        return StepResult.model_validate_json(content)

    @staticmethod
    def _metadata(state: ExecutionState) -> dict[str, Any]:
        return {
            "phase": state.phase.value,
            "plan": state.plan.to_dict(),
            "execution_state": state.to_dict(),
        }

    @staticmethod
    def _sync(context: RunContext, state: ExecutionState) -> None:
        context.execution_state = state
        serialized = state.to_dict()
        context.policy_state["execution_state"] = serialized
        # Kept as a read-compatible metadata field for 0.2 integrations.
        context.policy_state["plan"] = serialized["plan"]

    @staticmethod
    def _state(context: RunContext) -> ExecutionState:
        state = getattr(context, "execution_state", None)
        if isinstance(state, ExecutionState):
            return state
        serialized = context.policy_state.get("execution_state")
        if isinstance(serialized, Mapping):
            state = ExecutionState.from_dict(serialized)
            context.execution_state = state
            return state
        raise RuntimeError("PlanAndExecutePolicy.begin() must be called first")


class LegacyPlanAndExecutePolicy:
    """The implicit 0.2 completion behavior, retained for migration only."""

    requires_finalization = False
    strict_execution = False

    def __init__(
        self,
        plan_generator: PlanGenerator,
        *,
        revise_on_tool_failure: bool = True,
    ) -> None:
        warnings.warn(
            "LegacyPlanAndExecutePolicy uses unverified implicit step completion "
            "and will be removed in a future release; use PlanAndExecutePolicy",
            DeprecationWarning,
            stacklevel=2,
        )
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

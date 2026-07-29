from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from moduagent.messages import ToolCall
from moduagent.execution.planning.state import ToolRecoveryState

try:
    from moduagent.tools import (
        FailureProjector,
        SafeToolFailureView,
        ToolBatchOutcome,
        ToolRecoveryAction,
        ToolRepairConstraint,
        fingerprint_tool_arguments,
    )
except ImportError:  # pragma: no cover - staggered 0.4 package upgrade
    from moduagent.tools.arguments import fingerprint_tool_arguments
    from moduagent.tools.base import ToolRecoveryAction
    from moduagent.tools.failure import FailureProjector, SafeToolFailureView
    from moduagent.tools.runtime import ToolBatchOutcome, ToolRepairConstraint


class ToolRecoveryDecisionKind(str, Enum):
    CONTINUE = "continue"
    REPAIR = "repair"
    REPLAN = "replan"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class ToolRecoveryControllerConfig:
    fallback: Literal["replan", "fail"] = "replan"
    allow_same_step_repair: bool = True
    require_repair_safe: bool = True
    feedback_mode: Literal["type_only", "safe_message"] = "type_only"

    def __post_init__(self) -> None:
        if self.fallback not in {"replan", "fail"}:
            raise ValueError("fallback must be 'replan' or 'fail'")
        for field_name in (
            "allow_same_step_repair",
            "require_repair_safe",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        if self.feedback_mode not in {"type_only", "safe_message"}:
            raise ValueError("feedback_mode must be 'type_only' or 'safe_message'")


@dataclass(frozen=True, slots=True)
class ToolRecoveryDecision:
    kind: ToolRecoveryDecisionKind
    reason: str
    failure: SafeToolFailureView | None = None
    repair_constraint: ToolRepairConstraint | None = None
    repair_attempt: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ToolRecoveryDecisionKind):
            object.__setattr__(
                self,
                "kind",
                ToolRecoveryDecisionKind(str(self.kind)),
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("recovery decision reason cannot be empty")
        if self.failure is not None and not isinstance(
            self.failure,
            SafeToolFailureView,
        ):
            raise TypeError("failure must be a SafeToolFailureView")
        if self.repair_constraint is not None and not isinstance(
            self.repair_constraint,
            ToolRepairConstraint,
        ):
            raise TypeError("repair_constraint must be a ToolRepairConstraint")
        if type(self.repair_attempt) is not int:
            raise TypeError("repair_attempt must be an integer")
        if self.repair_attempt < 0:
            raise ValueError("repair_attempt cannot be negative")
        if self.kind is ToolRecoveryDecisionKind.REPAIR:
            if self.failure is None or self.repair_constraint is None:
                raise ValueError("repair decisions require failure and constraint")
            if self.repair_attempt < 1:
                raise ValueError("repair decisions require a positive attempt")
        elif self.repair_constraint is not None:
            raise ValueError("only repair decisions may contain a constraint")


class ToolRecoveryController:
    """Plan-owned repair, replan and failure decision boundary.

    Same-call retry remains entirely inside ToolRuntime. This controller sees
    only the post-retry ToolBatchOutcome and never invokes a Tool itself.
    """

    def __init__(
        self,
        config: ToolRecoveryControllerConfig | None = None,
        *,
        failure_projector: FailureProjector | None = None,
    ) -> None:
        if config is not None and not isinstance(
            config,
            ToolRecoveryControllerConfig,
        ):
            raise TypeError("config must be a ToolRecoveryControllerConfig")
        if failure_projector is not None and not isinstance(
            failure_projector,
            FailureProjector,
        ):
            raise TypeError("failure_projector must be a FailureProjector")
        self.config = config or ToolRecoveryControllerConfig()
        self.failure_projector = failure_projector or FailureProjector()

    def record_requested_calls(
        self,
        state: ToolRecoveryState,
        calls: tuple[ToolCall, ...],
    ) -> None:
        """Record call identity without retaining raw arguments."""

        if not isinstance(state, ToolRecoveryState):
            raise TypeError("state must be a ToolRecoveryState")
        if type(calls) is not tuple or any(
            not isinstance(call, ToolCall) for call in calls
        ):
            raise TypeError("calls must be a tuple of ToolCall")
        known = set(state.seen_call_ids)
        records: dict[str, dict[str, str]] = {}
        for call in calls:
            if call.id in known or call.id in records:
                raise ValueError("Tool calls must use new unique call IDs")
            records[call.id] = {
                "tool_name": call.name,
                "arguments_fingerprint": fingerprint_tool_arguments(call.arguments),
            }
        state.active_calls = records
        state.seen_call_ids.extend(records)

    def decide(
        self,
        outcome: ToolBatchOutcome,
        state: ToolRecoveryState,
        *,
        step_id: str,
        max_repair_attempts: int,
    ) -> ToolRecoveryDecision:
        if not isinstance(outcome, ToolBatchOutcome):
            raise TypeError("outcome must be a ToolBatchOutcome")
        if not isinstance(state, ToolRecoveryState):
            raise TypeError("state must be a ToolRecoveryState")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError("step_id cannot be empty")
        if type(max_repair_attempts) is not int:
            raise TypeError("max_repair_attempts must be an integer")
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative")

        if outcome.failure_count == 0:
            state.active_calls = {}
            state.pending_failure = None
            state.terminal_failure = None
            return ToolRecoveryDecision(
                ToolRecoveryDecisionKind.CONTINUE,
                "Tool batch completed",
            )

        primary = outcome.failures[0]
        failure_view = self.failure_projector.project(
            primary,
            include_safe_message=self.config.feedback_mode == "safe_message",
        )
        state.pending_failure = {
            **failure_view.to_dict(),
            "step_id": step_id,
            "repair_safe": (primary.safety_profile.changed_argument_repair_safe),
        }

        if outcome.partial_success:
            return self._terminal(
                state,
                failure_view,
                "Tool batch partially succeeded; automatic recovery is unsafe",
            )

        if outcome.failure_count != 1 or len(outcome.results) != 1:
            actions = {
                failure.classification.recovery_directive
                for failure in outcome.failures
            }
            if ToolRecoveryAction.FAIL in actions:
                return self._terminal(
                    state,
                    failure_view,
                    "Tool recovery action requires failure",
                )
            if ToolRecoveryAction.REPLAN in actions:
                return self._replan(
                    state,
                    failure_view,
                    "Tool recovery action requires replanning",
                )
            return self._fallback(
                state,
                failure_view,
                "Tool batches with multiple failures cannot be repaired "
                "inside one step",
            )

        action = primary.classification.recovery_directive
        if action is ToolRecoveryAction.FAIL:
            return self._terminal(
                state,
                failure_view,
                "Tool recovery action requires failure",
            )
        if action is ToolRecoveryAction.REPLAN:
            return self._replan(
                state,
                failure_view,
                "Tool recovery action requires replanning",
            )
        if action is ToolRecoveryAction.RETRY_CALL:
            return self._fallback(
                state,
                failure_view,
                "ToolRuntime exhausted same-call retry",
            )
        if action is not ToolRecoveryAction.REPAIR_CALL:
            return self._fallback(
                state,
                failure_view,
                "Tool failure has no same-step recovery action",
            )
        if not self.config.allow_same_step_repair:
            return self._fallback(
                state,
                failure_view,
                "Same-step Tool repair is not enabled",
            )

        if (
            self.config.require_repair_safe
            and not primary.safety_profile.changed_argument_repair_safe
        ):
            return self._fallback(
                state,
                failure_view,
                "Tool failure is not safe for changed-argument repair",
            )

        current_count = state.repair_count_by_step.get(step_id, 0)
        if current_count >= max_repair_attempts:
            return self._fallback(
                state,
                failure_view,
                "tool repair budget exhausted",
            )

        active = state.active_calls.get(primary.call_id)
        if (
            active is None
            or active.get("tool_name") != primary.tool_name
            or active.get("arguments_fingerprint")
            != primary.requested_arguments_fingerprint
        ):
            return self._fallback(
                state,
                failure_view,
                "Tool repair cannot verify the failed invocation",
            )
        requested_fingerprint = primary.requested_arguments_fingerprint
        effective_fingerprint = primary.effective_arguments_fingerprint
        if requested_fingerprint is None or effective_fingerprint is None:
            return self._fallback(
                state,
                failure_view,
                "Tool repair requires requested and effective argument fingerprints",
            )

        attempt = current_count + 1
        state.repair_count_by_step[step_id] = attempt
        state.total_repairs += 1
        state.active_calls = {}
        constraint = ToolRepairConstraint(
            failed_call_id=primary.call_id,
            expected_tool_name=primary.tool_name,
            seen_call_ids=frozenset({*state.seen_call_ids, primary.call_id}),
            previous_requested_fingerprint=requested_fingerprint,
            previous_effective_fingerprint=effective_fingerprint,
        )
        return ToolRecoveryDecision(
            ToolRecoveryDecisionKind.REPAIR,
            "Repair the failed Tool call with changed arguments",
            failure=failure_view,
            repair_constraint=constraint,
            repair_attempt=attempt,
        )

    def constraint_for_pending(
        self,
        state: ToolRecoveryState,
    ) -> ToolRepairConstraint | None:
        if not isinstance(state, ToolRecoveryState):
            raise TypeError("state must be a ToolRecoveryState")
        pending = state.pending_failure
        if pending is None:
            return None
        call_id = pending.get("call_id")
        tool_name = pending.get("tool_name")
        requested = pending.get("arguments_fingerprint")
        effective = pending.get("invocation_fingerprint")
        if not all(
            isinstance(value, str) and value
            for value in (call_id, tool_name, requested, effective)
        ):
            raise ValueError("pending Tool repair state is incomplete")
        return ToolRepairConstraint(
            failed_call_id=call_id,
            expected_tool_name=tool_name,
            seen_call_ids=frozenset(state.seen_call_ids),
            previous_requested_fingerprint=requested,
            previous_effective_fingerprint=effective,
        )

    def _fallback(
        self,
        state: ToolRecoveryState,
        failure: SafeToolFailureView,
        reason: str,
    ) -> ToolRecoveryDecision:
        if self.config.fallback == "fail":
            return self._terminal(state, failure, reason)
        return self._replan(state, failure, reason)

    @staticmethod
    def _replan(
        state: ToolRecoveryState,
        failure: SafeToolFailureView,
        reason: str,
    ) -> ToolRecoveryDecision:
        state.active_calls = {}
        return ToolRecoveryDecision(
            ToolRecoveryDecisionKind.REPLAN,
            reason,
            failure=failure,
        )

    @staticmethod
    def _terminal(
        state: ToolRecoveryState,
        failure: SafeToolFailureView,
        reason: str,
    ) -> ToolRecoveryDecision:
        state.active_calls = {}
        state.terminal_failure = {
            **failure.to_dict(),
            "terminal_reason": reason,
        }
        return ToolRecoveryDecision(
            ToolRecoveryDecisionKind.FAIL,
            reason,
            failure=failure,
        )


__all__ = [
    "ToolRecoveryController",
    "ToolRecoveryControllerConfig",
    "ToolRecoveryDecision",
    "ToolRecoveryDecisionKind",
]

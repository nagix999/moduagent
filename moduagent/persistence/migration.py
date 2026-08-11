from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from moduagent.errors import StateMigrationError
from moduagent.persistence.snapshot import (
    CommonRunState,
    EngineSnapshot,
    FinalizationMarkers,
    RunSnapshot,
    PREVIOUS_SNAPSHOT_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    current_runtime_version,
)
from moduagent.tools import fingerprint_tool_arguments


_LEGACY_CHECKPOINT_VERSIONS = frozenset({1, 2, 3})
_STANDARD_FINALIZATION_STATE_KEY = "_moduagent_structured_finalization"
_STANDARD_FINALIZATION_OUTPUT_KEY = "_moduagent_structured_output"
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "set_cookie",
        "token",
    }
)
_SAFE_FAILURE_IDENTIFIER_FIELDS = frozenset({"step_id", "call_id", "tool_name"})
_SAFE_FAILURE_CODE_FIELDS = frozenset({"error_type", "type", "reason", "recovery"})
_SAFE_FAILURE_BOOL_FIELDS = frozenset({"retryable", "repair_safe"})
_SAFE_FAILURE_COUNT_FIELDS = frozenset(
    {
        "failure_count",
        "success_count",
        "result_count",
        "repair_attempts",
    }
)
_SAFE_FAILURE_FINGERPRINT_FIELDS = frozenset(
    {"arguments_fingerprint", "invocation_fingerprint"}
)
_SAFE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def migrate_checkpoint_payload(
    value: Mapping[str, Any],
    *,
    agent_fingerprint: str = "legacy-unbound",
    runtime_version: str | None = None,
) -> RunSnapshot:
    """Copy and migrate a legacy checkpoint without mutating its source.

    Version 5 payloads are decoded and validated. Version 4 is copied and
    upgraded as a root run. Versions 1-3 are copied
    before any normalization, so a failed migration leaves the caller's object
    graph byte-for-byte representable as it was before the call.
    """

    if not isinstance(value, Mapping):
        raise StateMigrationError("checkpoint payload must be a JSON object")
    source = copy.deepcopy(dict(value))
    if "schema_version" in source:
        try:
            schema_version = _integer(source["schema_version"], "schema_version")
        except ValueError as exc:
            raise StateMigrationError(str(exc)) from exc
        if schema_version not in {
            PREVIOUS_SNAPSHOT_SCHEMA_VERSION,
            SNAPSHOT_SCHEMA_VERSION,
        }:
            raise StateMigrationError(
                f"unsupported snapshot schema version: {schema_version}"
            )
        try:
            return RunSnapshot.from_dict(source)
        except (TypeError, ValueError) as exc:
            raise StateMigrationError(str(exc)) from exc

    try:
        legacy_version = _integer(source.get("version", 1), "version")
    except ValueError as exc:
        raise StateMigrationError(str(exc)) from exc
    if legacy_version not in _LEGACY_CHECKPOINT_VERSIONS:
        raise StateMigrationError(f"unsupported checkpoint version: {legacy_version}")
    try:
        return _migrate_legacy(
            source,
            legacy_version=legacy_version,
            agent_fingerprint=agent_fingerprint,
            runtime_version=runtime_version or current_runtime_version(),
        )
    except StateMigrationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise StateMigrationError(str(exc)) from exc


def flatten_plan_engine_state(
    value: Mapping[str, Any],
    markers: FinalizationMarkers,
) -> dict[str, Any]:
    """Adapt v4 nested Plan state to the flat 0.3 compatibility facade."""

    state = _mapping_copy(value, "Plan engine state")
    progress = _mapping_copy(state.get("plan_progress", {}), "plan_progress")
    step = _mapping_copy(state.get("step_execution", {}), "step_execution")
    recovery = _mapping_copy(state.get("tool_recovery", {}), "tool_recovery")
    finalization = _mapping_copy(state.get("finalization", {}), "finalization")

    plan = _mapping_copy(progress.get("plan", {}), "plan")
    current_step_id = step.get("current_step_id")
    step_attempt_count = _integer(
        step.get("step_attempt_count", 0),
        "step_attempt_count",
    )
    if current_step_id is not None:
        raw_steps = plan.get("steps", [])
        if not isinstance(raw_steps, list):
            raise StateMigrationError("plan steps must be an array")
        matched = False
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                raise StateMigrationError("plan steps must contain objects")
            if str(raw_step.get("step_id", "")) == str(current_step_id):
                existing = _integer(
                    raw_step.get("attempt_count", 0),
                    "plan step attempt_count",
                )
                if existing != step_attempt_count:
                    raise StateMigrationError(
                        "step_attempt_count does not match the current Plan step"
                    )
                matched = True
                break
        if not matched:
            raise StateMigrationError("current_step_id does not exist in the plan")

    phase = str(state.get("phase", "plan"))
    if phase in {"act_tool", "tool_recovery"}:
        phase = "act"

    # The outer markers are authoritative. A mirrored engine value may be
    # present for engine-local diagnostics, but a disagreement is corruption.
    mirrored = _markers_from_plan_finalization(finalization)
    if mirrored != markers:
        raise StateMigrationError(
            "Plan finalization state does not match outer finalization markers"
        )

    return {
        "phase": phase,
        "plan": plan,
        "current_step_id": (None if current_step_id is None else str(current_step_id)),
        "committed_results": _mapping_copy(
            progress.get("committed_results", {}),
            "committed_results",
        ),
        "pending_step_result": copy.deepcopy(step.get("pending_step_result")),
        "validation_error": (
            None
            if step.get("validation_feedback") is None
            else str(step["validation_feedback"])
        ),
        "awaiting_step_result": bool(step.get("awaiting_step_result", False)),
        "replan_count": _integer(
            progress.get("replan_count", 0),
            "replan_count",
        ),
        "finalization_count": _integer(
            finalization.get("invocation_count", 0),
            "finalization invocation_count",
        ),
        "final_response": copy.deepcopy(markers.response),
        "final_persisted": markers.persisted,
        "final_emitted": markers.emitted,
        "tool_repair_counts": _mapping_copy(
            recovery.get("repair_count_by_step", {}),
            "repair_count_by_step",
        ),
        "pending_tool_failure": copy.deepcopy(recovery.get("pending_failure")),
        "total_tool_repairs": _integer(
            recovery.get("total_repairs", 0),
            "total_repairs",
        ),
        "failure": copy.deepcopy(recovery.get("terminal_failure")),
        "active_tool_calls": _mapping_copy(
            recovery.get("active_calls", {}),
            "active_calls",
        ),
        "seen_tool_call_ids": _string_array(
            recovery.get("seen_call_ids", ()),
            "seen_call_ids",
        ),
    }


def _migrate_legacy(
    source: dict[str, Any],
    *,
    legacy_version: int,
    agent_fingerprint: str,
    runtime_version: str,
) -> RunSnapshot:
    run_id = _required_identifier(source, "run_id")
    session_id = _required_identifier(source, "session_id")
    messages = _mapping_array(source.get("messages", ()), "messages")
    new_messages = _mapping_array(source.get("new_messages", ()), "new_messages")
    internal_messages = _mapping_array(
        source.get("internal_messages", ()),
        "internal_messages",
    )
    raw_policy_state = _mapping_copy(source.get("policy_state", {}), "policy_state")

    execution_state = _legacy_execution_state(
        source,
        raw_policy_state,
        legacy_version=legacy_version,
    )
    # This private compatibility carrier may be present in 0.3-era runtime
    # checkpoints. v4 stores the canonical opaque state only in ``engine``.
    raw_policy_state.pop("_moduagent_engine_snapshot", None)
    if execution_state is not None:
        markers = _legacy_plan_markers(execution_state)
        engine_state = _plan_state_v3_to_v4(execution_state, markers)
        engine = EngineSnapshot("plan", 1, engine_state)
        raw_policy_state.pop("execution_state", None)
        raw_policy_state.pop("plan", None)
        resume_safety = _plan_resume_safety(execution_state)
        terminal_reason = _plan_terminal_reason(execution_state)
    else:
        markers = _legacy_standard_markers(raw_policy_state)
        raw_policy_state.pop(_STANDARD_FINALIZATION_STATE_KEY, None)
        raw_policy_state.pop(_STANDARD_FINALIZATION_OUTPUT_KEY, None)
        status = str(source.get("status", "created"))
        phase = "done" if markers.emitted else "finalize" if markers.started else "act"
        engine = EngineSnapshot(
            "standard",
            1,
            {
                "phase": phase,
                "model_turn": _integer(source.get("step", 0), "step"),
                "tool_call_count": _integer(
                    source.get("tool_call_count", 0),
                    "tool_call_count",
                ),
                "finalization": {
                    **markers.to_dict(),
                    "invocation_count": (1 if markers.response_generated else 0),
                },
            },
        )
        # A legacy FAILED status records an interrupted run, not an Engine
        # terminal state. It must remain resumable so the failed operation can
        # be retried under the 0.4 Engine.
        resume_safety = "terminal" if status == "completed" else "resumable"
        terminal_reason = None

    request = {
        "input": str(source.get("input", "")),
        "user_context": _mapping_copy(
            source.get("user_context", {}),
            "user_context",
        ),
        "requested_skills": _string_array(
            source.get("requested_skills", ()),
            "requested_skills",
        ),
        "skill_mode": str(source.get("skill_mode", "disabled")),
    }
    usage = _usage(source.get("usage", {}))
    current_run_start = _integer(
        source.get("current_run_start", min(1, len(messages))),
        "current_run_start",
    )
    common = CommonRunState(
        request=request,
        messages=messages,
        new_messages=new_messages,
        internal_messages=internal_messages,
        status=str(source.get("status", "created")),
        step=_integer(source.get("step", 0), "step"),
        tool_call_count=_integer(
            source.get("tool_call_count", 0),
            "tool_call_count",
        ),
        usage=usage,
        current_run_start=current_run_start,
        compatibility_policy_state=raw_policy_state,
        terminal_reason=terminal_reason,
        resume_safety=resume_safety,
        event_sequence=_integer(
            source.get("event_sequence", 0),
            "event_sequence",
        ),
    )
    raw_skill_state = source.get("skill_state", {}) if legacy_version >= 2 else {}
    skill_state = _mapping_copy(raw_skill_state, "skill_state")
    raw_metadata = _mapping_copy(source.get("metadata", {}), "metadata")
    for canonical_key in (
        "_moduagent_engine",
        "_moduagent_engine_id",
        "_moduagent_engine_state_version",
        "_moduagent_agent_fingerprint",
        "_moduagent_runtime_version",
        "_moduagent_resume_safety",
        "_moduagent_event_sequence",
        "_moduagent_terminal_reason",
    ):
        raw_metadata.pop(canonical_key, None)
    metadata = _sanitize_runtime_metadata(raw_metadata)
    created_at = _timestamp(source.get("created_at"))
    updated_at = _timestamp(source.get("updated_at"))
    if updated_at < created_at:
        # Early checkpoints sometimes supplied only one timestamp. Preserve a
        # coherent envelope instead of creating an impossible ordering.
        updated_at = created_at
    return RunSnapshot(
        runtime_version=runtime_version,
        run_id=run_id,
        session_id=session_id,
        agent_fingerprint=agent_fingerprint,
        engine=engine,
        common_state=common,
        finalization_markers=markers,
        skill_state=skill_state,
        sanitized_runtime_metadata=metadata,
        created_at=created_at,
        updated_at=updated_at,
        migrated_from_schema_version=legacy_version,
    )


def _legacy_execution_state(
    source: Mapping[str, Any],
    policy_state: Mapping[str, Any],
    *,
    legacy_version: int,
) -> dict[str, Any] | None:
    if legacy_version < 3:
        return None
    top_level = source.get("execution_state")
    policy_copy = policy_state.get("execution_state")
    if top_level is not None and not isinstance(top_level, Mapping):
        raise StateMigrationError("execution_state must be an object")
    if policy_copy is not None and not isinstance(policy_copy, Mapping):
        raise StateMigrationError("policy execution_state must be an object")
    if top_level is not None and policy_copy is not None:
        if _canonical(top_level) != _canonical(policy_copy):
            raise StateMigrationError(
                "checkpoint contains inconsistent duplicate execution_state"
            )
    selected = top_level if top_level is not None else policy_copy
    if selected is None:
        return None
    result = _mapping_copy(selected, "execution_state")
    duplicate_plan = policy_state.get("plan")
    state_plan = result.get("plan")
    if duplicate_plan is not None:
        if not isinstance(duplicate_plan, Mapping):
            raise StateMigrationError("policy plan must be an object")
        if not isinstance(state_plan, Mapping) or _canonical(
            duplicate_plan
        ) != _canonical(state_plan):
            raise StateMigrationError("checkpoint contains inconsistent duplicate plan")
    _validate_v3_plan_state(result)
    return result


def _plan_state_v3_to_v4(
    state: Mapping[str, Any],
    markers: FinalizationMarkers,
) -> dict[str, Any]:
    plan = _mapping_copy(state.get("plan", {}), "plan")
    current_step_id = state.get("current_step_id")
    step_attempt_count = _current_step_attempt_count(plan, current_step_id)
    pending_failure = _migrate_safe_failure(
        state.get("pending_tool_failure"),
        "pending_tool_failure",
    )
    phase = str(state.get("phase", "plan"))
    if phase == "act" and pending_failure is not None:
        phase = "tool_recovery"

    return {
        "phase": phase,
        "plan_progress": {
            "plan": plan,
            "committed_results": _mapping_copy(
                state.get("committed_results", {}),
                "committed_results",
            ),
            "replan_count": _integer(
                state.get("replan_count", 0),
                "replan_count",
            ),
        },
        "step_execution": {
            "current_step_id": (
                None if current_step_id is None else str(current_step_id)
            ),
            "pending_step_result": copy.deepcopy(state.get("pending_step_result")),
            "validation_feedback": (
                None
                if state.get("validation_error") is None
                else str(state["validation_error"])
            ),
            "step_attempt_count": step_attempt_count,
            "awaiting_step_result": bool(state.get("awaiting_step_result", False)),
        },
        "tool_recovery": {
            "active_calls": _migrate_active_calls(
                state.get("active_tool_calls", {}),
                "active_tool_calls",
            ),
            "seen_call_ids": _string_array(
                state.get("seen_tool_call_ids", ()),
                "seen_tool_call_ids",
            ),
            "pending_failure": pending_failure,
            "repair_count_by_step": _mapping_copy(
                state.get("tool_repair_counts", {}),
                "tool_repair_counts",
            ),
            "total_repairs": _integer(
                state.get("total_tool_repairs", 0),
                "total_tool_repairs",
            ),
            "terminal_failure": _migrate_safe_failure(
                state.get("failure"),
                "failure",
            ),
        },
        # This engine-local mirror is retained because Plan codecs use it when
        # validating phase invariants. The envelope markers remain authoritative.
        "finalization": {
            "response": copy.deepcopy(markers.response),
            "invocation_count": _integer(
                state.get("finalization_count", 0),
                "finalization_count",
            ),
            "started": markers.started,
            "response_generated": markers.response_generated,
            "persisted": markers.persisted,
            "emitted": markers.emitted,
        },
    }


def _migrate_active_calls(value: Any, name: str) -> dict[str, dict[str, str]]:
    """Migrate only non-secret Tool invocation identity and fingerprints."""

    if not isinstance(value, Mapping):
        raise StateMigrationError(f"{name} must be an object")
    result: dict[str, dict[str, str]] = {}
    for raw_call_id, raw_call in value.items():
        call_id = str(raw_call_id)
        if not call_id.strip():
            raise StateMigrationError("active Tool call identity cannot be empty")
        if not isinstance(raw_call, Mapping):
            raise StateMigrationError("active Tool calls must contain objects")
        tool_name = str(raw_call.get("tool_name", ""))
        if not tool_name.strip():
            raise StateMigrationError("active Tool call identity cannot be empty")
        fingerprint = raw_call.get("arguments_fingerprint")
        _validate_fingerprint(
            fingerprint,
            "active Tool call arguments_fingerprint",
        )
        result[call_id] = {
            "tool_name": tool_name,
            "arguments_fingerprint": str(fingerprint),
        }
    return result


def _validate_v3_plan_state(state: Mapping[str, Any]) -> None:
    phase = str(state.get("phase", "plan"))
    if phase not in {
        "plan",
        "step_prepare",
        "act",
        "step_validate",
        "verify",
        "finalize",
        "done",
        "failed",
    }:
        raise StateMigrationError(f"unsupported legacy Plan phase: {phase}")
    plan = _mapping_copy(state.get("plan", {}), "plan")
    current_step_id = state.get("current_step_id")
    if current_step_id is not None:
        _current_step_attempt_count(plan, current_step_id)

    pending_failure = state.get("pending_tool_failure")
    if pending_failure is not None and not isinstance(pending_failure, Mapping):
        raise StateMigrationError("pending_tool_failure must be an object")
    terminal_failure = state.get("failure")
    if terminal_failure is not None and not isinstance(terminal_failure, Mapping):
        raise StateMigrationError("failure must be an object")

    active_calls = _mapping_copy(
        state.get("active_tool_calls", {}),
        "active_tool_calls",
    )
    seen_call_ids = _string_array(
        state.get("seen_tool_call_ids", ()),
        "seen_tool_call_ids",
    )
    if any(not call_id.strip() for call_id in seen_call_ids):
        raise StateMigrationError("seen Tool call IDs cannot be empty")
    if len(set(seen_call_ids)) != len(seen_call_ids):
        raise StateMigrationError("seen Tool call IDs cannot contain duplicates")
    if not set(active_calls).issubset(seen_call_ids):
        raise StateMigrationError("active Tool calls must be present in seen IDs")
    for call_id, raw_call in active_calls.items():
        if not isinstance(raw_call, Mapping):
            raise StateMigrationError("active Tool calls must contain objects")
        if not str(call_id).strip() or not str(raw_call.get("tool_name", "")).strip():
            raise StateMigrationError("active Tool call identity cannot be empty")
        _validate_fingerprint(
            raw_call.get("arguments_fingerprint"),
            "active Tool call arguments_fingerprint",
        )

    if pending_failure is not None:
        call_id = str(pending_failure.get("call_id", "")).strip()
        tool_name = str(pending_failure.get("tool_name", "")).strip()
        if not call_id or not tool_name:
            raise StateMigrationError("pending Tool failure identity cannot be empty")
        if call_id not in seen_call_ids:
            raise StateMigrationError(
                "pending Tool failure call must be present in seen IDs"
            )
        if active_calls:
            raise StateMigrationError(
                "pending Tool repair cannot contain ambiguous active Tool calls"
            )
        for field_name in (
            "arguments_fingerprint",
            "invocation_fingerprint",
        ):
            if pending_failure.get(field_name) is not None:
                _validate_fingerprint(
                    pending_failure[field_name],
                    f"pending Tool failure {field_name}",
                )


def _migrate_safe_failure(value: Any, name: str) -> dict[str, Any] | None:
    """Whitelist only data proven safe at the v4 recovery boundary."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise StateMigrationError(f"{name} must be an object")
    result: dict[str, Any] = {}
    for field_name in _SAFE_FAILURE_IDENTIFIER_FIELDS:
        raw = value.get(field_name)
        if raw is not None:
            bounded = _bounded_text(raw, limit=256)
            if bounded:
                result[field_name] = bounded
    for field_name in _SAFE_FAILURE_CODE_FIELDS:
        raw = value.get(field_name)
        if raw is None:
            continue
        code = str(raw).strip()
        if _SAFE_CODE_PATTERN.fullmatch(code):
            result[field_name] = code
    for field_name in _SAFE_FAILURE_BOOL_FIELDS:
        if field_name in value:
            result[field_name] = bool(value[field_name])
    for field_name in _SAFE_FAILURE_COUNT_FIELDS:
        if field_name in value:
            result[field_name] = min(
                _integer(value[field_name], f"{name} {field_name}"),
                1_000_000,
            )
    for field_name in _SAFE_FAILURE_FINGERPRINT_FIELDS:
        raw = value.get(field_name)
        if raw is None:
            continue
        _validate_fingerprint(raw, f"{name} {field_name}")
        result[field_name] = str(raw)
    return result or None


def _validate_fingerprint(value: Any, name: str) -> None:
    text = str(value)
    digest = text.removeprefix("sha256:")
    if not (
        text.startswith("sha256:")
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        raise StateMigrationError(f"{name} must use sha256")


def _legacy_plan_markers(state: Mapping[str, Any]) -> FinalizationMarkers:
    response = copy.deepcopy(state.get("final_response"))
    phase = str(state.get("phase", "plan"))
    started = (
        phase in {"finalize", "done"}
        or _integer(state.get("finalization_count", 0), "finalization_count") > 0
        or response is not None
    )
    return FinalizationMarkers(
        started=started,
        response_generated=response is not None,
        response=response,
        persisted=bool(state.get("final_persisted", False)),
        emitted=bool(state.get("final_emitted", False)),
    )


def _legacy_standard_markers(
    policy_state: Mapping[str, Any],
) -> FinalizationMarkers:
    phase = policy_state.get(_STANDARD_FINALIZATION_STATE_KEY)
    response = copy.deepcopy(policy_state.get(_STANDARD_FINALIZATION_OUTPUT_KEY))
    started = phase in {"pending", "completed"} or response is not None
    return FinalizationMarkers(
        started=started,
        response_generated=response is not None,
        response=response,
        persisted=False,
        emitted=False,
    )


def _markers_from_plan_finalization(
    value: Mapping[str, Any],
) -> FinalizationMarkers:
    response = copy.deepcopy(value.get("response"))
    return FinalizationMarkers(
        started=bool(value.get("started", False)),
        response_generated=bool(value.get("response_generated", response is not None)),
        response=response,
        persisted=bool(value.get("persisted", False)),
        emitted=bool(value.get("emitted", False)),
    )


def _plan_resume_safety(state: Mapping[str, Any]) -> str:
    phase = str(state.get("phase", "plan"))
    if phase in {"done", "failed"}:
        return "terminal"
    active_calls = state.get("active_tool_calls", {})
    if isinstance(active_calls, Mapping) and active_calls:
        return "manual_required"
    # A v3 pending failure can describe a mixed batch in which another Tool
    # already succeeded. Resuming that state at the repair turn could repeat or
    # supersede an externally visible side effect, so retain the sanitized
    # diagnostic state but never classify it as automatically resumable.
    for failure_name in ("pending_tool_failure", "failure"):
        failure = state.get(failure_name)
        if not isinstance(failure, Mapping):
            continue
        success_count = _integer(
            failure.get("success_count", 0),
            f"{failure_name} success_count",
        )
        if success_count > 0:
            return "manual_required"
    return "resumable"


def _plan_terminal_reason(state: Mapping[str, Any]) -> str | None:
    if str(state.get("phase", "")) not in {"done", "failed"}:
        return None
    failure = state.get("failure")
    if not isinstance(failure, Mapping):
        return None
    for key in ("reason", "error_type", "type"):
        value = failure.get(key)
        if value is not None and _SAFE_CODE_PATTERN.fullmatch(str(value).strip()):
            return str(value).strip()
    return "plan_execution_failed"


def _current_step_attempt_count(
    plan: Mapping[str, Any],
    current_step_id: Any,
) -> int:
    if current_step_id is None:
        return 0
    raw_steps = plan.get("steps", ())
    if not isinstance(raw_steps, (list, tuple)):
        raise StateMigrationError("plan steps must be an array")
    for step in raw_steps:
        if not isinstance(step, Mapping):
            raise StateMigrationError("plan steps must contain objects")
        if str(step.get("step_id", "")) == str(current_step_id):
            return _integer(step.get("attempt_count", 0), "step attempt_count")
    raise StateMigrationError("current_step_id does not exist in the plan")


def _usage(value: Any) -> dict[str, Any]:
    raw = _mapping_copy(value, "usage")
    return {
        "input_tokens": _integer(raw.get("input_tokens", 0), "input_tokens"),
        "output_tokens": _integer(raw.get("output_tokens", 0), "output_tokens"),
        "total_tokens": _integer(raw.get("total_tokens", 0), "total_tokens"),
        "provider": _mapping_copy(raw.get("provider", {}), "usage provider"),
    }


def _sanitize_runtime_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for raw_key, nested in item.items():
                key = str(raw_key)
                if key == "_moduagent_tool_trace":
                    result[key] = _checkpoint_tool_trace(nested)
                else:
                    result[key] = "[REDACTED]" if _is_sensitive(key) else visit(nested)
            return result
        if isinstance(item, (list, tuple)):
            return [visit(nested) for nested in item]
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise StateMigrationError("runtime metadata must contain only JSON-safe values")

    sanitized = visit(value)
    if not isinstance(sanitized, dict):
        raise StateMigrationError("runtime metadata must be an object")
    # Verify NaN/Infinity and non-standard numeric values are not retained.
    try:
        json.dumps(sanitized, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StateMigrationError(
            "runtime metadata must contain only JSON-safe values"
        ) from exc
    return sanitized


def _checkpoint_tool_trace(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise StateMigrationError("Tool trace must be an array")
    projected: list[dict[str, Any]] = []
    for raw_entry in value:
        if not isinstance(raw_entry, Mapping):
            raise StateMigrationError("Tool trace entries must be objects")
        entry: dict[str, Any] = {
            "step_id": (
                None
                if raw_entry.get("step_id") is None
                else _bounded_text(raw_entry["step_id"], limit=256)
            ),
            "call_id": _bounded_text(raw_entry.get("call_id", ""), limit=256),
            "tool_name": _bounded_text(raw_entry.get("tool_name", ""), limit=256),
            "success": bool(raw_entry.get("success", False)),
            "attempts": min(
                _integer(raw_entry.get("attempts", 0), "Tool trace attempts"),
                1_000_000,
            ),
            "duration_seconds": float(raw_entry.get("duration_seconds", 0.0)),
            "error": _checkpoint_tool_trace_error(raw_entry.get("error")),
        }
        recovery_of = raw_entry.get("recovery_of_call_id")
        if recovery_of is not None:
            entry["recovery_of_call_id"] = _bounded_text(recovery_of, limit=256)
        fingerprint = raw_entry.get("arguments_fingerprint")
        raw_arguments = raw_entry.get("arguments")
        if isinstance(raw_arguments, Mapping):
            fingerprint = fingerprint_tool_arguments(raw_arguments)
        if fingerprint is not None:
            _validate_fingerprint(
                fingerprint,
                "Tool trace arguments_fingerprint",
            )
            entry["arguments_fingerprint"] = str(fingerprint)
        projected.append(entry)
    return projected


def _checkpoint_tool_trace_error(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise StateMigrationError("Tool trace error must be an object")
    error_type = str(value.get("type", "execution_error"))
    if _SAFE_CODE_PATTERN.fullmatch(error_type) is None:
        error_type = "execution_error"
    result: dict[str, Any] = {
        "type": error_type,
        "retryable": bool(value.get("retryable", False)),
    }
    for field_name in ("reason", "recovery"):
        raw = value.get(field_name)
        if raw is None:
            continue
        code = str(raw)
        if _SAFE_CODE_PATTERN.fullmatch(code) is not None:
            result[field_name] = code
    return result


def _is_sensitive(key: str) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
        or normalized.endswith("_api_key")
        or normalized.endswith("_private_key")
    )


def _mapping_copy(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StateMigrationError(f"{name} must be an object")
    copied = copy.deepcopy(dict(value))
    try:
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StateMigrationError(f"{name} must contain JSON-safe values") from exc
    return copied


def _mapping_array(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        (list, tuple),
    ):
        raise StateMigrationError(f"{name} must be an array")
    if not all(isinstance(item, Mapping) for item in value):
        raise StateMigrationError(f"{name} must contain objects")
    return tuple(_mapping_copy(item, f"{name} item") for item in value)


def _string_array(value: Any, name: str) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        (list, tuple),
    ):
        raise StateMigrationError(f"{name} must be an array")
    return [str(item) for item in value]


def _required_identifier(value: Mapping[str, Any], name: str) -> str:
    result = str(value.get(name, ""))
    if not result.strip():
        raise StateMigrationError(f"{name} cannot be empty")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result


def _timestamp(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise StateMigrationError("checkpoint timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise StateMigrationError(
            "duplicate checkpoint state must contain JSON-safe values"
        ) from exc


def _bounded_text(value: Any, *, limit: int = 512) -> str:
    text = " ".join(
        "".join(
            character if character.isprintable() else " " for character in str(value)
        ).split()
    )
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


__all__ = [
    "StateMigrationError",
    "flatten_plan_engine_state",
    "migrate_checkpoint_payload",
]

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from moduagent.execution.state import (
    EngineSnapshot as ExecutionEngineSnapshot,
)
from moduagent.execution.standard import (
    StandardExecutionPhase,
    StandardStateCodec,
)
from moduagent.messages import FinishReason, Message, Usage
from moduagent.observability.sinks import event_to_dict
from moduagent.persistence import (
    EngineSnapshot,
    FinalizationMarkers,
    InMemoryCheckpointStore,
    RedisCheckpointStore,
    RunCheckpoint,
    RunSnapshot,
    SNAPSHOT_RUNTIME_VERSION,
    StateMigrationError,
    migrate_checkpoint_payload,
)
from moduagent.persistence.snapshot import current_runtime_version
from moduagent.runtime.context import AgentResult, RunContext, RunRequest
from moduagent.runtime.events import (
    AgentEvent,
    EventPublisher,
    EventType,
    EventVisibility,
)


def _plan_state(*, pending_repair: bool = False) -> dict[str, object]:
    fingerprint = "sha256:" + ("a" * 64)
    return {
        "phase": "act" if pending_repair else "step_prepare",
        "plan": {
            "steps": [
                {
                    "step_id": "lookup",
                    "objective": "look up a record",
                    "description": "look up a record",
                    "completion_criteria": ["a verified record is available"],
                    "expected_output": "record",
                    "dependencies": [],
                    "allowed_tools": ["lookup"],
                    "status": "in_progress" if pending_repair else "pending",
                    "attempt_count": 1,
                    "result_ref": None,
                    "metadata": {},
                }
            ],
            "current_index": 0,
            "version": 1,
        },
        "current_step_id": "lookup",
        "committed_results": {},
        "pending_step_result": None,
        "validation_error": (
            "Tool lookup failed (invalid_filter)" if pending_repair else None
        ),
        "awaiting_step_result": False,
        "replan_count": 0,
        "finalization_count": 0,
        "final_response": None,
        "final_persisted": False,
        "final_emitted": False,
        "tool_repair_counts": {"lookup": 1} if pending_repair else {},
        "pending_tool_failure": (
            {
                "step_id": "lookup",
                "call_id": "call-1",
                "tool_name": "lookup",
                "error_type": "execution_error",
                "reason": "invalid_filter",
                "recovery": "repair_call",
                "retryable": False,
                "repair_safe": True,
                "feedback": "Tool lookup failed (invalid_filter)",
                "arguments_fingerprint": fingerprint,
                "invocation_fingerprint": fingerprint,
            }
            if pending_repair
            else None
        ),
        "total_tool_repairs": 1 if pending_repair else 0,
        "failure": None,
        "active_tool_calls": {},
        "seen_tool_call_ids": ["call-1"] if pending_repair else [],
    }


def _v3_payload(*, pending_repair: bool = False) -> dict[str, object]:
    state = _plan_state(pending_repair=pending_repair)
    return {
        "version": 3,
        "run_id": "run-v3",
        "session_id": "session-v3",
        "input": "find it",
        "user_context": {"tenant": "company"},
        "requested_skills": [],
        "skill_mode": "disabled",
        "messages": [Message.user("find it").to_dict()],
        "new_messages": [Message.user("find it").to_dict()],
        "internal_messages": [],
        "execution_state": copy.deepcopy(state),
        "step": 1,
        "tool_call_count": 1,
        "status": "running",
        "policy_state": {
            "execution_state": copy.deepcopy(state),
            "plan": copy.deepcopy(state["plan"]),
        },
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "provider": {},
        },
        "metadata": {
            "trace_id": "trace-1",
            "nested": {"api_key": "must-not-survive"},
        },
        "current_run_start": 0,
        "skill_state": {},
        "created_at": "2026-07-29T00:00:00+00:00",
        "updated_at": "2026-07-29T00:00:01+00:00",
    }


def test_v4_snapshot_has_dual_version_guard_and_shared_engine_contract() -> None:
    checkpoint = RunCheckpoint(
        run_id="run-1",
        session_id="session-1",
        messages=(Message.user("hello"),),
        current_run_start=0,
    )

    payload = checkpoint.to_dict()
    restored = RunSnapshot.from_json(checkpoint.to_json())

    assert payload["schema_version"] == 4
    assert payload["version"] == 4
    assert "execution_state" not in payload
    assert restored.engine.engine_id == "standard"
    assert EngineSnapshot is ExecutionEngineSnapshot
    assert isinstance(restored.engine, ExecutionEngineSnapshot)


def test_snapshot_runtime_version_tracks_the_source_release() -> None:
    assert SNAPSHOT_RUNTIME_VERSION == "0.5.2"
    assert current_runtime_version() == SNAPSHOT_RUNTIME_VERSION


def test_v052_reads_a_v051a1_snapshot_envelope_without_rewriting_its_origin() -> None:
    checkpoint = RunCheckpoint(
        run_id="run-v051",
        session_id="session-v051",
        messages=(Message.user("resume"),),
        current_run_start=0,
    )
    payload = checkpoint.to_dict()
    payload["runtime_version"] = "0.5.1a1"

    restored = RunSnapshot.from_json(json.dumps(payload))

    assert restored.runtime_version == "0.5.1a1"
    assert restored.schema_version == 4


def test_event_wire_projection_is_finite_and_never_uses_opaque_repr() -> None:
    class SecretRepr:
        def __repr__(self) -> str:
            return "password=TOPSECRET"

    event = AgentEvent(
        EventType.POLICY_DECISION,
        "run-json-safe",
        {
            "not_a_number": float("nan"),
            "positive_infinity": float("inf"),
            "opaque": SecretRepr(),
        },
    )

    payload = event.to_dict()
    encoded = json.dumps(payload, allow_nan=False)

    assert payload["data"]["not_a_number"] is None
    assert payload["data"]["positive_infinity"] is None
    assert payload["data"]["opaque"]["unsupported_type"].endswith(".SecretRepr")
    assert "TOPSECRET" not in encoded


def test_run_creation_timestamp_survives_repeated_checkpoints_and_resume() -> None:
    created_at = datetime(2026, 7, 1, 12, 30, tzinfo=timezone.utc)
    context = RunContext(
        run_id="run-created-at",
        request=RunRequest("hello", "session-created-at"),
        messages=[Message.user("hello")],
        created_at=created_at,
    )

    first = RunCheckpoint.from_context(context)
    second = RunCheckpoint.from_context(context)
    restored = first.to_context()

    assert first.created_at == second.created_at == created_at
    assert restored.created_at == created_at


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"version": 3}, "must match"),
        ({"schema_version": 5, "version": 5}, "unsupported snapshot"),
    ],
)
def test_v4_snapshot_rejects_version_mismatch_and_future_schema(
    updates: dict[str, int],
    match: str,
) -> None:
    payload = RunCheckpoint(
        run_id="run-version",
        session_id="session-version",
        messages=(),
    ).to_dict()
    payload.update(updates)

    with pytest.raises((ValueError, StateMigrationError), match=match):
        RunSnapshot.from_dict(payload)


def test_v3_plan_migration_is_copy_only_nested_and_repair_resumable() -> None:
    source = _v3_payload(pending_repair=True)
    before = copy.deepcopy(source)

    snapshot = migrate_checkpoint_payload(
        source,
        agent_fingerprint="sha256:resolved-agent",
        runtime_version="0.4.0",
    )

    assert source == before
    assert snapshot.engine.engine_id == "plan"
    assert snapshot.engine.state_version == 1
    assert snapshot.engine.state["phase"] == "tool_recovery"
    assert (
        snapshot.engine.state["plan_progress"]["plan"]["steps"][0]["step_id"]
        == "lookup"
    )
    assert snapshot.engine.state["step_execution"]["step_attempt_count"] == 1
    recovery = snapshot.engine.state["tool_recovery"]
    assert recovery["pending_failure"]["call_id"] == "call-1"
    assert recovery["repair_count_by_step"] == {"lookup": 1}
    assert snapshot.common_state.resume_safety == "resumable"
    assert snapshot.finalization_markers == FinalizationMarkers()
    assert snapshot.sanitized_runtime_metadata["nested"]["api_key"] == "[REDACTED]"

    facade = RunCheckpoint.from_snapshot(snapshot)
    assert facade.execution_state["phase"] == "act"
    assert facade.execution_state["pending_tool_failure"]["call_id"] == "call-1"
    assert facade.to_context().execution_state.phase.value == "act"


def test_v3_migration_drops_untrusted_failure_payloads() -> None:
    source = _v3_payload(pending_repair=True)
    state = source["execution_state"]
    assert isinstance(state, dict)
    pending = state["pending_tool_failure"]
    assert isinstance(pending, dict)
    pending.update(
        {
            "message": "SQL select * from customers password=TOPSECRET",
            "feedback": "backend DSN=private",
            "raw_query": "select SSN from customers",
            "details": {"password": "TOPSECRET"},
            "raw_backend": "postgres://admin:secret@db",
        }
    )
    state["failure"] = {
        "reason": "execution_error",
        "terminal_reason": "password=TOPSECRET",
        "raw_backend": "private-backend",
    }
    policy_state = source["policy_state"]
    assert isinstance(policy_state, dict)
    policy_state["execution_state"] = copy.deepcopy(state)
    policy_state["plan"] = copy.deepcopy(state["plan"])

    snapshot = migrate_checkpoint_payload(source)
    serialized = snapshot.to_json()
    recovery = snapshot.engine.state["tool_recovery"]

    assert recovery["pending_failure"]["reason"] == "invalid_filter"
    assert recovery["terminal_failure"]["reason"] == "execution_error"
    assert {
        "message",
        "feedback",
        "raw_query",
        "details",
        "raw_backend",
        "terminal_reason",
    }.isdisjoint(recovery["pending_failure"])
    assert "TOPSECRET" not in serialized
    assert "select SSN" not in serialized
    assert "private-backend" not in serialized


def test_checkpoint_tool_trace_fingerprints_arguments_without_copying_them() -> None:
    source = _v3_payload()
    source["metadata"]["_moduagent_tool_trace"] = [
        {
            "step_id": "lookup",
            "call_id": "call-secret",
            "tool_name": "lookup",
            "success": True,
            "attempts": 1,
            "duration_seconds": 0.1,
            "error": None,
            "arguments": {"query": "customer-SSN-123"},
            "arguments_source": "validated",
        }
    ]

    snapshot = migrate_checkpoint_payload(source)
    trace = snapshot.sanitized_runtime_metadata["_moduagent_tool_trace"]

    assert "customer-SSN-123" not in snapshot.to_json()
    assert trace[0]["arguments_fingerprint"].startswith("sha256:")
    assert "arguments" not in trace[0]


def test_v3_standard_migration_restores_budget_and_finalization_state() -> None:
    payload = {
        "version": 3,
        "run_id": "run-standard-v3",
        "session_id": "session-standard-v3",
        "input": "finish it",
        "user_context": {},
        "requested_skills": [],
        "skill_mode": "disabled",
        "messages": [],
        "new_messages": [],
        "internal_messages": [],
        "execution_state": None,
        "step": 5,
        "tool_call_count": 9,
        "status": "running",
        "policy_state": {
            "custom_cursor": "keep-me",
            "_moduagent_structured_finalization": "completed",
            "_moduagent_structured_output": "already-final",
        },
        "usage": {},
        "metadata": {},
        "current_run_start": 0,
        "skill_state": {},
    }

    snapshot = migrate_checkpoint_payload(payload)
    state = StandardStateCodec().decode(snapshot.engine.state)
    restored = RunCheckpoint.from_snapshot(snapshot).to_context()

    assert state.phase is StandardExecutionPhase.FINALIZE
    assert state.model_turn == 5
    assert state.tool_call_count == 9
    assert state.finalization_response == "already-final"
    assert snapshot.finalization_markers.response == "already-final"
    assert restored.policy_state["custom_cursor"] == "keep-me"


def test_v3_standard_failed_checkpoint_remains_resumable_and_deduplicated() -> None:
    payload = {
        "version": 3,
        "run_id": "run-standard-retry",
        "session_id": "session-standard-retry",
        "input": "retry it",
        "user_context": {},
        "requested_skills": [],
        "skill_mode": "disabled",
        "messages": [],
        "new_messages": [],
        "internal_messages": [],
        "execution_state": None,
        "step": 1,
        "tool_call_count": 0,
        "status": "failed",
        "policy_state": {
            "custom_cursor": 3,
            "_moduagent_engine_snapshot": {
                "engine_id": "standard",
                "state_version": 1,
                "state": {"phase": "act"},
            },
        },
        "usage": {},
        "metadata": {},
        "current_run_start": 0,
        "skill_state": {},
    }

    snapshot = migrate_checkpoint_payload(payload)

    assert snapshot.common_state.resume_safety == "resumable"
    assert snapshot.engine.state["phase"] == "act"
    assert snapshot.common_state.compatibility_policy_state == {"custom_cursor": 3}


def test_v3_duplicate_execution_state_and_plan_mismatch_fail_closed() -> None:
    execution_mismatch = _v3_payload()
    execution_mismatch["policy_state"]["execution_state"]["phase"] = "failed"

    with pytest.raises(StateMigrationError, match="duplicate execution_state"):
        migrate_checkpoint_payload(execution_mismatch)

    plan_mismatch = _v3_payload()
    plan_mismatch["policy_state"]["plan"]["version"] = 99

    with pytest.raises(StateMigrationError, match="duplicate plan"):
        migrate_checkpoint_payload(plan_mismatch)


def test_v3_inflight_tool_state_is_classified_manual_required() -> None:
    payload = _v3_payload()
    fingerprint = "sha256:" + ("b" * 64)
    state = payload["execution_state"]
    state["active_tool_calls"] = {
        "call-live": {
            "tool_name": "lookup",
            "arguments_fingerprint": fingerprint,
            "arguments": {"password": "do-not-migrate"},
            "raw_query": "select secret from private_table",
        }
    }
    state["seen_tool_call_ids"] = ["call-live"]
    payload["policy_state"]["execution_state"] = copy.deepcopy(state)

    snapshot = migrate_checkpoint_payload(payload)

    assert snapshot.common_state.resume_safety == "manual_required"
    assert snapshot.engine.state["tool_recovery"]["active_calls"] == {
        "call-live": {
            "tool_name": "lookup",
            "arguments_fingerprint": fingerprint,
        }
    }
    assert "do-not-migrate" not in snapshot.to_json()
    assert "private_table" not in snapshot.to_json()


def test_v3_partial_success_batch_is_fail_closed_without_replay_arguments() -> None:
    payload = _v3_payload(pending_repair=True)
    state = payload["execution_state"]
    assert isinstance(state, dict)
    pending = state["pending_tool_failure"]
    assert isinstance(pending, dict)
    pending.update(
        {
            "success_count": 1,
            "failure_count": 1,
            "result_count": 2,
            "arguments": {"query": "customer-private-value"},
        }
    )
    policy_state = payload["policy_state"]
    assert isinstance(policy_state, dict)
    policy_state["execution_state"] = copy.deepcopy(state)
    policy_state["plan"] = copy.deepcopy(state["plan"])

    snapshot = migrate_checkpoint_payload(payload)
    migrated_failure = snapshot.engine.state["tool_recovery"]["pending_failure"]

    assert snapshot.common_state.resume_safety == "manual_required"
    assert snapshot.engine.state["phase"] == "tool_recovery"
    assert migrated_failure["success_count"] == 1
    assert migrated_failure["failure_count"] == 1
    assert migrated_failure["result_count"] == 2
    assert "arguments" not in migrated_failure
    assert "customer-private-value" not in snapshot.to_json()


def test_outer_finalization_markers_are_authoritative() -> None:
    state = _plan_state()
    state.update(
        {
            "phase": "done",
            "finalization_count": 1,
            "final_response": "done",
            "final_persisted": True,
            "final_emitted": True,
        }
    )
    payload = _v3_payload()
    payload["execution_state"] = state
    payload["policy_state"] = {
        "execution_state": copy.deepcopy(state),
        "plan": copy.deepcopy(state["plan"]),
    }
    snapshot = migrate_checkpoint_payload(payload)
    corrupted = snapshot.to_dict()
    corrupted["finalization_markers"]["emitted"] = False

    with pytest.raises(StateMigrationError, match="does not match"):
        RunCheckpoint.from_dict(corrupted)


def test_standard_finalization_mirror_must_match_outer_markers() -> None:
    snapshot = RunCheckpoint(
        run_id="run-standard-markers",
        session_id="session-standard-markers",
        messages=(),
    ).to_snapshot()
    outer_only = replace(
        snapshot,
        finalization_markers=FinalizationMarkers(
            started=True,
            response_generated=True,
            response="outer-only",
        ),
    )

    with pytest.raises(StateMigrationError, match="Standard finalization"):
        RunCheckpoint.from_snapshot(outer_only)

    state = dict(snapshot.engine.state)
    state["finalization"] = {
        "started": True,
        "response_generated": True,
        "response": "engine-only",
        "invocation_count": 1,
        "persisted": False,
        "emitted": False,
    }
    engine_only = replace(
        snapshot,
        engine=EngineSnapshot("standard", 1, state),
    )

    with pytest.raises(StateMigrationError, match="Standard finalization"):
        RunCheckpoint.from_snapshot(engine_only)


def test_in_memory_copy_on_migrate_preserves_exact_legacy_payload() -> None:
    async def scenario() -> None:
        store = InMemoryCheckpointStore()
        source = json.dumps(_v3_payload(), ensure_ascii=False, indent=2)
        await store.save_legacy_payload("run-v3", source)

        snapshot = await store.load_snapshot("run-v3")

        assert snapshot is not None
        assert snapshot.schema_version == 4
        assert await store.load_legacy_payload("run-v3") == source
        assert (await store.load_snapshot("run-v3")).to_json() == snapshot.to_json()

    asyncio.run(scenario())


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)
        self.expirations.pop(key, None)


def test_redis_migration_uses_additive_v4_key_and_preserves_legacy() -> None:
    async def scenario() -> None:
        client = _FakeRedis()
        store = RedisCheckpointStore(client, ttl_seconds=30)
        source = json.dumps(_v3_payload(), separators=(",", ":"))
        await store.save_legacy_payload("run-v3", source)

        snapshot = await store.load_snapshot("run-v3")

        assert snapshot is not None
        assert client.values["moduagent:checkpoint:run-v3"] == source
        assert (
            RunSnapshot.from_json(client.values["moduagent:checkpoint:v4:run-v3"])
            == snapshot
        )
        assert client.expirations["moduagent:checkpoint:v4:run-v3"] == 30

    asyncio.run(scenario())


def test_event_envelope_is_additive_and_wire_json_safe() -> None:
    result = AgentResult(
        run_id="run-event",
        output="done",
        messages=(Message.assistant("done"),),
        usage=Usage(2, 1, 3),
        finish_reason=FinishReason.COMPLETED,
    )
    occurred_at = datetime(2026, 7, 29, tzinfo=timezone.utc)
    event = AgentEvent(
        EventType.RUN_COMPLETED,
        "run-event",
        {"result": result},
        occurred_at,
        EventVisibility.PUBLIC,
        session_id="session-event",
        engine_id="plan",
        sequence=7,
    )

    payload = event_to_dict(event)

    assert event.event_type is EventType.RUN_COMPLETED
    assert event.timestamp == event.occurred_at
    assert payload["type"] == payload["event_type"] == "run_completed"
    assert payload["occurred_at"] == payload["timestamp"]
    assert payload["session_id"] == "session-event"
    assert payload["engine_id"] == "plan"
    assert payload["sequence"] == 7
    assert payload["data"]["result"]["finish_reason"] == "completed"
    json.dumps(payload, allow_nan=False)


def test_event_publisher_stamps_monotonic_run_scope() -> None:
    ids = iter(("event-1", "event-2"))
    publisher = EventPublisher(
        run_id="run-publisher",
        session_id="session-publisher",
        engine_id="standard",
        initial_sequence=10,
        event_id_factory=lambda: next(ids),
    )

    started = publisher.create(EventType.RUN_STARTED)
    stamped = publisher.stamp(
        AgentEvent(
            EventType.MODEL_STARTED,
            "run-publisher",
            visibility=EventVisibility.INTERNAL,
        )
    )

    assert (started.sequence, stamped.sequence) == (11, 12)
    assert started.event_id == "event-1"
    assert stamped.session_id == "session-publisher"
    assert stamped.engine_id == "standard"
    assert publisher.last_sequence == 12

    with pytest.raises(ValueError, match="monotonically"):
        publisher.stamp(
            AgentEvent(
                EventType.MODEL_COMPLETED,
                "run-publisher",
                sequence=12,
            )
        )


def test_event_wire_projection_handles_recursive_values() -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    event = AgentEvent(EventType.MODEL_COMPLETED, "run-cycle", {"value": cyclic})

    payload = event.to_dict()

    assert payload["data"]["value"]["self"] == {
        "unsupported_type": "recursive_reference"
    }
    json.dumps(payload, allow_nan=False)

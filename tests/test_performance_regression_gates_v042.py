from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from moduagent.decision import Plan, PlanStep
from moduagent.decision.planning import ExecutionState, RunPhase
from moduagent.execution.planning.state import PlanEngineState
from moduagent.observability import (
    AuditEventSink,
    CompositeEventSink,
    NoopEventSink,
)
from moduagent.runtime import AgentEvent, EventType


class _CopyProbe:
    def __init__(self, payload: str, copies: list[int]) -> None:
        self.payload = payload
        self.copies = copies

    def __deepcopy__(self, memo: dict[int, Any]) -> _CopyProbe:
        del memo
        self.copies.append(len(self.payload))
        return _CopyProbe(self.payload, self.copies)


class _CollectingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        self.events.append(event)


def test_large_plan_legacy_bridge_copies_graph_once_per_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copies: list[int] = []
    serialized_plans: list[int] = []
    probe = _CopyProbe("x" * (512 * 1024), copies)
    legacy = ExecutionState(
        phase=RunPhase.ACT,
        plan=Plan(
            [
                PlanStep(
                    step_id="extract",
                    objective="Extract the large payload.",
                    metadata={"payload": probe},
                )
            ]
        ),
        current_step_id="extract",
    )
    original_to_dict = Plan.to_dict

    def counting_to_dict(plan: Plan) -> dict[str, Any]:
        serialized_plans.append(1)
        return original_to_dict(plan)

    monkeypatch.setattr(Plan, "to_dict", counting_to_dict)

    nested = PlanEngineState.from_legacy(legacy)

    assert copies == [512 * 1024]
    assert serialized_plans == []
    nested_probe = nested.plan_progress.plan.steps[0].metadata["payload"]
    assert nested.plan_progress.plan is not legacy.plan
    assert nested_probe is not probe

    round_tripped = nested.to_legacy()

    assert copies == [512 * 1024, 512 * 1024]
    assert serialized_plans == []
    round_trip_probe = round_tripped.plan.steps[0].metadata["payload"]
    assert round_tripped.plan is not nested.plan_progress.plan
    assert round_trip_probe is not nested_probe


def test_nested_noop_composite_invokes_no_sink_and_makes_no_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noop_calls: list[int] = []
    copies: list[int] = []

    async def counting_noop(
        self: NoopEventSink,
        event: AgentEvent,
    ) -> None:
        del self, event
        noop_calls.append(1)

    monkeypatch.setattr(NoopEventSink, "publish", counting_noop)

    async def scenario() -> None:
        sink = CompositeEventSink(
            (
                NoopEventSink(),
                CompositeEventSink((NoopEventSink(), NoopEventSink())),
            )
        )
        event = AgentEvent(
            EventType.RUN_STARTED,
            "nested-noop",
            {"probe": _CopyProbe("x" * (256 * 1024), copies)},
        )

        await sink.publish(event)

    asyncio.run(scenario())

    assert noop_calls == []
    assert copies == []


def test_mixed_composite_calls_only_active_sinks_and_copies_custom_graph_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noop_calls: list[int] = []
    copies: list[int] = []

    async def counting_noop(
        self: NoopEventSink,
        event: AgentEvent,
    ) -> None:
        del self, event
        noop_calls.append(1)

    monkeypatch.setattr(NoopEventSink, "publish", counting_noop)

    async def scenario() -> tuple[AuditEventSink, _CollectingSink]:
        audit = AuditEventSink()
        custom = _CollectingSink()
        sink = CompositeEventSink((NoopEventSink(), audit, custom))
        event = AgentEvent(
            EventType.RUN_STARTED,
            "mixed-composite",
            {
                "agent": "benchmark",
                "probe": _CopyProbe("x" * (256 * 1024), copies),
            },
        )

        await sink.publish(event)
        return audit, custom

    audit, custom = asyncio.run(scenario())

    assert noop_calls == []
    assert copies == [256 * 1024]
    assert len(audit.records) == 1
    assert len(custom.events) == 1


def test_performance_microbenchmark_emits_stable_json_schema() -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks" / "performance_v042.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--payload-bytes",
            "1024",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)

    assert report["schema_version"] == 1
    assert report["benchmark"] == "moduagent.performance_v042"
    assert report["parameters"] == {
        "iterations": 1,
        "payload_bytes": 1024,
        "warmup": 0,
    }
    assert report["model_phase"]["calls"] == 1
    assert report["plan_state"]["encoded_bytes"] >= 1024
    configurations = report["runtime_configurations"]
    assert set(configurations) == {
        "checkpoint+composite",
        "checkpoint+noop",
        "no_checkpoint+composite",
        "no_checkpoint+noop",
    }
    assert configurations["no_checkpoint+noop"]["state_encode_calls"] == 0
    assert configurations["no_checkpoint+composite"]["state_encode_calls"] == 0
    assert configurations["checkpoint+noop"]["state_encode_calls"] > 0
    assert configurations["checkpoint+composite"]["state_encode_calls"] > 0

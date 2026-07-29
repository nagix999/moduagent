"""Dependency-free ModuAgent 0.4.2 microbenchmarks.

Run from a source checkout:

    python3 benchmarks/performance_v042.py --pretty

The report intentionally exposes measurements instead of pass/fail timing
thresholds. Structural performance invariants live in the test suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import moduagent  # noqa: E402
from moduagent.agent import Agent  # noqa: E402
from moduagent.config import AgentConfig  # noqa: E402
from moduagent.decision import Plan, PlanStep  # noqa: E402
from moduagent.decision.planning import ExecutionState, RunPhase  # noqa: E402
from moduagent.execution.planning.state import (  # noqa: E402
    PlanEngineState,
    PlanStateCodec,
)
from moduagent.messages import FinishReason, Message  # noqa: E402
from moduagent.models import (  # noqa: E402
    ModelCapabilities,
    ModelResponse,
)
from moduagent.observability import (  # noqa: E402
    AuditEventSink,
    CompositeEventSink,
    InMemoryMetricRecorder,
    MetricsEventSink,
    NoopEventSink,
)
from moduagent.persistence import InMemoryCheckpointStore  # noqa: E402
from moduagent.runtime import AgentEvent, EventType  # noqa: E402


class _ImmediateModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self) -> None:
        self.calls = 0
        self.invocation_ns: list[int] = []

    async def complete(self, request: Any) -> ModelResponse:
        del request
        started = perf_counter_ns()
        await asyncio.sleep(0)
        self.calls += 1
        self.invocation_ns.append(perf_counter_ns() - started)
        return ModelResponse(Message.assistant("done"))

    async def stream(self, request: Any) -> Any:
        del request
        raise AssertionError("stream should not be called")
        yield


class _PhaseSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        if event.type in {EventType.MODEL_STARTED, EventType.MODEL_COMPLETED}:
            self.events.append(event)


class _CountingCheckpointStore(InMemoryCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self.saves = 0
        self.deletes = 0

    async def save_snapshot(self, snapshot: Any) -> None:
        self.saves += 1
        await super().save_snapshot(snapshot)

    async def delete(self, run_id: str) -> None:
        self.deletes += 1
        await super().delete(run_id)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _milliseconds(samples_ns: list[int]) -> dict[str, float]:
    ordered = sorted(value / 1_000_000 for value in samples_ns)
    percentile_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "min_ms": round(ordered[0], 6),
        "mean_ms": round(statistics.fmean(ordered), 6),
        "median_ms": round(statistics.median(ordered), 6),
        "p95_ms": round(ordered[percentile_index], 6),
        "max_ms": round(ordered[-1], 6),
    }


def _plan_state(payload_bytes: int) -> ExecutionState:
    return ExecutionState(
        phase=RunPhase.ACT,
        plan=Plan(
            [
                PlanStep(
                    step_id="large-step",
                    objective="Process the benchmark payload.",
                    completion_criteria=["The payload is processed."],
                    metadata={"payload": "x" * payload_bytes},
                )
            ]
        ),
        current_step_id="large-step",
    )


def _benchmark_plan_state(
    *,
    iterations: int,
    warmup: int,
    payload_bytes: int,
) -> dict[str, Any]:
    legacy = _plan_state(payload_bytes)
    nested = PlanEngineState.from_legacy(legacy)
    codec = PlanStateCodec()

    for _ in range(warmup):
        PlanEngineState.from_legacy(legacy)
        nested.to_legacy()
        codec.decode(codec.encode(nested))

    from_legacy_ns: list[int] = []
    to_legacy_ns: list[int] = []
    codec_round_trip_ns: list[int] = []
    for _ in range(iterations):
        started = perf_counter_ns()
        converted = PlanEngineState.from_legacy(legacy)
        from_legacy_ns.append(perf_counter_ns() - started)

        started = perf_counter_ns()
        converted.to_legacy()
        to_legacy_ns.append(perf_counter_ns() - started)

        started = perf_counter_ns()
        codec.decode(codec.encode(converted))
        codec_round_trip_ns.append(perf_counter_ns() - started)

    encoded = codec.encode(nested)
    encoded_bytes = len(
        json.dumps(
            encoded,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "source_payload_bytes": payload_bytes,
        "encoded_bytes": encoded_bytes,
        "from_legacy": _milliseconds(from_legacy_ns),
        "to_legacy": _milliseconds(to_legacy_ns),
        "codec_round_trip": _milliseconds(codec_round_trip_ns),
    }


async def _run_agent(
    agent: Agent,
    *,
    session_prefix: str,
    count: int,
    sample: bool,
) -> list[int]:
    samples: list[int] = []
    for index in range(count):
        started = perf_counter_ns()
        result = await agent.run(
            "benchmark",
            session_id=f"{session_prefix}-{index}",
        )
        elapsed = perf_counter_ns() - started
        if result.finish_reason is not FinishReason.COMPLETED:
            raise RuntimeError(result.error or "benchmark Agent run failed")
        if sample:
            samples.append(elapsed)
    return samples


async def _benchmark_model_phase(
    *,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    model = _ImmediateModel()
    sink = _PhaseSink()
    agent = Agent(
        config=AgentConfig("performance-v042-model", "Answer directly."),
        model=model,
        event_sink=sink,
    )

    await _run_agent(
        agent,
        session_prefix="model-warmup",
        count=warmup,
        sample=False,
    )
    sink.events.clear()
    model.calls = 0
    model.invocation_ns.clear()
    await _run_agent(
        agent,
        session_prefix="model-sample",
        count=iterations,
        sample=False,
    )

    starts: dict[str, list[AgentEvent]] = {}
    completed: dict[str, list[AgentEvent]] = {}
    for event in sink.events:
        target = starts if event.type is EventType.MODEL_STARTED else completed
        target.setdefault(event.run_id, []).append(event)
    event_windows_ns: list[int] = []
    for run_id, start_events in starts.items():
        completed_events = completed.get(run_id, [])
        for start, end in zip(start_events, completed_events):
            event_windows_ns.append(
                max(
                    0,
                    int((end.occurred_at - start.occurred_at).total_seconds() * 1e9),
                )
            )
    if len(event_windows_ns) != iterations:
        raise RuntimeError("model phase events were not paired one-to-one")

    return {
        "calls": model.calls,
        "provider_stub": _milliseconds(model.invocation_ns),
        "runtime_event_window": _milliseconds(event_windows_ns),
    }


async def _benchmark_runtime_configuration(
    *,
    iterations: int,
    warmup: int,
    checkpoint: bool,
    composite: bool,
) -> dict[str, Any]:
    model = _ImmediateModel()
    store = _CountingCheckpointStore() if checkpoint else None
    audit: AuditEventSink | None = None
    if composite:
        audit = AuditEventSink()
        event_sink = CompositeEventSink(
            (
                audit,
                MetricsEventSink(InMemoryMetricRecorder()),
            )
        )
    else:
        event_sink = NoopEventSink()
    agent = Agent(
        config=AgentConfig(
            f"performance-v042-{'checkpoint' if checkpoint else 'memory'}",
            "Answer directly.",
        ),
        model=model,
        event_sink=event_sink,
        checkpoint_store=store,
    )
    state_encode_calls = 0
    original_encode = agent.engine.encode_state

    def counting_encode(state: Any) -> Any:
        nonlocal state_encode_calls
        state_encode_calls += 1
        return original_encode(state)

    agent.engine.encode_state = counting_encode  # type: ignore[method-assign]
    await _run_agent(
        agent,
        session_prefix="runtime-warmup",
        count=warmup,
        sample=False,
    )
    state_encode_calls = 0
    model.calls = 0
    model.invocation_ns.clear()
    if store is not None:
        store.saves = 0
        store.deletes = 0
    if audit is not None:
        audit.records.clear()

    samples = await _run_agent(
        agent,
        session_prefix="runtime-sample",
        count=iterations,
        sample=True,
    )
    return {
        "runs": iterations,
        "elapsed": _milliseconds(samples),
        "model_calls": model.calls,
        "state_encode_calls": state_encode_calls,
        "checkpoint_saves": 0 if store is None else store.saves,
        "checkpoint_deletes": 0 if store is None else store.deletes,
        "audit_records": 0 if audit is None else len(audit.records),
    }


async def _run_benchmarks(
    *,
    iterations: int,
    warmup: int,
    payload_bytes: int,
) -> dict[str, Any]:
    model_phase = await _benchmark_model_phase(
        iterations=iterations,
        warmup=warmup,
    )
    configurations: dict[str, Any] = {}
    for checkpoint, composite in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        name = (
            f"{'checkpoint' if checkpoint else 'no_checkpoint'}"
            f"+{'composite' if composite else 'noop'}"
        )
        configurations[name] = await _benchmark_runtime_configuration(
            iterations=iterations,
            warmup=warmup,
            checkpoint=checkpoint,
            composite=composite,
        )
    return {
        "schema_version": 1,
        "benchmark": "moduagent.performance_v042",
        "environment": {
            "moduagent_version": moduagent.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "parameters": {
            "iterations": iterations,
            "payload_bytes": payload_bytes,
            "warmup": warmup,
        },
        "model_phase": model_phase,
        "plan_state": _benchmark_plan_state(
            iterations=iterations,
            warmup=warmup,
            payload_bytes=payload_bytes,
        ),
        "runtime_configurations": configurations,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ModuAgent 0.4.2 local performance microbenchmarks.",
    )
    parser.add_argument("--iterations", type=_positive_int, default=10)
    parser.add_argument("--warmup", type=_non_negative_int, default=2)
    parser.add_argument("--payload-bytes", type=_positive_int, default=1_000_000)
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent the JSON report for interactive inspection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = asyncio.run(
        _run_benchmarks(
            iterations=args.iterations,
            warmup=args.warmup,
            payload_bytes=args.payload_bytes,
        )
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

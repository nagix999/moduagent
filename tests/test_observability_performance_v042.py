from __future__ import annotations

import asyncio
from typing import Any

from moduagent import (
    Agent,
    AgentConfig,
    AuditEventSink,
    CompositeEventSink,
    InMemoryMetricRecorder,
    MetricsEventSink,
)
from moduagent.messages import Message
from moduagent.models import ModelCapabilities, ModelResponse
from moduagent.observability import NoopEventSink
from moduagent.runtime import AgentEvent, EventType


class _ImmediateModel:
    capabilities = ModelCapabilities(streaming=False)

    async def complete(self, request: Any) -> ModelResponse:
        del request
        return ModelResponse(Message.assistant("done"))

    async def stream(self, request: Any) -> Any:
        del request
        raise AssertionError("stream should not be called")


class _CollectingSink:
    def __init__(self, *, mutate: bool = False) -> None:
        self.events: list[AgentEvent] = []
        self.mutate = mutate

    async def publish(self, event: AgentEvent) -> None:
        if self.mutate:
            user_context = event.data.get("user_context")
            if isinstance(user_context, dict):
                user_context["mutated"] = True
        self.events.append(event)


class _CopyProbe:
    def __init__(self, counter: list[int]) -> None:
        self.counter = counter

    def __deepcopy__(self, memo: dict[int, Any]) -> "_CopyProbe":
        del memo
        self.counter.append(1)
        return _CopyProbe(self.counter)


class _HostileOutput:
    def __deepcopy__(self, memo: dict[int, Any]) -> "_HostileOutput":
        del memo
        raise AssertionError("built-in observability must not deepcopy output")


class _HostileOutputCodec:
    def schema(self) -> None:
        return None

    def decode(self, response: Any) -> _HostileOutput:
        del response
        return _HostileOutput()


def test_noop_event_sink_skips_worker_queue_and_retains_only_small_stamps() -> None:
    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig("noop-fast-path", "Answer."),
            model=_ImmediateModel(),
            event_sink=NoopEventSink(),
        )
        stream = agent.stream_all(
            "run",
            session_id="noop-fast-path",
            user_context={"large": "x" * 100_000},
        )

        started = await anext(stream)

        assert started.type is EventType.RUN_STARTED
        assert agent.runtime._sink_queues == {}
        assert agent.runtime._sink_workers == {}
        assert agent.runtime._published_events
        assert all(
            not isinstance(value, AgentEvent) and not hasattr(value, "data")
            for value in agent.runtime._published_events.values()
        )
        context = agent.runtime._coordinator_contexts[started.run_id]
        assert context.metadata["_moduagent_session_queue_wait_seconds"] >= 0
        assert started.data["queue_wait_seconds"] >= 0

        await stream.aclose()

        assert agent.runtime._published_events == {}
        assert agent.runtime._sink_queues == {}
        assert agent.runtime._sink_workers == {}

    asyncio.run(scenario())


def test_non_noop_event_sink_uses_a_bounded_handoff_queue() -> None:
    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig("bounded-sink-queue", "Answer."),
            model=_ImmediateModel(),
            event_sink=_CollectingSink(),
        )
        stream = agent.stream_all("run", session_id="bounded-sink-queue")

        started = await anext(stream)
        queue = agent.runtime._sink_queues[started.run_id]

        assert queue.maxsize == 1_024

        await stream.aclose()

    asyncio.run(scenario())


def test_composite_removes_only_the_redundant_coordinator_copy() -> None:
    async def scenario() -> None:
        copies: list[int] = []
        mutating = _CollectingSink(mutate=True)
        observing = _CollectingSink()
        agent = Agent(
            config=AgentConfig("composite-copy", "Answer."),
            model=_ImmediateModel(),
            event_sink=CompositeEventSink((mutating, observing)),
        )

        result = await agent.run(
            "run",
            session_id="composite-copy",
            user_context={"probe": _CopyProbe(copies)},
        )

        assert result.output == "done"
        # RUN_STARTED contains the probe. Each custom child receives one private
        # graph; the Coordinator no longer makes an additional unused copy.
        assert len(copies) == 2
        observed_started = next(
            event for event in observing.events if event.type is EventType.RUN_STARTED
        )
        assert "mutated" not in observed_started.data["user_context"]

    asyncio.run(scenario())


def test_trusted_builtin_composite_does_not_copy_arbitrary_terminal_output() -> None:
    async def scenario() -> None:
        audit = AuditEventSink()
        metrics = MetricsEventSink(InMemoryMetricRecorder())
        agent = Agent(
            config=AgentConfig("builtin-copy-free", "Answer."),
            model=_ImmediateModel(),
            output_codec=_HostileOutputCodec(),  # type: ignore[arg-type]
            event_sink=CompositeEventSink((audit, metrics)),
        )

        result = await agent.run("run", session_id="builtin-copy-free")

        assert isinstance(result.output, _HostileOutput)
        assert audit.records[-1]["data"]["has_output"] is True

    asyncio.run(scenario())


def test_default_in_memory_observability_avoids_background_thread_hops(
    monkeypatch: Any,
) -> None:
    async def fail_thread_hop(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("trusted in-memory sink should execute inline")

    monkeypatch.setattr(
        "moduagent.observability.sinks.run_in_daemon_thread",
        fail_thread_hop,
    )

    async def scenario() -> None:
        event = AgentEvent(
            EventType.RUN_STARTED,
            "run-inline",
            {"queue_wait_seconds": 0.125},
        )
        audit = AuditEventSink()
        metrics = MetricsEventSink(InMemoryMetricRecorder())

        await audit.publish(event)
        await metrics.publish(event)

        assert len(audit.records) == 1
        assert metrics.last_error is None
        assert metrics.recorder.observations[
            ("moduagent.run.queue_wait_seconds", ())
        ] == [0.125]

    asyncio.run(scenario())


def test_performance_metrics_capture_phase_and_io_durations() -> None:
    async def scenario() -> None:
        recorder = InMemoryMetricRecorder()
        metrics = MetricsEventSink(recorder)
        events = (
            AgentEvent(
                EventType.MODEL_STARTED,
                "timed-run",
                {"phase": "act", "attempt": 1},
            ),
            AgentEvent(
                EventType.MODEL_COMPLETED,
                "timed-run",
                {"phase": "act", "duration_seconds": 0.25},
            ),
            AgentEvent(
                EventType.MEMORY_COMPACTED,
                "timed-run",
                {"phase": "act", "duration_seconds": 0.05},
            ),
            AgentEvent(
                EventType.CHECKPOINT_SAVED,
                "timed-run",
                {"duration_seconds": 0.02},
            ),
            AgentEvent(
                EventType.TOOL_COMPLETED,
                "timed-run",
                {
                    "tool_name": "query_db",
                    "success": True,
                    "duration_seconds": 0.1,
                },
            ),
        )
        for event in events:
            await metrics.publish(event)

        assert recorder.counters[("moduagent.model.calls", (("phase", "act"),))] == 1
        assert recorder.counters[("moduagent.checkpoint.saves", ())] == 1
        assert recorder.observations[
            ("moduagent.model.duration_seconds", (("phase", "act"),))
        ] == [0.25]
        assert recorder.observations[
            ("moduagent.memory.prepare_seconds", (("phase", "act"),))
        ] == [0.05]
        assert recorder.observations[("moduagent.checkpoint.duration_seconds", ())] == [
            0.02
        ]
        assert recorder.observations[
            ("moduagent.tool.duration_seconds", (("tool", "query_db"),))
        ] == [0.1]

    asyncio.run(scenario())

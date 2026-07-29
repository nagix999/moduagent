from __future__ import annotations

import asyncio
from types import SimpleNamespace

from moduagent.runtime.events import AgentEvent, EventType
from moduagent.runtime.services import (
    RuntimeServices,
    _MAX_PENDING_SERVICE_EVENTS,
)


def test_service_event_handoff_applies_bounded_backpressure() -> None:
    async def scenario() -> None:
        services = RuntimeServices(SimpleNamespace(), deadline=1.0)
        for sequence in range(_MAX_PENDING_SERVICE_EVENTS):
            await services._enqueue_event(
                AgentEvent(
                    EventType.MODEL_DELTA,
                    "bounded-run",
                    {"delta": str(sequence)},
                )
            )

        blocked = asyncio.create_task(
            services._enqueue_event(
                AgentEvent(
                    EventType.MODEL_DELTA,
                    "bounded-run",
                    {"delta": "overflow"},
                )
            )
        )
        await asyncio.sleep(0)

        assert blocked.done() is False
        assert len(services._pending_events) == _MAX_PENDING_SERVICE_EVENTS

        drained = services.drain_events()
        await asyncio.wait_for(blocked, timeout=0.1)

        assert len(drained) == _MAX_PENDING_SERVICE_EVENTS
        assert len(services._pending_events) == 1
        assert services._pending_events[0].data["delta"] == "overflow"

    asyncio.run(scenario())

"""Inspect a run without logging prompts, Tool results, or provider bodies."""

import asyncio
import logging
from typing import Literal

from moduagent import (
    Agent,
    ConsoleEventSink,
    InMemoryDiagnosticSink,
    LoggingEventSink,
    VLLMClient,
    tool,
)


SERVICE_STATUS = {
    "billing": {"status": "operational", "updated_at": "2026-07-30T09:00:00Z"},
    "orders": {"status": "degraded", "updated_at": "2026-07-30T09:05:00Z"},
}


@tool
def get_service_status(
    service: Literal["billing", "orders"],
) -> dict[str, str]:
    """Return the current status of a supported service."""

    return {"service": service, **SERVICE_STATUS[service]}


def build_agent(model, diagnostics: InMemoryDiagnosticSink, *, event_sink=None):
    return Agent.create(
        model=model,
        instructions=(
            "Answer service-status questions from get_service_status. "
            "Never invent operational status."
        ),
        tools=[get_service_status],
        event_sink=LoggingEventSink() if event_sink is None else event_sink,
        diagnostic_sink=diagnostics,
        tool_trace_mode="summary",
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    diagnostics = InMemoryDiagnosticSink(max_records=100)

    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        agent = build_agent(
            model,
            diagnostics,
            event_sink=ConsoleEventSink(detail="detailed"),
        )
        result = await agent.run("Is the orders service healthy?")

    print("output:", result.output)
    print("run usage:", dict(result.run_usage))

    print("Tool calls:")
    for entry in result.tool_trace:
        print(dict(entry))

    if result.error_summary:
        print("run error:", dict(result.error_summary))
    for failure in diagnostics.for_run(result.run_id):
        print(
            "diagnostic:",
            failure.failure_id,
            failure.component,
            failure.operation,
            failure.code,
        )

    result.raise_for_error()


if __name__ == "__main__":
    asyncio.run(main())

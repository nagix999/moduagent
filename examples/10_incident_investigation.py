"""Investigate a production incident with five read-only Tools."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from moduagent import (
    Agent,
    ConsoleEventSink,
    InMemoryDiagnosticSink,
    RunLimits,
    VLLMClient,
    tool,
)


ServiceName = Literal["checkout-api", "payments-api", "inventory-api"]
LogLevel = Literal["ALL", "INFO", "WARN", "ERROR"]

# This is only a local teaching aid. Production applications should send Tool
# telemetry to their normal observability system instead of a module-level list.
CALL_LOG: list[dict[str, object]] = []

INCIDENTS = {
    "INC-2042": {
        "incident_id": "INC-2042",
        "title": "Elevated checkout failures",
        "service": "checkout-api",
        "severity": "SEV-2",
        "status": "mitigated",
        "started_at": "2026-07-29T09:12:00Z",
        "mitigated_at": "2026-07-29T09:31:00Z",
        "investigation_window": {
            "start_time": "2026-07-29T09:00:00Z",
            "end_time": "2026-07-29T09:35:00Z",
        },
        "reported_impact": "Customers intermittently could not complete checkout.",
    }
}

METRIC_SNAPSHOTS = (
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:00:00Z",
        "error_rate_pct": 0.6,
        "p95_latency_ms": 210.0,
        "db_pool_utilization_pct": 54.0,
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:10:00Z",
        "error_rate_pct": 0.9,
        "p95_latency_ms": 260.0,
        "db_pool_utilization_pct": 74.0,
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:15:00Z",
        "error_rate_pct": 18.7,
        "p95_latency_ms": 1680.0,
        "db_pool_utilization_pct": 99.0,
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:20:00Z",
        "error_rate_pct": 31.4,
        "p95_latency_ms": 2410.0,
        "db_pool_utilization_pct": 100.0,
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:25:00Z",
        "error_rate_pct": 24.2,
        "p95_latency_ms": 1980.0,
        "db_pool_utilization_pct": 100.0,
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:30:00Z",
        "error_rate_pct": 1.1,
        "p95_latency_ms": 310.0,
        "db_pool_utilization_pct": 61.0,
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:35:00Z",
        "error_rate_pct": 0.7,
        "p95_latency_ms": 230.0,
        "db_pool_utilization_pct": 55.0,
    },
)

DEPLOYMENTS = (
    {
        "service": "checkout-api",
        "version": "4.17.0",
        "event": "deployment_completed",
        "timestamp": "2026-07-29T09:08:00Z",
        "change_summary": "Reduced DB pool max_size from 80 to 20.",
    },
    {
        "service": "checkout-api",
        "version": "4.16.3",
        "event": "rollback_completed",
        "timestamp": "2026-07-29T09:27:00Z",
        "change_summary": "Restored DB pool max_size to 80.",
    },
    {
        "service": "payments-api",
        "version": "8.4.1",
        "event": "deployment_completed",
        "timestamp": "2026-07-29T07:30:00Z",
        "change_summary": "Updated internal tracing labels.",
    },
)

SERVICE_LOGS = (
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:11:00Z",
        "level": "INFO",
        "code": "CONFIG_LOADED",
        "message": "Loaded db.pool.max_size=20 for version 4.17.0.",
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:12:00Z",
        "level": "ERROR",
        "code": "DB_POOL_TIMEOUT",
        "message": "Timed out acquiring an application DB connection.",
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:16:00Z",
        "level": "WARN",
        "code": "DB_POOL_SATURATED",
        "message": "Application DB connection pool reached 99 percent utilization.",
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:20:00Z",
        "level": "ERROR",
        "code": "CHECKOUT_UNAVAILABLE",
        "message": "Checkout request failed after DB pool acquisition timeout.",
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:29:00Z",
        "level": "INFO",
        "code": "DB_POOL_RECOVERED",
        "message": "Application DB connection pool returned below 70 percent.",
    },
)

DEPENDENCY_SNAPSHOTS = (
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:20:00Z",
        "dependency": "payments-api",
        "status": "healthy",
        "p95_latency_ms": 92.0,
        "error_rate_pct": 0.3,
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:20:00Z",
        "dependency": "inventory-api",
        "status": "healthy",
        "p95_latency_ms": 71.0,
        "error_rate_pct": 0.2,
    },
    {
        "service": "checkout-api",
        "timestamp": "2026-07-29T09:20:00Z",
        "dependency": "orders-db",
        "status": "healthy",
        "p95_latency_ms": 24.0,
        "error_rate_pct": 0.0,
    },
)


class IncidentEvidence(BaseModel):
    metrics: str
    deployments: str
    logs: str
    dependencies: str


class IncidentTimeline(BaseModel):
    deployment_at: str
    incident_started_at: str
    peak_impact_at: str
    rollback_at: str
    mitigated_at: str


class RunbookActions(BaseModel):
    """Select application-owned actions; the Agent never executes them."""

    verify_rollback: Literal["required", "not_required"]
    monitor_recovery: Literal["required", "not_required"]
    configuration_guardrail: Literal["required", "not_required"]
    predeployment_load_test: Literal["required", "not_required"]


class IncidentReport(BaseModel):
    incident_id: str
    severity: Literal["SEV-1", "SEV-2", "SEV-3", "SEV-4", "unknown"]
    affected_service: str
    status: Literal["active", "mitigated", "resolved", "unknown"]
    executive_summary: str
    customer_impact: str
    likely_root_cause: str
    evidence: IncidentEvidence
    timeline: IncidentTimeline
    runbook_actions: RunbookActions
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def preserve_incident_state(self) -> IncidentReport:
        """Do not silently upgrade a mitigated incident to resolved."""

        if (
            self.status != "resolved"
            and "resolved" in self.executive_summary.casefold()
        ):
            raise ValueError(
                "summary cannot claim resolution unless incident status is resolved"
            )
        return self


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("timestamps must use ISO 8601 format") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validated_window(start_time: str, end_time: str) -> tuple[datetime, datetime]:
    start = _parse_timestamp(start_time)
    end = _parse_timestamp(end_time)
    if start > end:
        raise ValueError("start_time must be on or before end_time")
    return start, end


def _record_call(tool_name: str, **arguments: object) -> None:
    CALL_LOG.append({"tool": tool_name, **arguments})


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=4096,
)
def get_incident(incident_id: str) -> dict[str, object]:
    """Read an incident record, including its investigation time window."""

    normalized = incident_id.strip().upper()
    _record_call("get_incident", incident_id=normalized)
    incident = INCIDENTS.get(normalized)
    if incident is None:
        return {"incident_id": normalized, "found": False}
    return {"found": True, **incident}


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=8192,
)
def query_service_metrics(
    service: ServiceName,
    start_time: str,
    end_time: str,
) -> dict[str, object]:
    """Read error, latency, and DB-pool metrics in an inclusive time window."""

    start, end = _validated_window(start_time, end_time)
    _record_call(
        "query_service_metrics",
        service=service,
        start_time=start_time,
        end_time=end_time,
    )
    points = [
        {key: value for key, value in snapshot.items() if key != "service"}
        for snapshot in METRIC_SNAPSHOTS
        if snapshot["service"] == service
        and start <= _parse_timestamp(str(snapshot["timestamp"])) <= end
    ]
    return {
        "service": service,
        "window": {"start_time": start_time, "end_time": end_time},
        "units": {
            "error_rate_pct": "percent",
            "p95_latency_ms": "milliseconds",
            "db_pool_utilization_pct": "percent",
        },
        "points": points,
    }


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=8192,
)
def list_deployments(
    service: ServiceName,
    start_time: str,
    end_time: str,
) -> dict[str, object]:
    """Read deployment and rollback events in an inclusive time window."""

    start, end = _validated_window(start_time, end_time)
    _record_call(
        "list_deployments",
        service=service,
        start_time=start_time,
        end_time=end_time,
    )
    events = [
        {key: value for key, value in deployment.items() if key != "service"}
        for deployment in DEPLOYMENTS
        if deployment["service"] == service
        and start <= _parse_timestamp(str(deployment["timestamp"])) <= end
    ]
    return {"service": service, "events": events}


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=8192,
)
def search_service_logs(
    service: ServiceName,
    start_time: str,
    end_time: str,
    level: LogLevel = "ALL",
    contains: str = "",
    limit: int = 20,
) -> dict[str, object]:
    """Read sanitized service logs using time, level, and text filters."""

    start, end = _validated_window(start_time, end_time)
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    needle = contains.strip().lower()
    _record_call(
        "search_service_logs",
        service=service,
        start_time=start_time,
        end_time=end_time,
        level=level,
        contains=contains,
        limit=limit,
    )
    matches = []
    for entry in SERVICE_LOGS:
        if entry["service"] != service:
            continue
        if not start <= _parse_timestamp(str(entry["timestamp"])) <= end:
            continue
        if level != "ALL" and entry["level"] != level:
            continue
        searchable = f"{entry['code']} {entry['message']}".lower()
        if needle and needle not in searchable:
            continue
        matches.append({key: value for key, value in entry.items() if key != "service"})
        if len(matches) == limit:
            break
    return {"service": service, "matches": matches, "match_count": len(matches)}


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=8192,
)
def inspect_dependency_health(
    service: ServiceName,
    start_time: str,
    end_time: str,
) -> dict[str, object]:
    """Read dependency health snapshots in an inclusive time window."""

    start, end = _validated_window(start_time, end_time)
    _record_call(
        "inspect_dependency_health",
        service=service,
        start_time=start_time,
        end_time=end_time,
    )
    snapshots = [
        {key: value for key, value in snapshot.items() if key != "service"}
        for snapshot in DEPENDENCY_SNAPSHOTS
        if snapshot["service"] == service
        and start <= _parse_timestamp(str(snapshot["timestamp"])) <= end
    ]
    return {"service": service, "snapshots": snapshots}


def build_agent(model, *, diagnostic_sink=None, event_sink=None):
    return Agent.create(
        name="incident-investigator",
        model=model,
        instructions=(
            "Investigate production incidents only from Tool evidence. Start with "
            "get_incident. If the incident exists, use its service and investigation "
            "window to call query_service_metrics, list_deployments, "
            "search_service_logs, and inspect_dependency_health. Use all five Tools "
            "before concluding. The four lookups after get_incident are independent, "
            "so request them together in one response when possible. Correlate "
            "timestamps, distinguish correlation from causation, and do not invent "
            "impact or evidence. Copy the exact incident status. Describe the 09:31 "
            "state as mitigated or recovered. Fill every named evidence and timeline "
            "field from Tool results. The runbook_actions object must contain all four "
            "keys: verify_rollback, monitor_recovery, configuration_guardrail, and "
            "predeployment_load_test. For this incident, set all four to required. These "
            "are advisory action codes; never claim to execute them. Keep the draft after "
            "Tool use under 120 words. In the final report, write the summary, impact, "
            "root cause, and each evidence value as one short sentence. Emit compact JSON "
            "without blank space between object fields."
        ),
        tools=[
            get_incident,
            query_service_metrics,
            list_deployments,
            search_service_logs,
            inspect_dependency_health,
        ],
        execution="standard",
        output=IncidentReport,
        limits=RunLimits(
            max_steps=6,
            max_tool_calls=7,
            timeout_seconds=180,
            parallel_tool_calls=True,
            max_parallel_tools=5,
            max_model_turns=10,
            no_progress_model_turn_threshold=3,
        ),
        tool_trace_mode="summary",
        diagnostic_sink=diagnostic_sink,
        event_sink=event_sink,
    )


async def main() -> None:
    CALL_LOG.clear()
    diagnostics = InMemoryDiagnosticSink(max_records=20)
    async with VLLMClient.from_env(
        # The report contains several nested evidence and timeline records, so
        # reserve more output space than the smaller beginner examples.
        default_options={"temperature": 0, "max_tokens": 8192},
    ) as model:
        agent = build_agent(
            model,
            diagnostic_sink=diagnostics,
            event_sink=ConsoleEventSink(),
        )
        result = await agent.run("Investigate incident INC-2042.")

    print("run usage:", dict(result.run_usage))
    print("tool trace:", [dict(entry) for entry in result.tool_trace])
    print("observed calls:", CALL_LOG)
    if result.error_summary:
        print("run error:", dict(result.error_summary))
        for failure in diagnostics.for_run(result.run_id):
            print("diagnostic:", failure.to_dict())
    result.raise_for_error()
    print(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())

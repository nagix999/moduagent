from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "10_incident_investigation.py"


class NoCallModel:
    async def complete(self, request: Any) -> Any:
        del request
        raise AssertionError("building the example must not call the model")


def _load_example() -> ModuleType:
    module_name = "_moduagent_example_10_incident_investigation"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load example: {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _invoke(tool: Any, arguments: dict[str, object]) -> object:
    return asyncio.run(tool.invoke(arguments))


def test_incident_example_imports_without_network_or_embedded_credentials() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    compile(source, str(EXAMPLE_PATH), "exec")
    module = _load_example()

    assert callable(module.build_agent)
    assert "async with VLLMClient.from_env(" in source
    assert '"max_tokens"' in source
    for forbidden in (
        "https://",
        "http://",
        "api_key=",
        "runpod-vllm-token",
        "t62y46bwfim0hq",
    ):
        assert forbidden not in source


def test_incident_agent_has_bounded_parallel_standard_configuration() -> None:
    module = _load_example()

    agent = module.build_agent(NoCallModel())
    spec = agent.inspect()

    assert spec.name == "incident-investigator"
    assert spec.execution_profile.kind == "standard"
    assert [tool.name for tool in agent.tool_registry] == [
        "get_incident",
        "query_service_metrics",
        "list_deployments",
        "search_service_logs",
        "inspect_dependency_health",
    ]
    assert spec.output_contract["structured"] is True
    assert spec.output_contract["staged_finalization"] is True
    assert agent.config.limits.max_tool_calls == 7
    assert agent.config.limits.max_model_turns == 10
    assert agent.config.limits.timeout_seconds == 180
    assert agent.config.limits.parallel_tool_calls is True
    assert agent.config.limits.max_parallel_tools == 5
    assert agent.config.tool_trace_mode == "summary"

    report = module.IncidentReport.model_validate(
        {
            "incident_id": "INC-2042",
            "severity": "SEV-2",
            "affected_service": "checkout-api",
            "status": "mitigated",
            "executive_summary": "Checkout recovered after rollback.",
            "customer_impact": "Some checkout attempts failed.",
            "likely_root_cause": "An undersized application DB pool.",
            "evidence": {
                "metrics": "Pool utilization reached 100%.",
                "deployments": "Rollback preceded recovery.",
                "logs": "DB pool timeouts occurred.",
                "dependencies": "All observed dependencies remained healthy.",
            },
            "timeline": {
                "deployment_at": "2026-07-29T09:08:00Z",
                "incident_started_at": "2026-07-29T09:12:00Z",
                "peak_impact_at": "2026-07-29T09:20:00Z",
                "rollback_at": "2026-07-29T09:27:00Z",
                "mitigated_at": "2026-07-29T09:31:00Z",
            },
            "runbook_actions": {
                "verify_rollback": "required",
                "monitor_recovery": "required",
                "configuration_guardrail": "required",
                "predeployment_load_test": "required",
            },
            "confidence": 0.94,
        }
    )
    assert report.incident_id == "INC-2042"
    assert report.confidence == 0.94


def test_incident_report_rejects_state_upgrades_and_unknown_action_codes() -> None:
    module = _load_example()
    payload = {
        "incident_id": "INC-2042",
        "severity": "SEV-2",
        "affected_service": "checkout-api",
        "status": "mitigated",
        "executive_summary": "Checkout resolved after rollback.",
        "customer_impact": "Some checkout attempts failed.",
        "likely_root_cause": "An undersized application DB pool.",
        "evidence": {
            "metrics": "Pool utilization reached 100%.",
            "deployments": "Rollback preceded recovery.",
            "logs": "DB pool timeouts occurred.",
            "dependencies": "All observed dependencies remained healthy.",
        },
        "timeline": {
            "deployment_at": "2026-07-29T09:08:00Z",
            "incident_started_at": "2026-07-29T09:12:00Z",
            "peak_impact_at": "2026-07-29T09:20:00Z",
            "rollback_at": "2026-07-29T09:27:00Z",
            "mitigated_at": "2026-07-29T09:31:00Z",
        },
        "runbook_actions": {
            "verify_rollback": "required",
            "monitor_recovery": "required",
            "configuration_guardrail": "required",
            "predeployment_load_test": "required",
        },
        "confidence": 0.94,
    }

    try:
        module.IncidentReport.model_validate(payload)
    except ValueError as exc:
        assert "cannot claim resolution" in str(exc)
    else:
        raise AssertionError("a mitigated incident must not be reported as resolved")

    payload["executive_summary"] = "Checkout recovered after rollback."
    payload["runbook_actions"]["monitor_recovery"] = "TODO placeholder"
    try:
        module.IncidentReport.model_validate(payload)
    except ValueError as exc:
        assert "literal_error" in str(exc)
    else:
        raise AssertionError("unknown runbook action codes must be rejected")


def test_get_incident_is_deterministic_and_records_normalized_input() -> None:
    module = _load_example()
    module.CALL_LOG.clear()

    found = _invoke(module.get_incident, {"incident_id": " inc-2042 "})
    missing = _invoke(module.get_incident, {"incident_id": "inc-9999"})

    assert found == {
        "found": True,
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
    assert missing == {"incident_id": "INC-9999", "found": False}
    assert module.CALL_LOG == [
        {"tool": "get_incident", "incident_id": "INC-2042"},
        {"tool": "get_incident", "incident_id": "INC-9999"},
    ]


def test_metrics_tool_filters_an_inclusive_window_deterministically() -> None:
    module = _load_example()
    module.CALL_LOG.clear()
    arguments = {
        "service": "checkout-api",
        "start_time": "2026-07-29T09:10:00Z",
        "end_time": "2026-07-29T09:20:00Z",
    }

    result = _invoke(module.query_service_metrics, arguments)

    assert result == {
        "service": "checkout-api",
        "window": {
            "start_time": "2026-07-29T09:10:00Z",
            "end_time": "2026-07-29T09:20:00Z",
        },
        "units": {
            "error_rate_pct": "percent",
            "p95_latency_ms": "milliseconds",
            "db_pool_utilization_pct": "percent",
        },
        "points": [
            {
                "timestamp": "2026-07-29T09:10:00Z",
                "error_rate_pct": 0.9,
                "p95_latency_ms": 260.0,
                "db_pool_utilization_pct": 74.0,
            },
            {
                "timestamp": "2026-07-29T09:15:00Z",
                "error_rate_pct": 18.7,
                "p95_latency_ms": 1680.0,
                "db_pool_utilization_pct": 99.0,
            },
            {
                "timestamp": "2026-07-29T09:20:00Z",
                "error_rate_pct": 31.4,
                "p95_latency_ms": 2410.0,
                "db_pool_utilization_pct": 100.0,
            },
        ],
    }
    assert module.CALL_LOG == [{"tool": "query_service_metrics", **arguments}]


def test_deployment_tool_returns_only_the_requested_service_and_window() -> None:
    module = _load_example()
    module.CALL_LOG.clear()
    arguments = {
        "service": "checkout-api",
        "start_time": "2026-07-29T09:00:00Z",
        "end_time": "2026-07-29T09:35:00Z",
    }

    result = _invoke(module.list_deployments, arguments)

    assert result == {
        "service": "checkout-api",
        "events": [
            {
                "version": "4.17.0",
                "event": "deployment_completed",
                "timestamp": "2026-07-29T09:08:00Z",
                "change_summary": "Reduced DB pool max_size from 80 to 20.",
            },
            {
                "version": "4.16.3",
                "event": "rollback_completed",
                "timestamp": "2026-07-29T09:27:00Z",
                "change_summary": "Restored DB pool max_size to 80.",
            },
        ],
    }
    assert module.CALL_LOG == [{"tool": "list_deployments", **arguments}]


def test_log_tool_combines_level_text_time_and_limit_filters() -> None:
    module = _load_example()
    module.CALL_LOG.clear()
    arguments = {
        "service": "checkout-api",
        "start_time": "2026-07-29T09:00:00Z",
        "end_time": "2026-07-29T09:35:00Z",
        "level": "ERROR",
        "contains": "pool",
        "limit": 10,
    }

    result = _invoke(module.search_service_logs, arguments)

    assert result == {
        "service": "checkout-api",
        "matches": [
            {
                "timestamp": "2026-07-29T09:12:00Z",
                "level": "ERROR",
                "code": "DB_POOL_TIMEOUT",
                "message": "Timed out acquiring an application DB connection.",
            },
            {
                "timestamp": "2026-07-29T09:20:00Z",
                "level": "ERROR",
                "code": "CHECKOUT_UNAVAILABLE",
                "message": "Checkout request failed after DB pool acquisition timeout.",
            },
        ],
        "match_count": 2,
    }
    assert module.CALL_LOG == [{"tool": "search_service_logs", **arguments}]


def test_dependency_tool_returns_independent_health_evidence() -> None:
    module = _load_example()
    module.CALL_LOG.clear()
    arguments = {
        "service": "checkout-api",
        "start_time": "2026-07-29T09:00:00Z",
        "end_time": "2026-07-29T09:35:00Z",
    }

    result = _invoke(module.inspect_dependency_health, arguments)

    assert result == {
        "service": "checkout-api",
        "snapshots": [
            {
                "timestamp": "2026-07-29T09:20:00Z",
                "dependency": "payments-api",
                "status": "healthy",
                "p95_latency_ms": 92.0,
                "error_rate_pct": 0.3,
            },
            {
                "timestamp": "2026-07-29T09:20:00Z",
                "dependency": "inventory-api",
                "status": "healthy",
                "p95_latency_ms": 71.0,
                "error_rate_pct": 0.2,
            },
            {
                "timestamp": "2026-07-29T09:20:00Z",
                "dependency": "orders-db",
                "status": "healthy",
                "p95_latency_ms": 24.0,
                "error_rate_pct": 0.0,
            },
        ],
    }
    assert module.CALL_LOG == [{"tool": "inspect_dependency_health", **arguments}]

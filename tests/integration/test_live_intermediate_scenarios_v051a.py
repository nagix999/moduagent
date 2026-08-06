from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from moduagent import FinishReason, VLLMClient


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "incident": ROOT / "examples" / "10_incident_investigation.py",
    "customer": ROOT / "examples" / "11_customer_case_resolution.py",
    "release": ROOT / "examples" / "12_release_readiness.py",
}


def _live_environment() -> Mapping[str, str]:
    if os.getenv("MODUAGENT_RUN_LIVE_INTERMEDIATE", "").strip() != "1":
        pytest.skip(
            "set MODUAGENT_RUN_LIVE_INTERMEDIATE=1 to run intermediate scenarios"
        )
    values = {
        name: os.getenv(name, "").strip() for name in ("VLLM_BASE_URL", "VLLM_MODEL")
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip(f"live vLLM environment is missing: {', '.join(missing)}")
    return values


def _client(*, max_tokens: int) -> VLLMClient:
    env = _live_environment()
    return VLLMClient(
        base_url=env["VLLM_BASE_URL"],
        model=env["VLLM_MODEL"],
        api_key=os.getenv("VLLM_API_KEY", "").strip() or None,
        timeout=60,
        default_options={"temperature": 0, "max_tokens": max_tokens},
    )


def _load_example(name: str) -> ModuleType:
    path = EXAMPLES[name]
    module_name = f"_moduagent_live_intermediate_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load example: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _assert_run_observability(
    result: Any,
    *,
    expected_tools: Sequence[str],
    max_model_turns: int,
    max_tool_calls: int,
) -> None:
    assert result.finish_reason is FinishReason.COMPLETED, result.explain()
    usage = result.run_usage
    assert 1 <= usage["model_turns"] <= max_model_turns
    assert usage["tool_calls"] == len(expected_tools)
    assert usage["tool_calls"] <= max_tool_calls
    assert usage["duration_seconds"] > 0

    trace = result.tool_trace
    trace_names = [entry["tool_name"] for entry in trace]
    assert len(trace_names) == len(set(trace_names)) == len(expected_tools)
    assert set(trace_names) == set(expected_tools)
    assert all(entry["success"] is True for entry in trace)
    assert all(entry.get("error") is None for entry in trace)


def test_vllm_investigates_incident_with_parallel_evidence_live() -> None:
    _live_environment()
    module = _load_example("incident")
    expected_tools = [
        "get_incident",
        "query_service_metrics",
        "list_deployments",
        "search_service_logs",
        "inspect_dependency_health",
    ]

    async def scenario() -> None:
        module.CALL_LOG.clear()
        async with _client(max_tokens=8192) as model:
            agent = module.build_agent(model)
            result = await agent.run("Investigate incident INC-2042.")

        _assert_run_observability(
            result,
            expected_tools=expected_tools,
            max_model_turns=agent.config.limits.max_model_turns,
            max_tool_calls=agent.config.limits.max_tool_calls,
        )
        assert isinstance(result.output, module.IncidentReport)
        assert result.output.status == "mitigated"
        assert isinstance(result.output.evidence, module.IncidentEvidence)
        assert isinstance(result.output.timeline, module.IncidentTimeline)
        assert result.output.timeline.model_dump() == {
            "deployment_at": "2026-07-29T09:08:00Z",
            "incident_started_at": "2026-07-29T09:12:00Z",
            "peak_impact_at": "2026-07-29T09:20:00Z",
            "rollback_at": "2026-07-29T09:27:00Z",
            "mitigated_at": "2026-07-29T09:31:00Z",
        }
        assert set(result.output.evidence.model_dump()) == {
            "metrics",
            "deployments",
            "logs",
            "dependencies",
        }
        assert result.output.runbook_actions.model_dump() == {
            "verify_rollback": "required",
            "monitor_recovery": "required",
            "configuration_guardrail": "required",
            "predeployment_load_test": "required",
        }

        call_names = [entry["tool"] for entry in module.CALL_LOG]
        assert call_names[0] == "get_incident"
        assert len(call_names) == len(set(call_names)) == len(expected_tools)
        assert set(call_names) == set(expected_tools)

    asyncio.run(scenario())


def test_vllm_prepares_safe_customer_resolution_live() -> None:
    _live_environment()
    module = _load_example("customer")
    expected_tools = [
        "lookup_case",
        "lookup_order",
        "get_return_policy",
        "assess_return_eligibility",
        "calculate_refund_quote",
    ]

    async def scenario() -> None:
        module.CALL_LOG.clear()
        async with _client(max_tokens=768) as model:
            agent = module.build_agent(model)
            result = await agent.run(
                "Review CASE-2048 and prepare a safe return and refund proposal. "
                "Do not execute any action."
            )

        _assert_run_observability(
            result,
            expected_tools=expected_tools,
            max_model_turns=agent.config.limits.max_model_turns,
            max_tool_calls=agent.config.limits.max_tool_calls,
        )
        assert isinstance(result.output, module.CaseResolution)
        assert result.output.decision == "eligible"
        assert result.output.recommended_resolution == "refund_after_return"
        assert result.output.refund_quote is not None
        assert result.output.refund_quote.total == 132_000
        assert result.output.human_approval_required is True
        assert result.output.write_action_performed is False

        call_names = [entry["tool"] for entry in module.CALL_LOG]
        assert call_names == expected_tools
        assert [entry["tool_name"] for entry in result.tool_trace] == expected_tools

    asyncio.run(scenario())


def test_vllm_holds_high_risk_security_blocked_release_live() -> None:
    _live_environment()
    module = _load_example("release")
    expected_tools = [
        "get_release_manifest",
        "get_ci_summary",
        "get_security_scan",
        "assess_change_risk",
        "get_deployment_capacity",
    ]

    async def scenario() -> None:
        module.CALL_LOG.clear()
        async with _client(max_tokens=768) as model:
            agent = module.build_agent(model)
            result = await agent.run(
                "Should payments-api-2026.08.03-rc1 ship to production now?"
            )

        _assert_run_observability(
            result,
            expected_tools=expected_tools,
            max_model_turns=agent.config.limits.max_model_turns,
            max_tool_calls=agent.config.limits.max_tool_calls,
        )
        assert isinstance(result.output, module.ReleaseDecision)
        assert result.output.decision == "hold"
        assert result.output.risk_level == "high"
        assert result.output.blocking_reasons
        assert any(
            "SEC-431" in text
            for text in [result.output.summary, *result.output.required_actions]
        )
        assert set(result.output.evidence_checked) == {
            "manifest",
            "ci",
            "security",
            "change_risk",
            "capacity",
        }

        call_names = [entry["tool_name"] for entry in module.CALL_LOG]
        assert call_names == expected_tools
        assert [entry["tool_name"] for entry in result.tool_trace] == expected_tools

    asyncio.run(scenario())

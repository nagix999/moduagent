from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "12_release_readiness.py"


class NoCallModel:
    async def complete(self, request: Any) -> Any:
        del request
        raise AssertionError("building the example must not make a model request")


def _load_example() -> ModuleType:
    module_name = "_moduagent_example_12_release_readiness"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load example: {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_release_example_imports_without_network_or_embedded_secrets() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    compile(source, str(EXAMPLE), "exec")
    module = _load_example()

    assert callable(module.build_agent)
    assert "async with VLLMClient.from_env(" in source
    assert '"max_tokens": 768' in source
    assert "http://" not in source
    assert "https://" not in source
    assert "t62y46bwfim0hq" not in source
    assert "runpod-vllm-token" not in source
    assert "sk-" not in source
    assert "requests" not in source
    assert "subprocess" not in source
    assert ".write_text(" not in source


def test_release_builder_has_five_read_only_tools_and_bounded_standard_config() -> None:
    module = _load_example()
    agent = module.build_agent(NoCallModel())

    assert agent.inspect().execution_profile.kind == "standard"
    assert agent.inspect().output_contract["structured"] is True
    assert [tool.name for tool in agent.tool_registry] == [
        "get_release_manifest",
        "get_ci_summary",
        "get_security_scan",
        "assess_change_risk",
        "get_deployment_capacity",
    ]
    assert all(tool.idempotent for tool in agent.tool_registry)
    assert all(tool.timeout_seconds == 2.0 for tool in agent.tool_registry)
    assert all(tool.max_result_bytes == 4096 for tool in agent.tool_registry)
    assert agent.config.name == "release-readiness"
    assert agent.config.tool_trace_mode == "summary"
    assert agent.config.limits.max_steps == 8
    assert agent.config.limits.max_tool_calls == 6
    assert agent.config.limits.timeout_seconds == 120.0
    assert agent.config.limits.parallel_tool_calls is False
    assert agent.config.limits.max_model_turns == 10
    assert agent.config.limits.no_progress_model_turn_threshold == 3


def test_release_tools_return_deterministic_evidence_and_record_safe_calls() -> None:
    module = _load_example()
    module.CALL_LOG.clear()

    manifest = asyncio.run(
        module.get_release_manifest.invoke(
            {"release_id": " payments-api-2026.08.03-rc1 "}
        )
    )
    ci = asyncio.run(
        module.get_ci_summary.invoke({"commit_sha": manifest["commit_sha"]})
    )
    security = asyncio.run(
        module.get_security_scan.invoke({"commit_sha": manifest["commit_sha"]})
    )
    risk = asyncio.run(
        module.assess_change_risk.invoke(
            {"change_set_id": manifest["change_set_id"].lower()}
        )
    )
    capacity = asyncio.run(
        module.get_deployment_capacity.invoke({"environment": "production"})
    )

    assert manifest == {
        "found": True,
        "release_id": "payments-api-2026.08.03-rc1",
        "commit_sha": "a18c93f7d2b1",
        "change_set_id": "CHG-2048",
        "artifact_signed": True,
        "approvals_required": 2,
        "approvals_received": 2,
    }
    assert ci["status"] == "passed"
    assert ci["passed_checks"] == ci["required_checks"] == 8
    assert security == {
        "found": True,
        "commit_sha": "a18c93f7d2b1",
        "policy_status": "blocked",
        "critical_findings": 0,
        "high_findings": 1,
        "waiver_approved": False,
        "blocking_findings": ["SEC-431"],
    }
    assert risk["risk_level"] == "high"
    assert risk["database_migration"] is True
    assert risk["rollback_tested"] is True
    assert capacity["change_freeze"] is False
    assert capacity["active_sev1_or_sev2_incidents"] == 0
    assert capacity["concurrent_changes"] < capacity["concurrent_change_limit"]
    assert module.CALL_LOG == [
        {
            "tool_name": "get_release_manifest",
            "release_id": "payments-api-2026.08.03-rc1",
        },
        {"tool_name": "get_ci_summary", "commit_sha": "a18c93f7d2b1"},
        {"tool_name": "get_security_scan", "commit_sha": "a18c93f7d2b1"},
        {"tool_name": "assess_change_risk", "change_set_id": "CHG-2048"},
        {"tool_name": "get_deployment_capacity", "environment": "production"},
    ]
    assert all(
        "token" not in entry and "api_key" not in entry for entry in module.CALL_LOG
    )


def test_release_tools_fail_closed_for_unknown_evidence_and_invalid_environment() -> (
    None
):
    module = _load_example()

    unknown_manifest = asyncio.run(
        module.get_release_manifest.invoke({"release_id": "unknown-release"})
    )
    unknown_ci = asyncio.run(module.get_ci_summary.invoke({"commit_sha": "deadbee"}))

    assert unknown_manifest["found"] is False
    assert unknown_manifest["artifact_signed"] is False
    assert unknown_ci["found"] is False
    assert unknown_ci["status"] == "unknown"
    with pytest.raises(ValidationError):
        asyncio.run(module.get_deployment_capacity.invoke({"environment": "disaster"}))


def test_release_decision_enforces_evidence_and_ship_hold_consistency() -> None:
    module = _load_example()
    evidence = ["manifest", "ci", "security", "change_risk", "capacity"]

    hold = module.ReleaseDecision(
        release_id="payments-api-2026.08.03-rc1",
        decision="hold",
        risk_level="high",
        summary="Security policy blocks this release.",
        blocking_reasons=["SEC-431 is an unwaived high finding."],
        required_actions=["Remediate SEC-431 and rerun the security scan."],
        evidence_checked=evidence,
    )

    assert hold.decision == "hold"
    with pytest.raises(ValidationError, match="ship decision"):
        module.ReleaseDecision(
            release_id="release-2",
            decision="ship",
            risk_level="low",
            summary="Ready.",
            blocking_reasons=["This conflicts with ship."],
            required_actions=[],
            evidence_checked=evidence,
        )
    with pytest.raises(ValidationError, match="every required check"):
        module.ReleaseDecision(
            release_id="release-3",
            decision="hold",
            risk_level="medium",
            summary="Evidence is incomplete.",
            blocking_reasons=["Capacity was not checked."],
            required_actions=["Check capacity."],
            evidence_checked=["manifest", "ci", "security", "change_risk", "ci"],
        )

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
EXAMPLE = ROOT / "examples" / "11_customer_case_resolution.py"


class NoCallModel:
    async def complete(self, request: Any) -> Any:
        del request
        raise AssertionError("building the example must not access the network")


def _load_example() -> ModuleType:
    module_name = "_moduagent_example_11_customer_case_resolution"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load example: {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_example_imports_without_network_and_has_bounded_safe_configuration() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    compile(source, str(EXAMPLE), "exec")
    module = _load_example()
    agent = module.build_agent(NoCallModel())

    assert agent.inspect().execution_profile.kind == "standard"
    assert agent.inspect().output_contract["structured"] is True
    assert [tool.name for tool in agent.tool_registry] == [
        "lookup_case",
        "lookup_order",
        "get_return_policy",
        "assess_return_eligibility",
        "calculate_refund_quote",
    ]
    assert agent.config.limits.max_steps == 8
    assert agent.config.limits.max_tool_calls == 7
    assert agent.config.limits.timeout_seconds == 180
    assert agent.config.limits.parallel_tool_calls is False
    assert agent.config.limits.max_model_turns == 10
    assert agent.config.limits.no_progress_model_turn_threshold == 3
    assert agent.config.retry.max_attempts == 1
    assert agent.config.tool_trace_mode == "summary"

    assert "async with VLLMClient.from_env(" in source
    assert '"max_tokens": 768' in source
    assert "t62y46bwfim0hq" not in source
    assert "runpod-vllm-token" not in source
    assert "VLLM_API_KEY" not in source
    assert "execute_refund" not in source


def test_all_tools_are_bounded_and_safe_for_read_only_recovery() -> None:
    module = _load_example()
    tools = (
        module.lookup_case,
        module.lookup_order,
        module.get_return_policy,
        module.assess_return_eligibility,
        module.calculate_refund_quote,
    )

    for tool in tools:
        assert tool.idempotent is True
        assert tool.repair_safe is True
        assert tool.timeout_retry_safe is True
        assert tool.timeout_seconds == 2
        assert tool.max_result_bytes == 4096

    output_schema = module.CaseResolution.model_json_schema()
    assert output_schema["properties"]["human_approval_required"]["const"] is True
    assert output_schema["properties"]["write_action_performed"]["const"] is False


def test_tools_coordinate_a_verified_case_and_record_safe_call_log() -> None:
    module = _load_example()
    module.CALL_LOG.clear()

    case = asyncio.run(module.lookup_case.invoke({"case_id": " case-2048 "}))
    order = asyncio.run(module.lookup_order.invoke({"order_id": case["order_id"]}))
    policy = asyncio.run(
        module.get_return_policy.invoke(
            {
                "product_category": order["product_category"],
                "issue_type": case["issue_type"],
                "country": order["country"],
            }
        )
    )
    eligibility = asyncio.run(
        module.assess_return_eligibility.invoke(
            {
                "delivered_on": order["delivered_on"],
                "reported_on": case["reported_on"],
                "return_window_days": policy["return_window_days"],
                "requires_evidence": policy["requires_evidence"],
                "evidence_count": case["evidence_count"],
            }
        )
    )
    quote = asyncio.run(
        module.calculate_refund_quote.invoke(
            {
                "unit_price": order["unit_price"],
                "quantity": order["quantity"],
                "shipping_paid": order["shipping_paid"],
                "original_shipping_refundable": policy["original_shipping_refundable"],
                "return_shipping_fee": policy["return_shipping_fee"],
                "currency": order["currency"],
            }
        )
    )

    assert case == {
        "case_id": "CASE-2048",
        "status": "open",
        **module.CASES["CASE-2048"],
    }
    assert order == {"order_id": "ORD-2048", **module.ORDERS["ORD-2048"]}
    assert policy == {
        "status": "found",
        **module.RETURN_POLICIES[("small_appliance", "damaged", "KR")],
    }
    assert eligibility == {
        "decision": "eligible",
        "reason": "policy_requirements_met",
        "days_elapsed": 3,
        "within_window": True,
        "evidence_satisfied": True,
    }
    assert quote == {
        "currency": "KRW",
        "merchandise_amount": 129_000,
        "shipping_credit": 3_000,
        "deductions": 0,
        "total": 132_000,
        "binding": False,
    }
    assert [entry["tool"] for entry in module.CALL_LOG] == [
        "lookup_case",
        "lookup_order",
        "get_return_policy",
        "assess_return_eligibility",
        "calculate_refund_quote",
    ]
    assert module.CALL_LOG[-1]["currency"] == "KRW"
    assert "customer_id" not in module.CALL_LOG[-1]


def test_calculation_tools_reject_invalid_business_inputs() -> None:
    module = _load_example()

    with pytest.raises(ValueError, match="before delivered_on"):
        asyncio.run(
            module.assess_return_eligibility.invoke(
                {
                    "delivered_on": "2026-08-01",
                    "reported_on": "2026-07-31",
                    "return_window_days": 30,
                    "requires_evidence": True,
                    "evidence_count": 1,
                }
            )
        )
    with pytest.raises(ValueError, match="quantity"):
        asyncio.run(
            module.calculate_refund_quote.invoke(
                {
                    "unit_price": 129_000,
                    "quantity": 0,
                    "shipping_paid": 3_000,
                    "original_shipping_refundable": True,
                    "return_shipping_fee": 0,
                    "currency": "KRW",
                }
            )
        )


def test_structured_result_cannot_claim_an_external_action() -> None:
    module = _load_example()
    valid = {
        "case_id": "CASE-2048",
        "order_id": "ORD-2048",
        "decision": "eligible",
        "verified_summary": "The damaged item was reported within three days.",
        "recommended_resolution": "refund_after_return",
        "refund_quote": {
            "currency": "KRW",
            "merchandise_amount": 129_000,
            "shipping_credit": 3_000,
            "deductions": 0,
            "total": 132_000,
        },
        "next_steps": ["Obtain human approval."],
        "customer_message": "Your request is eligible for review.",
        "human_approval_required": True,
        "write_action_performed": False,
    }

    assert module.CaseResolution.model_validate(valid).write_action_performed is False

    with pytest.raises(ValidationError):
        module.CaseResolution.model_validate(
            {**valid, "human_approval_required": False}
        )
    with pytest.raises(ValidationError):
        module.CaseResolution.model_validate({**valid, "write_action_performed": True})

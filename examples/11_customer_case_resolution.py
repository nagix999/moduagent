"""Prepare a safe resolution proposal for a damaged-product support case.

This intermediate example coordinates five application-owned Tools.  Every
Tool is read-only or performs a calculation: the Agent can propose a refund,
but it cannot execute one or make any other external change.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from moduagent import Agent, ConsoleEventSink, RetryConfig, RunLimits, VLLMClient, tool


CASES = {
    "CASE-2048": {
        "customer_id": "CUS-441",
        "order_id": "ORD-2048",
        "item_id": "COF-9000",
        "issue_type": "damaged",
        "reported_on": "2026-07-28",
        "evidence_count": 2,
        "requested_resolution": "refund",
    }
}

ORDERS = {
    "ORD-2048": {
        "status": "delivered",
        "delivered_on": "2026-07-25",
        "country": "KR",
        "item_id": "COF-9000",
        "product_name": "Pour-over coffee maker",
        "product_category": "small_appliance",
        "quantity": 1,
        "unit_price": 129_000,
        "shipping_paid": 3_000,
        "currency": "KRW",
    }
}

RETURN_POLICIES = {
    ("small_appliance", "damaged", "KR"): {
        "return_window_days": 30,
        "requires_evidence": True,
        "original_shipping_refundable": True,
        "return_shipping_fee": 0,
        "allowed_resolutions": ["replacement", "refund_after_return"],
    }
}

# A real application would send this safe audit information to its telemetry
# system.  It is intentionally local and contains no Tool results or secrets.
CALL_LOG: list[dict[str, object]] = []


class RefundQuote(BaseModel):
    currency: Literal["KRW"]
    merchandise_amount: int = Field(ge=0)
    shipping_credit: int = Field(ge=0)
    deductions: int = Field(ge=0)
    total: int = Field(ge=0)


class CaseResolution(BaseModel):
    case_id: str
    order_id: str
    decision: Literal["eligible", "not_eligible", "manual_review"]
    verified_summary: str
    recommended_resolution: Literal[
        "refund_after_return",
        "replacement",
        "request_more_evidence",
        "manual_review",
    ]
    refund_quote: RefundQuote | None
    next_steps: list[str] = Field(min_length=1, max_length=5)
    customer_message: str
    human_approval_required: Literal[True]
    write_action_performed: Literal[False] = False


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=4096,
)
def lookup_case(case_id: str) -> dict[str, object]:
    """Return verified details for one customer-support case."""

    normalized_id = case_id.strip().upper()
    CALL_LOG.append({"tool": "lookup_case", "case_id": normalized_id})
    case = CASES.get(normalized_id)
    if case is None:
        return {"case_id": normalized_id, "status": "not_found"}
    return {"case_id": normalized_id, "status": "open", **case}


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=4096,
)
def lookup_order(order_id: str) -> dict[str, object]:
    """Return verified delivery, item, and payment facts for one order."""

    normalized_id = order_id.strip().upper()
    CALL_LOG.append({"tool": "lookup_order", "order_id": normalized_id})
    order = ORDERS.get(normalized_id)
    if order is None:
        return {"order_id": normalized_id, "status": "not_found"}
    return {"order_id": normalized_id, **order}


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=4096,
)
def get_return_policy(
    product_category: Literal["small_appliance", "electronics"],
    issue_type: Literal["damaged", "defective", "changed_mind"],
    country: Literal["KR", "US"],
) -> dict[str, object]:
    """Return the applicable return policy without changing the case."""

    CALL_LOG.append(
        {
            "tool": "get_return_policy",
            "product_category": product_category,
            "issue_type": issue_type,
            "country": country,
        }
    )
    policy = RETURN_POLICIES.get((product_category, issue_type, country))
    if policy is None:
        return {"status": "not_found"}
    return {"status": "found", **policy}


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=4096,
)
def assess_return_eligibility(
    delivered_on: str,
    reported_on: str,
    return_window_days: int,
    requires_evidence: bool,
    evidence_count: int,
) -> dict[str, object]:
    """Calculate return eligibility from verified dates and policy limits."""

    delivered = date.fromisoformat(delivered_on)
    reported = date.fromisoformat(reported_on)
    if reported < delivered:
        raise ValueError("reported_on cannot be before delivered_on")
    if not 1 <= return_window_days <= 90:
        raise ValueError("return_window_days must be between 1 and 90")
    if not 0 <= evidence_count <= 20:
        raise ValueError("evidence_count must be between 0 and 20")

    CALL_LOG.append(
        {
            "tool": "assess_return_eligibility",
            "delivered_on": delivered_on,
            "reported_on": reported_on,
            "return_window_days": return_window_days,
            "requires_evidence": requires_evidence,
            "evidence_count": evidence_count,
        }
    )
    days_elapsed = (reported - delivered).days
    within_window = days_elapsed <= return_window_days
    evidence_satisfied = not requires_evidence or evidence_count > 0
    eligible = within_window and evidence_satisfied
    if not within_window:
        reason = "outside_return_window"
    elif not evidence_satisfied:
        reason = "evidence_required"
    else:
        reason = "policy_requirements_met"
    return {
        "decision": "eligible" if eligible else "not_eligible",
        "reason": reason,
        "days_elapsed": days_elapsed,
        "within_window": within_window,
        "evidence_satisfied": evidence_satisfied,
    }


@tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=True,
    timeout_seconds=2,
    max_result_bytes=4096,
)
def calculate_refund_quote(
    unit_price: int,
    quantity: int,
    shipping_paid: int,
    original_shipping_refundable: bool,
    return_shipping_fee: int,
    currency: Literal["KRW"],
) -> dict[str, object]:
    """Calculate a non-binding refund quote; never issue a refund."""

    if not 1 <= quantity <= 20:
        raise ValueError("quantity must be between 1 and 20")
    if min(unit_price, shipping_paid, return_shipping_fee) < 0:
        raise ValueError("money inputs cannot be negative")

    CALL_LOG.append(
        {
            "tool": "calculate_refund_quote",
            "unit_price": unit_price,
            "quantity": quantity,
            "shipping_paid": shipping_paid,
            "original_shipping_refundable": original_shipping_refundable,
            "return_shipping_fee": return_shipping_fee,
            "currency": currency,
        }
    )
    merchandise_amount = unit_price * quantity
    shipping_credit = shipping_paid if original_shipping_refundable else 0
    subtotal = merchandise_amount + shipping_credit
    deductions = min(return_shipping_fee, subtotal)
    return {
        "currency": currency,
        "merchandise_amount": merchandise_amount,
        "shipping_credit": shipping_credit,
        "deductions": deductions,
        "total": subtotal - deductions,
        "binding": False,
    }


def build_agent(model, *, event_sink=None):
    return Agent.create(
        model=model,
        name="customer-case-resolution",
        instructions=(
            "Prepare a customer-support resolution proposal from verified Tool "
            "results. Call exactly one Tool per response in this order: "
            "lookup_case, lookup_order, get_return_policy, "
            "assess_return_eligibility, and calculate_refund_quote. Copy IDs, "
            "dates, policy values, and money amounts exactly between Tools. If "
            "any lookup is not_found, return manual_review and do not guess. "
            "This Agent is advisory only: never claim that a refund, return, "
            "replacement, case update, email, or other write action happened. "
            "Set human_approval_required=true and write_action_performed=false. "
            "Finish with the requested CaseResolution."
        ),
        tools=[
            lookup_case,
            lookup_order,
            get_return_policy,
            assess_return_eligibility,
            calculate_refund_quote,
        ],
        execution="standard",
        output=CaseResolution,
        limits=RunLimits(
            max_steps=8,
            max_tool_calls=7,
            timeout_seconds=180,
            parallel_tool_calls=False,
            max_model_turns=10,
            no_progress_model_turn_threshold=3,
        ),
        retry=RetryConfig(max_attempts=1),
        tool_trace_mode="summary",
        event_sink=event_sink,
    )


async def main() -> None:
    CALL_LOG.clear()
    async with VLLMClient.from_env(
        timeout=60,
        default_options={"temperature": 0, "max_tokens": 768},
    ) as model:
        agent = build_agent(model, event_sink=ConsoleEventSink())
        result = await agent.run(
            "Review CASE-2048 and prepare a safe return and refund proposal. "
            "Do not execute any action."
        )

    result.raise_for_error()
    resolution: CaseResolution = result.unwrap()
    print(resolution.model_dump_json(indent=2))
    print("tools:", [entry["tool"] for entry in CALL_LOG])
    print("run usage:", dict(result.run_usage))
    print("tool trace:", [entry["tool_name"] for entry in result.tool_trace])


if __name__ == "__main__":
    asyncio.run(main())

"""Decide whether a software release is ready without deploying anything."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

from moduagent import Agent, ConsoleEventSink, RunLimits, VLLMClient, tool


ReleaseId = Annotated[str, Field(min_length=3, max_length=80)]
CommitSha = Annotated[str, Field(min_length=7, max_length=40)]
ChangeSetId = Annotated[str, Field(min_length=3, max_length=40)]

CheckName = Literal["manifest", "ci", "security", "change_risk", "capacity"]


class ManifestResult(TypedDict, total=False):
    found: bool
    release_id: str
    commit_sha: str
    change_set_id: str
    artifact_signed: bool
    approvals_required: int
    approvals_received: int


class CISummary(TypedDict, total=False):
    found: bool
    commit_sha: str
    status: Literal["passed", "failed", "unknown"]
    required_checks: int
    passed_checks: int


class SecuritySummary(TypedDict, total=False):
    found: bool
    commit_sha: str
    policy_status: Literal["passed", "blocked", "unknown"]
    critical_findings: int
    high_findings: int
    waiver_approved: bool
    blocking_findings: list[str]


class ChangeRisk(TypedDict, total=False):
    found: bool
    change_set_id: str
    risk_level: Literal["low", "medium", "high", "critical", "unknown"]
    database_migration: bool
    rollback_tested: bool
    feature_flagged: bool


class DeploymentCapacity(TypedDict):
    environment: Literal["staging", "production"]
    change_freeze: bool
    active_sev1_or_sev2_incidents: int
    on_call_primary_available: bool
    concurrent_changes: int
    concurrent_change_limit: int


class ReleaseDecision(BaseModel):
    release_id: str
    decision: Literal["ship", "hold"]
    risk_level: Literal["low", "medium", "high", "critical"]
    summary: str = Field(min_length=1)
    blocking_reasons: list[str]
    required_actions: list[str]
    evidence_checked: list[CheckName] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_decision(self) -> ReleaseDecision:
        expected = {"manifest", "ci", "security", "change_risk", "capacity"}
        if set(self.evidence_checked) != expected:
            raise ValueError("evidence_checked must contain every required check once")
        if self.decision == "ship" and self.blocking_reasons:
            raise ValueError("a ship decision cannot contain blocking reasons")
        if self.decision == "hold" and not self.blocking_reasons:
            raise ValueError("a hold decision requires at least one blocking reason")
        return self


RELEASE_ID = "payments-api-2026.08.03-rc1"
COMMIT_SHA = "a18c93f7d2b1"
CHANGE_SET_ID = "CHG-2048"

# This is only for demonstrating and testing Tool orchestration. Do not record
# credentials, full Tool results, or other sensitive values in application logs.
CALL_LOG: list[dict[str, object]] = []


@tool(idempotent=True, timeout_seconds=2.0, max_result_bytes=4096)
def get_release_manifest(release_id: ReleaseId) -> ManifestResult:
    """Return immutable build identity, signing, and approval evidence."""

    normalized = release_id.strip()
    CALL_LOG.append({"tool_name": "get_release_manifest", "release_id": normalized})
    if normalized == RELEASE_ID:
        return {
            "found": True,
            "release_id": RELEASE_ID,
            "commit_sha": COMMIT_SHA,
            "change_set_id": CHANGE_SET_ID,
            "artifact_signed": True,
            "approvals_required": 2,
            "approvals_received": 2,
        }
    return {
        "found": False,
        "release_id": normalized,
        "commit_sha": "",
        "change_set_id": "",
        "artifact_signed": False,
        "approvals_required": 0,
        "approvals_received": 0,
    }


@tool(idempotent=True, timeout_seconds=2.0, max_result_bytes=4096)
def get_ci_summary(commit_sha: CommitSha) -> CISummary:
    """Return the required-check and test summary for an exact commit."""

    normalized = commit_sha.strip().lower()
    CALL_LOG.append({"tool_name": "get_ci_summary", "commit_sha": normalized})
    if normalized == COMMIT_SHA:
        return {
            "found": True,
            "commit_sha": COMMIT_SHA,
            "status": "passed",
            "required_checks": 8,
            "passed_checks": 8,
        }
    return {
        "found": False,
        "commit_sha": normalized,
        "status": "unknown",
        "required_checks": 0,
        "passed_checks": 0,
    }


@tool(idempotent=True, timeout_seconds=2.0, max_result_bytes=4096)
def get_security_scan(commit_sha: CommitSha) -> SecuritySummary:
    """Return policy-level vulnerability evidence for an exact commit."""

    normalized = commit_sha.strip().lower()
    CALL_LOG.append({"tool_name": "get_security_scan", "commit_sha": normalized})
    if normalized == COMMIT_SHA:
        return {
            "found": True,
            "commit_sha": COMMIT_SHA,
            "policy_status": "blocked",
            "critical_findings": 0,
            "high_findings": 1,
            "waiver_approved": False,
            "blocking_findings": ["SEC-431"],
        }
    return {
        "found": False,
        "commit_sha": normalized,
        "policy_status": "unknown",
        "critical_findings": 0,
        "high_findings": 0,
        "waiver_approved": False,
        "blocking_findings": [],
    }


@tool(idempotent=True, timeout_seconds=2.0, max_result_bytes=4096)
def assess_change_risk(change_set_id: ChangeSetId) -> ChangeRisk:
    """Return rollout and rollback risk for an approved change set."""

    normalized = change_set_id.strip().upper()
    CALL_LOG.append({"tool_name": "assess_change_risk", "change_set_id": normalized})
    if normalized == CHANGE_SET_ID:
        return {
            "found": True,
            "change_set_id": CHANGE_SET_ID,
            "risk_level": "high",
            "database_migration": True,
            "rollback_tested": True,
            "feature_flagged": True,
        }
    return {
        "found": False,
        "change_set_id": normalized,
        "risk_level": "unknown",
        "database_migration": False,
        "rollback_tested": False,
        "feature_flagged": False,
    }


@tool(idempotent=True, timeout_seconds=2.0, max_result_bytes=4096)
def get_deployment_capacity(
    environment: Literal["staging", "production"],
) -> DeploymentCapacity:
    """Return incident, staffing, freeze, and concurrency capacity."""

    CALL_LOG.append(
        {"tool_name": "get_deployment_capacity", "environment": environment}
    )
    production = environment == "production"
    return {
        "environment": environment,
        "change_freeze": False,
        "active_sev1_or_sev2_incidents": 0,
        "on_call_primary_available": True,
        "concurrent_changes": 2 if production else 0,
        "concurrent_change_limit": 3 if production else 2,
    }


def build_agent(model, *, event_sink=None):
    return Agent.create(
        name="release-readiness",
        model=model,
        instructions=(
            "Evaluate release readiness using evidence, without deploying or "
            "changing any system. For the requested release, call each Tool "
            "exactly once. Start with get_release_manifest. Copy its exact "
            "commit_sha into get_ci_summary and get_security_scan, and its exact "
            "change_set_id into assess_change_risk. Check production with "
            "get_deployment_capacity. HOLD if evidence is missing, the artifact "
            "is unsigned, approvals or CI checks are incomplete, CI failed, the "
            "security policy is blocked, an unsafe migration cannot roll back, "
            "a freeze or severity-1/2 incident is active, on-call is unavailable, "
            "or concurrent capacity is exhausted. Otherwise SHIP. Never invent "
            "identifiers or evidence. Finish with ReleaseDecision and list all "
            "five evidence names. A ship decision has no blocking reasons; a "
            "hold decision has at least one."
        ),
        tools=[
            get_release_manifest,
            get_ci_summary,
            get_security_scan,
            assess_change_risk,
            get_deployment_capacity,
        ],
        execution="standard",
        output=ReleaseDecision,
        limits=RunLimits(
            max_steps=8,
            max_tool_calls=6,
            timeout_seconds=120.0,
            parallel_tool_calls=False,
            max_model_turns=10,
            no_progress_model_turn_threshold=3,
        ),
        tool_trace_mode="summary",
        event_sink=event_sink,
    )


async def main() -> None:
    CALL_LOG.clear()
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 768},
    ) as model:
        result = await build_agent(model, event_sink=ConsoleEventSink()).run(
            "Should payments-api-2026.08.03-rc1 ship to production now?"
        )

    result.raise_for_error()
    decision = ReleaseDecision.model_validate(result.output)
    print(decision.model_dump_json(indent=2))
    print("checks:", [entry["tool_name"] for entry in CALL_LOG])
    print("run usage:", dict(result.run_usage))
    print("tool trace:", [entry["tool_name"] for entry in result.tool_trace])


if __name__ == "__main__":
    asyncio.run(main())

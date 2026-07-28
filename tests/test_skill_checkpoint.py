from __future__ import annotations

import pytest

from moduagent.decision.planning import ExecutionState, Plan, PlanStep, RunPhase
from moduagent.messages import Message
from moduagent.persistence.checkpoint import RunCheckpoint
from moduagent.runtime.context import (
    RunContext,
    RunRequest,
    SkillActivationState,
    SkillRunState,
)
from moduagent.runtime.events import EventType


def _skill_state() -> SkillRunState:
    return SkillRunState(
        catalog_digest="sha256:catalog",
        active_skills=(
            SkillActivationState(
                name="invoice-review",
                version="1.2.0",
                digest="sha256:skill",
                source_id="filesystem:./skills",
                selected_by="explicit",
                allowed_tools=("lookup-invoice", "lookup-vendor"),
                metadata={"tenant": "company"},
            ),
        ),
        resource_reads=2,
        instruction_tokens=120,
        resource_tokens=40,
    )


def test_checkpoint_v3_round_trips_skill_and_strict_execution_state() -> None:
    execution_state = ExecutionState(
        phase=RunPhase.STEP_PREPARE,
        plan=Plan(
            [
                PlanStep(
                    step_id="review",
                    objective="Review the invoice",
                    completion_criteria=["The invoice was reviewed"],
                )
            ]
        ),
        current_step_id="review",
    )
    context = RunContext(
        run_id="run-skills",
        request=RunRequest(
            input="검토해줘",
            session_id="session-skills",
            requested_skills=("invoice-review",),
            skill_mode="explicit",
        ),
        messages=[Message.user("검토해줘")],
        internal_messages=[Message.assistant("private executor draft")],
        execution_state=execution_state,
        current_run_start=0,
        skill_state=_skill_state(),
    )

    encoded = RunCheckpoint.from_context(context).to_dict()
    decoded = RunCheckpoint.from_dict(encoded)
    restored = decoded.to_context()

    assert encoded["version"] == 3
    assert restored.request.requested_skills == ("invoice-review",)
    assert restored.request.skill_mode == "explicit"
    assert restored.skill_state == _skill_state()
    assert restored.internal_messages == [
        Message.assistant("private executor draft"),
    ]
    assert isinstance(restored.execution_state, ExecutionState)
    assert restored.execution_state.phase is RunPhase.STEP_PREPARE
    assert restored.execution_state.current_step_id == "review"
    assert restored.skill_state.active_skills[0].allowed_tools == (
        "lookup-invoice",
        "lookup-vendor",
    )


def test_version_1_checkpoint_resumes_with_skills_disabled() -> None:
    legacy = {
        "version": 1,
        "run_id": "legacy-run",
        "session_id": "legacy-session",
        "input": "hello",
        "messages": [Message.user("hello").to_dict()],
        "status": "running",
    }

    checkpoint = RunCheckpoint.from_dict(legacy)
    context = checkpoint.to_context()

    assert context.request.requested_skills == ()
    assert context.request.skill_mode == "disabled"
    assert context.skill_state == SkillRunState()
    assert context.execution_state is None
    assert checkpoint.to_dict()["version"] == 3


def test_version_2_checkpoint_restores_skills_but_not_strict_execution_state() -> None:
    legacy = {
        "version": 2,
        "run_id": "version-2-run",
        "session_id": "version-2-session",
        "input": "review",
        "requested_skills": ["invoice-review"],
        "skill_mode": "explicit",
        "messages": [Message.user("review").to_dict()],
        "status": "running",
        "current_run_start": 0,
        "skill_state": _skill_state().to_dict(),
        # Version 2 did not define this field. It must not be interpreted as
        # resumable strict state even if a producer happened to include it.
        "execution_state": {
            "phase": "done",
            "final_response": "untrusted legacy final",
            "final_emitted": True,
        },
    }

    context = RunCheckpoint.from_dict(legacy).to_context()

    assert context.request.requested_skills == ("invoice-review",)
    assert context.request.skill_mode == "explicit"
    assert context.skill_state == _skill_state()
    assert context.execution_state is None


def test_checkpoint_rejects_unknown_version_and_invalid_skill_state() -> None:
    payload = {
        "version": 4,
        "run_id": "future-run",
        "session_id": "future-session",
        "messages": [],
    }
    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        RunCheckpoint.from_dict(payload)

    with pytest.raises(ValueError, match="duplicate names"):
        SkillRunState(
            active_skills=(
                SkillActivationState("same"),
                SkillActivationState("same"),
            )
        )


def test_skill_event_types_have_stable_wire_values() -> None:
    assert EventType.SKILLS_DISCOVERED.value == "skills_discovered"
    assert EventType.SKILL_SELECTION_COMPLETED.value == "skill_selection_completed"
    assert EventType.SKILL_ACTIVATED.value == "skill_activated"
    assert EventType.SKILL_RESOURCE_READ.value == "skill_resource_read"
    assert EventType.SKILL_DENIED.value == "skill_denied"
    assert EventType.SKILL_ERROR.value == "skill_error"

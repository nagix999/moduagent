from __future__ import annotations

import pytest

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


def test_checkpoint_v2_round_trips_skill_request_and_run_state() -> None:
    context = RunContext(
        run_id="run-skills",
        request=RunRequest(
            input="검토해줘",
            session_id="session-skills",
            requested_skills=("invoice-review",),
            skill_mode="explicit",
        ),
        messages=[Message.user("검토해줘")],
        current_run_start=0,
        skill_state=_skill_state(),
    )

    encoded = RunCheckpoint.from_context(context).to_dict()
    decoded = RunCheckpoint.from_dict(encoded)
    restored = decoded.to_context()

    assert encoded["version"] == 2
    assert restored.request.requested_skills == ("invoice-review",)
    assert restored.request.skill_mode == "explicit"
    assert restored.skill_state == _skill_state()
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
    assert checkpoint.to_dict()["version"] == 2


def test_checkpoint_rejects_unknown_version_and_invalid_skill_state() -> None:
    payload = {
        "version": 3,
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

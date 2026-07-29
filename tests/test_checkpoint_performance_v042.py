from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from moduagent.agent import Agent
from moduagent.config import AgentConfig
from moduagent.decision import (
    Plan,
    PlanAndExecutePolicy,
    PlanStep,
)
from moduagent.execution import (
    DurableBoundary,
    EngineSnapshot,
)
from moduagent.messages import FinishReason, Message, Usage
from moduagent.models import ModelCapabilities, ModelRequest, ModelResponse
from moduagent.persistence import (
    InMemoryCheckpointStore,
    RunCheckpoint,
)
from moduagent.persistence.checkpoint import _build_run_snapshot
from moduagent.runtime.context import RunContext, RunRequest, RunStatus
from moduagent.runtime.events import EventType


class ScriptedModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request: ModelRequest):
        raise AssertionError("stream should not be called")
        yield


class StaticPlanGenerator:
    def __init__(self, steps: list[PlanStep]) -> None:
        self.plan = Plan(steps)

    async def create(self, context: RunContext) -> Plan:
        del context
        return Plan.from_dict(self.plan.to_dict())

    async def revise(
        self,
        context: RunContext,
        plan: Plan,
        feedback: str,
    ) -> Plan:
        del context, feedback
        return Plan.from_dict(plan.to_dict())


def _count_engine_encodes(agent: Agent) -> list[Any]:
    encoded_states: list[Any] = []
    original = agent.engine.encode_state

    def counting_encode(state: Any) -> Mapping[str, Any]:
        encoded_states.append(state)
        return original(state)

    agent.engine.encode_state = counting_encode  # type: ignore[method-assign]
    return encoded_states


def test_standard_engine_skips_state_encoding_without_checkpoint_store() -> None:
    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig("no-checkpoint-standard", "Answer."),
            model=ScriptedModel([ModelResponse(Message.assistant("done"))]),
        )
        encoded_states = _count_engine_encodes(agent)

        result = await agent.run("work", session_id="no-checkpoint-standard")

        assert result.finish_reason is FinishReason.COMPLETED
        assert result.output == "done"
        assert encoded_states == []

    asyncio.run(scenario())


def test_plan_engine_skips_state_encoding_without_checkpoint_store() -> None:
    async def scenario() -> None:
        step_result = (
            '{"step_id":"inspect","status":"completed",'
            '"facts":["checked"],"completion_evidence":["checked"]}'
        )
        agent = Agent(
            config=AgentConfig("no-checkpoint-plan", "Inspect and answer."),
            model=ScriptedModel(
                [
                    ModelResponse(Message.assistant(step_result)),
                    ModelResponse(Message.assistant("inspection complete")),
                ]
            ),
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="inspect",
                            objective="inspect",
                            completion_criteria=["checked"],
                        )
                    ]
                )
            ),
        )
        encoded_states = _count_engine_encodes(agent)

        result = await agent.run("inspect", session_id="no-checkpoint-plan")

        assert result.finish_reason is FinishReason.COMPLETED
        assert result.output == "inspection complete"
        assert encoded_states == []

    asyncio.run(scenario())


@pytest.mark.parametrize("engine_id", ["standard", "plan"])
def test_native_v4_builder_matches_legacy_runtime_projection(
    engine_id: str,
) -> None:
    created_at = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    updated_at = datetime(2026, 7, 1, 12, 1, tzinfo=timezone.utc)
    system = Message.system("Work safely.")
    ephemeral = Message.assistant(
        "private protocol",
        metadata={"moduagent.ephemeral": True},
    )
    user = Message.user("work")
    internal = Message.assistant(
        "internal retained",
        metadata={"moduagent.internal": True},
    )
    context = RunContext(
        run_id=f"native-{engine_id}",
        request=RunRequest(
            "work",
            f"session-{engine_id}",
            user_context={"tenant": "acme"},
            requested_skills=("reporting",),
            skill_mode="explicit",
        ),
        messages=[system, ephemeral, user],
        new_messages=[user],
        internal_messages=[ephemeral, internal],
        status=RunStatus.RUNNING,
        step=3,
        tool_call_count=1,
        policy_state={
            "custom": {"value": 1},
            "_moduagent_engine_snapshot": {"discard": True},
        },
        usage=Usage(10, 4, 14, {"cached": 2}),
        metadata={
            "visible": {"value": 1},
            "database_password": "private",
            "_moduagent_agent_fingerprint": "sha256:old",
            "_moduagent_event_sequence": 7,
            "_moduagent_resume_safety": "resumable",
            "_moduagent_runtime_version": "0.4.2",
        },
        current_run_start=2,
        created_at=created_at,
    )
    finalization = {
        "started": True,
        "response_generated": True,
        "response": "stable answer",
        "invocation_count": 1,
        "persisted": False,
        "emitted": False,
    }
    engine = EngineSnapshot(
        engine_id,
        1,
        {
            "phase": "finalize",
            "finalization": finalization,
            **(
                {"model_turn": 3, "tool_call_count": 1, "terminal": {}}
                if engine_id == "standard"
                else {
                    "plan_progress": {
                        "plan": {
                            "steps": [],
                            "current_index": 0,
                            "version": 1,
                        },
                        "committed_results": {},
                        "replan_count": 0,
                    },
                    "step_execution": {
                        "current_step_id": None,
                        "pending_step_result": None,
                        "validation_feedback": None,
                        "step_attempt_count": 0,
                    },
                    "tool_recovery": {
                        "active_calls": {},
                        "seen_call_ids": [],
                        "pending_failure": None,
                        "repair_count_by_step": {},
                        "total_repairs": 0,
                        "terminal_failure": None,
                    },
                    "terminal": {},
                }
            ),
        },
    )
    context.metadata["_moduagent_engine"] = {
        "engine_id": engine.engine_id,
        "state_version": engine.state_version,
        "state": dict(engine.state),
        "durable_boundary": DurableBoundary.FINALIZATION_RESPONSE.value,
    }
    compatibility_policy_state = {
        "custom": {"value": 1},
        "_moduagent_engine_initialized": True,
    }
    fingerprint = "sha256:current"

    checkpoint = replace(
        RunCheckpoint.from_context(context),
        execution_state=None,
        policy_state=compatibility_policy_state,
        engine_id=engine.engine_id,
        engine_state_version=engine.state_version,
        engine_state=engine.state,
        updated_at=updated_at,
    )
    legacy = checkpoint.to_snapshot()
    legacy = replace(
        legacy,
        agent_fingerprint=fingerprint,
        common_state=replace(
            legacy.common_state,
            compatibility_policy_state=compatibility_policy_state,
        ),
        engine=engine,
    )
    native = _build_run_snapshot(
        context,
        engine,
        compatibility_policy_state=compatibility_policy_state,
        agent_fingerprint=fingerprint,
        updated_at=updated_at,
    )

    assert native.to_dict() == legacy.to_dict()
    assert native.sanitized_runtime_metadata["database_password"] == "[REDACTED]"
    assert "private protocol" not in native.to_json()
    assert native.common_state.current_run_start == 1


def test_snapshot_store_runtime_does_not_use_legacy_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import moduagent.persistence.checkpoint as checkpoint_module

        def fail_legacy_migration(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise AssertionError("native v4 saves must not use legacy migration")

        monkeypatch.setattr(
            checkpoint_module,
            "migrate_checkpoint_payload",
            fail_legacy_migration,
        )
        agent = Agent(
            config=AgentConfig("native-runtime", "Answer."),
            model=ScriptedModel([ModelResponse(Message.assistant("done"))]),
            checkpoint_store=InMemoryCheckpointStore(),
        )

        result = await agent.run("work", session_id="native-runtime")

        assert result.finish_reason is FinishReason.COMPLETED
        assert result.output == "done"

    asyncio.run(scenario())


def test_legacy_checkpoint_store_save_contract_remains_supported() -> None:
    class LegacyCheckpointStore:
        def __init__(self) -> None:
            self.saved: list[tuple[str, RunContext]] = []
            self.deleted: list[str] = []

        async def load(self, run_id: str) -> RunCheckpoint | None:
            del run_id
            return None

        async def save(self, run_id: str, context: RunContext) -> None:
            self.saved.append((run_id, context))

        async def delete(self, run_id: str) -> None:
            self.deleted.append(run_id)

    async def scenario() -> None:
        store = LegacyCheckpointStore()
        agent = Agent(
            config=AgentConfig("legacy-store", "Answer."),
            model=ScriptedModel([ModelResponse(Message.assistant("done"))]),
            checkpoint_store=store,
        )

        result = await agent.run("work", session_id="legacy-store")

        assert result.finish_reason is FinishReason.COMPLETED
        assert store.saved
        assert all(run_id == result.run_id for run_id, _ in store.saved)
        assert store.deleted == [result.run_id]

    asyncio.run(scenario())


def test_plan_finalization_started_boundary_is_saved_once() -> None:
    async def scenario() -> None:
        step_result = (
            '{"step_id":"inspect","status":"completed",'
            '"facts":["checked"],"completion_evidence":["checked"]}'
        )
        agent = Agent(
            config=AgentConfig("single-finalization-start", "Inspect and answer."),
            model=ScriptedModel(
                [
                    ModelResponse(Message.assistant(step_result)),
                    ModelResponse(Message.assistant("inspection complete")),
                ]
            ),
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(
                    [
                        PlanStep(
                            step_id="inspect",
                            objective="inspect",
                            completion_criteria=["checked"],
                        )
                    ]
                )
            ),
            checkpoint_store=InMemoryCheckpointStore(),
        )

        events = [
            event
            async for event in agent.stream_all(
                "inspect",
                session_id="single-finalization-start",
            )
        ]
        starts = [
            event
            for event in events
            if event.type is EventType.CHECKPOINT_SAVED
            and event.data.get("boundary") == DurableBoundary.FINALIZATION_STARTED.value
        ]

        assert events[-1].type is EventType.RUN_COMPLETED
        assert len(starts) == 1

    asyncio.run(scenario())

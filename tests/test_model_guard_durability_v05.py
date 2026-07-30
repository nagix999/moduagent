from __future__ import annotations

import asyncio
from contextlib import suppress

from moduagent import (
    Agent,
    AgentConfig,
    InMemoryCheckpointStore,
    LLMPlanGenerator,
    Message,
    ModelCapabilities,
    ModelResponse,
    PlanAndExecutePolicy,
    RetryConfig,
    RunLimits,
    ToolCall,
    tool,
)
from moduagent.persistence import RunSnapshot


_MODEL_GUARD_POLICY_KEY = "_moduagent_model_guard"


class _RecordingCheckpointStore(InMemoryCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self.latest: RunSnapshot | None = None

    async def save_snapshot(self, snapshot: RunSnapshot) -> None:
        await super().save_snapshot(snapshot)
        self.latest = snapshot


class _BlockingFirstAttemptModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self) -> None:
        self.calls = 0
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def complete(self, request: object) -> ModelResponse:
        del request
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            await self.release_first.wait()
        return ModelResponse(Message.assistant("unexpected second provider call"))


class _RetryThenBlockModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self) -> None:
        self.calls = 0
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()

    async def complete(self, request: object) -> ModelResponse:
        del request
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("retry this attempt")
        self.second_started.set()
        await self.release_second.wait()
        return ModelResponse(Message.assistant("done"))


class _BlockingPlanModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self) -> None:
        self.calls = 0
        self.first_started = asyncio.Event()
        self.second_started = asyncio.Event()

    async def complete(self, request: object) -> ModelResponse:
        del request
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
        elif self.calls == 2:
            self.second_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _agent(
    model: object,
    checkpoints: InMemoryCheckpointStore,
    *,
    max_model_turns: int,
    max_attempts: int,
) -> Agent:
    return Agent(
        config=AgentConfig(
            "durable-model-guard",
            "Answer briefly.",
            limits=RunLimits(
                max_model_turns=max_model_turns,
                no_progress_model_turn_threshold=10,
            ),
            retry=RetryConfig(
                max_attempts=max_attempts,
                initial_delay=0,
                max_delay=0,
            ),
            finalization_mode="disabled",
        ),
        model=model,  # type: ignore[arg-type]
        checkpoint_store=checkpoints,
    )


def test_provider_attempt_reservation_survives_hard_crash_and_resume() -> None:
    async def scenario() -> None:
        checkpoints = _RecordingCheckpointStore()
        model = _BlockingFirstAttemptModel()
        agent = _agent(
            model,
            checkpoints,
            max_model_turns=1,
            max_attempts=1,
        )

        running = asyncio.create_task(
            agent.run("work", session_id="hard-crash-session")
        )
        await asyncio.wait_for(model.first_started.wait(), timeout=2)

        latest = checkpoints.latest
        assert latest is not None
        crashed_snapshot = await checkpoints.load_snapshot(latest.run_id)
        assert crashed_snapshot is not None
        guard_state = crashed_snapshot.common_state.compatibility_policy_state[
            _MODEL_GUARD_POLICY_KEY
        ]
        assert guard_state["model_turns"] == 1

        # Cancelling performs normal cleanup, so restore the exact snapshot
        # captured while the provider request was in flight. This models a
        # process hard crash where no finally block can run.
        running.cancel()
        with suppress(asyncio.CancelledError):
            await running
        await checkpoints.save_snapshot(crashed_snapshot)

        resumed = await agent.resume(
            crashed_snapshot.run_id,
            session_id="hard-crash-session",
        )

        assert resumed.finish_reason.value == "max_model_turns"
        assert resumed.metadata["error_summary"]["resumable"] is False
        assert model.calls == 1
        terminal_snapshot = await checkpoints.load_snapshot(crashed_snapshot.run_id)
        assert terminal_snapshot is not None
        assert terminal_snapshot.common_state.resume_safety == "not_resumable"

        rejected = await agent.resume(
            crashed_snapshot.run_id,
            session_id="hard-crash-session",
        )

        assert rejected.finish_reason.value == "error"
        assert rejected.metadata["error_summary"]["code"] == (
            "checkpoint_migration_failed"
        )
        assert rejected.metadata["error_summary"]["resumable"] is False
        assert model.calls == 1

    asyncio.run(scenario())


def test_each_retry_attempt_is_durably_reserved_before_provider_io() -> None:
    async def scenario() -> None:
        checkpoints = _RecordingCheckpointStore()
        model = _RetryThenBlockModel()
        agent = _agent(
            model,
            checkpoints,
            max_model_turns=5,
            max_attempts=2,
        )

        running = asyncio.create_task(
            agent.run("work", session_id="retry-reservation-session")
        )
        await asyncio.wait_for(model.second_started.wait(), timeout=2)

        latest = checkpoints.latest
        assert latest is not None
        snapshot = await checkpoints.load_snapshot(latest.run_id)
        assert snapshot is not None
        guard_state = snapshot.common_state.compatibility_policy_state[
            _MODEL_GUARD_POLICY_KEY
        ]
        assert guard_state["model_turns"] == 2
        assert model.calls == 2

        running.cancel()
        with suppress(asyncio.CancelledError):
            await running

    asyncio.run(scenario())


def test_plan_creation_uses_non_initialized_bootstrap_on_crash_and_resume() -> None:
    async def scenario() -> None:
        checkpoints = _RecordingCheckpointStore()
        model = _BlockingPlanModel()
        agent = Agent(
            config=AgentConfig(
                "durable-plan-bootstrap",
                "Create and execute a small plan.",
                limits=RunLimits(
                    max_steps=1,
                    max_model_turns=5,
                    no_progress_model_turn_threshold=5,
                ),
                retry=RetryConfig(max_attempts=1),
            ),
            model=model,
            decision_policy=PlanAndExecutePolicy(LLMPlanGenerator(model, max_steps=1)),
            checkpoint_store=checkpoints,
        )

        running = asyncio.create_task(
            agent.run("plan it", session_id="plan-bootstrap-session")
        )
        await asyncio.wait_for(model.first_started.wait(), timeout=2)

        latest = checkpoints.latest
        assert latest is not None
        crashed_snapshot = await checkpoints.load_snapshot(latest.run_id)
        assert crashed_snapshot is not None
        assert crashed_snapshot.engine.state == {}
        assert (
            crashed_snapshot.common_state.compatibility_policy_state[
                "_moduagent_engine_initialized"
            ]
            is False
        )
        assert (
            crashed_snapshot.common_state.compatibility_policy_state[
                _MODEL_GUARD_POLICY_KEY
            ]["model_turns"]
            == 1
        )

        running.cancel()
        with suppress(asyncio.CancelledError):
            await running
        await checkpoints.save_snapshot(crashed_snapshot)

        resumed = asyncio.create_task(
            agent.resume(
                crashed_snapshot.run_id,
                session_id="plan-bootstrap-session",
            )
        )
        await asyncio.wait_for(model.second_started.wait(), timeout=2)

        resumed_snapshot = await checkpoints.load_snapshot(crashed_snapshot.run_id)
        assert resumed_snapshot is not None
        assert resumed_snapshot.engine.state == {}
        assert (
            resumed_snapshot.common_state.compatibility_policy_state[
                "_moduagent_engine_initialized"
            ]
            is False
        )
        assert (
            resumed_snapshot.common_state.compatibility_policy_state[
                _MODEL_GUARD_POLICY_KEY
            ]["model_turns"]
            == 2
        )
        assert model.calls == 2

        resumed.cancel()
        with suppress(asyncio.CancelledError):
            await resumed

    asyncio.run(scenario())


def test_no_progress_guard_terminal_is_not_resumable() -> None:
    @tool
    def lookup(query: str) -> str:
        """Look up a value."""

        del query
        raise ValueError("invalid input")

    class _RepeatingModel:
        capabilities = ModelCapabilities(streaming=False)

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: object) -> ModelResponse:
            del request
            self.calls += 1
            return ModelResponse(
                Message.assistant(
                    None,
                    (
                        ToolCall(
                            id=f"provider-{self.calls}",
                            name="lookup",
                            arguments={"query": "same"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            )

    async def scenario() -> None:
        model = _RepeatingModel()
        agent = Agent(
            config=AgentConfig(
                "terminal-no-progress",
                "Use lookup.",
                limits=RunLimits(
                    max_model_turns=10,
                    no_progress_model_turn_threshold=2,
                ),
                finalization_mode="disabled",
            ),
            model=model,
            tools=(lookup,),
            checkpoint_store=InMemoryCheckpointStore(),
        )

        result = await agent.run(
            "find it",
            session_id="terminal-no-progress-session",
        )

        assert result.finish_reason.value == "no_progress"
        assert result.metadata["error_summary"]["resumable"] is False
        snapshot = await agent.runtime.checkpoint_store.load_snapshot(result.run_id)
        assert snapshot is not None
        assert snapshot.common_state.resume_safety == "not_resumable"
        calls_before_resume = model.calls

        rejected = await agent.resume(
            result.run_id,
            session_id="terminal-no-progress-session",
        )

        assert rejected.finish_reason.value == "error"
        assert rejected.metadata["error_summary"]["code"] == (
            "checkpoint_migration_failed"
        )
        assert model.calls == calls_before_resume

    asyncio.run(scenario())

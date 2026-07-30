from __future__ import annotations

import asyncio
import time
from typing import Any

from moduagent.agent import Agent
from moduagent.config import AgentConfig, RetryConfig, RunLimits
from moduagent.decision import (
    DecisionKind,
    ExecutionDecision,
    StandardDecisionPolicy,
)
from moduagent.execution import (
    CodecBackedEngine,
    EngineEmission,
    EngineOutcome,
    EngineStateCodec,
)
from moduagent.execution.standard import StandardExecutionEngine
from moduagent.messages import FinishReason, Message, ToolCall, Usage
from moduagent.models import (
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
)
from moduagent.observability import AuditEventSink, NoopEventSink
from moduagent.output import TextOutputCodec
from moduagent.persistence import (
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    RunCheckpoint,
)
from moduagent.runtime.context import RunRequest
from moduagent.runtime.coordinator import RunCoordinator
from moduagent.runtime.events import AgentEvent, EventType
from moduagent.skills import InMemorySkillSource, SkillRegistry
from moduagent.tools import ToolExecutor, ToolRegistry, function_tool


class ScriptedModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request: ModelRequest) -> Any:
        raise AssertionError("stream should not be called")
        yield


class StreamingModel:
    capabilities = ModelCapabilities(streaming=True)

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("complete should not be called")

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        yield ModelChunk(delta="hel")
        yield ModelChunk(delta="lo")
        yield ModelChunk(response=ModelResponse(Message.assistant("hello")))


def _coordinator(
    model: Any,
    *,
    config: AgentConfig | None = None,
    tools: tuple[Any, ...] = (),
    checkpoints: InMemoryCheckpointStore | None = None,
    conversations: InMemoryConversationStore | None = None,
    engine: Any | None = None,
    event_sink: Any | None = None,
) -> RunCoordinator:
    return RunCoordinator(
        config=config or AgentConfig("coordinator", "Answer."),
        model=model,
        decision_policy=StandardDecisionPolicy(),
        tool_executor=ToolExecutor(ToolRegistry(tools)),
        conversation_store=conversations or InMemoryConversationStore(),
        output_codec=TextOutputCodec(),
        event_sink=event_sink or NoopEventSink(),
        checkpoint_store=checkpoints,
        engine=engine,
    )


def test_coordinator_owns_terminal_event_and_monotonic_envelope() -> None:
    async def scenario() -> None:
        runtime = _coordinator(
            ScriptedModel([ModelResponse(Message.assistant("done"))])
        )
        request = RunRequest("work", "session-basic")

        events = [
            event async for event in runtime.stream(request, include_internal=True)
        ]

        assert events[0].type is EventType.RUN_STARTED
        assert events[-1].type is EventType.RUN_COMPLETED
        assert events[-1].data["result"].output == "done"
        assert (
            sum(
                event.type in {EventType.RUN_COMPLETED, EventType.RUN_FAILED}
                for event in events
            )
            == 1
        )
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        assert {event.session_id for event in events} == {"session-basic"}
        assert {event.engine_id for event in events} == {"standard"}

    asyncio.run(scenario())


def test_custom_engine_events_cannot_bypass_coordinator_publication() -> None:
    class CustomCodec(EngineStateCodec[dict[str, int]]):
        engine_id = "event-engine"
        state_version = 1

        def encode(self, state: dict[str, int]) -> dict[str, int]:
            return dict(state)

        def decode(self, payload: Any) -> dict[str, int]:
            return {"value": int(payload.get("value", 0))}

    class EventEngine(CodecBackedEngine[dict[str, int]]):
        engine_id = "event-engine"
        state_version = 1
        state_codec = CustomCodec()

        async def initialize(self, context: Any, services: Any) -> dict[str, int]:
            del context, services
            return {"value": 0}

        async def execute(self, context: Any, state: Any, services: Any):
            del state, services
            yield EngineEmission(
                event=AgentEvent(
                    EventType.POLICY_DECISION,
                    context.run.run_id,
                    {"secret": "internal-only"},
                )
            )
            yield EngineEmission(
                outcome=EngineOutcome(FinishReason.COMPLETED, output="done")
            )

    async def scenario() -> None:
        runtime = _coordinator(ScriptedModel([]), engine=EventEngine())
        request = RunRequest("work", "session-custom-event")

        internal = [
            event async for event in runtime.stream(request, include_internal=True)
        ]

        decision = next(
            event for event in internal if event.type is EventType.POLICY_DECISION
        )
        assert decision.visibility.value == "internal"
        assert decision.sequence > 0
        assert decision.session_id == "session-custom-event"
        assert decision.engine_id == "event-engine"
        assert [event.sequence for event in internal] == list(
            range(1, len(internal) + 1)
        )

    asyncio.run(scenario())


def test_hanging_event_sink_is_isolated_from_the_run_deadline() -> None:
    class HangingSink:
        async def publish(self, event: AgentEvent) -> None:
            del event
            await asyncio.Event().wait()

    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig(
                "bounded-sink",
                "Answer.",
                limits=RunLimits(timeout_seconds=0.05),
            ),
            model=ScriptedModel([ModelResponse(Message.assistant("too late"))]),
            event_sink=HangingSink(),
        )

        result = await asyncio.wait_for(
            agent.run("work", session_id="session-bounded-sink"),
            timeout=0.5,
        )

        assert result.finish_reason.value == "completed"
        assert result.output == "too late"
        assert result.error is None

    asyncio.run(scenario())


def test_coordinator_standard_engine_executes_tool_batch() -> None:
    async def scenario() -> None:
        @function_tool
        def add(a: int, b: int) -> int:
            return a + b

        call = ToolCall("call-add", "add", {"a": 2, "b": 3})
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                ModelResponse(Message.assistant("5")),
            ]
        )
        runtime = _coordinator(model, tools=(add,))

        result = await runtime.execute(RunRequest("2+3", "session-tool"))

        assert result.output == "5"
        assert len(model.requests) == 2
        assert model.requests[1].messages[-1].role.value == "tool"
        assert '"value": 5' in (model.requests[1].messages[-1].content or "")
        assert result.metadata["tool_trace"][0]["tool_name"] == "add"

    asyncio.run(scenario())


def test_standard_failure_projection_excludes_raw_tool_error_everywhere() -> None:
    async def scenario() -> None:
        secret = "SQL password=TOPSECRET syntax error"

        @function_tool
        def query_database(query: str) -> str:
            del query
            raise ValueError(secret)

        call = ToolCall(
            "call-query",
            "query_database",
            {"query": "select broken"},
        )
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                ModelResponse(Message.assistant("조회에 실패했습니다.")),
            ]
        )
        agent = Agent(
            config=AgentConfig(
                "safe-standard",
                "Answer safely.",
                finalization_mode="disabled",
            ),
            model=model,
            tools=(query_database,),
        )

        events = [
            event
            async for event in agent.stream_all(
                "데이터를 조회해줘.",
                session_id="session-safe-tool-failure",
            )
        ]

        assert secret not in repr([event.to_dict() for event in events])
        assert secret not in repr(model.requests[1])
        completed = next(
            event for event in events if event.type is EventType.TOOL_COMPLETED
        )
        assert "tool_call" not in completed.data
        assert "result" not in completed.data
        assert "error" not in completed.data
        assert completed.data["failure"]["reason"] == "execution_error"
        terminal = events[-1].data["result"]
        assert secret not in repr(terminal.metadata.get("tool_trace"))

    asyncio.run(scenario())


def test_model_retry_event_excludes_raw_provider_exception() -> None:
    class RetryOnceModel:
        capabilities = ModelCapabilities(streaming=False)

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("provider token=TOPSECRET")
            return ModelResponse(Message.assistant("recovered"))

        async def stream(self, request: ModelRequest):
            del request
            raise AssertionError("stream should not be called")
            yield

    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig(
                "safe-retry",
                "Answer safely.",
                retry=RetryConfig(
                    max_attempts=2,
                    initial_delay=0,
                    max_delay=0,
                ),
            ),
            model=RetryOnceModel(),
        )

        events = [
            event
            async for event in agent.stream_all(
                "retry",
                session_id="session-safe-retry",
            )
        ]

        retry = next(event for event in events if event.type is EventType.RETRY)
        assert retry.data["error"] == "model request failed"
        assert retry.data["error_type"] == "ConnectionError"
        assert "TOPSECRET" not in repr([event.to_dict() for event in events])

    asyncio.run(scenario())


def test_terminal_model_failure_excludes_raw_provider_exception() -> None:
    class FailingModel:
        capabilities = ModelCapabilities(streaming=False)

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise ValueError("Authorization bearer TOPSECRET provider error")

        async def stream(self, request: ModelRequest):
            del request
            raise AssertionError("stream should not be called")
            yield

    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig("safe-terminal", "Answer safely."),
            model=FailingModel(),
        )

        events = [
            event
            async for event in agent.stream_all(
                "fail",
                session_id="session-safe-terminal",
            )
        ]

        result = events[-1].data["result"]
        assert result.error == "model invocation failed"
        assert "TOPSECRET" not in repr([event.to_dict() for event in events])

    asyncio.run(scenario())


def test_audit_terminal_summary_excludes_tool_arguments_and_results() -> None:
    async def scenario() -> None:
        @function_tool
        def lookup_customer(query: str) -> str:
            del query
            return "customer_ssn=123-45-6789"

        call = ToolCall(
            "call-customer",
            "lookup_customer",
            {"query": "private-customer-query"},
        )
        audit = AuditEventSink()
        agent = Agent(
            config=AgentConfig("safe-audit", "Answer safely."),
            model=ScriptedModel(
                [
                    ModelResponse(Message.assistant(None, (call,)), (call,)),
                    ModelResponse(Message.assistant("done")),
                ]
            ),
            tools=(lookup_customer,),
            event_sink=audit,
        )

        result = await agent.run("lookup", session_id="session-safe-audit")

        assert result.output == "done"
        serialized = repr(audit.records)
        assert "private-customer-query" not in serialized
        assert "customer_ssn=123-45-6789" not in serialized
        terminal = audit.records[-1]["data"]
        assert terminal["finish_reason"] == "completed"
        assert terminal["message_count"] > 0
        assert "result" not in terminal

    asyncio.run(scenario())


def test_coordinator_preserves_explicit_finalization_model_count() -> None:
    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        model = ScriptedModel(
            [
                ModelResponse(
                    Message.assistant("draft"),
                    usage=Usage(10, 2, 12),
                ),
                ModelResponse(
                    Message.assistant("public"),
                    usage=Usage(7, 1, 8),
                ),
            ]
        )
        runtime = _coordinator(
            model,
            config=AgentConfig(
                "finalizing",
                "Answer.",
                finalization_mode="always",
            ),
            conversations=conversations,
        )

        result = await runtime.execute(RunRequest("work", "session-finalize"))

        assert result.output == "public"
        assert result.usage == Usage(17, 3, 20)
        assert len(model.requests) == 2
        assert model.requests[1].tools == ()
        assert [
            message.content for message in await conversations.load("session-finalize")
        ] == ["work", "public"]

    asyncio.run(scenario())


def test_finalization_resume_does_not_duplicate_an_already_appended_batch() -> None:
    class AppendThenInterruptStore(InMemoryConversationStore):
        def __init__(self) -> None:
            super().__init__()
            self.interruptions = 2

        async def append_once(
            self,
            session_id: str,
            idempotency_key: str,
            messages: Any,
        ) -> bool:
            written = await super().append_once(
                session_id,
                idempotency_key,
                messages,
            )
            if self.interruptions:
                self.interruptions -= 1
                raise asyncio.TimeoutError
            return written

    async def scenario() -> None:
        conversations = AppendThenInterruptStore()
        checkpoints = InMemoryCheckpointStore()
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant("draft")),
                ModelResponse(Message.assistant("stable public answer")),
            ]
        )
        agent = Agent(
            config=AgentConfig(
                "durable-final",
                "Answer.",
                finalization_mode="always",
            ),
            model=model,
            conversation_store=conversations,
            checkpoint_store=checkpoints,
        )

        failed = await agent.run("work", session_id="session-durable-final")
        assert failed.finish_reason.value == "timeout"
        assert await checkpoints.load(failed.run_id) is not None

        resumed = await agent.resume(
            failed.run_id,
            session_id="session-durable-final",
        )
        stored = await conversations.load("session-durable-final")

        assert resumed.output == "stable public answer"
        assert [message.content for message in stored].count(
            "stable public answer"
        ) == 1
        assert len(model.requests) == 2

    asyncio.run(scenario())


def test_coordinator_streams_provider_deltas_once() -> None:
    async def scenario() -> None:
        runtime = _coordinator(StreamingModel())

        events = [
            event
            async for event in runtime.stream(
                RunRequest("greet", "session-stream"),
                include_internal=True,
            )
        ]

        assert [
            event.data["delta"]
            for event in events
            if event.type is EventType.MODEL_DELTA
        ] == ["hel", "lo"]
        assert events[-1].data["result"].output == "hello"

    asyncio.run(scenario())


def test_engine_cannot_publish_a_forged_terminal_event() -> None:
    async def scenario() -> None:
        class ForgingEngine(StandardExecutionEngine):
            async def execute(self, context, state, services):
                await services.publish_event(
                    context,
                    AgentEvent(
                        EventType.RUN_COMPLETED,
                        context.run.run_id,
                        {"forged": True},
                    ),
                )
                raise AssertionError("terminal publication should be rejected")
                yield

        runtime = _coordinator(
            ScriptedModel([]),
            engine=ForgingEngine(StandardDecisionPolicy()),
        )

        events = [
            event
            async for event in runtime.stream(
                RunRequest("work", "session-forged-terminal"),
                include_internal=True,
            )
        ]

        assert events[-1].type is EventType.RUN_FAILED
        assert events[-1].data["result"].error == (
            "terminal events are owned by RunCoordinator"
        )
        assert (
            sum(
                event.type in {EventType.RUN_COMPLETED, EventType.RUN_FAILED}
                for event in events
            )
            == 1
        )

    asyncio.run(scenario())


def test_skill_resource_event_follows_its_tool_completion() -> None:
    async def scenario() -> None:
        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "policy-guide": {
                        "SKILL.md": (
                            "---\n"
                            "name: policy-guide\n"
                            "description: Read the policy reference.\n"
                            "---\n"
                            "Read references/policy.md before answering."
                        ),
                        "references/policy.md": "Manager approval is required.",
                    }
                }
            )
        )
        call = ToolCall(
            "read-policy",
            "moduagent_skill_read",
            {
                "skill_name": "policy-guide",
                "path": "references/policy.md",
            },
        )
        agent = Agent(
            config=AgentConfig("skill-order", "Use the selected Skill."),
            model=ScriptedModel(
                [
                    ModelResponse(Message.assistant(None, (call,)), (call,)),
                    ModelResponse(Message.assistant("Approval is required.")),
                ]
            ),
            skill_registry=registry,
        )

        events = [
            event
            async for event in agent.stream_all(
                "Check the policy.",
                session_id="session-skill-order",
                skills=["policy-guide"],
            )
        ]
        event_types = [event.type for event in events]

        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        assert event_types.index(EventType.TOOL_COMPLETED) < event_types.index(
            EventType.SKILL_RESOURCE_READ
        )

    asyncio.run(scenario())


def test_resume_continues_durable_event_sequence_and_engine_state() -> None:
    async def scenario() -> None:
        class FailingModel:
            capabilities = ModelCapabilities(streaming=False)

            async def complete(self, request: ModelRequest) -> ModelResponse:
                raise RuntimeError("temporary provider failure")

            async def stream(self, request: ModelRequest):
                raise AssertionError("stream should not be called")
                yield

        checkpoints = InMemoryCheckpointStore()
        failed_runtime = _coordinator(
            FailingModel(),
            checkpoints=checkpoints,
        )
        first = [
            event
            async for event in failed_runtime.stream(
                RunRequest("work", "session-resume"),
                include_internal=True,
            )
        ]
        failed = first[-1].data["result"]
        snapshot = await checkpoints.load_snapshot(failed.run_id)

        assert first[-1].type is EventType.RUN_FAILED
        assert snapshot is not None
        assert snapshot.common_state.event_sequence == first[-1].sequence
        assert "_moduagent_engine_snapshot" not in (
            snapshot.common_state.compatibility_policy_state
        )
        assert "execution_state" not in (
            snapshot.common_state.compatibility_policy_state
        )

        resumed_runtime = _coordinator(
            ScriptedModel([ModelResponse(Message.assistant("recovered"))]),
            checkpoints=checkpoints,
        )
        resumed = [
            event
            async for event in resumed_runtime.stream(
                RunRequest(
                    "",
                    "session-resume",
                    resume_run_id=failed.run_id,
                ),
                include_internal=True,
            )
        ]

        assert resumed[0].type is EventType.RUN_STARTED
        assert resumed[0].sequence == first[-1].sequence + 1
        assert resumed[1].type is EventType.CHECKPOINT_LOADED
        assert resumed[-1].data["result"].output == "recovered"

    asyncio.run(scenario())


def test_tool_timeout_persists_fail_closed_intent_before_invocation() -> None:
    async def scenario() -> None:
        @function_tool
        def slow_lookup() -> str:
            time.sleep(0.3)
            return "late"

        call = ToolCall("slow-1", "slow_lookup", {})
        checkpoints = InMemoryCheckpointStore()
        runtime = _coordinator(
            ScriptedModel([ModelResponse(Message.assistant(None, (call,)), (call,))]),
            config=AgentConfig(
                "coordinator",
                "Use the Tool.",
                RunLimits(max_steps=2, timeout_seconds=0.08),
            ),
            tools=(slow_lookup,),
            checkpoints=checkpoints,
        )

        events = [
            event
            async for event in runtime.stream(
                RunRequest("work", "session-tool-timeout"),
                include_internal=True,
            )
        ]
        result = events[-1].data["result"]
        snapshot = await checkpoints.load_snapshot(result.run_id)

        assert result.finish_reason is FinishReason.TIMEOUT
        assert result.metadata["error_summary"]["resumable"] is False
        assert snapshot is not None
        assert snapshot.common_state.resume_safety == "manual_required"
        assert any(
            event.type is EventType.CHECKPOINT_SAVED
            and event.data["boundary"] == "tool_invocation_pending"
            for event in events
        )

    asyncio.run(scenario())


def test_tool_is_not_invoked_when_intent_checkpoint_fails() -> None:
    async def scenario() -> None:
        invocations = 0

        @function_tool
        def guarded_write() -> str:
            nonlocal invocations
            invocations += 1
            return "written"

        class FailingIntentStore(InMemoryCheckpointStore):
            async def save_snapshot(self, snapshot):
                if snapshot.common_state.resume_safety == "manual_required":
                    raise RuntimeError("PRIVATE-STORE-DETAIL")
                await super().save_snapshot(snapshot)

        call = ToolCall("write-1", "guarded_write", {})
        runtime = _coordinator(
            ScriptedModel([ModelResponse(Message.assistant(None, (call,)), (call,))]),
            tools=(guarded_write,),
            checkpoints=FailingIntentStore(),
        )

        events = [
            event
            async for event in runtime.stream(
                RunRequest("write", "session-intent-failure"),
                include_internal=True,
            )
        ]
        result = events[-1].data["result"]

        assert invocations == 0
        assert result.error == "persistence operation failed"
        assert result.metadata["error_summary"]["category"] == "persistence"
        assert "PRIVATE-STORE-DETAIL" not in repr(result)

    asyncio.run(scenario())


def test_conversation_append_failure_has_persistence_error_taxonomy() -> None:
    async def scenario() -> None:
        class FailingConversationStore(InMemoryConversationStore):
            async def append_once(
                self,
                session_id,
                idempotency_key,
                messages,
            ):
                del session_id, idempotency_key, messages
                raise RuntimeError("PRIVATE-CONVERSATION-STORE-DETAIL")

        runtime = _coordinator(
            ScriptedModel([ModelResponse(Message.assistant("done"))]),
            conversations=FailingConversationStore(),
        )

        events = [
            event
            async for event in runtime.stream(
                RunRequest("work", "session-conversation-failure"),
                include_internal=True,
            )
        ]
        result = events[-1].data["result"]

        assert result.error == "persistence operation failed"
        assert result.metadata["error_summary"]["category"] == "persistence"
        assert result.metadata["error_summary"]["retryable"] is True
        assert "PRIVATE-CONVERSATION-STORE-DETAIL" not in repr(result)

    asyncio.run(scenario())


def test_unsafe_resume_rejection_is_not_reported_as_resumable() -> None:
    async def scenario() -> None:
        checkpoints = InMemoryCheckpointStore()
        checkpoint = RunCheckpoint(
            run_id="unsafe-resume",
            session_id="unsafe-session",
            messages=(Message.user("work"),),
            input="work",
            resume_safety="manual_required",
        )
        await checkpoints.save(checkpoint)
        runtime = _coordinator(
            ScriptedModel([]),
            checkpoints=checkpoints,
        )

        events = [
            event
            async for event in runtime.stream(
                RunRequest(
                    "",
                    "unsafe-session",
                    resume_run_id="unsafe-resume",
                ),
                include_internal=True,
            )
        ]
        result = events[-1].data["result"]

        assert result.error == "checkpoint state migration failed"
        assert result.metadata["error_summary"]["category"] == "state_migration"
        assert result.metadata["error_summary"]["resumable"] is False

    asyncio.run(scenario())


def test_history_load_failure_cannot_resume_with_silently_missing_history() -> None:
    async def scenario() -> None:
        class FlakyHistoryStore(InMemoryConversationStore):
            def __init__(self) -> None:
                super().__init__()
                self.fail_next_load = True

            async def load(self, session_id):
                if self.fail_next_load:
                    self.fail_next_load = False
                    raise RuntimeError("history backend unavailable")
                return await super().load(session_id)

        conversations = FlakyHistoryStore()
        conversations.fail_next_load = False
        await conversations.append(
            "history-session",
            (Message.user("prior"), Message.assistant("prior answer")),
        )
        conversations.fail_next_load = True
        checkpoints = InMemoryCheckpointStore()
        runtime = _coordinator(
            ScriptedModel([]),
            conversations=conversations,
            checkpoints=checkpoints,
        )

        failed_events = [
            event
            async for event in runtime.stream(
                RunRequest("new", "history-session"),
                include_internal=True,
            )
        ]
        failed = failed_events[-1].data["result"]
        snapshot = await checkpoints.load_snapshot(failed.run_id)

        assert snapshot is None
        assert failed.metadata["error_summary"]["resumable"] is False

    asyncio.run(scenario())


def test_resume_load_failure_never_overwrites_existing_checkpoint() -> None:
    async def scenario() -> None:
        class FlakyCheckpointStore(InMemoryCheckpointStore):
            def __init__(self) -> None:
                super().__init__()
                self.fail_next_load = False

            async def load_snapshot(self, run_id):
                if self.fail_next_load:
                    self.fail_next_load = False
                    raise RuntimeError("checkpoint backend unavailable")
                return await super().load_snapshot(run_id)

        checkpoints = FlakyCheckpointStore()
        checkpoint = RunCheckpoint(
            run_id="preserved-resume",
            session_id="preserved-session",
            input="ORIGINAL",
            messages=(Message.user("ORIGINAL"),),
            metadata={"sentinel": "keep"},
        )
        await checkpoints.save(checkpoint)
        original = await checkpoints.load_snapshot(checkpoint.run_id)
        checkpoints.fail_next_load = True
        runtime = _coordinator(
            ScriptedModel([]),
            checkpoints=checkpoints,
        )

        events = [
            event
            async for event in runtime.stream(
                RunRequest(
                    "",
                    checkpoint.session_id,
                    resume_run_id=checkpoint.run_id,
                ),
                include_internal=True,
            )
        ]
        result = events[-1].data["result"]
        preserved = await checkpoints.load_snapshot(checkpoint.run_id)

        assert result.error == "persistence operation failed"
        assert result.metadata["error_summary"]["resumable"] is False
        assert preserved == original
        assert preserved is not None
        assert preserved.common_state.request["input"] == "ORIGINAL"
        assert preserved.sanitized_runtime_metadata["sentinel"] == "keep"

    asyncio.run(scenario())


def test_direct_standard_response_resumes_after_conversation_write_failure() -> None:
    async def scenario() -> None:
        class ToggleConversationStore(InMemoryConversationStore):
            def __init__(self) -> None:
                super().__init__()
                self.fail_writes = True

            async def append_once(
                self,
                session_id,
                idempotency_key,
                messages,
            ):
                if self.fail_writes:
                    raise RuntimeError("conversation unavailable")
                return await super().append_once(
                    session_id,
                    idempotency_key,
                    messages,
                )

        model = ScriptedModel([ModelResponse(Message.assistant("answer"))])
        conversations = ToggleConversationStore()
        checkpoints = InMemoryCheckpointStore()
        runtime = _coordinator(
            model,
            conversations=conversations,
            checkpoints=checkpoints,
        )

        failed_events = [
            event
            async for event in runtime.stream(
                RunRequest("question", "direct-resume-session"),
                include_internal=True,
            )
        ]
        failed = failed_events[-1].data["result"]
        snapshot = await checkpoints.load_snapshot(failed.run_id)

        assert failed.error == "persistence operation failed"
        assert failed.metadata["error_summary"]["resumable"] is True
        assert snapshot is not None
        assert snapshot.engine.state["phase"] == "done"
        assert snapshot.engine.state["finalization"]["response"] == "answer"

        conversations.fail_writes = False
        resumed_events = [
            event
            async for event in runtime.stream(
                RunRequest(
                    "",
                    "direct-resume-session",
                    resume_run_id=failed.run_id,
                ),
                include_internal=True,
            )
        ]
        resumed = resumed_events[-1].data["result"]

        assert resumed.output == "answer"
        assert resumed.error is None
        assert len(model.requests) == 1
        assert await checkpoints.load_snapshot(failed.run_id) is None

    asyncio.run(scenario())


def test_standard_final_response_after_tool_is_safely_resumable() -> None:
    async def scenario() -> None:
        for finalization_mode in ("structured_only", "always"):
            tool_calls = 0

            @function_tool
            def lookup() -> str:
                nonlocal tool_calls
                tool_calls += 1
                return "verified"

            class ToggleConversationStore(InMemoryConversationStore):
                def __init__(self) -> None:
                    super().__init__()
                    self.fail_writes = True

                async def append_once(
                    self,
                    session_id,
                    idempotency_key,
                    messages,
                ):
                    if self.fail_writes:
                        raise RuntimeError("conversation unavailable")
                    return await super().append_once(
                        session_id,
                        idempotency_key,
                        messages,
                    )

            call = ToolCall(f"lookup-{finalization_mode}", "lookup", {})
            responses = [
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                ModelResponse(Message.assistant("draft answer")),
            ]
            expected_output = "draft answer"
            if finalization_mode == "always":
                responses.append(ModelResponse(Message.assistant("public answer")))
                expected_output = "public answer"

            model = ScriptedModel(responses)
            conversations = ToggleConversationStore()
            checkpoints = InMemoryCheckpointStore()
            runtime = _coordinator(
                model,
                config=AgentConfig(
                    f"tool-final-{finalization_mode}",
                    "Use the Tool.",
                    finalization_mode=finalization_mode,
                ),
                tools=(lookup,),
                conversations=conversations,
                checkpoints=checkpoints,
            )
            session_id = f"tool-final-{finalization_mode}"

            failed_events = [
                event
                async for event in runtime.stream(
                    RunRequest("question", session_id),
                    include_internal=True,
                )
            ]
            failed = failed_events[-1].data["result"]
            snapshot = await checkpoints.load_snapshot(failed.run_id)

            assert failed.error == "persistence operation failed"
            assert failed.metadata["error_summary"]["resumable"] is True
            assert snapshot is not None
            assert snapshot.common_state.resume_safety == "resumable"
            assert snapshot.engine.state["finalization"]["response"] == expected_output

            conversations.fail_writes = False
            resumed_events = [
                event
                async for event in runtime.stream(
                    RunRequest(
                        "",
                        session_id,
                        resume_run_id=failed.run_id,
                    ),
                    include_internal=True,
                )
            ]
            resumed = resumed_events[-1].data["result"]

            assert resumed.output == expected_output
            assert resumed.error is None
            assert tool_calls == 1
            assert len(model.requests) == (3 if finalization_mode == "always" else 2)
            assert await checkpoints.load_snapshot(failed.run_id) is None

    asyncio.run(scenario())


def test_custom_terminal_output_without_durable_codec_fails_closed() -> None:
    async def scenario() -> None:
        class CustomOutputPolicy:
            async def begin(self, context):
                del context

            async def decide(self, context, response):
                del context, response
                return ExecutionDecision(
                    DecisionKind.FINISH,
                    final_output={"source": "policy"},
                )

            async def observe(self, context, results):
                del context, results
                return None

            def should_stop(self, context):
                del context
                return False

        class ToggleConversationStore(InMemoryConversationStore):
            def __init__(self) -> None:
                super().__init__()
                self.fail_writes = True

            async def append_once(
                self,
                session_id,
                idempotency_key,
                messages,
            ):
                if self.fail_writes:
                    raise RuntimeError("conversation unavailable")
                return await super().append_once(
                    session_id,
                    idempotency_key,
                    messages,
                )

        model = ScriptedModel([ModelResponse(Message.assistant("model text"))])
        conversations = ToggleConversationStore()
        checkpoints = InMemoryCheckpointStore()
        agent = Agent(
            config=AgentConfig("custom-output", "Use the custom policy."),
            model=model,
            decision_policy=CustomOutputPolicy(),
            conversation_store=conversations,
            checkpoint_store=checkpoints,
        )

        failed = await agent.run("question", session_id="custom-output-session")
        snapshot = await checkpoints.load_snapshot(failed.run_id)

        assert failed.error == "persistence operation failed"
        assert failed.metadata["error_summary"]["resumable"] is False
        assert [message.role.value for message in failed.messages] == [
            "system",
            "user",
        ]
        assert "model text" not in repr(failed.messages)
        assert await conversations.load("custom-output-session") == []
        assert snapshot is not None
        assert snapshot.common_state.resume_safety == "manual_required"
        assert "model text" not in snapshot.to_json()

        conversations.fail_writes = False
        resumed = await agent.resume(
            failed.run_id,
            session_id="custom-output-session",
        )

        assert resumed.error == "checkpoint state migration failed"
        assert resumed.metadata["error_summary"]["resumable"] is False
        assert len(model.requests) == 1

    asyncio.run(scenario())


def test_missing_checkpoint_has_stable_non_resumable_error() -> None:
    async def scenario() -> None:
        runtime = _coordinator(
            ScriptedModel([]),
            checkpoints=InMemoryCheckpointStore(),
        )

        events = [
            event
            async for event in runtime.stream(
                RunRequest(
                    "",
                    "missing-session",
                    resume_run_id="missing-run",
                ),
                include_internal=True,
            )
        ]
        result = events[-1].data["result"]

        assert result.error == "checkpoint not found"
        assert result.metadata["error_summary"] == {
            "category": "persistence",
            "code": "checkpoint_not_found",
            "retryable": False,
            "resumable": False,
        }

    asyncio.run(scenario())


def test_event_sink_mutation_cannot_change_streamed_terminal_result() -> None:
    async def scenario() -> None:
        class MutatingSink:
            async def publish(self, event):
                event.data.clear()

        runtime = _coordinator(
            ScriptedModel([ModelResponse(Message.assistant("done"))]),
            event_sink=MutatingSink(),
        )

        result = await runtime.execute(RunRequest("work", "sink-mutation-session"))

        assert result.output == "done"
        assert result.finish_reason is FinishReason.COMPLETED

    asyncio.run(scenario())

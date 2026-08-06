from __future__ import annotations

import asyncio
import logging
from threading import Event
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import pytest
from pydantic import BaseModel, ValidationError

from moduagent.messages import FinishReason, Message, ToolCall, Usage
from moduagent.observability import (
    AuditEventSink,
    CompositeEventSink,
    InMemoryMetricRecorder,
    LoggingEventSink,
    MetricsEventSink,
)
from moduagent.output import PydanticOutputCodec, TextOutputCodec
from moduagent.persistence import (
    ConversationStoreCapacityError,
    DatabaseConversationStore,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    RedisCheckpointStore,
    RedisConversationStore,
    RunCheckpoint,
)
from moduagent.runtime import AgentEvent, AgentResult, EventType, RunStatus


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)
        self.expirations.pop(key, None)


class FakeConversationRepository:
    def __init__(self) -> None:
        self.rows: dict[str, list[str]] = {}

    async def load_messages(self, session_id: str) -> list[str]:
        return list(self.rows.get(session_id, ()))

    async def append_messages(self, session_id: str, messages: list[str]) -> None:
        self.rows.setdefault(session_id, []).extend(messages)

    async def clear_messages(self, session_id: str) -> None:
        self.rows.pop(session_id, None)


def test_conversation_stores_round_trip_json_and_expire_independently() -> None:
    async def scenario() -> None:
        now = [10.0]
        memory = InMemoryConversationStore(ttl_seconds=2, clock=lambda: now[0])
        message = Message.assistant(
            None, (ToolCall("call-1", "lookup", {"employee": 7}),)
        )
        await memory.append("s1", [message])
        assert await memory.load("s1") == [message]
        now[0] = 12.0
        assert await memory.load("s1") == []

        redis = FakeRedis()
        store = RedisConversationStore(redis, ttl_seconds=30)
        await store.append("s2", [Message.user("안녕"), message])
        assert await store.load("s2") == [Message.user("안녕"), message]
        key = "moduagent:conversation:s2"
        assert redis.expirations[key] == 30
        assert '"tool_calls"' in redis.values[key]
        assert store._fallback_locks == {}
        assert store._fallback_lock_users == {}
        await store.clear("s2")
        assert await store.load("s2") == []

    asyncio.run(scenario())


def test_in_memory_conversation_append_once_is_atomic_and_detects_key_reuse() -> None:
    async def scenario() -> None:
        store = InMemoryConversationStore()
        messages = [Message.user("question"), Message.assistant("answer")]

        assert await store.append_once("session", "run:batch:0", messages) is True
        assert await store.append_once("session", "run:batch:0", messages) is False
        assert await store.load("session") == messages
        with pytest.raises(ValueError, match="different messages"):
            await store.append_once(
                "session",
                "run:batch:0",
                [Message.assistant("different")],
            )

    asyncio.run(scenario())


def test_in_memory_conversation_store_evicts_sessions_by_lru_capacity() -> None:
    async def scenario() -> None:
        store = InMemoryConversationStore(max_sessions=2)
        await store.append("session-1", [Message.user("one")])
        await store.append("session-2", [Message.user("two")])

        # A read refreshes recency without extending the session TTL.
        assert await store.load("session-1") == [Message.user("one")]
        await store.append("session-3", [Message.user("three")])

        assert await store.load("session-1") == [Message.user("one")]
        assert await store.load("session-2") == []
        assert await store.load("session-3") == [Message.user("three")]
        stats = await store.stats()
        assert stats["sessions"] == 2
        assert stats["total_bytes"] > 0

    asyncio.run(scenario())


def test_in_memory_conversation_store_bounds_serialized_bytes_atomically() -> None:
    async def scenario() -> None:
        message = Message.user("bounded")
        probe = InMemoryConversationStore()
        await probe.append("probe", [message])
        row_bytes = (await probe.stats())["total_bytes"]

        store = InMemoryConversationStore(max_total_bytes=row_bytes)
        await store.append("session-1", [message])
        await store.append("session-2", [message])
        assert await store.load("session-1") == []
        assert await store.load("session-2") == [message]

        before = await store.stats()
        with pytest.raises(
            ConversationStoreCapacityError,
            match="exceeds max_total_bytes",
        ):
            await store.append("session-2", [Message.user("too large")])
        assert await store.load("session-2") == [message]
        assert await store.stats() == before

    asyncio.run(scenario())


def test_in_memory_conversation_store_sweeps_expired_sessions_lazily() -> None:
    async def scenario() -> None:
        now = [1.0]
        store = InMemoryConversationStore(
            ttl_seconds=2,
            ttl_sweep_interval_seconds=10,
            clock=lambda: now[0],
        )
        await store.append("session-1", [Message.user("one")])
        await store.append("session-2", [Message.user("two")])
        now[0] = 3.0

        assert await store.sweep_expired() == 2
        assert await store.stats() == {"sessions": 0, "total_bytes": 0}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_sessions": 0}, "max_sessions"),
        ({"max_total_bytes": True}, "max_total_bytes"),
        ({"ttl_sweep_interval_seconds": 0}, "ttl_sweep_interval_seconds"),
        ({"ttl_sweep_interval_seconds": float("nan")}, "ttl_sweep_interval_seconds"),
    ],
)
def test_in_memory_conversation_store_rejects_invalid_capacity(
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        InMemoryConversationStore(**options)  # type: ignore[arg-type]


def test_database_conversation_store_uses_json_repository_rows() -> None:
    async def scenario() -> None:
        repository = FakeConversationRepository()
        store = DatabaseConversationStore(repository)
        messages = [Message.user("질문"), Message.assistant("답변")]

        await store.append("db-session", messages)

        assert repository.rows["db-session"][0].startswith('{"role":"user"')
        assert await store.load("db-session") == messages
        await store.clear("db-session")
        assert await store.load("db-session") == []

    asyncio.run(scenario())


def test_sync_database_repository_does_not_block_the_event_loop() -> None:
    class BlockingRepository:
        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()

        def load_messages(self, session_id: str) -> list[str]:
            del session_id
            self.started.set()
            if not self.release.wait(timeout=0.5):
                raise AssertionError("synchronous repository blocked the event loop")
            return []

        def append_messages(self, session_id: str, messages: list[str]) -> None:
            del session_id, messages

        def clear_messages(self, session_id: str) -> None:
            del session_id

    async def scenario() -> None:
        repository = BlockingRepository()
        store = DatabaseConversationStore(repository)  # type: ignore[arg-type]
        loading = asyncio.create_task(store.load("sync-db"))

        while not repository.started.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        repository.release.set()

        assert await loading == []

    asyncio.run(scenario())


def _checkpoint() -> RunCheckpoint:
    return RunCheckpoint(
        run_id="run-1",
        session_id="session-1",
        input="휴가 규정을 알려줘",
        user_context={"user_id": "employee-1"},
        messages=(
            Message.system("인사 규정을 안내한다."),
            Message.user("휴가 규정을 알려줘"),
        ),
        new_messages=(Message.user("휴가 규정을 알려줘"),),
        step=2,
        tool_call_count=1,
        status=RunStatus.WAITING_FOR_MODEL,
        policy_state={"phase": "react"},
        usage=Usage(10, 2, 12, {"provider": "fake"}),
        metadata={"trace_id": "trace-1"},
        current_run_start=1,
    )


def test_checkpoint_json_context_and_stores_round_trip() -> None:
    async def scenario() -> None:
        checkpoint = _checkpoint()
        decoded = RunCheckpoint.from_json(checkpoint.to_json())
        context = decoded.to_context()

        assert decoded == checkpoint
        assert context.request.session_id == "session-1"
        assert context.policy_state == {"phase": "react"}
        assert context.current_run_start == 1
        assert RunCheckpoint.from_context(context).messages == checkpoint.messages

        legacy_payload = {
            "version": 1,
            "run_id": checkpoint.run_id,
            "session_id": checkpoint.session_id,
            "messages": [message.to_dict() for message in checkpoint.messages],
        }
        legacy = RunCheckpoint.from_dict(legacy_payload)
        assert legacy.current_run_start == 1

        memory = InMemoryCheckpointStore()
        await memory.save("run-1", checkpoint.to_context())
        restored = await memory.load("run-1")
        assert restored is not None
        assert restored.to_context().messages == checkpoint.to_context().messages
        await memory.delete("run-1")
        assert await memory.load("run-1") is None

        redis = FakeRedis()
        store = RedisCheckpointStore(redis, ttl_seconds=5)
        await store.save("run-1", checkpoint)
        assert await store.load("run-1") == checkpoint
        assert redis.expirations["moduagent:checkpoint:v4:run-1"] == 5
        assert redis.expirations.get("moduagent:conversation:session-1") is None

    asyncio.run(scenario())


def test_in_memory_checkpoint_ttl() -> None:
    async def scenario() -> None:
        now = [1.0]
        store = InMemoryCheckpointStore(ttl_seconds=1, clock=lambda: now[0])
        await store.save("run-1", _checkpoint())
        now[0] = 2.0
        assert await store.load("run-1") is None

    asyncio.run(scenario())


class Answer(BaseModel):
    answer: str
    confidence: float


@dataclass
class FakeModelResponse:
    message: Message


def test_text_and_pydantic_output_codecs() -> None:
    text = TextOutputCodec()
    assert text.schema() is None
    assert text.decode(FakeModelResponse(Message.assistant("hello"))) == "hello"

    structured = PydanticOutputCodec(Answer)
    output = structured.decode(
        FakeModelResponse(Message.assistant('{"answer":"가능합니다","confidence":0.9}'))
    )
    assert output == Answer(answer="가능합니다", confidence=0.9)
    assert structured.schema()["properties"]["answer"]["type"] == "string"

    with pytest.raises(ValidationError):
        structured.decode(FakeModelResponse(Message.assistant('{"answer":3}')))


def test_composite_sink_is_non_intrusive() -> None:
    class BrokenSink:
        async def publish(self, event: AgentEvent) -> None:
            raise RuntimeError("telemetry unavailable")

    class CollectingSink:
        def __init__(self) -> None:
            self.events: list[AgentEvent] = []

        async def publish(self, event: AgentEvent) -> None:
            self.events.append(event)

    async def scenario() -> None:
        collector = CollectingSink()
        composite = CompositeEventSink([BrokenSink(), collector])
        event = AgentEvent(EventType.RUN_STARTED, "run-1")

        await composite.publish(event)

        assert collector.events == [event]
        assert len(composite.last_errors) == 1

    asyncio.run(scenario())


def test_logging_and_audit_sinks_mask_sensitive_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        logger = logging.getLogger("test.moduagent.audit")
        logging_sink = LoggingEventSink(logger)
        audit_sink = AuditEventSink()
        event = AgentEvent(
            EventType.TOOL_STARTED,
            "run-secret",
            {
                "tool_name": "lookup",
                "arguments": {
                    "employee_id": "e-1",
                    "api_key": "must-not-leak",
                    "nested": {"access_token": "also-secret"},
                    "input_tokens": 11,
                },
            },
        )

        await logging_sink.publish(event)
        await audit_sink.publish(event)
        terminal = AgentEvent(
            EventType.RUN_COMPLETED,
            "run-secret",
            {
                "result": AgentResult(
                    run_id="run-secret",
                    output="customer_ssn=123-45-6789",
                    messages=(
                        Message.assistant(
                            None,
                            (
                                ToolCall(
                                    "call-secret",
                                    "lookup",
                                    {"query": "private-query"},
                                ),
                            ),
                        ),
                        Message.tool(
                            "customer_ssn=123-45-6789",
                            call_id="call-secret",
                            name="lookup",
                        ),
                    ),
                    usage=Usage(1, 1, 2),
                    finish_reason=FinishReason.COMPLETED,
                )
            },
        )
        await logging_sink.publish(terminal)

        assert "must-not-leak" not in caplog.text
        assert "also-secret" not in caplog.text
        assert "private-query" not in caplog.text
        assert "customer_ssn=123-45-6789" not in caplog.text
        assert "arguments" not in audit_sink.records[0]["data"]

    with caplog.at_level(logging.INFO, logger="test.moduagent.audit"):
        asyncio.run(scenario())


def test_metrics_sink_records_run_tool_latency_error_and_usage() -> None:
    async def scenario() -> None:
        recorder = InMemoryMetricRecorder()
        sink = MetricsEventSink(recorder)
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = AgentResult(
            run_id="run-1",
            output="done",
            messages=(),
            usage=Usage(7, 3, 10),
            finish_reason=FinishReason.COMPLETED,
        )

        await sink.publish(
            AgentEvent(EventType.RUN_STARTED, "run-1", occurred_at=started)
        )
        await sink.publish(
            AgentEvent(
                EventType.TOOL_STARTED,
                "run-1",
                {"tool_name": "search"},
                occurred_at=started + timedelta(seconds=1),
            )
        )
        await sink.publish(
            AgentEvent(
                EventType.TOOL_COMPLETED,
                "run-1",
                {"tool_name": "search", "success": False},
                occurred_at=started + timedelta(seconds=2),
            )
        )
        await sink.publish(
            AgentEvent(
                EventType.RUN_COMPLETED,
                "run-1",
                {"result": result},
                occurred_at=started + timedelta(seconds=4),
            )
        )

        assert (
            recorder.counters[("moduagent.runs.total", (("status", "completed"),))] == 1
        )
        assert (
            recorder.counters[("moduagent.tool_calls.failed", (("tool", "search"),))]
            == 1
        )
        assert recorder.counters[("moduagent.tokens.total", (("type", "total"),))] == 10
        assert recorder.observations[
            (
                "moduagent.run.duration_seconds",
                (("status", "completed"),),
            )
        ] == [4.0]

    asyncio.run(scenario())

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from typing import Any

from moduagent import (
    Agent,
    AgentConfig,
    EventType,
    FinishReason,
    InMemoryConversationStore,
    InMemoryDiagnosticSink,
    InMemoryMemoryStateStore,
    Message,
    ModelOutputIncompleteError,
    ModelProtocolError,
    ModelRequest,
    ModelResponse,
    RetryConfig,
    RunLimits,
    Usage,
)
from moduagent.decision import DecisionKind, ExecutionDecision
from moduagent.memory import (
    ModelConversationSummarizer,
    SummarizingConversationMemoryPolicy,
    TokenBudget,
)
from moduagent.runtime.context import RunContext
from moduagent.tools import ToolResult


class _SequenceModel:
    def __init__(self, outcomes: Sequence[ModelResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _AlwaysOneCounter:
    async def count_request(self, request: ModelRequest) -> int:
        del request
        return 1


class _MessageCountCounter:
    async def count_request(self, request: ModelRequest) -> int:
        return len(request.messages)


class _OneRecordPerBatchCounter:
    async def count_request(self, request: ModelRequest) -> int:
        content = request.messages[-1].content or ""
        payload = json.loads(content.split("\n", 1)[1])
        return len(payload["conversation_record_fragments"])


class _AlwaysContinuePolicy:
    async def begin(self, context: RunContext) -> None:
        del context

    async def decide(
        self,
        context: RunContext,
        response: ModelResponse,
    ) -> ExecutionDecision:
        del context, response
        return ExecutionDecision(DecisionKind.CONTINUE)

    async def observe(
        self,
        context: RunContext,
        results: Sequence[ToolResult],
    ) -> ExecutionDecision | None:
        del context, results
        return None

    def should_stop(self, context: RunContext) -> bool:
        del context
        return False


async def _conversation(session_id: str) -> InMemoryConversationStore:
    store = InMemoryConversationStore()
    await store.append(
        session_id,
        (
            Message.user("PRIVATE-OLD-QUESTION"),
            Message.assistant("PRIVATE-OLD-ANSWER"),
        ),
    )
    return store


def _memory_policy(
    summary_model: Any,
    *,
    token_counter: Any | None = None,
    summary_token_counter: Any | None = None,
    state_store: InMemoryMemoryStateStore | None = None,
) -> SummarizingConversationMemoryPolicy:
    outer_counter = token_counter or _AlwaysOneCounter()
    return SummarizingConversationMemoryPolicy(
        budget=TokenBudget(100),
        token_counter=outer_counter,
        summarizer=ModelConversationSummarizer(
            model=summary_model,
            token_counter=summary_token_counter or _AlwaysOneCounter(),
            max_input_tokens=1,
        ),
        state_store=state_store,
        max_history_turns=0,
    )


def test_recovered_summary_failure_does_not_replace_later_terminal_cause() -> None:
    async def scenario() -> None:
        session_id = "memory-fallback-terminal"
        summary_model = _SequenceModel(
            [ConnectionError("PRIVATE-SUMMARY-CONNECTION-DETAIL")]
        )
        main_model = _SequenceModel([ModelResponse(Message.assistant("continue once"))])
        diagnostics = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig(
                "memory-fallback-terminal",
                "Continue until the run limit is reached.",
                limits=RunLimits(max_steps=1, max_model_turns=4),
                retry=RetryConfig(max_attempts=1),
            ),
            model=main_model,  # type: ignore[arg-type]
            decision_policy=_AlwaysContinuePolicy(),
            conversation_store=await _conversation(session_id),
            conversation_memory_policy=_memory_policy(summary_model),
            diagnostic_sink=diagnostics,
        )

        events = [
            event
            async for event in agent.stream_all(
                "new question",
                session_id=session_id,
            )
        ]
        result = events[-1].data["result"]

        assert result.finish_reason is FinishReason.MAX_STEPS
        assert result.error_summary["code"] == "max_steps_exceeded"
        assert result.error_summary["category"] == "limit"
        assert "phase" not in result.error_summary
        assert len(summary_model.requests) == 1
        assert len(main_model.requests) == 1

        summary_failures = [
            event
            for event in events
            if event.type is EventType.MODEL_FAILED
            and event.data.get("phase") == "memory_summary"
        ]
        assert len(summary_failures) == 1
        assert summary_failures[0].data["code"] == "model_connection_error"
        assert summary_failures[0].data["terminal"] is True

        records = diagnostics.for_run(result.run_id)
        assert len(records) == 1
        assert records[0].phase == "memory_summary"
        assert records[0].code == "model_connection_error"
        assert records[0].terminal is True

        compacted = [
            event for event in events if event.type is EventType.MEMORY_COMPACTED
        ]
        assert len(compacted) == 1
        assert compacted[0].data["summary_error"] == "ModelInvocationError"
        assert compacted[0].data["original_tokens"] == 1
        assert compacted[0].data["selected_tokens"] == 1
        assert compacted[0].data["dropped_messages"] == 2
        assert math.isfinite(compacted[0].data["duration_seconds"])
        assert compacted[0].data["duration_seconds"] >= 0

        projected = repr(
            (
                result.error_summary,
                summary_failures[0].to_dict(),
                compacted[0].to_dict(),
                tuple(record.to_dict() for record in records),
            )
        )
        assert "PRIVATE-SUMMARY-CONNECTION-DETAIL" not in projected
        assert "PRIVATE-OLD-QUESTION" not in projected
        assert "PRIVATE-OLD-ANSWER" not in projected

    asyncio.run(scenario())


def test_summary_protocol_failure_keeps_terminal_primary_semantics() -> None:
    async def scenario() -> None:
        session_id = "memory-protocol-terminal"
        summary_model = _SequenceModel(
            [ModelProtocolError("PRIVATE-SUMMARY-PROTOCOL-DETAIL")]
        )
        main_model = _SequenceModel([ModelResponse(Message.assistant("must not run"))])
        diagnostics = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig(
                "memory-protocol-terminal",
                "Answer from bounded context.",
                retry=RetryConfig(max_attempts=3, initial_delay=0, max_delay=0),
            ),
            model=main_model,  # type: ignore[arg-type]
            conversation_store=await _conversation(session_id),
            conversation_memory_policy=_memory_policy(summary_model),
            diagnostic_sink=diagnostics,
        )

        events = [
            event
            async for event in agent.stream_all(
                "new question",
                session_id=session_id,
            )
        ]
        result = events[-1].data["result"]

        assert result.finish_reason is FinishReason.ERROR
        assert result.error_summary["code"] == "model_protocol_error"
        assert result.error_summary["component"] == "model"
        assert result.error_summary["operation"] == "complete"
        assert result.error_summary["phase"] == "memory_summary"
        assert len(summary_model.requests) == 1
        assert main_model.requests == []
        assert not any(event.type is EventType.MEMORY_COMPACTED for event in events)

        records = diagnostics.for_run(result.run_id)
        assert len(records) == 1
        assert records[0].code == "model_protocol_error"
        assert records[0].phase == "memory_summary"
        assert "PRIVATE-SUMMARY-PROTOCOL-DETAIL" not in repr(
            (
                result,
                events[-1].to_dict(),
                tuple(record.to_dict() for record in records),
            )
        )

    asyncio.run(scenario())


def test_summary_model_incomplete_exception_is_terminal_and_not_cached() -> None:
    async def scenario() -> None:
        session_id = "memory-incomplete-exception"
        summary_model = _SequenceModel([ModelOutputIncompleteError("length")])
        main_model = _SequenceModel([ModelResponse(Message.assistant("must not run"))])
        state_store = InMemoryMemoryStateStore()
        diagnostics = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig(
                "memory-incomplete-exception",
                "Answer from bounded context.",
                retry=RetryConfig(max_attempts=3, initial_delay=0, max_delay=0),
            ),
            model=main_model,  # type: ignore[arg-type]
            conversation_store=await _conversation(session_id),
            conversation_memory_policy=_memory_policy(
                summary_model,
                state_store=state_store,
            ),
            diagnostic_sink=diagnostics,
        )

        events = [
            event
            async for event in agent.stream_all(
                "new question",
                session_id=session_id,
            )
        ]
        result = events[-1].data["result"]

        assert result.finish_reason is FinishReason.ERROR
        assert result.error_summary["code"] == "model_output_incomplete"
        assert result.error_summary["category"] == "model_protocol"
        assert result.error_summary["provider_finish_reason"] == "length"
        assert len(summary_model.requests) == 1
        assert main_model.requests == []
        assert await state_store.load(session_id) is None
        assert not any(event.type is EventType.MEMORY_COMPACTED for event in events)
        failures = [
            event
            for event in events
            if event.type is EventType.MODEL_FAILED
            and event.data.get("phase") == "memory_summary"
        ]
        assert len(failures) == 1
        assert failures[0].data["code"] == "model_output_incomplete"

    asyncio.run(scenario())


def test_partial_summary_response_is_terminal_and_not_cached() -> None:
    async def scenario() -> None:
        session_id = "memory-incomplete-response"
        summary_model = _SequenceModel(
            [
                ModelResponse(
                    Message.assistant("PARTIAL-SUMMARY-MUST-NOT-BE-CACHED"),
                    finish_reason="max_tokens",
                )
            ]
        )
        main_model = _SequenceModel([ModelResponse(Message.assistant("must not run"))])
        state_store = InMemoryMemoryStateStore()
        agent = Agent(
            config=AgentConfig(
                "memory-incomplete-response",
                "Answer from bounded context.",
            ),
            model=main_model,  # type: ignore[arg-type]
            conversation_store=await _conversation(session_id),
            conversation_memory_policy=_memory_policy(
                summary_model,
                state_store=state_store,
            ),
        )

        events = [
            event
            async for event in agent.stream_all(
                "new question",
                session_id=session_id,
            )
        ]
        result = events[-1].data["result"]

        assert result.finish_reason is FinishReason.ERROR
        assert result.error_summary["code"] == "model_output_incomplete"
        assert result.error_summary["category"] == "model_protocol"
        assert result.error_summary["provider_finish_reason"] == "max_tokens"
        assert len(summary_model.requests) == 1
        assert main_model.requests == []
        assert await state_store.load(session_id) is None
        assert not any(event.type is EventType.MEMORY_COMPACTED for event in events)
        assert "PARTIAL-SUMMARY-MUST-NOT-BE-CACHED" not in repr(
            tuple(event.to_dict() for event in events)
        )

    asyncio.run(scenario())


def test_summary_model_guard_is_not_converted_to_optional_fallback() -> None:
    async def scenario() -> None:
        session_id = "memory-summary-guard"
        summary_model = _SequenceModel(
            [
                ModelResponse(
                    Message.assistant("first folded summary"),
                    usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
                )
            ]
        )
        main_model = _SequenceModel([ModelResponse(Message.assistant("must not run"))])
        diagnostics = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig(
                "memory-summary-guard",
                "Answer from bounded context.",
                limits=RunLimits(max_model_turns=1),
            ),
            model=main_model,  # type: ignore[arg-type]
            conversation_store=await _conversation(session_id),
            conversation_memory_policy=_memory_policy(
                summary_model,
                summary_token_counter=_OneRecordPerBatchCounter(),
            ),
            diagnostic_sink=diagnostics,
        )

        events = [
            event
            async for event in agent.stream_all(
                "new question",
                session_id=session_id,
            )
        ]
        result = events[-1].data["result"]

        assert result.finish_reason is FinishReason.MAX_MODEL_TURNS
        assert result.error_summary["code"] == "max_model_turns_exceeded"
        assert result.error_summary["max_model_turns"] == 1
        assert len(summary_model.requests) == 1
        assert main_model.requests == []
        assert not any(event.type is EventType.MEMORY_COMPACTED for event in events)
        assert not any(
            event.type is EventType.MODEL_STARTED
            and event.data.get("phase") != "memory_summary"
            for event in events
        )

    asyncio.run(scenario())


def test_memory_compaction_event_has_consistent_content_free_measurements() -> None:
    async def scenario() -> None:
        session_id = "memory-event-contract"
        summary_model = _SequenceModel(
            [
                ModelResponse(
                    Message.assistant("PRIVATE-SUMMARY-BODY"),
                    usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
                )
            ]
        )
        main_model = _SequenceModel(
            [ModelResponse(Message.assistant("public final answer"))]
        )
        agent = Agent(
            config=AgentConfig(
                "memory-event-contract",
                "Answer from bounded context.",
                limits=RunLimits(max_model_turns=4),
            ),
            model=main_model,  # type: ignore[arg-type]
            conversation_store=await _conversation(session_id),
            conversation_memory_policy=_memory_policy(
                summary_model,
                token_counter=_MessageCountCounter(),
            ),
        )

        events = [
            event
            async for event in agent.stream_all(
                "new question",
                session_id=session_id,
            )
        ]
        result = events[-1].data["result"]
        compacted = [
            event for event in events if event.type is EventType.MEMORY_COMPACTED
        ]

        assert result.finish_reason is FinishReason.COMPLETED
        assert len(compacted) == 1
        event = compacted[0]
        assert event.event_schema_version == 2
        assert event.data["phase"] == "act"
        assert event.data["original_tokens"] == 4
        assert event.data["selected_tokens"] == 3
        assert event.data["summarized_messages"] == 2
        assert event.data["dropped_messages"] == 0
        assert event.data["budget_tokens"] == 100
        assert type(event.data["duration_seconds"]) is float
        assert math.isfinite(event.data["duration_seconds"])
        assert event.data["duration_seconds"] >= 0
        assert not {"messages", "prompt", "summary"}.intersection(event.data)
        projected = repr(event.to_dict())
        assert "PRIVATE-SUMMARY-BODY" not in projected
        assert "PRIVATE-OLD-QUESTION" not in projected
        assert "PRIVATE-OLD-ANSWER" not in projected

    asyncio.run(scenario())

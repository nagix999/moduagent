from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from moduagent.memory import (
    ApproximateTokenCounter,
    CachingTokenCounter,
    ConversationMemoryOverflowError,
    InMemoryMemoryStateStore,
    MemoryIntegrityError,
    MemoryPhase,
    MemoryRequest,
    ModelConversationSummarizer,
    RecentTurnsConversationMemoryPolicy,
    SummaryResult,
    SummarizingConversationMemoryPolicy,
    TokenBudget,
    TokenBudgetConversationMemoryPolicy,
)
from moduagent.messages import Message, ToolCall, Usage
from moduagent.models import ModelRequest, ModelResponse


class SimpleTokenCounter:
    async def count_request(self, request: ModelRequest) -> int:
        message_tokens = sum(
            1
            + len(message.content or "")
            + len(json.dumps(message.to_dict(), ensure_ascii=False)) // 10
            for message in request.messages
        )
        tool_tokens = len(json.dumps(list(request.tools), ensure_ascii=False))
        schema_tokens = len(
            json.dumps(request.output_schema, ensure_ascii=False)
            if request.output_schema is not None
            else ""
        )
        return message_tokens + tool_tokens + schema_tokens


class PromptLengthCounter:
    async def count_request(self, request: ModelRequest) -> int:
        return sum(len(message.content or "") for message in request.messages)


class RecordingSummaryModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        number = len(self.requests)
        return ModelResponse(
            Message.assistant(f"summary-{number}"),
            usage=Usage(1, 1, 2),
        )


def _memory_request(
    messages: tuple[Message, ...],
    *,
    protected_from: int,
    tools: tuple[object, ...] = (),
    output_schema: dict[str, object] | None = None,
) -> MemoryRequest:
    return MemoryRequest(
        run_id="run-1",
        session_id="session-1",
        phase=MemoryPhase.ACT,
        model_request=ModelRequest(
            messages,
            tools=tools,
            output_schema=output_schema,
        ),
        protected_from=protected_from,
    )


def test_recent_turns_keeps_latest_complete_parallel_tool_turn() -> None:
    async def scenario() -> None:
        call_a = ToolCall("call-a", "first", {})
        call_b = ToolCall("call-b", "second", {})
        messages = (
            Message("system", "system"),
            Message("user", "old question"),
            Message("assistant", "old answer"),
            Message("user", "recent question"),
            Message("assistant", None, (call_a, call_b)),
            Message("tool", "B", tool_call_id="call-b", name="second"),
            Message("tool", "A", tool_call_id="call-a", name="first"),
            Message("assistant", "recent answer"),
            Message("user", "current question"),
        )
        request = _memory_request(messages, protected_from=8)

        result = await RecentTurnsConversationMemoryPolicy(max_turns=1).prepare(request)

        assert [message.content for message in result.messages] == [
            "system",
            "recent question",
            None,
            "B",
            "A",
            "recent answer",
            "current question",
        ]
        assert result.dropped_messages == 2
        assert request.model_request.messages == messages

    asyncio.run(scenario())


def test_recent_turns_zero_removes_all_history() -> None:
    async def scenario() -> None:
        messages = (
            Message.system("system"),
            Message.user("old"),
            Message.assistant("answer"),
            Message.user("current"),
        )
        result = await RecentTurnsConversationMemoryPolicy(max_turns=0).prepare(
            _memory_request(messages, protected_from=3)
        )

        assert [message.content for message in result.messages] == [
            "system",
            "current",
        ]

    asyncio.run(scenario())


def test_invalid_historical_tool_turn_drops_the_older_prefix() -> None:
    async def scenario() -> None:
        call = ToolCall("missing", "lookup", {})
        messages = (
            Message.system("system"),
            Message.user("old valid"),
            Message.assistant("old answer"),
            Message.user("broken"),
            Message.assistant(None, (call,)),
            Message.user("latest valid"),
            Message.assistant("latest answer"),
            Message.user("current"),
        )

        result = await RecentTurnsConversationMemoryPolicy(max_turns=10).prepare(
            _memory_request(messages, protected_from=7)
        )

        assert [message.content for message in result.messages] == [
            "system",
            "latest valid",
            "latest answer",
            "current",
        ]

    asyncio.run(scenario())


def test_active_tool_chain_integrity_error_is_not_silently_compacted() -> None:
    async def scenario() -> None:
        call = ToolCall("missing", "lookup", {})
        messages = (
            Message.system("system"),
            Message.user("current"),
            Message.assistant(None, (call,)),
        )

        try:
            await RecentTurnsConversationMemoryPolicy(max_turns=1).prepare(
                _memory_request(messages, protected_from=1)
            )
        except MemoryIntegrityError as exc:
            assert "missing a Tool result" in str(exc)
        else:
            raise AssertionError("an incomplete active Tool chain must fail")

    asyncio.run(scenario())


def test_token_budget_selects_the_newest_contiguous_suffix() -> None:
    async def scenario() -> None:
        counter = SimpleTokenCounter()
        messages = (
            Message.system("system"),
            Message.user("old question " * 8),
            Message.assistant("old answer " * 8),
            Message.user("recent question"),
            Message.assistant("recent answer"),
            Message.user("current"),
        )
        request = _memory_request(
            messages,
            protected_from=5,
            tools=({"name": "lookup"},),
        )
        expected_messages = (messages[0], messages[3], messages[4], messages[5])
        expected_tokens = await counter.count_request(
            replace(request.model_request, messages=expected_messages)
        )
        policy = TokenBudgetConversationMemoryPolicy(
            budget=TokenBudget(expected_tokens),
            token_counter=counter,
        )

        result = await policy.prepare(request)

        assert result.messages == expected_messages
        assert result.selected_tokens <= policy.budget.input_tokens
        assert result.dropped_messages == 2

    asyncio.run(scenario())


def test_token_budget_max_history_turns_applies_below_budget() -> None:
    async def scenario() -> None:
        messages = (
            Message.system("system"),
            Message.user("old"),
            Message.assistant("old answer"),
            Message.user("recent"),
            Message.assistant("recent answer"),
            Message.user("current"),
        )
        result = await TokenBudgetConversationMemoryPolicy(
            budget=TokenBudget(100_000),
            token_counter=SimpleTokenCounter(),
            max_history_turns=1,
        ).prepare(_memory_request(messages, protected_from=5))

        assert [message.content for message in result.messages] == [
            "system",
            "recent",
            "recent answer",
            "current",
        ]

    asyncio.run(scenario())


def test_token_budget_counts_an_unchanged_request_once() -> None:
    async def scenario() -> None:
        class CountingTokenCounter(SimpleTokenCounter):
            def __init__(self) -> None:
                self.calls = 0

            async def count_request(self, request: ModelRequest) -> int:
                self.calls += 1
                return await super().count_request(request)

        counter = CountingTokenCounter()
        messages = (
            Message.system("system"),
            Message.user("history"),
            Message.assistant("answer"),
            Message.user("current"),
        )

        result = await TokenBudgetConversationMemoryPolicy(
            budget=TokenBudget(100_000),
            token_counter=counter,
        ).prepare(_memory_request(messages, protected_from=3))

        assert result.messages == messages
        assert result.original_tokens == result.selected_tokens
        assert counter.calls == 1

    asyncio.run(scenario())


def test_token_budget_memoizes_binary_search_and_final_selection() -> None:
    async def scenario() -> None:
        class MessageCountTokenCounter:
            def __init__(self) -> None:
                self.calls = 0

            async def count_request(self, request: ModelRequest) -> int:
                self.calls += 1
                return len(request.messages)

        counter = MessageCountTokenCounter()
        messages = (
            Message.system("system"),
            Message.user("turn-1"),
            Message.assistant("answer-1"),
            Message.user("turn-2"),
            Message.assistant("answer-2"),
            Message.user("turn-3"),
            Message.assistant("answer-3"),
            Message.user("turn-4"),
            Message.assistant("answer-4"),
            Message.user("current"),
        )

        result = await TokenBudgetConversationMemoryPolicy(
            budget=TokenBudget(6),
            token_counter=counter,
        ).prepare(_memory_request(messages, protected_from=9))

        assert [message.content for message in result.messages] == [
            "system",
            "turn-3",
            "answer-3",
            "turn-4",
            "answer-4",
            "current",
        ]
        # Original, last two turns, and last three turns are three distinct
        # requests. The selected last-two view is not counted again.
        assert counter.calls == 3

    asyncio.run(scenario())


def test_token_budget_rejects_oversized_protected_input() -> None:
    async def scenario() -> None:
        messages = (
            Message.system("system instruction"),
            Message.user("current input that cannot fit"),
        )
        policy = TokenBudgetConversationMemoryPolicy(
            budget=TokenBudget(5),
            token_counter=SimpleTokenCounter(),
        )

        try:
            await policy.prepare(_memory_request(messages, protected_from=1))
        except ConversationMemoryOverflowError as exc:
            assert exc.required_tokens > exc.available_tokens
            assert exc.available_tokens == 5
        else:
            raise AssertionError("oversized protected input must fail")

    asyncio.run(scenario())


def test_summary_snapshot_is_reused_without_changing_raw_messages() -> None:
    async def scenario() -> None:
        class Summarizer:
            def __init__(self) -> None:
                self.calls = 0

            async def summarize(
                self,
                messages: tuple[Message, ...],
                *,
                previous_summary: str | None = None,
            ) -> SummaryResult:
                self.calls += 1
                assert previous_summary is None
                return SummaryResult("remembered fact", Usage(3, 2, 5))

        messages = (
            Message.system("system"),
            Message.user("old"),
            Message.assistant("old answer"),
            Message.user("recent"),
            Message.assistant("recent answer"),
            Message.user("current"),
        )
        request = _memory_request(messages, protected_from=5)
        summarizer = Summarizer()
        policy = SummarizingConversationMemoryPolicy(
            budget=TokenBudget(100_000),
            token_counter=SimpleTokenCounter(),
            summarizer=summarizer,
            state_store=InMemoryMemoryStateStore(),
            max_history_turns=1,
        )

        first = await policy.prepare(request)
        second = await policy.prepare(request)

        assert summarizer.calls == 1
        assert first.summarized_messages == 2
        assert first.usage.total_tokens == 5
        assert second.usage.total_tokens == 0
        assert second.metadata["cache_hit"] is True
        assert first.messages[1].metadata["moduagent.memory"] == "summary"
        assert request.model_request.messages == messages

    asyncio.run(scenario())


def test_summary_failure_falls_back_to_recent_turns() -> None:
    async def scenario() -> None:
        class FailingSummarizer:
            async def summarize(
                self,
                messages: tuple[Message, ...],
                *,
                previous_summary: str | None = None,
            ) -> SummaryResult:
                raise RuntimeError("unavailable")

        messages = (
            Message.system("system"),
            Message.user("old"),
            Message.assistant("old answer"),
            Message.user("recent"),
            Message.assistant("recent answer"),
            Message.user("current"),
        )
        result = await SummarizingConversationMemoryPolicy(
            budget=TokenBudget(100_000),
            token_counter=SimpleTokenCounter(),
            summarizer=FailingSummarizer(),
            max_history_turns=1,
        ).prepare(_memory_request(messages, protected_from=5))

        assert [message.content for message in result.messages] == [
            "system",
            "recent",
            "recent answer",
            "current",
        ]
        assert result.metadata["summary_error"] == "RuntimeError"

    asyncio.run(scenario())


def test_summarizing_policy_does_not_call_model_below_limits() -> None:
    async def scenario() -> None:
        class UnexpectedSummarizer:
            async def summarize(
                self,
                messages: tuple[Message, ...],
                *,
                previous_summary: str | None = None,
            ) -> SummaryResult:
                raise AssertionError("summarizer must not be called")

        messages = (
            Message.system("system"),
            Message.user("history"),
            Message.assistant("answer"),
            Message.user("current"),
        )
        result = await SummarizingConversationMemoryPolicy(
            budget=TokenBudget(100_000),
            token_counter=SimpleTokenCounter(),
            summarizer=UnexpectedSummarizer(),
        ).prepare(_memory_request(messages, protected_from=3))

        assert result.messages == messages
        assert result.summarized_messages == 0
        assert result.usage.total_tokens == 0

    asyncio.run(scenario())


def test_model_summarizer_batches_and_folds_a_large_single_message() -> None:
    async def scenario() -> None:
        model = RecordingSummaryModel()
        counter = PromptLengthCounter()
        summarizer = ModelConversationSummarizer(
            model=model,
            token_counter=counter,
            max_input_tokens=900,
            max_output_tokens=64,
        )
        source = Message.tool(
            "x" * 2_400,
            call_id="call-large",
            name="large_result",
        )

        result = await summarizer.summarize((source,))

        assert len(model.requests) >= 3
        assert result.summary == f"summary-{len(model.requests)}"
        assert result.usage.total_tokens == len(model.requests) * 2
        fragments: list[str] = []
        for index, request in enumerate(model.requests):
            assert request.tools == ()
            assert request.output_schema is None
            assert request.options["max_tokens"] == 64
            assert await counter.count_request(request) <= 900
            payload = json.loads((request.messages[-1].content or "").split("\n", 1)[1])
            fragments.extend(payload["conversation_record_fragments"])
            if index:
                assert payload["previous_summary"] == f"summary-{index}"
        expected = json.dumps(
            source.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        assert "".join(fragments) == expected

    asyncio.run(scenario())


def test_model_summarizer_rejects_tool_calls_and_blank_output() -> None:
    async def scenario() -> None:
        call = ToolCall("call-summary", "unexpected", {})

        class InvalidModel:
            def __init__(self, response: ModelResponse) -> None:
                self.response = response

            async def complete(self, request: ModelRequest) -> ModelResponse:
                return self.response

        for response, expected in (
            (
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                "Tool Calls",
            ),
            (ModelResponse(Message.assistant("  ")), "empty summary"),
        ):
            summarizer = ModelConversationSummarizer(
                model=InvalidModel(response),  # type: ignore[arg-type]
                token_counter=PromptLengthCounter(),
                max_input_tokens=2_000,
            )
            try:
                await summarizer.summarize((Message.user("old"),))
            except RuntimeError as exc:
                assert expected in str(exc)
            else:
                raise AssertionError("invalid summary response must fail")

    asyncio.run(scenario())


def test_approximate_counter_includes_tools_and_output_schema() -> None:
    async def scenario() -> None:
        counter = ApproximateTokenCounter(bytes_per_token=2)
        plain = ModelRequest((Message.user("안녕하세요"),))
        constrained = ModelRequest(
            plain.messages,
            tools=({"name": "lookup", "description": "검색"},),
            output_schema={"type": "object", "properties": {"value": {}}},
        )

        assert await counter.count_request(constrained) > await counter.count_request(
            plain
        )

    asyncio.run(scenario())


def test_caching_token_counter_singleflights_and_retains_no_prompt() -> None:
    async def scenario() -> None:
        class SlowTokenCounter:
            def __init__(self) -> None:
                self.calls = 0
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def count_request(self, request: ModelRequest) -> int:
                self.calls += 1
                self.started.set()
                await self.release.wait()
                return len(request.messages)

        secret = "private-customer-prompt-42"
        delegate = SlowTokenCounter()
        counter = CachingTokenCounter(
            delegate,
            max_entries=4,
            ttl_seconds=None,
        )
        request = ModelRequest((Message.user(secret),))
        tasks = [asyncio.create_task(counter.count_request(request)) for _ in range(20)]
        await delegate.started.wait()
        delegate.release.set()

        assert await asyncio.gather(*tasks) == [1] * 20
        assert await counter.count_request(request) == 1
        assert delegate.calls == 1
        assert counter._inflight == {}
        assert len(counter._cache) == 1
        assert all(isinstance(key, bytes) and len(key) == 32 for key in counter._cache)
        assert secret not in repr(counter.__dict__)

    asyncio.run(scenario())


def test_caching_token_counter_caches_only_successful_counts() -> None:
    async def scenario() -> None:
        class FlakyTokenCounter:
            def __init__(self) -> None:
                self.calls = 0

            async def count_request(self, request: ModelRequest) -> int:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("tokenizer unavailable")
                return 17

        delegate = FlakyTokenCounter()
        counter = CachingTokenCounter(delegate)
        request = ModelRequest((Message.user("hello"),))

        try:
            await counter.count_request(request)
        except RuntimeError as exc:
            assert str(exc) == "tokenizer unavailable"
        else:
            raise AssertionError("delegate failure must be propagated")

        assert await counter.count_request(request) == 17
        assert await counter.count_request(request) == 17
        assert delegate.calls == 2

    asyncio.run(scenario())


def test_caching_token_counter_uses_a_bounded_lru() -> None:
    async def scenario() -> None:
        class CountingTokenCounter:
            def __init__(self) -> None:
                self.calls = 0

            async def count_request(self, request: ModelRequest) -> int:
                self.calls += 1
                return len(request.messages[0].content or "")

        delegate = CountingTokenCounter()
        counter = CachingTokenCounter(
            delegate,
            max_entries=2,
            ttl_seconds=None,
        )
        requests = {
            value: ModelRequest((Message.user(value),)) for value in ("a", "bb", "ccc")
        }

        assert await counter.count_request(requests["a"]) == 1
        assert await counter.count_request(requests["bb"]) == 2
        assert await counter.count_request(requests["a"]) == 1
        assert await counter.count_request(requests["ccc"]) == 3
        assert await counter.count_request(requests["bb"]) == 2
        assert delegate.calls == 4
        assert len(counter._cache) == 2

    asyncio.run(scenario())


def test_caching_token_counter_keys_the_complete_model_request() -> None:
    async def scenario() -> None:
        class SequenceTokenCounter:
            def __init__(self) -> None:
                self.calls = 0

            async def count_request(self, request: ModelRequest) -> int:
                self.calls += 1
                return self.calls

        delegate = SequenceTokenCounter()
        counter = CachingTokenCounter(delegate, ttl_seconds=None)
        base = ModelRequest((Message.user("hello", metadata={"opaque": object()}),))
        requests = (
            base,
            replace(base, tools=({"name": "lookup"},)),
            replace(base, output_schema={"type": "object"}),
            replace(base, options={"temperature": 0}),
            replace(base, provider_options={"chat_template": "custom"}),
        )

        for expected, request in enumerate(requests, start=1):
            assert await counter.count_request(request) == expected
            assert await counter.count_request(request) == expected
        assert delegate.calls == len(requests)

    asyncio.run(scenario())

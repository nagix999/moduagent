from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from moduagent.memory.base import (
    ConversationMemoryOverflowError,
    MemoryIntegrityError,
    MemoryRequest,
    MemoryResult,
)
from moduagent.memory.state import (
    InMemoryMemoryStateStore,
    MemorySnapshot,
    MemoryStateStore,
)
from moduagent.memory.summarizer import (
    ConversationSummarizer,
    GatewayConversationSummarizer,
)
from moduagent.models import ModelGateway
from moduagent.memory.token import (
    ApproximateTokenCounter,
    TokenBudget,
    TokenCounter,
)
from moduagent.messages import Message, MessageRole, Usage


@dataclass(frozen=True, slots=True)
class _ConversationParts:
    system: tuple[Message, ...]
    history_turns: tuple[tuple[Message, ...], ...]
    protected: tuple[Message, ...]
    invalid_history_messages: int = 0
    invalid_history_turns: int = 0


class FullConversationMemoryPolicy:
    """Identity policy preserving the framework's original behavior."""

    async def prepare(self, request: MemoryRequest) -> MemoryResult:
        if request.protected_from > len(request.model_request.messages):
            raise ValueError("protected_from exceeds the message count")
        return MemoryResult(messages=request.model_request.messages)


class _ImmutableSemanticConfiguration:
    _semantic_fields: frozenset[str] = frozenset()

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._semantic_fields and name in self.__dict__:
            raise AttributeError(f"{name} is immutable after policy construction")
        object.__setattr__(self, name, value)


class RecentTurnsConversationMemoryPolicy(_ImmutableSemanticConfiguration):
    """Keep at most the most recent complete historical user turns."""

    _semantic_fields = frozenset({"max_turns"})

    def __init__(self, max_turns: int) -> None:
        if max_turns < 0:
            raise ValueError("max_turns cannot be negative")
        self.max_turns = max_turns

    async def prepare(self, request: MemoryRequest) -> MemoryResult:
        parts = _conversation_parts(request)
        turns = parts.history_turns[-self.max_turns :] if self.max_turns else ()
        messages = _compose(parts, turns)
        selected_history_messages = sum(len(turn) for turn in turns)
        total_history_messages = sum(len(turn) for turn in parts.history_turns)
        dropped = (
            total_history_messages
            - selected_history_messages
            + parts.invalid_history_messages
        )
        return MemoryResult(
            messages=messages,
            dropped_messages=dropped,
            metadata={
                "phase": request.phase.value,
                "selected_history_turns": len(turns),
                "dropped_history_turns": len(parts.history_turns) - len(turns),
                "invalid_history_turns": parts.invalid_history_turns,
            },
        )


class TokenBudgetConversationMemoryPolicy(_ImmutableSemanticConfiguration):
    """Select a contiguous suffix of complete turns within a token budget."""

    _semantic_fields = frozenset(
        {
            "budget",
            "token_counter",
            "summarizer",
            "state_store",
            "max_history_turns",
            "policy_fingerprint",
        }
    )

    def __init__(
        self,
        *,
        budget: TokenBudget,
        token_counter: TokenCounter | None = None,
        summarizer: ConversationSummarizer | None = None,
        state_store: MemoryStateStore | None = None,
        max_history_turns: int | None = None,
    ) -> None:
        if max_history_turns is not None and max_history_turns < 0:
            raise ValueError("max_history_turns cannot be negative")
        if summarizer is None and state_store is not None:
            raise ValueError("state_store requires a summarizer")
        self.budget = budget
        self.token_counter = token_counter or ApproximateTokenCounter()
        self.summarizer = summarizer
        self.state_store = (
            state_store
            if state_store is not None
            else (InMemoryMemoryStateStore() if summarizer is not None else None)
        )
        self.max_history_turns = max_history_turns
        fingerprint_value = {
            "budget": {
                "context": budget.context_window_tokens,
                "output": budget.reserved_output_tokens,
                "margin": budget.safety_margin_tokens,
            },
            "max_history_turns": max_history_turns,
            "counter": type(self.token_counter).__qualname__,
            "summarizer": (
                str(
                    getattr(
                        summarizer,
                        "cache_fingerprint",
                        type(summarizer).__qualname__,
                    )
                )
                if summarizer is not None
                else None
            ),
        }
        self.policy_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_value, sort_keys=True).encode()
        ).hexdigest()

    async def prepare(self, request: MemoryRequest) -> MemoryResult:
        parts = _conversation_parts(request)
        original_tokens = await self.token_counter.count_request(request.model_request)
        all_turns = parts.history_turns
        if self.max_history_turns is None:
            selected = list(all_turns)
        elif self.max_history_turns == 0:
            selected = []
        else:
            selected = list(all_turns[-self.max_history_turns :])

        selected_messages = _compose(parts, selected)
        selected_tokens = (
            original_tokens
            if selected_messages == request.model_request.messages
            else await self._count(request, selected_messages)
        )
        if selected and selected_tokens > self.budget.input_tokens:
            # Token counts are monotonic for a suffix of complete turns. Find the
            # smallest removable prefix with O(log N) tokenizer calls; this is
            # important when the counter uses vLLM's remote /tokenize endpoint.
            low = 1
            high = len(selected)
            while low < high:
                middle = (low + high) // 2
                candidate = selected[middle:]
                candidate_tokens = await self._count(
                    request, _compose(parts, candidate)
                )
                if candidate_tokens <= self.budget.input_tokens:
                    high = middle
                else:
                    low = middle + 1
            selected = selected[low:]
            selected_tokens = await self._count(request, _compose(parts, selected))

        if selected_tokens > self.budget.input_tokens:
            await self._raise_overflow(request, _compose(parts, ()), selected_tokens)

        excluded_count = len(all_turns) - len(selected)
        excluded_turns = all_turns[:excluded_count]
        summary_message: Message | None = None
        summary_usage = Usage()
        cache_hit = False
        summary_error: str | None = None
        fallback_selected = list(selected)
        fallback_excluded_count = excluded_count
        fallback_tokens = selected_tokens

        if self.summarizer is not None and excluded_turns:
            try:
                summary_message, summary_usage, cache_hit = await self._summary(
                    request.session_id,
                    excluded_turns,
                    model_gateway=request.model_gateway,
                )
                with_summary = _compose(parts, selected, summary=summary_message)
                summary_tokens = await self._count(request, with_summary)
                while selected and summary_tokens > self.budget.input_tokens:
                    selected.pop(0)
                    excluded_count += 1
                    excluded_turns = all_turns[:excluded_count]
                    summary_message, extra_usage, cache_hit = await self._summary(
                        request.session_id,
                        excluded_turns,
                        model_gateway=request.model_gateway,
                    )
                    summary_usage = summary_usage + extra_usage
                    with_summary = _compose(parts, selected, summary=summary_message)
                    summary_tokens = await self._count(request, with_summary)
                if summary_tokens <= self.budget.input_tokens:
                    selected_tokens = summary_tokens
                else:
                    summary_message = None
            except Exception as exc:
                summary_message = None
                summary_error = type(exc).__name__

            if summary_message is None:
                # Trying to fit a summary may have removed additional recent
                # turns. If the summary is unusable, restore the most informative
                # recent-only view that was already proven to fit.
                selected = fallback_selected
                excluded_count = fallback_excluded_count
                excluded_turns = all_turns[:excluded_count]
                selected_tokens = fallback_tokens

        messages = _compose(parts, selected, summary=summary_message)
        selected_tokens = await self._count(request, messages)
        represented = sum(len(turn) for turn in excluded_turns)
        summarized_messages = represented if summary_message is not None else 0
        valid_history_messages = sum(len(turn) for turn in all_turns)
        selected_history_messages = sum(len(turn) for turn in selected)
        dropped_messages = (
            valid_history_messages
            - selected_history_messages
            - summarized_messages
            + parts.invalid_history_messages
        )
        metadata: dict[str, object] = {
            "phase": request.phase.value,
            "budget_tokens": self.budget.input_tokens,
            "selected_history_turns": len(selected),
            "invalid_history_turns": parts.invalid_history_turns,
            "cache_hit": cache_hit,
        }
        if summary_error is not None:
            metadata["summary_error"] = summary_error
        return MemoryResult(
            messages=messages,
            usage=summary_usage,
            original_tokens=original_tokens,
            selected_tokens=selected_tokens,
            summarized_messages=summarized_messages,
            dropped_messages=dropped_messages,
            metadata=metadata,
        )

    async def _count(
        self, request: MemoryRequest, messages: tuple[Message, ...]
    ) -> int:
        return await self.token_counter.count_request(
            replace(request.model_request, messages=messages)
        )

    async def _raise_overflow(
        self,
        request: MemoryRequest,
        messages: tuple[Message, ...],
        required_tokens: int,
    ) -> None:
        original = request.model_request
        message_tokens = await self.token_counter.count_request(
            replace(original, messages=messages, tools=(), output_schema=None)
        )
        messages_and_tools = await self.token_counter.count_request(
            replace(original, messages=messages, output_schema=None)
        )
        tool_tokens = max(0, messages_and_tools - message_tokens)
        schema_tokens = max(0, required_tokens - messages_and_tools)
        raise ConversationMemoryOverflowError(
            required_tokens=required_tokens,
            available_tokens=self.budget.input_tokens,
            message_tokens=message_tokens,
            tool_tokens=tool_tokens,
            schema_tokens=schema_tokens,
        )

    async def _summary(
        self,
        session_id: str,
        turns: tuple[tuple[Message, ...], ...],
        *,
        model_gateway: ModelGateway | None,
    ) -> tuple[Message, Usage, bool]:
        if self.summarizer is None:
            raise RuntimeError("summarizer is not configured")
        messages = tuple(message for turn in turns for message in turn)
        digest = _messages_digest(messages)
        snapshot = (
            await self.state_store.load(session_id)
            if self.state_store is not None
            else None
        )
        previous_summary: str | None = None
        new_messages = messages
        cache_hit = False
        if (
            snapshot is not None
            and snapshot.policy_fingerprint == self.policy_fingerprint
            and snapshot.covered_message_count <= len(messages)
            and _messages_digest(messages[: snapshot.covered_message_count])
            == snapshot.covered_prefix_digest
        ):
            previous_summary = snapshot.summary
            new_messages = messages[snapshot.covered_message_count :]
            if not new_messages:
                cache_hit = True
                return _summary_message(snapshot.summary), Usage(), cache_hit

        if model_gateway is not None and isinstance(
            self.summarizer,
            GatewayConversationSummarizer,
        ):
            result = await self.summarizer.summarize_with_gateway(
                new_messages,
                previous_summary=previous_summary,
                gateway=model_gateway,
            )
        else:
            result = await self.summarizer.summarize(
                new_messages,
                previous_summary=previous_summary,
            )
        if self.state_store is not None:
            await self.state_store.save(
                session_id,
                MemorySnapshot(
                    summary=result.summary,
                    covered_message_count=len(messages),
                    covered_prefix_digest=digest,
                    policy_fingerprint=self.policy_fingerprint,
                ),
            )
        return _summary_message(result.summary), result.usage, cache_hit


class SummarizingConversationMemoryPolicy(TokenBudgetConversationMemoryPolicy):
    """Token-budget policy that requires and applies a conversation summarizer."""

    def __init__(
        self,
        *,
        budget: TokenBudget,
        summarizer: ConversationSummarizer,
        token_counter: TokenCounter | None = None,
        state_store: MemoryStateStore | None = None,
        max_history_turns: int | None = None,
    ) -> None:
        super().__init__(
            budget=budget,
            token_counter=token_counter,
            summarizer=summarizer,
            state_store=state_store,
            max_history_turns=max_history_turns,
        )


def _conversation_parts(request: MemoryRequest) -> _ConversationParts:
    messages = request.model_request.messages
    if request.protected_from > len(messages):
        raise ValueError("protected_from exceeds the message count")

    historical = messages[: request.protected_from]
    protected = messages[request.protected_from :]
    integrity_error = _tool_integrity_error(
        protected, index_offset=request.protected_from
    )
    if integrity_error is not None:
        raise integrity_error

    system_end = 0
    while (
        system_end < len(historical)
        and historical[system_end].role == MessageRole.SYSTEM
    ):
        system_end += 1
    system = historical[:system_end]
    raw_history = historical[system_end:]

    raw_turns: list[tuple[Message, ...]] = []
    current: list[Message] = []
    orphaned: list[Message] = []
    for message in raw_history:
        if message.role == MessageRole.USER:
            if current:
                raw_turns.append(tuple(current))
            current = [message]
        elif current:
            current.append(message)
        else:
            orphaned.append(message)
    if current:
        raw_turns.append(tuple(current))

    validated_turns: list[tuple[tuple[Message, ...], bool]] = []
    last_invalid_turn = -1
    turn_offset = system_end
    for turn_index, turn in enumerate(raw_turns):
        error = _tool_integrity_error(turn, index_offset=turn_offset)
        valid = error is None
        validated_turns.append((turn, valid))
        if not valid:
            last_invalid_turn = turn_index
        turn_offset += len(turn)

    # A malformed historical Tool block breaks the semantic continuity of all
    # preceding turns. Retain only the newest valid contiguous suffix.
    history_turns = tuple(
        turn for turn, valid in validated_turns[last_invalid_turn + 1 :] if valid
    )
    excluded_prefix = raw_turns[: last_invalid_turn + 1]
    invalid_messages = len(orphaned) + sum(len(turn) for turn in excluded_prefix)
    invalid_turns = (1 if orphaned else 0) + len(excluded_prefix)

    return _ConversationParts(
        system=system,
        history_turns=history_turns,
        protected=protected,
        invalid_history_messages=invalid_messages,
        invalid_history_turns=invalid_turns,
    )


def _tool_integrity_error(
    messages: tuple[Message, ...], *, index_offset: int
) -> MemoryIntegrityError | None:
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == MessageRole.TOOL:
            return MemoryIntegrityError(
                "Tool result has no preceding assistant Tool Call",
                message_index=index_offset + index,
            )
        if message.role != MessageRole.ASSISTANT or not message.tool_calls:
            index += 1
            continue

        call_ids = [call.id for call in message.tool_calls]
        if any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(
            call_ids
        ):
            return MemoryIntegrityError(
                "assistant Tool Call identifiers must be non-empty and unique",
                message_index=index_offset + index,
            )
        expected = {call.id: call.name for call in message.tool_calls}
        seen: set[str] = set()
        position = index + 1
        while position < len(messages) and messages[position].role == MessageRole.TOOL:
            result = messages[position]
            call_id = result.tool_call_id
            if call_id not in expected or call_id in seen:
                return MemoryIntegrityError(
                    "Tool result identifier does not match its Tool Call",
                    message_index=index_offset + position,
                )
            if result.name is not None and result.name != expected[call_id]:
                return MemoryIntegrityError(
                    "Tool result name does not match its Tool Call",
                    message_index=index_offset + position,
                )
            seen.add(call_id)
            position += 1
        if seen != set(expected):
            return MemoryIntegrityError(
                "assistant Tool Call is missing a Tool result",
                message_index=index_offset + index,
            )
        index = position
    return None


def _compose(
    parts: _ConversationParts,
    turns: list[tuple[Message, ...]] | tuple[tuple[Message, ...], ...],
    *,
    summary: Message | None = None,
) -> tuple[Message, ...]:
    history = tuple(message for turn in turns for message in turn)
    summary_messages = () if summary is None else (summary,)
    return (*parts.system, *summary_messages, *history, *parts.protected)


def _summary_message(summary: str) -> Message:
    return Message(
        MessageRole.ASSISTANT,
        f"Summary of earlier conversation:\n{summary}",
        metadata={"moduagent.memory": "summary"},
    )


def _messages_digest(messages: tuple[Message, ...]) -> str:
    payload = [message.to_dict() for message in messages]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


__all__ = [
    "FullConversationMemoryPolicy",
    "RecentTurnsConversationMemoryPolicy",
    "SummarizingConversationMemoryPolicy",
    "TokenBudgetConversationMemoryPolicy",
]

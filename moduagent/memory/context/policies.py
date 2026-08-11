from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from moduagent.memory.base import MemoryRequest, MemoryResult
from moduagent.memory.policies import (
    _RequestTokenMemo,
    _compose,
    _conversation_parts,
    SummarizingConversationMemoryPolicy as LegacySummarizingConversationMemoryPolicy,
)
from moduagent.memory.summarizer import (
    ConversationSummarizer,
    GatewayConversationSummarizer,
)
from moduagent.memory.token import TokenBudget, TokenCounter
from moduagent.messages import Message, MessageRole, Usage
from moduagent.models import ModelGateway, ModelProtocolError
from moduagent.persistence.conversation import ConversationStore

from .assembler import ContextAssembler
from .errors import (
    ContextMemoryError,
    ContextMemoryIntegrityError,
    ContextMemoryWriteConflictError,
)
from .history import (
    ContextHistoryView,
    DurableContextHistoryLoader,
    _extend_prefix_digest,
    _is_summary_boundary,
    _message_cursor,
    _summary_boundary_messages,
    _summary_boundary_prefix,
)
from .models import (
    MAX_SUMMARY_SOURCE_MESSAGE_IDS,
    ConversationSummary,
    ConversationSummarySnapshot,
    MemoryStateKey,
)
from .migration import ScopedLegacyMemoryStateStore
from .runtime_assembly import (
    assemble_runtime_memory_result,
    select_runtime_context,
)
from .stores import ContextMemoryStateStore


@dataclass(frozen=True, slots=True)
class _SummaryCandidate:
    message: Message
    usage: Usage
    cache_hit: bool
    expected_version: int | None = None
    next_snapshot: ConversationSummarySnapshot | None = None


class DurableSummarizingConversationMemoryPolicy(
    LegacySummarizingConversationMemoryPolicy
):
    """Cursor-aware summarizing policy with one bounded request budget.

    Protected Tool-block handling, exact token counting, model-guard
    propagation, and optional transport-failure fallback retain the legacy
    policy's behavior. This policy additionally owns a bounded history loader,
    persists v2 snapshots with an absolute message cursor and CAS, and delegates
    summary/recent-turn selection to ContextAssembler v1.

    The runtime must call ``history_loader.load_history(...)`` at bootstrap and
    use ``ContextHistoryView.messages`` instead of loading the complete session.
    Legacy/Full policies remain untouched and may continue to use ``load()``.
    """

    _semantic_fields = (
        LegacySummarizingConversationMemoryPolicy._semantic_fields
        | frozenset(
            {
                "tenant_id",
                "agent_id",
                "context_assembler",
                "history_loader",
                "legacy_state_store",
            }
        )
    )

    def __init__(
        self,
        *,
        budget: TokenBudget,
        summarizer: ConversationSummarizer,
        state_store: ContextMemoryStateStore,
        tenant_id: str,
        agent_id: str,
        token_counter: TokenCounter | None = None,
        max_history_turns: int | None = None,
        history_page_size: int = 256,
        max_uncompacted_messages: int = 2_048,
        legacy_state_store: ScopedLegacyMemoryStateStore | None = None,
        max_legacy_migration_messages: int = 100_000,
    ) -> None:
        if not isinstance(state_store, ContextMemoryStateStore):
            raise TypeError("state_store must implement ContextMemoryStateStore")
        # The superclass only calls this store in _summary(), which this class
        # overrides with the v2 CAS contract. Passing it through keeps the
        # established state_store attribute available to profile validation.
        super().__init__(
            budget=budget,
            summarizer=summarizer,
            token_counter=token_counter,
            state_store=state_store,  # type: ignore[arg-type]
            max_history_turns=max_history_turns,
        )
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.context_assembler = ContextAssembler()
        self.legacy_state_store = legacy_state_store
        self.history_loader = DurableContextHistoryLoader(
            state_store=state_store,
            tenant_id=tenant_id,
            agent_id=agent_id,
            policy_fingerprint=self.policy_fingerprint,
            page_size=history_page_size,
            max_tail_messages=max_uncompacted_messages,
            legacy_state_store=legacy_state_store,
            max_legacy_migration_messages=max_legacy_migration_messages,
        )

    async def load_history(
        self,
        conversation_store: ConversationStore,
        session_id: str,
    ) -> ContextHistoryView:
        """Convenience hook equivalent to calling the owned loader directly."""

        return await self.history_loader.load_history(
            conversation_store,
            session_id,
        )

    async def clear_history(
        self,
        conversation_store: ConversationStore,
        session_id: str,
    ) -> None:
        """Clear this policy's derived state and canonical session safely."""

        await self.history_loader.clear_history(conversation_store, session_id)

    def _is_terminal_summary_error(self, error: Exception) -> bool:
        # Preserve the original typed failure, including subclasses declared by
        # third-party stores after this module was imported. Class-name matching
        # is unsafe at an extensible persistence boundary.
        return isinstance(error, ContextMemoryError)

    async def prepare(self, request: MemoryRequest) -> MemoryResult:
        """Prepare one durable view with ContextAssembler-owned selection.

        The established policy still discovers the largest recent suffix that
        fits without a summary. When older turns need compaction, summary,
        recent turns, system/task/current-run input and request schemas become
        typed Context items under one budget. ContextAssembler then owns the
        optional priority/atomic decision; exact recounting handles provider
        chat templates that are not perfectly additive.
        """

        token_counter = _RequestTokenMemo(self.token_counter)
        parts = _conversation_parts(request)
        original_tokens = await token_counter.count_request(request.model_request)
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
            else await self._count(request, selected_messages, token_counter)
        )
        if selected and selected_tokens > self.budget.input_tokens:
            low = 1
            high = len(selected)
            while low < high:
                middle = (low + high) // 2
                candidate = selected[middle:]
                candidate_tokens = await self._count(
                    request,
                    _compose(parts, candidate),
                    token_counter,
                )
                if candidate_tokens <= self.budget.input_tokens:
                    high = middle
                else:
                    low = middle + 1
            selected = selected[low:]
            selected_tokens = await self._count(
                request,
                _compose(parts, selected),
                token_counter,
            )

        if selected_tokens > self.budget.input_tokens:
            await self._raise_overflow(
                request,
                _compose(parts, ()),
                selected_tokens,
                token_counter,
            )

        excluded_count = len(all_turns) - len(selected)
        excluded_turns = all_turns[:excluded_count]
        fallback_selected = list(selected)
        fallback_excluded_count = excluded_count
        fallback_tokens = selected_tokens
        summary_message: Message | None = None
        summary_usage = Usage()
        cache_hit = False
        summary_error: str | None = None
        if excluded_turns:
            try:
                candidate = await self._summary_candidate(
                    request.session_id,
                    excluded_turns,
                    model_gateway=request.model_gateway,
                )
                summary_usage = summary_usage + candidate.usage
                cache_hit = candidate.cache_hit
                selection = await select_runtime_context(
                    assembler=self.context_assembler,
                    model_request=request.model_request,
                    system=parts.system,
                    summary=candidate.message,
                    turns=tuple(selected),
                    protected=parts.protected,
                    token_counter=token_counter,
                    token_budget=self.budget.input_tokens,
                )
                if not selection.summary_selected:
                    # Summary is optional. Never CAS a new candidate that was
                    # not selected. An existing snapshot is left untouched and
                    # omitted for this exact request; it remains a valid cache
                    # boundary for a later request with more space.
                    summary_message = None
                else:
                    committed_message, cache_hit = await self._commit_summary_candidate(
                        candidate
                    )
                    # A concurrent CAS winner may contain different summary text
                    # for the same verified prefix. Revalidate the exact boundary
                    # before exposing it; a persisted winner cannot be rolled back.
                    if committed_message != candidate.message:
                        selection = await select_runtime_context(
                            assembler=self.context_assembler,
                            model_request=request.model_request,
                            system=parts.system,
                            summary=committed_message,
                            turns=tuple(selected),
                            protected=parts.protected,
                            token_counter=token_counter,
                            token_budget=self.budget.input_tokens,
                        )
                        if not selection.summary_selected:
                            # CAS is irreversible at this boundary, but request
                            # inclusion is not: keep the verified winner in the
                            # cache and continue with the proven recent-only view.
                            summary_message = None
                        else:
                            summary_message = committed_message
                    else:
                        summary_message = committed_message
                    if summary_message is not None:
                        selected = list(selection.selected_turns)
                        selected_tokens = selection.selected_tokens
            except ModelProtocolError:
                raise
            except Exception as exc:
                from moduagent.runtime.model_guard import ModelGuardTripped

                if isinstance(exc, ModelGuardTripped):
                    raise
                if self._is_terminal_summary_error(exc):
                    raise
                summary_message = None
                summary_error = type(exc).__name__

            if summary_message is None:
                selected = fallback_selected
                excluded_count = fallback_excluded_count
                excluded_turns = all_turns[:excluded_count]
                selected_tokens = fallback_tokens

        messages = _compose(parts, selected, summary=summary_message)
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
            "context_selection_owner": "context-assembler-v1",
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

    async def assemble_runtime_context(
        self,
        request: MemoryRequest,
        result: MemoryResult,
    ) -> MemoryResult:
        """Apply ContextAssembler v1 to the prepared model request.

        This additive runtime hook is intentionally absent from legacy policies.
        It therefore preserves their request behavior while making the durable
        policy's system/task/history/Tool/schema allocation explicit and
        auditable under the same tokenizer-aware input budget.
        """

        return assemble_runtime_memory_result(
            assembler=self.context_assembler,
            request=request,
            result=result,
            token_budget=self.budget.input_tokens,
        )

    async def _summary_candidate(
        self,
        session_id: str,
        turns: tuple[tuple[Message, ...], ...],
        *,
        model_gateway: ModelGateway | None,
    ) -> _SummaryCandidate:
        if self.summarizer is None:  # Defensive; constructor requires it.
            raise RuntimeError("summarizer is not configured")
        flattened = tuple(message for turn in turns for message in turn)
        source_messages, loaded_cursor, loaded_digest = _validated_source_messages(
            flattened
        )
        cursors = tuple(_required_cursor(message) for message in source_messages)
        if cursors:
            _validate_contiguous_cursors(cursors)
            if cursors[0][0] != loaded_cursor + 1:
                raise ContextMemoryIntegrityError(
                    "summary input has a gap after the loaded message cursor"
                )
            target_cursor = cursors[-1][0]
        else:
            target_cursor = loaded_cursor
        key = MemoryStateKey(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            session_id=session_id,
            policy_fingerprint=self.policy_fingerprint,
        )
        state_store = cast(ContextMemoryStateStore, self.state_store)
        current = await state_store.load(key)

        if not cursors:
            if current is None:
                raise ContextMemoryIntegrityError(
                    "summary boundary exists without persisted Context Memory state"
                )
            _require_matching_prefix(
                current,
                expected_cursor=loaded_cursor,
                expected_digest=loaded_digest,
            )
            return _SummaryCandidate(_summary_message(current), Usage(), True)

        view_records = tuple(
            (message, sequence, message_id)
            for message, (sequence, message_id) in zip(
                source_messages,
                cursors,
                strict=True,
            )
        )
        target_digest = _extend_prefix_digest(loaded_digest, view_records)

        if current is None:
            if loaded_cursor != 0 or any(
                _is_summary_boundary(message) for message in flattened
            ):
                raise ContextMemoryIntegrityError(
                    "persisted Context Memory state disappeared after history load"
                )
            expected_sequence = 1
            previous_summary = None
            expected_version = 0
            previous_digest = "context-summary-v2:origin"
            previous_structured = None
            previous_source_ids: tuple[str, ...] = ()
        else:
            if current.covered_through_sequence < loaded_cursor:
                raise ContextMemoryWriteConflictError(
                    "Context Memory state moved behind the loaded history cursor"
                )
            if current.covered_through_sequence > target_cursor:
                raise ContextMemoryWriteConflictError(
                    "Context Memory state advanced beyond this history view"
                )
            current_records = tuple(
                record
                for record in view_records
                if record[1] <= current.covered_through_sequence
            )
            if current.covered_through_sequence == loaded_cursor:
                expected_current_digest = loaded_digest
            else:
                if (
                    not current_records
                    or current_records[-1][1] != current.covered_through_sequence
                ):
                    raise ContextMemoryWriteConflictError(
                        "Context Memory state cursor is outside this history view"
                    )
                expected_current_digest = _extend_prefix_digest(
                    loaded_digest,
                    current_records,
                )
            _require_matching_prefix(
                current,
                expected_cursor=current.covered_through_sequence,
                expected_digest=expected_current_digest,
            )
            if current.covered_through_sequence == target_cursor:
                if current.covered_prefix_digest != target_digest:
                    raise ContextMemoryWriteConflictError(
                        "Context Memory state has a divergent prefix digest"
                    )
                return _SummaryCandidate(
                    _summary_message(current),
                    Usage(),
                    True,
                )
            expected_sequence = current.covered_through_sequence + 1
            previous_summary = current.structured_summary.summary
            expected_version = current.version
            previous_digest = current.covered_prefix_digest
            previous_structured = current.structured_summary
            previous_source_ids = current.source_message_ids

        new_records = tuple(
            record for record in view_records if record[1] >= expected_sequence
        )
        if not new_records or new_records[0][1] != expected_sequence:
            raise ContextMemoryIntegrityError(
                "summary input has a gap after the persisted message cursor"
            )

        new_messages = tuple(record[0] for record in new_records)
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

        structured = _updated_structured_summary(
            result.summary,
            previous=previous_structured,
        )
        source_ids = _merge_source_ids(
            previous_source_ids,
            tuple(record[2] for record in new_records),
        )
        next_snapshot = ConversationSummarySnapshot(
            tenant_id=key.tenant_id,
            agent_id=key.agent_id,
            session_id=key.session_id,
            policy_fingerprint=key.policy_fingerprint,
            covered_through_sequence=target_cursor,
            covered_prefix_digest=_extend_prefix_digest(
                previous_digest,
                new_records,
            ),
            structured_summary=structured,
            source_message_ids=source_ids,
            version=expected_version + 1,
        )
        return _SummaryCandidate(
            message=_summary_message(next_snapshot),
            usage=result.usage,
            cache_hit=False,
            expected_version=expected_version,
            next_snapshot=next_snapshot,
        )

    async def _commit_summary_candidate(
        self,
        candidate: _SummaryCandidate,
    ) -> tuple[Message, bool]:
        next_snapshot = candidate.next_snapshot
        expected_version = candidate.expected_version
        if next_snapshot is None:
            return candidate.message, candidate.cache_hit
        if expected_version is None:
            raise ContextMemoryIntegrityError(
                "summary candidate is missing its expected CAS version"
            )
        state_store = cast(ContextMemoryStateStore, self.state_store)
        if await state_store.save_if_version(expected_version, next_snapshot):
            return candidate.message, False

        # Reuse a concurrent writer only for the exact same canonical prefix.
        # The caller revalidates the winner's actual boundary under its current
        # ModelRequest budget before exposing it.
        winner = await state_store.load(next_snapshot.key)
        if (
            winner is not None
            and winner.covered_through_sequence
            == next_snapshot.covered_through_sequence
            and winner.covered_prefix_digest == next_snapshot.covered_prefix_digest
        ):
            return _summary_message(winner), False
        raise ContextMemoryWriteConflictError(
            "Context Memory CAS lost to a state that does not cover this summary"
        )

    async def _summary(
        self,
        session_id: str,
        turns: tuple[tuple[Message, ...], ...],
        *,
        model_gateway: ModelGateway | None,
    ) -> tuple[Message, Usage, bool]:
        """Compatibility seam for direct extension/tests: build, then commit."""

        candidate = await self._summary_candidate(
            session_id,
            turns,
            model_gateway=model_gateway,
        )
        message, cache_hit = await self._commit_summary_candidate(candidate)
        return message, candidate.usage, cache_hit


def _validated_source_messages(
    messages: tuple[Message, ...],
) -> tuple[tuple[Message, ...], int, str]:
    boundary_positions = tuple(
        index for index, message in enumerate(messages) if _is_summary_boundary(message)
    )
    if boundary_positions not in {(), (0,)}:
        raise ContextMemoryIntegrityError(
            "Context Memory summary boundary must be one complete leading turn"
        )
    if boundary_positions:
        first = messages[0]
        if first.role is not MessageRole.USER:
            raise ContextMemoryIntegrityError(
                "Context Memory summary boundary has invalid message roles"
            )
        first_prefix = _summary_boundary_prefix(first)
        if first_prefix is None:
            raise ContextMemoryIntegrityError(
                "Context Memory summary boundary has invalid prefix metadata"
            )
        loaded_cursor, loaded_digest = first_prefix
    else:
        loaded_cursor = 0
        loaded_digest = "context-summary-v2:origin"
    return messages[len(boundary_positions) :], loaded_cursor, loaded_digest


def _require_matching_prefix(
    snapshot: ConversationSummarySnapshot,
    *,
    expected_cursor: int,
    expected_digest: str,
) -> None:
    if (
        snapshot.covered_through_sequence != expected_cursor
        or snapshot.covered_prefix_digest != expected_digest
    ):
        raise ContextMemoryWriteConflictError(
            "Context Memory state does not match this history prefix"
        )


def _required_cursor(message: Message) -> tuple[int, str]:
    cursor = _message_cursor(message)
    if cursor is None:
        raise ContextMemoryIntegrityError(
            "durable summary input is missing store-assigned cursor metadata"
        )
    return cursor


def _validate_contiguous_cursors(cursors: tuple[tuple[int, str], ...]) -> None:
    expected = cursors[0][0]
    seen_ids: set[str] = set()
    for sequence, message_id in cursors:
        if sequence != expected:
            raise ContextMemoryIntegrityError(
                "durable summary input message cursors are not contiguous"
            )
        if message_id in seen_ids:
            raise ContextMemoryIntegrityError(
                "durable summary input contains duplicate message identifiers"
            )
        seen_ids.add(message_id)
        expected += 1


def _updated_structured_summary(
    summary: str,
    *,
    previous: ConversationSummary | None,
) -> ConversationSummary:
    try:
        if previous is None:
            return ConversationSummary(summary=summary)
        # The existing summarizer emits one verified free-form summary. Preserve
        # already structured fields verbatim rather than guessing classifications
        # from model prose. A future structured summarizer can populate new fields.
        return ConversationSummary(
            summary=summary,
            facts=previous.facts,
            decisions=previous.decisions,
            preferences=previous.preferences,
            open_items=previous.open_items,
            tool_observations=previous.tool_observations,
        )
    except (TypeError, ValueError) as exc:
        # The superclass treats ordinary summarizer failures as an optional
        # recent-turn fallback. A bounded-schema violation is an integrity
        # failure: never persist it or silently continue with detached state.
        raise ContextMemoryIntegrityError(
            "generated conversation summary violates the bounded v2 schema"
        ) from exc


def _merge_source_ids(
    previous: tuple[str, ...],
    additions: tuple[str, ...],
) -> tuple[str, ...]:
    merged = tuple(dict.fromkeys((*previous, *additions)))
    if len(merged) <= MAX_SUMMARY_SOURCE_MESSAGE_IDS:
        return merged
    # The monotonic cursor and chained prefix digest cover the complete prefix.
    # Keep one origin anchor plus the newest exact source IDs so the snapshot
    # remains O(1) in long sessions while retaining useful provenance samples.
    return (
        merged[0],
        *merged[-(MAX_SUMMARY_SOURCE_MESSAGE_IDS - 1) :],
    )


def _summary_message(snapshot: ConversationSummarySnapshot) -> Message:
    return _summary_boundary_messages(snapshot)[0]


__all__ = ["DurableSummarizingConversationMemoryPolicy"]

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from moduagent.memory.base import MemoryRequest, MemoryResult
from moduagent.messages import Message, MessageRole
from moduagent.models import ModelRequest

from .assembler import ContextAssembler, ContextItem
from .errors import ContextMemoryIntegrityError
from .history import _is_summary_boundary


_REQUIRED_PRIORITY = 10_000
_SUMMARY_PRIORITY = 700
_HISTORY_PRIORITY_BASE = 800


@dataclass(frozen=True, slots=True)
class RuntimeContextSelection:
    messages: tuple[Message, ...]
    selected_turns: tuple[tuple[Message, ...], ...]
    summary_selected: bool
    selected_tokens: int


async def select_runtime_context(
    *,
    assembler: ContextAssembler,
    model_request: ModelRequest,
    system: tuple[Message, ...],
    summary: Message | None,
    turns: tuple[tuple[Message, ...], ...],
    protected: tuple[Message, ...],
    token_counter: Any,
    token_budget: int,
) -> RuntimeContextSelection:
    """Select optional summary/recent groups around an exact required core.

    ContextAssembler owns the priority and atomic-group decision. Two exact
    whole-request counts calibrate its additive item costs; a final exact count
    conservatively removes the lowest-priority complete turn when a provider's
    chat template is non-additive. No Tool/protocol block is split.
    """

    required_messages = (*system, *protected)
    required_tokens = await token_counter.count_request(
        replace(model_request, messages=required_messages)
    )
    all_messages = (
        *system,
        *((summary,) if summary is not None else ()),
        *(message for turn in turns for message in turn),
        *protected,
    )
    all_tokens = await token_counter.count_request(
        replace(model_request, messages=all_messages)
    )

    candidates: list[ContextItem] = [
        replace(
            _item(
                item_id="request-contract",
                source="request_contract",
                payload=None,
                priority=_REQUIRED_PRIORITY,
                required=True,
                atomic_group="request-contract",
                authority="runtime",
            ),
            min_tokens=required_tokens,
            max_tokens=required_tokens,
        )
    ]
    candidates.extend(
        replace(
            _item(
                item_id=f"system-{index}",
                source="system_policy",
                payload=("system", index),
                priority=_REQUIRED_PRIORITY,
                required=True,
                atomic_group="system-policy",
                authority="system",
            ),
            min_tokens=0,
            max_tokens=0,
        )
        for index, _ in enumerate(system)
    )
    protected_rows = _protected_candidates(protected, start_index=0)
    candidates.extend(
        replace(item, payload=("protected", index), min_tokens=0, max_tokens=0)
        for index, (item, _) in enumerate(protected_rows)
    )

    group_weights: list[int] = []
    group_descriptors: list[tuple[str, tuple[Message, ...], int]] = []
    if summary is not None:
        group_weights.append(_message_weight(summary))
        group_descriptors.append(
            ("conversation-summary", (summary,), _SUMMARY_PRIORITY + 9_000)
        )
    for index, turn in enumerate(turns):
        group_weights.append(sum(_message_weight(message) for message in turn))
        group_descriptors.append(
            (f"history-turn-{index}", turn, _HISTORY_PRIORITY_BASE + index)
        )
    optional_costs = _allocate_integer_costs(
        tuple(group_weights),
        max(0, all_tokens - required_tokens),
    )
    for (group, group_messages, priority), group_cost in zip(
        group_descriptors,
        optional_costs,
        strict=True,
    ):
        source = (
            "conversation_summary" if group == "conversation-summary" else "recent_turn"
        )
        for member_index, _ in enumerate(group_messages):
            cost = group_cost if member_index == 0 else 0
            candidates.append(
                replace(
                    _item(
                        item_id=f"{group}-{member_index}",
                        source=source,
                        payload=(group, member_index),
                        priority=priority,
                        required=False,
                        atomic_group=group,
                        authority="untrusted_history",
                    ),
                    min_tokens=cost,
                    max_tokens=cost,
                )
            )

    assembly = assembler.assemble(candidates, token_budget=token_budget)
    selected_groups = {entry.item.atomic_group for entry in assembly.items}
    summary_selected = summary is not None and "conversation-summary" in selected_groups

    selected_indices = {
        index
        for index in range(len(turns))
        if f"history-turn-{index}" in selected_groups
    }
    # Conversation turns are semantically ordered. ContextAssembler may skip a
    # large high-priority group then fit a smaller old one; retain only the
    # newest contiguous suffix so the model never sees a reordered history.
    suffix_start = len(turns)
    for index in range(len(turns) - 1, -1, -1):
        if index not in selected_indices:
            break
        suffix_start = index
    selected_turns = turns[suffix_start:]

    async def count(
        use_summary: bool,
        selected: tuple[tuple[Message, ...], ...],
    ) -> tuple[tuple[Message, ...], int]:
        messages = (
            *system,
            *((summary,) if use_summary and summary is not None else ()),
            *(message for turn in selected for message in turn),
            *protected,
        )
        tokens = await token_counter.count_request(
            replace(model_request, messages=messages)
        )
        return messages, tokens

    messages, exact_tokens = await count(summary_selected, selected_turns)
    while exact_tokens > token_budget and selected_turns:
        selected_turns = selected_turns[1:]
        suffix_start = len(turns) - len(selected_turns)
        messages, exact_tokens = await count(summary_selected, selected_turns)
    if exact_tokens > token_budget and summary_selected:
        summary_selected = False
        messages, exact_tokens = await count(False, selected_turns)
    if exact_tokens > token_budget:
        raise ContextMemoryIntegrityError(
            "required Context items exceed the exact model request budget"
        )

    # Correct conservative cost estimates without violating priority: summary
    # outranks recent history, then older turns are admitted only as a suffix.
    if summary is not None and not summary_selected:
        with_summary, with_summary_tokens = await count(True, selected_turns)
        while with_summary_tokens > token_budget and selected_turns:
            selected_turns = selected_turns[1:]
            suffix_start = len(turns) - len(selected_turns)
            with_summary, with_summary_tokens = await count(True, selected_turns)
        if with_summary_tokens <= token_budget:
            summary_selected = True
            messages = with_summary
            exact_tokens = with_summary_tokens

    while suffix_start > 0:
        candidate_turns = (turns[suffix_start - 1], *selected_turns)
        candidate_messages, candidate_tokens = await count(
            summary_selected,
            candidate_turns,
        )
        if candidate_tokens > token_budget:
            break
        suffix_start -= 1
        selected_turns = candidate_turns
        messages = candidate_messages
        exact_tokens = candidate_tokens

    return RuntimeContextSelection(
        messages=messages,
        selected_turns=selected_turns,
        summary_selected=summary_selected,
        selected_tokens=exact_tokens,
    )


def assemble_runtime_memory_result(
    *,
    assembler: ContextAssembler,
    request: MemoryRequest,
    result: MemoryResult,
    token_budget: int,
) -> MemoryResult:
    """Project a prepared durable-memory view through ContextAssembler v1.

    The established token-budget policy remains responsible for tokenizer-aware
    turn selection and summarization. This final pass gives the *actual* model
    request a single typed allocation: system/policy input, the task and active
    run, Tool/protocol blocks, and the request schemas are required; summary and
    complete historical turns are optional atomic groups.

    Item costs are a deterministic allocation of the already measured complete
    request token count. No additional remote tokenizer calls are introduced.
    Because the established policy has already fitted the request, every item
    must survive this pass. A result outside that invariant fails closed rather
    than silently trimming a required current-run or protocol block.
    """

    if not isinstance(assembler, ContextAssembler):
        raise TypeError("assembler must be a ContextAssembler")
    if not isinstance(request, MemoryRequest):
        raise TypeError("request must be a MemoryRequest")
    if not isinstance(result, MemoryResult):
        raise TypeError("result must be a MemoryResult")
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError("token_budget must be a positive integer")
    if result.selected_tokens > token_budget:
        raise ContextMemoryIntegrityError(
            "prepared durable Context Memory exceeds its configured token budget"
        )

    candidates = _context_candidates(request, result)
    allocations = _allocate_integer_costs(
        tuple(weight for _, weight in candidates),
        result.selected_tokens,
    )
    items = tuple(
        replace(item, min_tokens=cost, max_tokens=cost)
        for (item, _), cost in zip(candidates, allocations, strict=True)
    )
    assembly = assembler.assemble(items, token_budget=token_budget)
    if assembly.dropped_item_ids:
        # With costs normalized to the exact, already-fitted request count this
        # indicates an assembler or policy contract violation, not a legitimate
        # opportunity to discard model input.
        raise ContextMemoryIntegrityError(
            "ContextAssembler dropped an item from an already-fitted request"
        )

    sources = Counter(entry.item.source for entry in assembly.items)
    atomic_groups = {
        entry.item.atomic_group
        for entry in assembly.items
        if entry.item.atomic_group is not None
    }
    metadata = dict(result.metadata)
    metadata.update(
        {
            "context_assembly_algorithm": (
                f"context-assembler-v{assembler.algorithm_version}"
            ),
            "context_assembly_budget_tokens": assembly.token_budget,
            "context_assembly_used_tokens": assembly.used_tokens,
            "context_assembly_candidate_items": len(items),
            "context_assembly_selected_items": len(assembly.items),
            "context_assembly_dropped_items": len(assembly.dropped_item_ids),
            "context_assembly_required_items": sum(
                1 for item in items if item.required
            ),
            "context_assembly_optional_items": sum(
                1 for item in items if not item.required
            ),
            "context_assembly_atomic_groups": len(atomic_groups),
            # Only fixed source labels and aggregate counts are observable. Raw
            # message content, IDs, Tool arguments and provenance never enter
            # Memory events/result metadata.
            "context_assembly_source_counts": dict(sorted(sources.items())),
        }
    )
    return replace(result, metadata=metadata)


def _context_candidates(
    request: MemoryRequest,
    result: MemoryResult,
) -> tuple[tuple[ContextItem, int], ...]:
    original = request.model_request.messages
    if request.protected_from > len(original):
        raise ContextMemoryIntegrityError(
            "protected Context boundary exceeds the original model request"
        )
    protected = original[request.protected_from :]
    if protected:
        if len(result.messages) < len(protected) or (
            result.messages[-len(protected) :] != protected
        ):
            raise ContextMemoryIntegrityError(
                "prepared durable Context Memory changed a protected current-run block"
            )
        protected_start = len(result.messages) - len(protected)
    else:
        protected_start = len(result.messages)

    system_end = 0
    while (
        system_end < protected_start
        and result.messages[system_end].role is MessageRole.SYSTEM
    ):
        system_end += 1

    rows: list[tuple[ContextItem, int]] = []
    rows.append(
        (
            _item(
                item_id="request-contract",
                source="request_contract",
                payload=None,
                priority=_REQUIRED_PRIORITY,
                required=True,
                atomic_group="request-contract",
                authority="runtime",
            ),
            _request_contract_weight(request),
        )
    )

    for index, message in enumerate(result.messages[:system_end]):
        rows.append(
            (
                _item(
                    item_id=f"system-{index}",
                    source="system_policy",
                    payload=index,
                    priority=_REQUIRED_PRIORITY,
                    required=True,
                    atomic_group="system-policy",
                    authority="system",
                ),
                _message_weight(message),
            )
        )

    history = result.messages[system_end:protected_start]
    history_groups = _historical_groups(history)
    for group_index, (source, messages) in enumerate(history_groups):
        group = (
            "conversation-summary"
            if source == "conversation_summary"
            else f"history-turn-{group_index}"
        )
        priority = (
            _SUMMARY_PRIORITY
            if source == "conversation_summary"
            else _HISTORY_PRIORITY_BASE + group_index
        )
        offset = system_end + sum(
            len(group_messages) for _, group_messages in history_groups[:group_index]
        )
        for member_index, message in enumerate(messages):
            absolute_index = offset + member_index
            rows.append(
                (
                    _item(
                        item_id=f"history-{absolute_index}",
                        source=source,
                        payload=absolute_index,
                        priority=priority,
                        required=False,
                        atomic_group=group,
                        authority="untrusted_history",
                    ),
                    _message_weight(message),
                )
            )

    rows.extend(
        _protected_candidates(
            protected,
            start_index=protected_start,
        )
    )
    return tuple(rows)


def _historical_groups(
    messages: tuple[Message, ...],
) -> tuple[tuple[str, tuple[Message, ...]], ...]:
    groups: list[tuple[str, tuple[Message, ...]]] = []
    position = 0
    if messages and _is_summary_boundary(messages[0]):
        groups.append(("conversation_summary", (messages[0],)))
        position = 1

    current: list[Message] = []
    for message in messages[position:]:
        if message.role is MessageRole.USER and current:
            groups.append(("recent_turn", tuple(current)))
            current = []
        current.append(message)
    if current:
        groups.append(("recent_turn", tuple(current)))
    return tuple(groups)


def _protected_candidates(
    messages: tuple[Message, ...],
    *,
    start_index: int,
) -> list[tuple[ContextItem, int]]:
    rows: list[tuple[ContextItem, int]] = []
    position = 0
    task_assigned = False
    while position < len(messages):
        message = messages[position]
        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            call_ids = {call.id for call in message.tool_calls}
            end = position + 1
            while (
                end < len(messages)
                and messages[end].role is MessageRole.TOOL
                and messages[end].tool_call_id in call_ids
            ):
                end += 1
            group_end = end
            source = "tool_protocol"
            group = f"active-tool-block-{position}"
        else:
            group_end = position + 1
            if message.role is MessageRole.TOOL:
                # The established policy rejects an orphaned Tool result before
                # this hook. Keep the classification defensive and required.
                source = "tool_protocol"
                group = f"active-tool-block-{position}"
            elif message.role is MessageRole.USER and not task_assigned:
                source = "task_projection"
                group = "active-task"
                task_assigned = True
            else:
                source = "current_run"
                group = f"active-run-{position}"

        for member in range(position, group_end):
            absolute_index = start_index + member
            item_message = messages[member]
            rows.append(
                (
                    _item(
                        item_id=f"active-{absolute_index}",
                        source=source,
                        payload=absolute_index,
                        priority=_REQUIRED_PRIORITY,
                        required=True,
                        atomic_group=group,
                        authority="runtime",
                    ),
                    _message_weight(item_message),
                )
            )
        position = group_end
    return rows


def _item(
    *,
    item_id: str,
    source: str,
    payload: int | None,
    priority: int,
    required: bool,
    atomic_group: str,
    authority: str,
) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        source=source,
        payload=payload,
        priority=priority,
        required=required,
        atomic_group=atomic_group,
        compressible=False,
        min_tokens=0,
        max_tokens=0,
        authority=authority,
    )


def _allocate_integer_costs(weights: tuple[int, ...], total: int) -> tuple[int, ...]:
    if type(total) is not int or total < 0:
        raise ContextMemoryIntegrityError(
            "prepared durable Context Memory returned an invalid token count"
        )
    if not weights:
        if total:
            raise ContextMemoryIntegrityError(
                "prepared durable Context Memory has tokens without Context items"
            )
        return ()
    weight_total = sum(weights)
    if weight_total < 1:
        weights = tuple(1 for _ in weights)
        weight_total = len(weights)
    allocated: list[int] = []
    cumulative = 0
    previous = 0
    for weight in weights:
        cumulative += weight
        current = cumulative * total // weight_total
        allocated.append(current - previous)
        previous = current
    return tuple(allocated)


def _message_weight(message: Message) -> int:
    return _serialized_weight(message.to_dict()) + 16


def _request_contract_weight(request: MemoryRequest) -> int:
    model_request = request.model_request
    return (
        _serialized_weight(
            {
                "tools": model_request.tools,
                "output_schema": model_request.output_schema,
                "options": model_request.options,
                "provider_options": model_request.provider_options,
            }
        )
        + 16
    )


def _serialized_weight(value: Any) -> int:
    return max(
        1,
        len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=repr,
            ).encode("utf-8")
        ),
    )


__all__ = ["assemble_runtime_memory_result"]

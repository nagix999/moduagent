from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import ContextBudgetExceededError


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One pre-counted candidate for a bounded model request context."""

    item_id: str
    source: str
    payload: Any
    priority: int
    required: bool
    atomic_group: str | None
    compressible: bool
    min_tokens: int
    max_tokens: int
    authority: str
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("item_id", "source", "authority"):
            _validate_name(getattr(self, name), name)
        if type(self.priority) is not int:
            raise TypeError("priority must be an integer")
        if type(self.required) is not bool:
            raise TypeError("required must be a bool")
        if type(self.compressible) is not bool:
            raise TypeError("compressible must be a bool")
        if self.atomic_group is not None:
            _validate_name(self.atomic_group, "atomic_group")
        if type(self.min_tokens) is not int or self.min_tokens < 0:
            raise ValueError("min_tokens must be a non-negative integer")
        if type(self.max_tokens) is not int or self.max_tokens < self.min_tokens:
            raise ValueError("max_tokens must be an integer at least min_tokens")
        if not self.compressible and self.min_tokens != self.max_tokens:
            raise ValueError(
                "non-compressible Context items require min_tokens == max_tokens"
            )
        refs = _validated_names(self.provenance_refs, "provenance_refs")
        object.__setattr__(self, "provenance_refs", refs)


@dataclass(frozen=True, slots=True)
class AssembledContextItem:
    item: ContextItem
    allocated_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.item, ContextItem):
            raise TypeError("item must be a ContextItem")
        if not self.item.min_tokens <= self.allocated_tokens <= self.item.max_tokens:
            raise ValueError("allocated_tokens is outside the item's token bounds")

    @property
    def compressed(self) -> bool:
        return self.allocated_tokens < self.item.max_tokens


@dataclass(frozen=True, slots=True)
class ContextAssemblyResult:
    """Deterministic selection and allocation produced by ContextAssembler v1."""

    items: tuple[AssembledContextItem, ...]
    dropped_item_ids: tuple[str, ...]
    token_budget: int
    used_tokens: int

    def __post_init__(self) -> None:
        if type(self.token_budget) is not int or self.token_budget < 0:
            raise ValueError("token_budget cannot be negative")
        if (
            type(self.used_tokens) is not int
            or not 0 <= self.used_tokens <= self.token_budget
        ):
            raise ValueError("used_tokens must fit inside token_budget")

    @property
    def remaining_tokens(self) -> int:
        return self.token_budget - self.used_tokens


@dataclass(slots=True)
class _Group:
    key: str
    member_indices: list[int]
    required: bool
    priority: int
    first_index: int


class ContextAssembler:
    """Allocate one token budget while preserving required and atomic items."""

    algorithm_version = 1

    def assemble(
        self,
        items: Iterable[ContextItem],
        *,
        token_budget: int,
    ) -> ContextAssemblyResult:
        if type(token_budget) is not int or token_budget < 0:
            raise ValueError("token_budget must be a non-negative integer")
        candidates = tuple(items)
        if not all(isinstance(item, ContextItem) for item in candidates):
            raise TypeError("items must contain ContextItem instances")
        identifiers = [item.item_id for item in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Context item_id values must be unique")

        groups = _groups(candidates)
        required_minimum = sum(
            _group_minimum(group, candidates) for group in groups if group.required
        )
        if required_minimum > token_budget:
            raise ContextBudgetExceededError(
                required_tokens=required_minimum,
                available_tokens=token_budget,
            )

        selected_groups = {group.key for group in groups if group.required}
        allocations = {
            index: candidates[index].min_tokens
            for group in groups
            if group.required
            for index in group.member_indices
        }
        remaining = token_budget - required_minimum

        for group in sorted(
            groups,
            key=lambda value: (-value.priority, value.first_index, value.key),
        ):
            if not group.required:
                minimum = _group_minimum(group, candidates)
                if minimum > remaining:
                    continue
                selected_groups.add(group.key)
                for index in group.member_indices:
                    allocations[index] = candidates[index].min_tokens
                remaining -= minimum

            if group.key not in selected_groups:
                continue
            for index in sorted(
                group.member_indices,
                key=lambda value: (-candidates[value].priority, value),
            ):
                item = candidates[index]
                extra = min(remaining, item.max_tokens - allocations[index])
                allocations[index] += extra
                remaining -= extra
                if remaining == 0:
                    break

        selected = tuple(
            AssembledContextItem(item=item, allocated_tokens=allocations[index])
            for index, item in enumerate(candidates)
            if index in allocations
        )
        dropped = tuple(
            item.item_id
            for index, item in enumerate(candidates)
            if index not in allocations
        )
        used = sum(item.allocated_tokens for item in selected)
        return ContextAssemblyResult(
            items=selected,
            dropped_item_ids=dropped,
            token_budget=token_budget,
            used_tokens=used,
        )


def _groups(items: Sequence[ContextItem]) -> tuple[_Group, ...]:
    groups: dict[str, _Group] = {}
    order: list[str] = []
    for index, item in enumerate(items):
        key = (
            f"atomic:{item.atomic_group}"
            if item.atomic_group is not None
            else f"item:{item.item_id}"
        )
        group = groups.get(key)
        if group is None:
            group = _Group(
                key=key,
                member_indices=[],
                required=False,
                priority=item.priority,
                first_index=index,
            )
            groups[key] = group
            order.append(key)
        group.member_indices.append(index)
        group.required = group.required or item.required
        group.priority = max(group.priority, item.priority)
    return tuple(groups[key] for key in order)


def _group_minimum(group: _Group, items: Sequence[ContextItem]) -> int:
    return sum(items[index].min_tokens for index in group.member_indices)


def _validate_name(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} cannot be empty or padded")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} cannot contain control characters")


def _validated_names(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(value)
    for item in result:
        _validate_name(item, f"{name} item")
    return result


__all__ = [
    "AssembledContextItem",
    "ContextAssembler",
    "ContextAssemblyResult",
    "ContextItem",
]

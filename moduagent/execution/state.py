from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable


StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """Opaque, versioned state owned by one ExecutionEngine."""

    engine_id: str
    state_version: int
    state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.engine_id, str) or not self.engine_id.strip():
            raise ValueError("engine_id cannot be empty")
        if type(self.state_version) is not int:
            raise TypeError("state_version must be an integer")
        if self.state_version < 1:
            raise ValueError("state_version must be at least 1")
        if not isinstance(self.state, Mapping):
            raise TypeError("state must be a mapping")
        object.__setattr__(self, "state", dict(self.state))


@runtime_checkable
class EngineStateCodec(Protocol[StateT]):
    """Versioned data contract for one Engine's durable state."""

    engine_id: str
    state_version: int

    def encode(self, state: StateT) -> Mapping[str, Any]: ...

    def decode(self, payload: Mapping[str, Any]) -> StateT: ...

    def migrate(
        self,
        from_version: int,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


__all__ = ["EngineSnapshot", "EngineStateCodec"]

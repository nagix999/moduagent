from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

from moduagent.execution.state import EngineSnapshot, EngineStateCodec

SNAPSHOT_SCHEMA_VERSION = 4
DEFAULT_ENGINE_STATE_VERSION = 1
SNAPSHOT_RUNTIME_VERSION = "0.5.2"
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class CommonRunState:
    """Engine-neutral state required to reconstruct a run context."""

    request: Mapping[str, Any]
    messages: tuple[Mapping[str, Any], ...] = ()
    new_messages: tuple[Mapping[str, Any], ...] = ()
    internal_messages: tuple[Mapping[str, Any], ...] = ()
    status: str = "created"
    step: int = 0
    tool_call_count: int = 0
    usage: Mapping[str, Any] = field(default_factory=dict)
    current_run_start: int = 0
    compatibility_policy_state: Mapping[str, Any] = field(default_factory=dict)
    terminal_reason: str | None = None
    resume_safety: str = "resumable"
    event_sequence: int = 0

    def __post_init__(self) -> None:
        request = _json_mapping_copy(self.request, "common request")
        messages = _message_sequence(self.messages, "messages")
        new_messages = _message_sequence(self.new_messages, "new_messages")
        internal_messages = _message_sequence(
            self.internal_messages,
            "internal_messages",
        )
        if self.status not in {
            "created",
            "running",
            "waiting_for_model",
            "waiting_for_tools",
            "completed",
            "failed",
            "cancelled",
        }:
            raise ValueError(f"unsupported run status: {self.status}")
        for value, field_name in (
            (self.step, "step"),
            (self.tool_call_count, "tool_call_count"),
            (self.event_sequence, "event_sequence"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not 0 <= self.current_run_start <= len(messages):
            raise ValueError("current_run_start must reference the message sequence")
        if not isinstance(self.resume_safety, str) or not self.resume_safety.strip():
            raise ValueError("resume_safety cannot be empty")
        if self.terminal_reason is not None and not isinstance(
            self.terminal_reason,
            str,
        ):
            raise TypeError("terminal_reason must be a string or None")

        object.__setattr__(self, "request", request)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "new_messages", new_messages)
        object.__setattr__(self, "internal_messages", internal_messages)
        object.__setattr__(
            self,
            "usage",
            _json_mapping_copy(self.usage, "common usage"),
        )
        object.__setattr__(
            self,
            "compatibility_policy_state",
            _json_mapping_copy(
                self.compatibility_policy_state,
                "compatibility policy state",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "request": _json_mapping_copy(self.request, "common request"),
            "messages": [dict(message) for message in self.messages],
            "new_messages": [dict(message) for message in self.new_messages],
            "internal_messages": [dict(message) for message in self.internal_messages],
            "status": self.status,
            "step": self.step,
            "tool_call_count": self.tool_call_count,
            "usage": _json_mapping_copy(self.usage, "common usage"),
            "current_run_start": self.current_run_start,
            "compatibility_policy_state": _json_mapping_copy(
                self.compatibility_policy_state,
                "compatibility policy state",
            ),
            "resume_safety": self.resume_safety,
            "event_sequence": self.event_sequence,
        }
        if self.terminal_reason is not None:
            value["terminal_reason"] = self.terminal_reason
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CommonRunState":
        if not isinstance(value, Mapping):
            raise ValueError("snapshot common_state must be an object")
        return cls(
            request=_mapping(value.get("request", {}), "common request"),
            messages=_mapping_sequence(value.get("messages", ()), "messages"),
            new_messages=_mapping_sequence(
                value.get("new_messages", ()),
                "new_messages",
            ),
            internal_messages=_mapping_sequence(
                value.get("internal_messages", ()),
                "internal_messages",
            ),
            status=str(value.get("status", "created")),
            step=_integer(value.get("step", 0), "step"),
            tool_call_count=_integer(
                value.get("tool_call_count", 0),
                "tool_call_count",
            ),
            usage=_mapping(value.get("usage", {}), "common usage"),
            current_run_start=_integer(
                value.get("current_run_start", 0),
                "current_run_start",
            ),
            compatibility_policy_state=_mapping(
                value.get("compatibility_policy_state", {}),
                "compatibility policy state",
            ),
            terminal_reason=(
                None
                if value.get("terminal_reason") is None
                else str(value["terminal_reason"])
            ),
            resume_safety=str(value.get("resume_safety", "resumable")),
            event_sequence=_integer(
                value.get("event_sequence", 0),
                "event_sequence",
            ),
        )


@dataclass(frozen=True, slots=True)
class FinalizationMarkers:
    """Durable, engine-neutral at-most-once finalization markers."""

    started: bool = False
    response_generated: bool = False
    response: Any = None
    persisted: bool = False
    emitted: bool = False

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.started, "started"),
            (self.response_generated, "response_generated"),
            (self.persisted, "persisted"),
            (self.emitted, "emitted"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"finalization {field_name} must be a bool")
        if self.response_generated and self.response is None:
            raise ValueError("response_generated requires a response")
        if self.response is not None and not self.response_generated:
            raise ValueError("a finalization response requires response_generated")
        if (self.persisted or self.emitted) and not self.response_generated:
            raise ValueError("finalization persistence/emission requires a response")
        if self.response_generated and not self.started:
            raise ValueError("response_generated requires finalization to be started")
        if self.response is not None:
            object.__setattr__(
                self,
                "response",
                _json_value_copy(self.response, "finalization response"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "response_generated": self.response_generated,
            "response": (
                None
                if self.response is None
                else _json_value_copy(self.response, "finalization response")
            ),
            "persisted": self.persisted,
            "emitted": self.emitted,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FinalizationMarkers":
        if not isinstance(value, Mapping):
            raise ValueError("snapshot finalization_markers must be an object")
        return cls(
            started=_boolean(value.get("started", False), "finalization started"),
            response_generated=_boolean(
                value.get("response_generated", False),
                "finalization response_generated",
            ),
            response=value.get("response"),
            persisted=_boolean(
                value.get("persisted", False),
                "finalization persisted",
            ),
            emitted=_boolean(value.get("emitted", False), "finalization emitted"),
        )


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Version 4 checkpoint envelope.

    ``schema_version`` is the canonical discriminator. ``version`` is emitted
    as a matching compatibility guard so 0.3 readers reject v4 instead of
    silently interpreting it as their default v1 shape.
    """

    runtime_version: str
    run_id: str
    session_id: str
    agent_fingerprint: str
    engine: EngineSnapshot
    common_state: CommonRunState
    finalization_markers: FinalizationMarkers = field(
        default_factory=FinalizationMarkers
    )
    skill_state: Mapping[str, Any] = field(default_factory=dict)
    sanitized_runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported snapshot schema version: {self.schema_version}"
            )
        for value, field_name in (
            (self.runtime_version, "runtime_version"),
            (self.run_id, "run_id"),
            (self.session_id, "session_id"),
            (self.agent_fingerprint, "agent_fingerprint"),
        ):
            _validate_identifier(value, field_name)
        if not isinstance(self.engine, EngineSnapshot):
            raise TypeError("engine must be an EngineSnapshot")
        object.__setattr__(
            self,
            "engine",
            EngineSnapshot(
                engine_id=self.engine.engine_id,
                state_version=self.engine.state_version,
                state=_json_mapping_copy(self.engine.state, "engine state"),
            ),
        )
        if not isinstance(self.common_state, CommonRunState):
            raise TypeError("common_state must be a CommonRunState")
        if not isinstance(self.finalization_markers, FinalizationMarkers):
            raise TypeError("finalization_markers must be FinalizationMarkers")
        object.__setattr__(
            self,
            "skill_state",
            _json_mapping_copy(self.skill_state, "skill_state"),
        )
        object.__setattr__(
            self,
            "sanitized_runtime_metadata",
            _json_mapping_copy(
                self.sanitized_runtime_metadata,
                "sanitized runtime metadata",
            ),
        )
        object.__setattr__(self, "created_at", _as_utc(self.created_at))
        object.__setattr__(self, "updated_at", _as_utc(self.updated_at))
        if self.updated_at < self.created_at:
            raise ValueError("snapshot updated_at cannot precede created_at")

    @property
    def version(self) -> int:
        """Deprecated alias retained for storage diagnostics."""

        return self.schema_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.schema_version,
            "runtime_version": self.runtime_version,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "agent_fingerprint": self.agent_fingerprint,
            "engine": _engine_to_dict(self.engine),
            "common_state": self.common_state.to_dict(),
            "finalization_markers": self.finalization_markers.to_dict(),
            "skill_state": _json_mapping_copy(self.skill_state, "skill_state"),
            "sanitized_runtime_metadata": _json_mapping_copy(
                self.sanitized_runtime_metadata,
                "sanitized runtime metadata",
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunSnapshot":
        if not isinstance(value, Mapping):
            raise ValueError("snapshot payload must be a JSON object")
        if "schema_version" not in value or "version" not in value:
            raise ValueError("v4 snapshot requires schema_version and version")
        schema_version = _integer(value["schema_version"], "schema_version")
        compatibility_version = _integer(value["version"], "version")
        if schema_version != compatibility_version:
            raise ValueError("snapshot schema_version and version must match")
        if schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported snapshot schema version: {schema_version}")
        return cls(
            runtime_version=str(value.get("runtime_version", "")),
            run_id=str(value.get("run_id", "")),
            session_id=str(value.get("session_id", "")),
            agent_fingerprint=str(value.get("agent_fingerprint", "")),
            engine=_engine_from_dict(
                _mapping(value.get("engine", {}), "snapshot engine")
            ),
            common_state=CommonRunState.from_dict(
                _mapping(value.get("common_state", {}), "snapshot common_state")
            ),
            finalization_markers=FinalizationMarkers.from_dict(
                _mapping(
                    value.get("finalization_markers", {}),
                    "snapshot finalization_markers",
                )
            ),
            skill_state=_mapping(value.get("skill_state", {}), "skill_state"),
            sanitized_runtime_metadata=_mapping(
                value.get("sanitized_runtime_metadata", {}),
                "sanitized runtime metadata",
            ),
            created_at=_parse_datetime(value.get("created_at")),
            updated_at=_parse_datetime(value.get("updated_at")),
            schema_version=schema_version,
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "RunSnapshot":
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise ValueError("snapshot payload must be a JSON object")
        return cls.from_dict(value)


def current_runtime_version() -> str:
    # Keep checkpoints tied to the source runtime. Importlib distribution
    # metadata can be stale while developing from an editable checkout.
    return SNAPSHOT_RUNTIME_VERSION


def encode_engine_snapshot(
    codec: EngineStateCodec[StateT],
    state: StateT,
) -> EngineSnapshot:
    """Encode and immediately decode-check one Engine state boundary."""

    _validate_codec(codec)
    payload = _json_mapping_copy(codec.encode(state), "encoded engine state")
    codec.decode(payload)
    return EngineSnapshot(
        engine_id=codec.engine_id,
        state_version=codec.state_version,
        state=payload,
    )


def decode_engine_snapshot(
    codec: EngineStateCodec[StateT],
    snapshot: EngineSnapshot,
) -> StateT:
    """Decode current state or migrate an older state without downgrading."""

    _validate_codec(codec)
    if not isinstance(snapshot, EngineSnapshot):
        raise TypeError("snapshot must be an EngineSnapshot")
    if snapshot.engine_id != codec.engine_id:
        raise ValueError(
            f"snapshot engine {snapshot.engine_id!r} does not match {codec.engine_id!r}"
        )
    if snapshot.state_version > codec.state_version:
        raise ValueError(
            f"unsupported future engine state version: {snapshot.state_version}"
        )
    payload = _json_mapping_copy(snapshot.state, "engine state")
    if snapshot.state_version < codec.state_version:
        payload = _json_mapping_copy(
            codec.migrate(snapshot.state_version, payload),
            "migrated engine state",
        )
    decoded = codec.decode(payload)
    # A successful re-encode proves the decoded object remains serializable
    # under the current codec without retaining the returned duplicate.
    _json_mapping_copy(codec.encode(decoded), "re-encoded engine state")
    return decoded


def _validate_codec(codec: EngineStateCodec[Any]) -> None:
    if not isinstance(codec, EngineStateCodec):
        raise TypeError("codec must implement EngineStateCodec")
    _validate_identifier(codec.engine_id, "codec engine_id")
    if type(codec.state_version) is not int or codec.state_version < 1:
        raise ValueError("codec state_version must be a positive integer")


def _engine_to_dict(engine: EngineSnapshot) -> dict[str, Any]:
    return {
        "engine_id": engine.engine_id,
        "state_version": engine.state_version,
        "state": _json_mapping_copy(engine.state, "engine state"),
    }


def _engine_from_dict(value: Mapping[str, Any]) -> EngineSnapshot:
    return EngineSnapshot(
        engine_id=str(value.get("engine_id", "")),
        state_version=_integer(value.get("state_version"), "engine state_version"),
        state=_mapping(value.get("state", {}), "engine state"),
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _mapping_sequence(
    value: Any,
    name: str,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        (list, tuple),
    ):
        raise ValueError(f"{name} must be an array")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{name} must contain objects")
    return tuple(value)


def _message_sequence(
    value: Any,
    name: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _json_mapping_copy(item, f"{name} item")
        for item in _mapping_sequence(value, name)
    )


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _json_mapping_copy(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied = _json_value_copy(dict(value), name)
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must be a JSON object")
    return copied


def _json_value_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain only JSON-safe values") from exc


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("snapshot timestamps must be datetime values")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("snapshot timestamp must be ISO-8601") from exc
    return _as_utc(parsed)


__all__ = [
    "CommonRunState",
    "DEFAULT_ENGINE_STATE_VERSION",
    "EngineSnapshot",
    "EngineStateCodec",
    "FinalizationMarkers",
    "RunSnapshot",
    "SNAPSHOT_SCHEMA_VERSION",
    "SNAPSHOT_RUNTIME_VERSION",
    "current_runtime_version",
    "decode_engine_snapshot",
    "encode_engine_snapshot",
]

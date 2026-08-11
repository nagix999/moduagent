from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

from moduagent.execution.state import EngineSnapshot, EngineStateCodec

SNAPSHOT_SCHEMA_VERSION = 5
PREVIOUS_SNAPSHOT_SCHEMA_VERSION = 4
DEFAULT_ENGINE_STATE_VERSION = 1
SNAPSHOT_RUNTIME_VERSION = "0.6.0"
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
    """Version 5 checkpoint envelope with delegation lineage.

    ``schema_version`` is the canonical discriminator. ``version`` is emitted
    as a matching compatibility guard so older readers reject v5 instead of
    silently interpreting it as their default shape. Version 4 payloads are
    upgraded on read and are always interpreted as root runs.
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
    # v5 fields are appended after the complete v4 positional surface.
    run_lineage: Mapping[str, Any] = field(default_factory=dict)
    execution_group_id: str | None = None
    agent_ref: Mapping[str, Any] = field(default_factory=dict)
    agent_definition_fingerprint: str | None = None
    delegation_id: str | None = None
    parent_tool_call_id: str | None = None
    budget_lease_id: str | None = None
    migrated_from_schema_version: int | None = None
    tenant_scope_digest: str | None = None
    principal_scope_digest: str | None = None

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
        lineage = _json_mapping_copy(self.run_lineage, "run_lineage")
        if not lineage:
            lineage = {
                "root_run_id": self.run_id,
                "parent_run_id": None,
                "depth": 0,
                "agent_path": [],
            }
        allowed_lineage_keys = {
            "root_run_id",
            "parent_run_id",
            "delegation_id",
            "parent_tool_call_id",
            "caller_agent_id",
            "agent_id",
            "agent_version",
            "agent_path",
            "depth",
        }
        unknown_lineage_keys = set(lineage).difference(allowed_lineage_keys)
        if unknown_lineage_keys:
            raise ValueError("run_lineage contains unsupported fields")
        root_run_id = lineage.get("root_run_id")
        if not isinstance(root_run_id, str) or not root_run_id.strip():
            raise ValueError("run_lineage root_run_id cannot be empty")
        parent_run_id = lineage.get("parent_run_id")
        if parent_run_id is not None and (
            not isinstance(parent_run_id, str) or not parent_run_id.strip()
        ):
            raise ValueError("run_lineage parent_run_id cannot be empty")
        depth = lineage.get("depth", 0)
        if type(depth) is not int or depth < 0:
            raise ValueError("run_lineage depth must be a non-negative integer")
        agent_path = lineage.get("agent_path", [])
        if isinstance(agent_path, (str, bytes)) or not isinstance(
            agent_path, (list, tuple)
        ):
            raise ValueError("run_lineage agent_path must be an array")
        if not all(isinstance(item, str) and item.strip() for item in agent_path):
            raise ValueError("run_lineage agent_path must contain non-empty strings")
        lineage["agent_path"] = list(agent_path)
        execution_group_id = (
            root_run_id if self.execution_group_id is None else self.execution_group_id
        )
        _validate_identifier(execution_group_id, "execution_group_id")
        object.__setattr__(self, "execution_group_id", execution_group_id)
        agent_ref = _json_mapping_copy(self.agent_ref, "agent_ref")
        if set(agent_ref).difference({"agent_id", "version"}):
            raise ValueError("agent_ref contains unsupported fields")
        if agent_ref and set(agent_ref) != {"agent_id", "version"}:
            raise ValueError("agent_ref requires agent_id and version")
        for key in ("agent_id", "version"):
            if key in agent_ref:
                _validate_identifier(agent_ref[key], f"agent_ref {key}")
        object.__setattr__(self, "agent_ref", agent_ref)
        if depth == 0:
            if root_run_id != self.run_id:
                raise ValueError("root run_lineage root_run_id must match run_id")
            if execution_group_id != root_run_id:
                raise ValueError(
                    "root execution_group_id must match run_lineage root_run_id"
                )
            for field_name in (
                "parent_run_id",
                "delegation_id",
                "parent_tool_call_id",
                "caller_agent_id",
            ):
                if lineage.get(field_name) is not None:
                    raise ValueError(f"root run_lineage cannot contain {field_name}")
            for value, field_name in (
                (self.delegation_id, "delegation_id"),
                (self.parent_tool_call_id, "parent_tool_call_id"),
                (self.budget_lease_id, "budget_lease_id"),
            ):
                if value is not None:
                    raise ValueError(f"root snapshot cannot contain {field_name}")
            if agent_ref:
                if len(agent_path) != 1:
                    raise ValueError(
                        "definition-bound root agent_path must contain one Agent"
                    )
                if (
                    lineage.get("agent_id") != agent_ref["agent_id"]
                    or lineage.get("agent_version") != agent_ref["version"]
                ):
                    raise ValueError(
                        "root run_lineage current Agent does not match agent_ref"
                    )
        else:
            _validate_identifier(self.budget_lease_id, "budget_lease_id")
            if len(agent_path) != depth + 1:
                raise ValueError("child agent_path length must equal depth + 1")
            lineage.setdefault("delegation_id", self.delegation_id)
            lineage.setdefault("parent_tool_call_id", self.parent_tool_call_id)
            if agent_ref:
                lineage.setdefault("agent_id", agent_ref["agent_id"])
                lineage.setdefault("agent_version", agent_ref["version"])
            if len(agent_path) >= 2:
                prior_id = str(agent_path[-2]).rpartition("@")[0]
                lineage.setdefault("caller_agent_id", prior_id or None)
            for field_name in (
                "parent_run_id",
                "delegation_id",
                "parent_tool_call_id",
                "caller_agent_id",
                "agent_id",
                "agent_version",
            ):
                _validate_identifier(
                    lineage.get(field_name), f"run_lineage {field_name}"
                )
            expected_tail = f"{lineage['agent_id']}@{lineage['agent_version']}"
            if agent_path[-1] != expected_tail:
                raise ValueError("run_lineage agent_path does not end at current Agent")
            if agent_ref and (
                lineage["agent_id"] != agent_ref["agent_id"]
                or lineage["agent_version"] != agent_ref["version"]
            ):
                raise ValueError("run_lineage current Agent does not match agent_ref")
            if self.delegation_id != lineage["delegation_id"]:
                raise ValueError("delegation_id does not match run_lineage")
            if self.parent_tool_call_id != lineage["parent_tool_call_id"]:
                raise ValueError("parent_tool_call_id does not match run_lineage")
        if agent_ref and agent_path:
            expected_tail = f"{agent_ref['agent_id']}@{agent_ref['version']}"
            if agent_path[-1] != expected_tail:
                raise ValueError("agent_ref does not match run_lineage agent_path")
        object.__setattr__(self, "run_lineage", lineage)
        definition_fingerprint = (
            self.agent_fingerprint
            if self.agent_definition_fingerprint is None
            else self.agent_definition_fingerprint
        )
        _validate_identifier(
            definition_fingerprint,
            "agent_definition_fingerprint",
        )
        object.__setattr__(
            self,
            "agent_definition_fingerprint",
            definition_fingerprint,
        )
        for value, field_name in (
            (self.delegation_id, "delegation_id"),
            (self.parent_tool_call_id, "parent_tool_call_id"),
            (self.budget_lease_id, "budget_lease_id"),
        ):
            if value is not None:
                _validate_identifier(value, field_name)
        if self.migrated_from_schema_version is not None and (
            type(self.migrated_from_schema_version) is not int
            or self.migrated_from_schema_version not in {1, 2, 3, 4}
        ):
            raise ValueError("migrated_from_schema_version must be 1-4 or None")
        for value, field_name in (
            (self.tenant_scope_digest, "tenant_scope_digest"),
            (self.principal_scope_digest, "principal_scope_digest"),
        ):
            if (
                value is not None
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    value,
                )
                is None
            ):
                raise ValueError(f"{field_name} must use sha256")
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
            "run_lineage": _json_mapping_copy(self.run_lineage, "run_lineage"),
            "execution_group_id": self.execution_group_id,
            "agent_ref": _json_mapping_copy(self.agent_ref, "agent_ref"),
            "agent_definition_fingerprint": self.agent_definition_fingerprint,
            "delegation_id": self.delegation_id,
            "parent_tool_call_id": self.parent_tool_call_id,
            "budget_lease_id": self.budget_lease_id,
            "migrated_from_schema_version": self.migrated_from_schema_version,
            "tenant_scope_digest": self.tenant_scope_digest,
            "principal_scope_digest": self.principal_scope_digest,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunSnapshot":
        if not isinstance(value, Mapping):
            raise ValueError("snapshot payload must be a JSON object")
        if "schema_version" not in value or "version" not in value:
            raise ValueError("snapshot requires schema_version and version")
        schema_version = _integer(value["schema_version"], "schema_version")
        compatibility_version = _integer(value["version"], "version")
        if schema_version != compatibility_version:
            raise ValueError("snapshot schema_version and version must match")
        if schema_version not in {
            PREVIOUS_SNAPSHOT_SCHEMA_VERSION,
            SNAPSHOT_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported snapshot schema version: {schema_version}")
        if schema_version == SNAPSHOT_SCHEMA_VERSION:
            required_v5_fields = {
                "run_lineage",
                "execution_group_id",
                "agent_ref",
                "agent_definition_fingerprint",
                "delegation_id",
                "parent_tool_call_id",
                "budget_lease_id",
                "migrated_from_schema_version",
                "tenant_scope_digest",
                "principal_scope_digest",
            }
            if not required_v5_fields.issubset(value):
                raise ValueError("native v5 snapshot is missing identity fields")
        migrated = dict(value)
        if schema_version == PREVIOUS_SNAPSHOT_SCHEMA_VERSION:
            run_id = str(value.get("run_id", ""))
            migrated.update(
                {
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
                    "version": SNAPSHOT_SCHEMA_VERSION,
                    "run_lineage": {
                        "root_run_id": run_id,
                        "parent_run_id": None,
                        "depth": 0,
                        "agent_path": [],
                    },
                    "execution_group_id": run_id,
                    "agent_ref": {},
                    "agent_definition_fingerprint": str(
                        value.get("agent_fingerprint", "legacy-unbound")
                    ),
                    "delegation_id": None,
                    "parent_tool_call_id": None,
                    "budget_lease_id": None,
                    "migrated_from_schema_version": schema_version,
                    "tenant_scope_digest": None,
                    "principal_scope_digest": None,
                }
            )
        value = migrated
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
            run_lineage=_mapping(value.get("run_lineage", {}), "run_lineage"),
            execution_group_id=(
                None
                if value.get("execution_group_id") is None
                else str(value["execution_group_id"])
            ),
            agent_ref=_mapping(value.get("agent_ref", {}), "agent_ref"),
            agent_definition_fingerprint=(
                None
                if value.get("agent_definition_fingerprint") is None
                else str(value["agent_definition_fingerprint"])
            ),
            delegation_id=(
                None
                if value.get("delegation_id") is None
                else str(value["delegation_id"])
            ),
            parent_tool_call_id=(
                None
                if value.get("parent_tool_call_id") is None
                else str(value["parent_tool_call_id"])
            ),
            budget_lease_id=(
                None
                if value.get("budget_lease_id") is None
                else str(value["budget_lease_id"])
            ),
            migrated_from_schema_version=(
                None
                if value.get("migrated_from_schema_version") is None
                else _integer(
                    value["migrated_from_schema_version"],
                    "migrated_from_schema_version",
                )
            ),
            tenant_scope_digest=(
                None
                if value.get("tenant_scope_digest") is None
                else str(value["tenant_scope_digest"])
            ),
            principal_scope_digest=(
                None
                if value.get("principal_scope_digest") is None
                else str(value["principal_scope_digest"])
            ),
            created_at=_parse_datetime(value.get("created_at")),
            updated_at=_parse_datetime(value.get("updated_at")),
            schema_version=SNAPSHOT_SCHEMA_VERSION,
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


def identity_scope_digest(kind: str, value: str) -> str:
    """Return a stable, content-free binding for a trusted identity claim."""

    if kind not in {"tenant", "principal"}:
        raise ValueError("identity scope kind must be tenant or principal")
    _validate_identifier(value, f"{kind} identity")
    encoded = f"moduagent.identity-scope.v1\0{kind}\0{value}".encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
    "PREVIOUS_SNAPSHOT_SCHEMA_VERSION",
    "RunSnapshot",
    "SNAPSHOT_SCHEMA_VERSION",
    "SNAPSHOT_RUNTIME_VERSION",
    "current_runtime_version",
    "decode_engine_snapshot",
    "encode_engine_snapshot",
    "identity_scope_digest",
]

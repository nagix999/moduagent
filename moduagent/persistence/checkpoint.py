from __future__ import annotations

import asyncio
import copy
import inspect
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from moduagent.messages import Message, Usage
from moduagent.persistence.migration import (
    StateMigrationError,
    flatten_plan_engine_state,
    migrate_checkpoint_payload,
)
from moduagent.persistence.snapshot import (
    EngineSnapshot,
    FinalizationMarkers,
    RunSnapshot,
    current_runtime_version,
)
from moduagent.runtime.context import (
    RunContext,
    RunRequest,
    RunStatus,
    SkillRunState,
)


_LEGACY_CHECKPOINT_VERSION = 3
_STANDARD_FINALIZATION_STATE_KEY = "_moduagent_structured_finalization"
_STANDARD_FINALIZATION_OUTPUT_KEY = "_moduagent_structured_output"


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    """Backward-compatible facade over a v4 :class:`RunSnapshot`.

    Construction and context conversion retain the 0.3 API. Serialization is
    always the v4 envelope; legacy v1-v3 payloads are copy-migrated on read.
    """

    run_id: str
    session_id: str
    messages: tuple[Message, ...]
    input: str = ""
    user_context: Mapping[str, Any] = field(default_factory=dict)
    requested_skills: tuple[str, ...] = ()
    skill_mode: str = "disabled"
    new_messages: tuple[Message, ...] = ()
    internal_messages: tuple[Message, ...] = ()
    execution_state: Any = None
    step: int = 0
    tool_call_count: int = 0
    status: RunStatus = RunStatus.CREATED
    policy_state: Mapping[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_run_start: int = 0
    skill_state: SkillRunState = field(default_factory=SkillRunState)
    # Additive v4 compatibility fields. They are intentionally appended so old
    # positional construction keeps its exact meaning.
    runtime_version: str = field(default_factory=current_runtime_version)
    agent_fingerprint: str = "legacy-unbound"
    engine_id: str | None = None
    engine_state_version: int = 1
    finalization_markers: FinalizationMarkers | None = None
    terminal_reason: str | None = None
    resume_safety: str = "resumable"
    event_sequence: int = 0
    engine_state: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        _validate_identifier(self.session_id, "session_id")
        _validate_identifier(self.runtime_version, "runtime_version")
        _validate_identifier(self.agent_fingerprint, "agent_fingerprint")
        if self.step < 0 or self.tool_call_count < 0 or self.event_sequence < 0:
            raise ValueError("checkpoint counters cannot be negative")
        if type(self.engine_state_version) is not int or self.engine_state_version < 1:
            raise ValueError("engine_state_version must be a positive integer")
        if not 0 <= self.current_run_start <= len(self.messages):
            raise ValueError("current_run_start must reference the message sequence")
        if not isinstance(self.status, RunStatus):
            object.__setattr__(self, "status", RunStatus(str(self.status)))
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "new_messages", tuple(self.new_messages))
        object.__setattr__(self, "internal_messages", tuple(self.internal_messages))
        object.__setattr__(self, "requested_skills", tuple(self.requested_skills))
        if not all(isinstance(message, Message) for message in self.messages):
            raise TypeError("messages must contain Message instances")
        if not all(isinstance(message, Message) for message in self.new_messages):
            raise TypeError("new_messages must contain Message instances")
        if not all(isinstance(message, Message) for message in self.internal_messages):
            raise TypeError("internal_messages must contain Message instances")
        if not isinstance(self.skill_state, SkillRunState):
            raise TypeError("skill_state must be a SkillRunState")
        if not isinstance(self.user_context, Mapping):
            raise TypeError("user_context must be a mapping")
        if not isinstance(self.policy_state, Mapping):
            raise TypeError("policy_state must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "user_context", copy.deepcopy(dict(self.user_context)))
        object.__setattr__(self, "policy_state", copy.deepcopy(dict(self.policy_state)))
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))
        execution_state = _execution_state_to_dict(self.execution_state)
        object.__setattr__(self, "execution_state", execution_state)
        if self.engine_state is not None:
            if not isinstance(self.engine_state, Mapping):
                raise TypeError("engine_state must be a mapping or None")
            object.__setattr__(
                self,
                "engine_state",
                copy.deepcopy(dict(self.engine_state)),
            )
        engine_id = self.engine_id or (
            "plan" if execution_state is not None else "standard"
        )
        _validate_identifier(engine_id, "engine_id")
        object.__setattr__(self, "engine_id", engine_id)
        markers = self.finalization_markers or _compatibility_markers(
            execution_state,
            self.policy_state,
            engine_state=self.engine_state,
        )
        if not isinstance(markers, FinalizationMarkers):
            raise TypeError("finalization_markers must be FinalizationMarkers")
        object.__setattr__(self, "finalization_markers", markers)
        if not isinstance(self.resume_safety, str) or not self.resume_safety.strip():
            raise ValueError("resume_safety cannot be empty")
        if self.terminal_reason is not None and not isinstance(
            self.terminal_reason,
            str,
        ):
            raise TypeError("terminal_reason must be a string or None")
        object.__setattr__(self, "created_at", _as_utc(self.created_at))
        object.__setattr__(self, "updated_at", _as_utc(self.updated_at))

    @classmethod
    def from_context(
        cls,
        context: RunContext,
        *,
        created_at: datetime | None = None,
    ) -> "RunCheckpoint":
        now = datetime.now(timezone.utc)
        messages = tuple(
            message
            for message in context.messages
            if not _ephemeral_protocol_message(message)
        )
        new_messages = tuple(
            message
            for message in context.new_messages
            if not _ephemeral_protocol_message(message)
        )
        internal_messages = tuple(
            message
            for message in context.internal_messages
            if not _ephemeral_protocol_message(message)
        )
        current_run_start = sum(
            not _ephemeral_protocol_message(message)
            for message in context.messages[: context.current_run_start]
        )
        execution_state = _execution_state_to_dict(context.execution_state)
        policy_state = copy.deepcopy(dict(context.policy_state))
        raw_engine = context.metadata.get("_moduagent_engine")
        engine_state: Mapping[str, Any] | None = None
        engine_id: str | None = None
        engine_state_version = 1
        if isinstance(raw_engine, Mapping):
            raw_state = raw_engine.get("state", {})
            if not isinstance(raw_state, Mapping):
                raise TypeError("runtime engine state must be a mapping")
            engine_state = copy.deepcopy(dict(raw_state))
            engine_id = str(raw_engine.get("engine_id", "")).strip() or None
            engine_state_version = int(raw_engine.get("state_version", 1))
        # Existing Plan policies use these copies to restore their domain state.
        # Keep them consistent at the facade boundary; the v4 envelope stores
        # only one nested engine state.
        if execution_state is not None:
            policy_state["execution_state"] = copy.deepcopy(execution_state)
            if isinstance(execution_state.get("plan"), Mapping):
                policy_state["plan"] = copy.deepcopy(execution_state["plan"])
        return cls(
            run_id=context.run_id,
            session_id=context.request.session_id,
            input=context.request.input,
            user_context=dict(context.request.user_context),
            requested_skills=context.request.requested_skills,
            skill_mode=context.request.skill_mode,
            messages=messages,
            new_messages=new_messages,
            internal_messages=internal_messages,
            execution_state=execution_state,
            step=context.step,
            tool_call_count=context.tool_call_count,
            status=context.status,
            policy_state=policy_state,
            usage=context.usage,
            metadata=dict(context.metadata),
            created_at=created_at or context.created_at,
            updated_at=now,
            current_run_start=current_run_start,
            skill_state=context.skill_state,
            runtime_version=str(
                context.metadata.get(
                    "_moduagent_runtime_version",
                    current_runtime_version(),
                )
            ),
            agent_fingerprint=str(
                context.metadata.get(
                    "_moduagent_agent_fingerprint",
                    "legacy-unbound",
                )
            ),
            engine_id=engine_id
            or (
                str(context.metadata.get("_moduagent_engine_id"))
                if context.metadata.get("_moduagent_engine_id")
                else ("plan" if execution_state is not None else "standard")
            ),
            engine_state_version=(
                engine_state_version
                if engine_state is not None
                else int(context.metadata.get("_moduagent_engine_state_version", 1))
            ),
            terminal_reason=(
                None
                if context.metadata.get("_moduagent_terminal_reason") is None
                else str(context.metadata["_moduagent_terminal_reason"])
            ),
            resume_safety=str(
                context.metadata.get("_moduagent_resume_safety", "resumable")
            ),
            event_sequence=int(context.metadata.get("_moduagent_event_sequence", 0)),
            engine_state=engine_state,
        )

    @classmethod
    def from_snapshot(cls, snapshot: RunSnapshot) -> "RunCheckpoint":
        if not isinstance(snapshot, RunSnapshot):
            raise TypeError("snapshot must be a RunSnapshot")
        common = snapshot.common_state
        request = common.request
        policy_state = copy.deepcopy(dict(common.compatibility_policy_state))
        execution_state: dict[str, Any] | None = None
        engine_initialized = policy_state.get(
            "_moduagent_engine_initialized",
            True,
        )
        if type(engine_initialized) is not bool:
            raise StateMigrationError("_moduagent_engine_initialized must be a bool")
        if snapshot.engine.engine_id == "plan" and engine_initialized:
            execution_state = flatten_plan_engine_state(
                snapshot.engine.state,
                snapshot.finalization_markers,
            )
            policy_state["execution_state"] = copy.deepcopy(execution_state)
            policy_state["plan"] = copy.deepcopy(execution_state["plan"])
        elif snapshot.engine.engine_id == "standard":
            engine_markers = _compatibility_markers(
                None,
                {},
                engine_state=snapshot.engine.state,
            )
            if engine_markers != snapshot.finalization_markers:
                raise StateMigrationError(
                    "Standard finalization state does not match outer "
                    "finalization markers"
                )
            engine_policy = snapshot.engine.state.get("policy_state", {})
            if not isinstance(engine_policy, Mapping):
                raise StateMigrationError(
                    "standard engine policy_state must be an object"
                )
            policy_state.update(copy.deepcopy(dict(engine_policy)))
            markers = snapshot.finalization_markers
            if markers.started:
                policy_state[_STANDARD_FINALIZATION_STATE_KEY] = (
                    "completed" if markers.response_generated else "pending"
                )
            if markers.response_generated:
                policy_state[_STANDARD_FINALIZATION_OUTPUT_KEY] = copy.deepcopy(
                    markers.response
                )

        usage = common.usage
        raw_request_skills = request.get("requested_skills", ())
        if isinstance(raw_request_skills, (str, bytes, bytearray)) or not isinstance(
            raw_request_skills,
            (list, tuple),
        ):
            raise StateMigrationError("requested_skills must be an array")
        return cls(
            run_id=snapshot.run_id,
            session_id=snapshot.session_id,
            input=str(request.get("input", "")),
            user_context=_mapping_copy(
                request.get("user_context", {}),
                "user_context",
            ),
            requested_skills=tuple(str(skill) for skill in raw_request_skills),
            skill_mode=str(request.get("skill_mode", "disabled")),
            messages=tuple(Message.from_dict(message) for message in common.messages),
            new_messages=tuple(
                Message.from_dict(message) for message in common.new_messages
            ),
            internal_messages=tuple(
                Message.from_dict(message) for message in common.internal_messages
            ),
            execution_state=execution_state,
            step=common.step,
            tool_call_count=common.tool_call_count,
            status=RunStatus(common.status),
            policy_state=policy_state,
            usage=Usage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                provider=_mapping_copy(
                    usage.get("provider", {}),
                    "usage provider",
                ),
            ),
            metadata=dict(snapshot.sanitized_runtime_metadata),
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
            current_run_start=common.current_run_start,
            skill_state=SkillRunState.from_dict(snapshot.skill_state),
            runtime_version=snapshot.runtime_version,
            agent_fingerprint=snapshot.agent_fingerprint,
            engine_id=snapshot.engine.engine_id,
            engine_state_version=snapshot.engine.state_version,
            finalization_markers=snapshot.finalization_markers,
            terminal_reason=common.terminal_reason,
            resume_safety=common.resume_safety,
            event_sequence=common.event_sequence,
            engine_state=snapshot.engine.state,
        )

    def to_snapshot(self) -> RunSnapshot:
        migrated = migrate_checkpoint_payload(
            self._to_legacy_dict(),
            agent_fingerprint=self.agent_fingerprint,
            runtime_version=self.runtime_version,
        )
        if self.engine_id not in {"plan", "standard"} and self.engine_state is None:
            raise StateMigrationError(
                "custom Engine checkpoints require an explicit engine_state"
            )
        engine = migrated.engine
        if self.engine_state is not None:
            engine = EngineSnapshot(
                engine_id=str(self.engine_id),
                state_version=self.engine_state_version,
                state=self.engine_state,
            )
        if engine.state_version != self.engine_state_version:
            engine = EngineSnapshot(
                engine_id=engine.engine_id,
                state_version=self.engine_state_version,
                state=engine.state,
            )
        markers = self.finalization_markers or migrated.finalization_markers
        bootstrap = self.policy_state.get("_moduagent_engine_initialized") is False
        if engine.engine_id == "plan" and not bootstrap:
            engine_state = copy.deepcopy(dict(engine.state))
            finalization = engine_state.get("finalization")
            if not isinstance(finalization, Mapping):
                raise StateMigrationError("Plan engine finalization must be an object")
            engine_state["finalization"] = {
                **dict(finalization),
                **markers.to_dict(),
            }
            engine = EngineSnapshot(
                engine_id=engine.engine_id,
                state_version=engine.state_version,
                state=engine_state,
            )
        common = replace(
            migrated.common_state,
            terminal_reason=self.terminal_reason,
            resume_safety=self.resume_safety,
            event_sequence=self.event_sequence,
        )
        return replace(
            migrated,
            agent_fingerprint=self.agent_fingerprint,
            runtime_version=self.runtime_version,
            engine=engine,
            common_state=common,
            finalization_markers=markers,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def to_context(self) -> RunContext:
        request = RunRequest(
            input=self.input,
            session_id=self.session_id,
            user_context=dict(self.user_context),
            resume_run_id=self.run_id,
            requested_skills=self.requested_skills,
            skill_mode=self.skill_mode,
        )
        metadata = dict(self.metadata)
        metadata["_moduagent_runtime_version"] = self.runtime_version
        metadata["_moduagent_agent_fingerprint"] = self.agent_fingerprint
        metadata["_moduagent_engine_id"] = self.engine_id
        metadata["_moduagent_engine_state_version"] = self.engine_state_version
        # The compatibility runtime reconstructs its mutable policy state from
        # the facade fields below. Re-injecting an EngineSnapshot here would
        # become stale as that legacy policy advances. New Engines persist their
        # current snapshot explicitly through RuntimeServices.checkpoint().
        metadata.pop("_moduagent_engine", None)
        metadata["_moduagent_resume_safety"] = self.resume_safety
        metadata["_moduagent_event_sequence"] = self.event_sequence
        if self.terminal_reason is not None:
            metadata["_moduagent_terminal_reason"] = self.terminal_reason
        return RunContext(
            run_id=self.run_id,
            request=request,
            messages=list(self.messages),
            new_messages=list(self.new_messages),
            internal_messages=list(self.internal_messages),
            execution_state=_execution_state_from_value(self.execution_state),
            step=self.step,
            tool_call_count=self.tool_call_count,
            status=self.status,
            policy_state=dict(self.policy_state),
            usage=self.usage,
            metadata=metadata,
            current_run_start=self.current_run_start,
            skill_state=self.skill_state,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_snapshot().to_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunCheckpoint":
        snapshot = migrate_checkpoint_payload(value)
        return cls.from_snapshot(snapshot)

    def to_json(self) -> str:
        return self.to_snapshot().to_json()

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "RunCheckpoint":
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        try:
            value = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint payload must be valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint payload must be a JSON object")
        return cls.from_dict(value)

    def _to_legacy_dict(self) -> dict[str, Any]:
        execution_state = (
            None
            if self.execution_state is None
            else copy.deepcopy(dict(self.execution_state))
        )
        markers = self.finalization_markers or FinalizationMarkers()
        policy_state = copy.deepcopy(dict(self.policy_state))
        if execution_state is not None:
            execution_state.update(
                {
                    "final_response": copy.deepcopy(markers.response),
                    "final_persisted": markers.persisted,
                    "final_emitted": markers.emitted,
                }
            )
            policy_state["execution_state"] = copy.deepcopy(execution_state)
            if isinstance(execution_state.get("plan"), Mapping):
                policy_state["plan"] = copy.deepcopy(execution_state["plan"])
        else:
            if markers.started:
                policy_state[_STANDARD_FINALIZATION_STATE_KEY] = (
                    "completed" if markers.response_generated else "pending"
                )
            if markers.response_generated:
                policy_state[_STANDARD_FINALIZATION_OUTPUT_KEY] = copy.deepcopy(
                    markers.response
                )
        return {
            "version": _LEGACY_CHECKPOINT_VERSION,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "input": self.input,
            "user_context": copy.deepcopy(dict(self.user_context)),
            "requested_skills": list(self.requested_skills),
            "skill_mode": self.skill_mode,
            "messages": [message.to_dict() for message in self.messages],
            "new_messages": [message.to_dict() for message in self.new_messages],
            "internal_messages": [
                message.to_dict() for message in self.internal_messages
            ],
            "execution_state": execution_state,
            "step": self.step,
            "tool_call_count": self.tool_call_count,
            "status": self.status.value,
            "policy_state": policy_state,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
                "provider": dict(self.usage.provider),
            },
            "metadata": copy.deepcopy(dict(self.metadata)),
            "current_run_start": self.current_run_start,
            "skill_state": self.skill_state.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "terminal_reason": self.terminal_reason,
            "resume_safety": self.resume_safety,
            "event_sequence": self.event_sequence,
        }


@runtime_checkable
class CheckpointStore(Protocol):
    async def load(self, run_id: str) -> RunCheckpoint | None: ...

    async def save(
        self,
        run_id: str,
        context: RunContext | RunCheckpoint,
    ) -> None: ...

    async def delete(self, run_id: str) -> None: ...


@runtime_checkable
class SnapshotStore(Protocol):
    """Additive v4 store contract used by the new PersistenceCoordinator."""

    async def load_snapshot(self, run_id: str) -> RunSnapshot | None: ...

    async def save_snapshot(self, snapshot: RunSnapshot) -> None: ...

    async def delete(self, run_id: str) -> None: ...


class InMemoryCheckpointStore:
    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _validate_ttl(ttl_seconds)
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._snapshots: dict[str, str] = {}
        self._legacy_payloads: dict[str, str] = {}
        self._expires_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def load(self, run_id: str) -> RunCheckpoint | None:
        snapshot = await self.load_snapshot(run_id)
        return None if snapshot is None else RunCheckpoint.from_snapshot(snapshot)

    async def load_snapshot(self, run_id: str) -> RunSnapshot | None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            if self._expired(run_id):
                self._delete_locked(run_id)
                return None
            payload = self._snapshots.get(run_id)
            if payload is not None:
                return RunSnapshot.from_json(payload)
            legacy_payload = self._legacy_payloads.get(run_id)
            if legacy_payload is None:
                return None
            snapshot = _snapshot_from_json(legacy_payload)
            # Copy-on-migrate: write and verify the v4 candidate while retaining
            # the exact legacy payload under its original slot.
            encoded = snapshot.to_json()
            verified = RunSnapshot.from_json(encoded)
            self._snapshots[run_id] = encoded
            return verified

    async def save(
        self,
        run_id: str | RunCheckpoint | RunSnapshot,
        context: RunContext | RunCheckpoint | RunSnapshot | None = None,
    ) -> None:
        snapshot = _coerce_snapshot(run_id, context)
        await self.save_snapshot(snapshot)

    async def save_snapshot(self, snapshot: RunSnapshot) -> None:
        if not isinstance(snapshot, RunSnapshot):
            raise TypeError("snapshot must be a RunSnapshot")
        encoded = snapshot.to_json()
        stored = RunSnapshot.from_json(encoded)
        async with self._lock:
            self._snapshots[snapshot.run_id] = stored.to_json()
            if self._ttl_seconds is not None:
                self._expires_at[snapshot.run_id] = self._clock() + self._ttl_seconds

    async def save_legacy_payload(
        self,
        run_id: str,
        payload: str | bytes | bytearray | Mapping[str, Any],
    ) -> None:
        """Import an exact legacy payload for migration rehearsal."""

        _validate_identifier(run_id, "run_id")
        encoded = _legacy_payload_json(payload)
        value = json.loads(encoded)
        if str(value.get("run_id", "")) != run_id:
            raise ValueError("run_id does not match legacy checkpoint payload")
        # Dry-run before making the legacy payload active.
        migrate_checkpoint_payload(value)
        async with self._lock:
            self._legacy_payloads[run_id] = encoded
            self._snapshots.pop(run_id, None)
            if self._ttl_seconds is not None:
                self._expires_at[run_id] = self._clock() + self._ttl_seconds

    async def load_legacy_payload(self, run_id: str) -> str | None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            if self._expired(run_id):
                self._delete_locked(run_id)
                return None
            return self._legacy_payloads.get(run_id)

    async def delete(self, run_id: str) -> None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            self._delete_locked(run_id)

    def _expired(self, run_id: str) -> bool:
        expires_at = self._expires_at.get(run_id)
        return expires_at is not None and expires_at <= self._clock()

    def _delete_locked(self, run_id: str) -> None:
        self._snapshots.pop(run_id, None)
        self._legacy_payloads.pop(run_id, None)
        self._expires_at.pop(run_id, None)


class RedisCheckpointStore:
    """Versioned checkpoint storage over a sync or async Redis-style client."""

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "moduagent:checkpoint:",
        ttl_seconds: int | None = None,
    ) -> None:
        _validate_ttl(ttl_seconds)
        for method in ("get", "set", "delete"):
            if not callable(getattr(client, method, None)):
                raise TypeError(f"Redis client must provide {method}()")
        self._client = client
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    async def load(self, run_id: str) -> RunCheckpoint | None:
        snapshot = await self.load_snapshot(run_id)
        return None if snapshot is None else RunCheckpoint.from_snapshot(snapshot)

    async def load_snapshot(self, run_id: str) -> RunSnapshot | None:
        _validate_identifier(run_id, "run_id")
        v4_payload = await _call(self._client.get, self._v4_key(run_id))
        if v4_payload is not None:
            return RunSnapshot.from_json(v4_payload)
        legacy_payload = await _call(self._client.get, self._legacy_key(run_id))
        if legacy_payload is None:
            return None
        snapshot = _snapshot_from_json(legacy_payload)
        encoded = snapshot.to_json()
        await _redis_set(
            self._client,
            self._v4_key(run_id),
            encoded,
            self._ttl_seconds,
        )
        verified_payload = await _call(self._client.get, self._v4_key(run_id))
        if verified_payload is None:
            raise RuntimeError("v4 checkpoint migration read-back failed")
        verified = RunSnapshot.from_json(verified_payload)
        if verified.to_json() != encoded:
            raise RuntimeError("v4 checkpoint migration verification failed")
        # The legacy key is deliberately retained.
        return verified

    async def save(
        self,
        run_id: str | RunCheckpoint | RunSnapshot,
        context: RunContext | RunCheckpoint | RunSnapshot | None = None,
    ) -> None:
        await self.save_snapshot(_coerce_snapshot(run_id, context))

    async def save_snapshot(self, snapshot: RunSnapshot) -> None:
        if not isinstance(snapshot, RunSnapshot):
            raise TypeError("snapshot must be a RunSnapshot")
        await _redis_set(
            self._client,
            self._v4_key(snapshot.run_id),
            snapshot.to_json(),
            self._ttl_seconds,
        )

    async def save_legacy_payload(
        self,
        run_id: str,
        payload: str | bytes | bytearray | Mapping[str, Any],
    ) -> None:
        _validate_identifier(run_id, "run_id")
        encoded = _legacy_payload_json(payload)
        value = json.loads(encoded)
        if str(value.get("run_id", "")) != run_id:
            raise ValueError("run_id does not match legacy checkpoint payload")
        migrate_checkpoint_payload(value)
        await _redis_set(
            self._client,
            self._legacy_key(run_id),
            encoded,
            self._ttl_seconds,
        )

    async def load_legacy_payload(self, run_id: str) -> str | None:
        _validate_identifier(run_id, "run_id")
        payload = await _call(self._client.get, self._legacy_key(run_id))
        if payload is None:
            return None
        if isinstance(payload, bytes):
            return payload.decode("utf-8")
        return str(payload)

    async def delete(self, run_id: str) -> None:
        _validate_identifier(run_id, "run_id")
        await _call(self._client.delete, self._v4_key(run_id))
        await _call(self._client.delete, self._legacy_key(run_id))

    def _legacy_key(self, run_id: str) -> str:
        _validate_identifier(run_id, "run_id")
        return f"{self._key_prefix}{run_id}"

    def _v4_key(self, run_id: str) -> str:
        _validate_identifier(run_id, "run_id")
        return f"{self._key_prefix}v4:{run_id}"


async def _redis_set(
    client: Any,
    key: str,
    payload: str,
    ttl_seconds: int | None,
) -> None:
    if ttl_seconds is None:
        await _call(client.set, key, payload)
        return
    try:
        await _call(client.set, key, payload, ex=ttl_seconds)
    except TypeError:
        await _call(client.set, key, payload)
        expire = getattr(client, "expire", None)
        if not callable(expire):
            raise TypeError(
                "Redis client must provide expire when set(ex=) is unsupported"
            )
        await _call(expire, key, ttl_seconds)


async def _call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = function(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _coerce_snapshot(
    run_id: str | RunCheckpoint | RunSnapshot,
    context: RunContext | RunCheckpoint | RunSnapshot | None,
) -> RunSnapshot:
    if isinstance(run_id, RunSnapshot) and context is None:
        return run_id
    if isinstance(run_id, RunCheckpoint) and context is None:
        return run_id.to_snapshot()
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    _validate_identifier(run_id, "run_id")
    if isinstance(context, RunContext):
        snapshot = RunCheckpoint.from_context(context).to_snapshot()
    elif isinstance(context, RunCheckpoint):
        snapshot = context.to_snapshot()
    elif isinstance(context, RunSnapshot):
        snapshot = context
    else:
        raise TypeError("context must be a RunContext, RunCheckpoint, or RunSnapshot")
    if snapshot.run_id != run_id:
        raise ValueError("run_id does not match context.run_id")
    return snapshot


def _snapshot_from_json(payload: str | bytes | bytearray) -> RunSnapshot:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint payload must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint payload must be a JSON object")
    return migrate_checkpoint_payload(value)


def _legacy_payload_json(
    payload: str | bytes | bytearray | Mapping[str, Any],
) -> str:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        value = json.loads(payload)
        encoded = payload
    elif isinstance(payload, Mapping):
        value = copy.deepcopy(dict(payload))
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    else:
        raise TypeError("legacy payload must be JSON text or a mapping")
    if not isinstance(value, Mapping):
        raise ValueError("legacy checkpoint payload must be a JSON object")
    if "schema_version" in value or int(value.get("version", 1)) not in {1, 2, 3}:
        raise ValueError("legacy checkpoint payload must use version 1, 2, or 3")
    return encoded


def _compatibility_markers(
    execution_state: Mapping[str, Any] | None,
    policy_state: Mapping[str, Any],
    *,
    engine_state: Mapping[str, Any] | None = None,
) -> FinalizationMarkers:
    if engine_state is not None:
        raw_finalization = engine_state.get("finalization", {})
        if isinstance(raw_finalization, Mapping):
            response = copy.deepcopy(raw_finalization.get("response"))
            return FinalizationMarkers(
                started=bool(
                    raw_finalization.get(
                        "started",
                        response is not None
                        or int(raw_finalization.get("invocation_count", 0)) > 0,
                    )
                ),
                response_generated=bool(
                    raw_finalization.get(
                        "response_generated",
                        response is not None,
                    )
                ),
                response=response,
                persisted=bool(raw_finalization.get("persisted", False)),
                emitted=bool(raw_finalization.get("emitted", False)),
            )
    if execution_state is not None:
        response = copy.deepcopy(execution_state.get("final_response"))
        phase = str(execution_state.get("phase", "plan"))
        count = int(execution_state.get("finalization_count", 0))
        return FinalizationMarkers(
            started=phase in {"finalize", "done"} or count > 0 or response is not None,
            response_generated=response is not None,
            response=response,
            persisted=bool(execution_state.get("final_persisted", False)),
            emitted=bool(execution_state.get("final_emitted", False)),
        )
    state = policy_state.get(_STANDARD_FINALIZATION_STATE_KEY)
    response = copy.deepcopy(policy_state.get(_STANDARD_FINALIZATION_OUTPUT_KEY))
    return FinalizationMarkers(
        started=state in {"pending", "completed"} or response is not None,
        response_generated=response is not None,
        response=response,
    )


def _execution_state_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise TypeError("execution_state must provide to_dict()")
    if not isinstance(payload, Mapping):
        raise TypeError("execution_state.to_dict() must return a mapping")
    return copy.deepcopy(dict(payload))


def _execution_state_from_value(value: Any) -> Any:
    """Legacy facade adapter; v4 persistence itself keeps engine state opaque."""

    if value is None:
        return None
    from moduagent.decision.planning import ExecutionState

    if isinstance(value, ExecutionState):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint execution_state must be an object")
    return ExecutionState.from_dict(value)


def _ephemeral_protocol_message(message: Message) -> bool:
    return (
        message.metadata.get("moduagent.ephemeral") is True
        or message.metadata.get("moduagent.checkpoint_excluded") is True
    )


def _mapping_copy(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StateMigrationError(f"{name} must be an object")
    return copy.deepcopy(dict(value))


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _validate_ttl(value: float | None) -> None:
    if value is not None and value <= 0:
        raise ValueError("ttl_seconds must be positive")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "RedisCheckpointStore",
    "RunCheckpoint",
    "SnapshotStore",
]

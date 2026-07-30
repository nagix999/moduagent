from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from moduagent.errors import ModuAgentError
from moduagent.models import ModelResponse


GuardTripReason = Literal["model_turn_budget", "no_progress"]
# Version 1 stored an unsalted observation digest. It was never released and
# is intentionally not migrated because carrying that digest forward would
# preserve the cross-run correlation this schema revision removes.
_STATE_VERSION = 2
_SALT_BYTES = 32


@dataclass(frozen=True, slots=True)
class ModelGuardSnapshot:
    """Secret-free counters suitable for diagnostics and tests."""

    model_turns: int
    max_model_turns: int
    no_progress_model_turns: int
    no_progress_model_turn_threshold: int
    trip_reason: GuardTripReason | None

    @property
    def tripped(self) -> bool:
        return self.trip_reason is not None

    @property
    def remaining_model_turns(self) -> int:
        return max(0, self.max_model_turns - self.model_turns)


class ModelGuardTripped(ModuAgentError, RuntimeError):
    """Base class for secret-safe model execution circuit-breaker failures."""

    code: str

    def __init__(
        self,
        message: str,
        *,
        code: str,
        snapshot: ModelGuardSnapshot,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.snapshot = snapshot


class ModelTurnBudgetExceeded(ModelGuardTripped):
    """Raised before an attempt would exceed the per-run model turn budget."""

    def __init__(self, snapshot: ModelGuardSnapshot) -> None:
        super().__init__(
            "model turn budget exhausted",
            code="max_model_turns_exceeded",
            snapshot=snapshot,
        )


class ModelNoProgressError(ModelGuardTripped):
    """Raised after repeated identical state/response observations."""

    def __init__(self, snapshot: ModelGuardSnapshot) -> None:
        super().__init__(
            "model execution made no semantic progress",
            code="model_no_progress",
            snapshot=snapshot,
        )


class NoProgressCircuitBreaker:
    """Bound model calls and stop consecutive identical semantic observations.

    Call ``before_model_attempt`` with a small JSON-like projection of the
    current semantic state, then call ``observe_model_response`` with the
    normalized response. The breaker retains only a random salt, SHA-256
    digests, and numeric counters; prompt text, Tool arguments, output text,
    usage and provider metadata are never retained.

    A durable per-run random salt makes otherwise identical observation
    digests unlinkable across runs. ``no_progress_model_turn_threshold`` counts
    completed observations including the first one. A value of three therefore
    trips on the third consecutive identical semantic-state/semantic-response
    pair. Call ``mark_progress`` only after independently verified semantic
    progress.
    """

    __slots__ = (
        "_last_observation_digest",
        "_max_model_turns",
        "_no_progress_model_turn_threshold",
        "_model_turns",
        "_no_progress_model_turns",
        "_pending_state_digest",
        "_salt",
        "_trip_reason",
    )

    def __init__(
        self,
        *,
        max_model_turns: int = 32,
        no_progress_model_turn_threshold: int = 3,
    ) -> None:
        _validate_positive_integer(max_model_turns, "max_model_turns")
        _validate_positive_integer(
            no_progress_model_turn_threshold,
            "no_progress_model_turn_threshold",
        )
        if no_progress_model_turn_threshold < 2:
            raise ValueError("no_progress_model_turn_threshold must be at least 2")
        self._max_model_turns = max_model_turns
        self._no_progress_model_turn_threshold = no_progress_model_turn_threshold
        self._salt = b""
        self.reset()

    @property
    def snapshot(self) -> ModelGuardSnapshot:
        return ModelGuardSnapshot(
            model_turns=self._model_turns,
            max_model_turns=self._max_model_turns,
            no_progress_model_turns=self._no_progress_model_turns,
            no_progress_model_turn_threshold=(self._no_progress_model_turn_threshold),
            trip_reason=self._trip_reason,
        )

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        max_model_turns: int,
        no_progress_model_turn_threshold: int,
    ) -> NoProgressCircuitBreaker:
        """Restore a secret-free guard state from a run checkpoint."""

        if not isinstance(state, Mapping):
            raise TypeError("model guard state must be a mapping")
        if state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported model guard state version")
        breaker = cls(
            max_model_turns=max_model_turns,
            no_progress_model_turn_threshold=no_progress_model_turn_threshold,
        )
        model_turns = _non_negative_integer(
            state.get("model_turns"),
            "model_turns",
        )
        no_progress_model_turns = _non_negative_integer(
            state.get("no_progress_model_turns"),
            "no_progress_model_turns",
        )
        if model_turns > max_model_turns:
            raise ValueError("model guard state exceeds max_model_turns")
        if no_progress_model_turns > no_progress_model_turn_threshold:
            raise ValueError(
                "model guard state exceeds no_progress_model_turn_threshold"
            )
        if no_progress_model_turns > model_turns:
            raise ValueError("model guard no_progress_model_turns exceeds model_turns")
        raw_salt = state.get("salt")
        if not _is_sha256_digest(raw_salt):
            raise ValueError("model guard salt is invalid")
        salt = bytes.fromhex(raw_salt)
        digest = state.get("last_observation_digest")
        if digest is not None and not _is_sha256_digest(digest):
            raise ValueError("model guard observation digest is invalid")
        if (no_progress_model_turns == 0) != (digest is None):
            raise ValueError(
                "model guard observation count and digest are inconsistent"
            )
        trip_reason = state.get("trip_reason")
        if trip_reason not in (None, "model_turn_budget", "no_progress"):
            raise ValueError("model guard trip reason is invalid")
        if trip_reason == "model_turn_budget" and model_turns < max_model_turns:
            raise ValueError("model turn budget trip has an invalid count")
        if (
            trip_reason == "no_progress"
            and no_progress_model_turns < no_progress_model_turn_threshold
        ):
            raise ValueError("no-progress trip has an invalid count")

        breaker._model_turns = model_turns
        breaker._no_progress_model_turns = no_progress_model_turns
        breaker._last_observation_digest = digest
        breaker._pending_state_digest = None
        breaker._salt = salt
        breaker._trip_reason = trip_reason
        return breaker

    def to_state(self) -> Mapping[str, Any]:
        """Return durable salted hashes and counters without raw model data."""

        return {
            "version": _STATE_VERSION,
            "model_turns": self._model_turns,
            "no_progress_model_turns": self._no_progress_model_turns,
            "salt": self._salt.hex(),
            "last_observation_digest": self._last_observation_digest,
            "trip_reason": self._trip_reason,
        }

    def before_model_attempt(
        self,
        semantic_state: Mapping[str, Any],
    ) -> ModelGuardSnapshot:
        """Reserve one model turn and retain only the state's digest."""

        self._raise_if_tripped()
        if not isinstance(semantic_state, Mapping):
            raise TypeError("semantic_state must be a mapping")
        if self._pending_state_digest is not None:
            raise RuntimeError(
                "previous model attempt requires a response or abandonment"
            )
        if self._model_turns >= self._max_model_turns:
            self._trip_reason = "model_turn_budget"
            raise ModelTurnBudgetExceeded(self.snapshot)

        state_digest = _fingerprint_json_value(
            semantic_state,
            field_name="semantic_state",
        )
        self._model_turns += 1
        self._pending_state_digest = state_digest
        return self.snapshot

    def observe_model_response(
        self,
        response: ModelResponse,
    ) -> ModelGuardSnapshot:
        """Record one completed attempt and trip on a repeated observation."""

        self._raise_if_tripped()
        state_digest = self._pending_state_digest
        if state_digest is None:
            raise RuntimeError("model response has no matching attempt")
        if not isinstance(response, ModelResponse):
            raise TypeError("response must be a ModelResponse")

        response_digest = _model_response_fingerprint(response)
        observation_digest = hmac.new(
            self._salt,
            f"{state_digest}\0{response_digest}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        self._pending_state_digest = None

        if observation_digest == self._last_observation_digest:
            self._no_progress_model_turns += 1
        else:
            self._last_observation_digest = observation_digest
            self._no_progress_model_turns = 1

        if self._no_progress_model_turns >= self._no_progress_model_turn_threshold:
            self._trip_reason = "no_progress"
            raise ModelNoProgressError(self.snapshot)
        return self.snapshot

    def abandon_model_attempt(self) -> ModelGuardSnapshot:
        """Close an attempt that failed before producing a normalized response."""

        self._pending_state_digest = None
        return self.snapshot

    def mark_progress(self) -> ModelGuardSnapshot:
        """Reset only no-progress history while preserving the turn budget."""

        self._last_observation_digest = None
        self._no_progress_model_turns = 0
        if self._trip_reason == "no_progress":
            self._trip_reason = None
        return self.snapshot

    def reset(self) -> ModelGuardSnapshot:
        """Reset the complete breaker for a new run."""

        self._salt = secrets.token_bytes(_SALT_BYTES)
        self._model_turns = 0
        self._no_progress_model_turns = 0
        self._pending_state_digest: str | None = None
        self._last_observation_digest: str | None = None
        self._trip_reason: GuardTripReason | None = None
        return self.snapshot

    def _raise_if_tripped(self) -> None:
        if self._trip_reason == "model_turn_budget":
            raise ModelTurnBudgetExceeded(self.snapshot)
        if self._trip_reason == "no_progress":
            raise ModelNoProgressError(self.snapshot)


def _model_response_fingerprint(response: ModelResponse) -> str:
    calls = response.tool_calls or response.message.tool_calls
    semantic_response = {
        "content": response.message.content,
        "finish_reason": response.finish_reason,
        # Provider-generated call IDs are deliberately excluded: a model that
        # emits the same Tool name and arguments has made the same decision.
        "tool_calls": [
            {
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in calls
        ],
    }
    return _fingerprint_json_value(
        semantic_response,
        field_name="response",
    )


def _fingerprint_json_value(value: Any, *, field_name: str) -> str:
    normalized = _normalize_json_value(value, field_name=field_name)
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_json_value(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{field_name} must contain finite JSON values")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} mapping keys must be strings")
        return {
            key: _normalize_json_value(
                item,
                field_name=field_name,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_normalize_json_value(item, field_name=field_name) for item in value]
    raise TypeError(f"{field_name} must contain only JSON-like values")


def _validate_positive_integer(value: Any, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")


def _non_negative_integer(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return value


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ModelGuardSnapshot",
    "ModelGuardTripped",
    "ModelNoProgressError",
    "ModelTurnBudgetExceeded",
    "NoProgressCircuitBreaker",
]

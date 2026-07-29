from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from moduagent.tools.arguments import is_tool_argument_fingerprint
from moduagent.tools.base import (
    ToolError,
    ToolErrorType,
    ToolRecoveryAction,
)

_STABLE_REASON_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


@dataclass(frozen=True, slots=True)
class ToolSafetyProfile:
    """The automatic recovery operations proven safe for one Tool.

    This profile does not claim transactionality or exactly-once execution. It
    only controls which recovery operations ModuAgent may initiate.
    """

    same_call_retry_safe: bool = False
    changed_argument_repair_safe: bool = False
    timeout_retry_safe: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "same_call_retry_safe",
            "changed_argument_repair_safe",
            "timeout_retry_safe",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")


@dataclass(frozen=True, slots=True)
class ToolFailureClassification:
    """Stable, policy-facing classification of a Tool invocation failure."""

    error_type: ToolErrorType
    stable_reason: str
    retryable: bool = False
    recovery_directive: ToolRecoveryAction | None = None
    safe_message: str | None = None
    diagnostic_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.error_type, ToolErrorType):
            object.__setattr__(
                self,
                "error_type",
                ToolErrorType(str(self.error_type)),
            )
        reason = str(self.stable_reason).strip()
        if not reason:
            raise ValueError("stable_reason cannot be empty")
        if _STABLE_REASON_PATTERN.fullmatch(reason) is None:
            raise ValueError(
                "stable_reason must be a machine-readable code "
                "(letters, digits, '_', '.', ':', '-' only; max 128)"
            )
        object.__setattr__(self, "stable_reason", reason)
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be a bool")
        if self.recovery_directive is not None and not isinstance(
            self.recovery_directive,
            ToolRecoveryAction,
        ):
            object.__setattr__(
                self,
                "recovery_directive",
                ToolRecoveryAction(str(self.recovery_directive)),
            )
        for field_name in ("safe_message", "diagnostic_ref"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")


@dataclass(frozen=True, slots=True)
class InternalToolFailure:
    """Ephemeral Tool failure record used inside the execution boundary.

    Raw exceptions and raw arguments are deliberately absent. ``diagnostic_ref``
    is an opaque local reference and must never be projected to model, public,
    event, trace, or checkpoint payloads.
    """

    call_id: str
    tool_name: str
    classification: ToolFailureClassification
    safety_profile: ToolSafetyProfile
    requested_arguments_fingerprint: str | None = None
    effective_arguments_fingerprint: str | None = None
    attempts: int = 0
    same_call_retry_exhausted: bool = False
    diagnostic_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("failure call_id cannot be empty")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("failure tool_name cannot be empty")
        if not isinstance(self.classification, ToolFailureClassification):
            raise TypeError("classification must be a ToolFailureClassification")
        if not isinstance(self.safety_profile, ToolSafetyProfile):
            raise TypeError("safety_profile must be a ToolSafetyProfile")
        for field_name in (
            "requested_arguments_fingerprint",
            "effective_arguments_fingerprint",
        ):
            value = getattr(self, field_name)
            if value is not None and not is_tool_argument_fingerprint(value):
                raise ValueError(f"{field_name} must use sha256")
        if type(self.attempts) is not int:
            raise TypeError("attempts must be an integer")
        if self.attempts < 0:
            raise ValueError("attempts cannot be negative")
        if type(self.same_call_retry_exhausted) is not bool:
            raise TypeError("same_call_retry_exhausted must be a bool")
        if self.diagnostic_ref is not None and not isinstance(
            self.diagnostic_ref,
            str,
        ):
            raise TypeError("diagnostic_ref must be a string")


@dataclass(frozen=True, slots=True)
class SafeToolFailureView:
    """Bounded failure data safe for model and persistence boundaries."""

    error_type: ToolErrorType
    reason: str
    recovery: ToolRecoveryAction | None
    retryable: bool
    call_id: str
    tool_name: str
    requested_arguments_fingerprint: str | None = None
    effective_arguments_fingerprint: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.error_type, ToolErrorType):
            object.__setattr__(
                self,
                "error_type",
                ToolErrorType(str(self.error_type)),
            )
        if not self.reason:
            raise ValueError("failure reason cannot be empty")
        if self.recovery is not None and not isinstance(
            self.recovery,
            ToolRecoveryAction,
        ):
            object.__setattr__(
                self,
                "recovery",
                ToolRecoveryAction(str(self.recovery)),
            )
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be a bool")
        if not self.call_id or not self.tool_name:
            raise ValueError("failure identifiers cannot be empty")
        for field_name in (
            "requested_arguments_fingerprint",
            "effective_arguments_fingerprint",
        ):
            value = getattr(self, field_name)
            if value is not None and not is_tool_argument_fingerprint(value):
                raise ValueError(f"{field_name} must use sha256")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": self.error_type.value,
            "reason": self.reason,
            "retryable": self.retryable,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
        }
        if self.recovery is not None:
            value["recovery"] = self.recovery.value
        if self.requested_arguments_fingerprint is not None:
            value["arguments_fingerprint"] = self.requested_arguments_fingerprint
        if self.effective_arguments_fingerprint is not None:
            value["invocation_fingerprint"] = self.effective_arguments_fingerprint
        if self.message is not None:
            value["message"] = self.message
        return value


class FailureProjector:
    """Project internal failures through one bounded, model-safe boundary."""

    def __init__(
        self,
        *,
        max_identifier_chars: int = 256,
        max_reason_chars: int = 256,
        max_message_chars: int = 512,
    ) -> None:
        for field_name, value in (
            ("max_identifier_chars", max_identifier_chars),
            ("max_reason_chars", max_reason_chars),
            ("max_message_chars", max_message_chars),
        ):
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if value < 1:
                raise ValueError(f"{field_name} must be at least 1")
        self.max_identifier_chars = max_identifier_chars
        self.max_reason_chars = max_reason_chars
        self.max_message_chars = max_message_chars

    @staticmethod
    def _bounded_text(value: Any, *, limit: int, fallback: str) -> str:
        try:
            raw = str(value)
        except Exception:
            raw = fallback
        text = " ".join(
            "".join(
                character if character.isprintable() else " " for character in raw
            ).split()
        )
        text = text or fallback
        if len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        return f"{text[: limit - 3]}..."

    def project(
        self,
        failure: InternalToolFailure,
        *,
        include_safe_message: bool = False,
    ) -> SafeToolFailureView:
        if not isinstance(failure, InternalToolFailure):
            raise TypeError("failure must be an InternalToolFailure")
        if type(include_safe_message) is not bool:
            raise TypeError("include_safe_message must be a bool")
        classification = failure.classification
        message = None
        if include_safe_message and classification.safe_message is not None:
            message = self._bounded_text(
                classification.safe_message,
                limit=self.max_message_chars,
                fallback="Tool execution failed",
            )
        return SafeToolFailureView(
            error_type=classification.error_type,
            reason=self._bounded_text(
                classification.stable_reason,
                limit=self.max_reason_chars,
                fallback=classification.error_type.value,
            ),
            recovery=classification.recovery_directive,
            retryable=classification.retryable,
            call_id=self._bounded_text(
                failure.call_id,
                limit=self.max_identifier_chars,
                fallback="unknown_call",
            ),
            tool_name=self._bounded_text(
                failure.tool_name,
                limit=self.max_identifier_chars,
                fallback="unknown_tool",
            ),
            requested_arguments_fingerprint=(failure.requested_arguments_fingerprint),
            effective_arguments_fingerprint=(failure.effective_arguments_fingerprint),
            message=message,
        )


def resolve_tool_safety_profile(tool: Any) -> ToolSafetyProfile:
    """Resolve an explicit profile or adapt the 0.3.2 boolean capabilities."""

    explicit = getattr(tool, "safety_profile", None)
    if explicit is not None:
        if not isinstance(explicit, ToolSafetyProfile):
            raise TypeError("tool safety_profile must be a ToolSafetyProfile")
        return explicit
    return ToolSafetyProfile(
        same_call_retry_safe=bool(getattr(tool, "idempotent", False)),
        changed_argument_repair_safe=bool(getattr(tool, "repair_safe", False)),
        timeout_retry_safe=bool(getattr(tool, "timeout_retry_safe", False)),
    )


def classification_from_tool_error(
    error: ToolError,
    *,
    diagnostic_ref: str | None = None,
) -> ToolFailureClassification:
    """Adapt the 0.3.2 ``ToolError`` contract without exposing its details."""

    if not isinstance(error, ToolError):
        raise TypeError("error must be a ToolError")
    reason = error.reason or error.type.value
    if _STABLE_REASON_PATTERN.fullmatch(reason) is None:
        reason = error.type.value
    return ToolFailureClassification(
        error_type=error.type,
        stable_reason=reason,
        retryable=error.retryable,
        recovery_directive=error.recovery,
        safe_message=error.message,
        diagnostic_ref=diagnostic_ref,
    )


def tool_error_from_classification(
    classification: ToolFailureClassification,
) -> ToolError:
    """Create a backward-compatible ToolError from a new classification."""

    if not isinstance(classification, ToolFailureClassification):
        raise TypeError("classification must be a ToolFailureClassification")
    message = classification.safe_message or {
        ToolErrorType.NOT_FOUND: "unknown tool",
        ToolErrorType.INVALID_ARGUMENTS: "invalid tool arguments",
        ToolErrorType.UNAUTHORIZED: "tool authorization failed",
        ToolErrorType.TIMEOUT: "tool timed out",
        ToolErrorType.RESULT_TOO_LARGE: "tool result is too large",
        ToolErrorType.CANCELLED: "tool execution was cancelled",
    }.get(classification.error_type, "tool execution failed")
    return ToolError(
        classification.error_type,
        message,
        retryable=classification.retryable,
        reason=classification.stable_reason,
        recovery=classification.recovery_directive,
    )

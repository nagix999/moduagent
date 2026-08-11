from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import ContextMemorySerializationError


CONTEXT_SUMMARY_SCHEMA_VERSION = 2
MAX_SUMMARY_SOURCE_MESSAGE_IDS = 256
MAX_CONVERSATION_SUMMARY_TEXT_BYTES = 128 * 1_024
MAX_CONVERSATION_SUMMARY_FIELD_ITEMS = 128
MAX_CONVERSATION_SUMMARY_ITEM_BYTES = 1_024
MAX_CONVERSATION_SUMMARY_SERIALIZED_BYTES = 256 * 1_024
MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES = 1_024 * 1_024
_STORAGE_KEY_PREFIX = "summary-v2"
_MAX_IDENTIFIER_CHARS = 512
_SUMMARY_FIELDS = (
    "facts",
    "decisions",
    "preferences",
    "open_items",
    "tool_observations",
)


@dataclass(frozen=True, slots=True)
class MemoryStateKey:
    """Tenant-bound identity for one Context Memory policy state."""

    tenant_id: str
    agent_id: str
    session_id: str
    policy_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "agent_id",
            "session_id",
            "policy_fingerprint",
        ):
            _validate_identifier(getattr(self, name), name)

    def to_storage_key(self) -> str:
        """Return a reversible, delimiter-safe composite storage key."""

        components = (
            self.tenant_id,
            self.agent_id,
            self.session_id,
            self.policy_fingerprint,
        )
        return ".".join((_STORAGE_KEY_PREFIX, *map(_encode_component, components)))

    @classmethod
    def from_storage_key(cls, value: str) -> MemoryStateKey:
        if not isinstance(value, str):
            raise TypeError("storage key must be a string")
        parts = value.split(".")
        if len(parts) != 5 or parts[0] != _STORAGE_KEY_PREFIX:
            raise ValueError("invalid Context Memory storage key")
        try:
            components = tuple(_decode_component(part) for part in parts[1:])
        except (UnicodeDecodeError, binascii.Error, ValueError) as exc:
            raise ValueError("invalid Context Memory storage key") from exc
        return cls(*components)


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """Structured, instruction-free, size-bounded conversation summary.

    Limits are measured on UTF-8 bytes, not Python character counts. They keep
    custom or compromised summarizers from persisting an unbounded value before
    request-level token fitting runs.
    """

    summary: str
    facts: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    open_items: tuple[str, ...] = ()
    tool_observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(
            self.summary,
            "summary",
            max_bytes=MAX_CONVERSATION_SUMMARY_TEXT_BYTES,
        )
        for name in _SUMMARY_FIELDS:
            object.__setattr__(
                self,
                name,
                _validated_string_tuple(
                    getattr(self, name),
                    name,
                    max_items=MAX_CONVERSATION_SUMMARY_FIELD_ITEMS,
                    max_item_bytes=MAX_CONVERSATION_SUMMARY_ITEM_BYTES,
                ),
            )
        _validate_serialized_size(
            self.to_dict(),
            name="structured conversation summary",
            max_bytes=MAX_CONVERSATION_SUMMARY_SERIALIZED_BYTES,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            **{name: list(getattr(self, name)) for name in _SUMMARY_FIELDS},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConversationSummary:
        if not isinstance(value, Mapping):
            raise TypeError("structured_summary must be an object")
        expected = {"summary", *_SUMMARY_FIELDS}
        _require_exact_fields(value, expected, "structured_summary")
        return cls(
            summary=value["summary"],
            **{name: value[name] for name in _SUMMARY_FIELDS},
        )


@dataclass(frozen=True, slots=True)
class ConversationSummarySnapshot:
    """Versioned v2 summary state protected by a composite key and CAS.

    ``source_message_ids`` is ordered. Up to the fixed cap it contains a
    contiguous suffix ending at ``covered_through_sequence``; once capped it
    contains the first-message anchor followed by the newest IDs. The cursor
    and chained prefix digest represent the complete compacted prefix, while
    this bounded set supports provenance sampling and reset detection without
    making snapshot size grow with session length.
    """

    tenant_id: str
    agent_id: str
    session_id: str
    policy_fingerprint: str
    covered_through_sequence: int
    covered_prefix_digest: str
    structured_summary: ConversationSummary
    source_message_ids: tuple[str, ...]
    version: int
    summary_schema_version: int = field(
        default=CONTEXT_SUMMARY_SCHEMA_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        # Constructing the key applies the same validation used at the store boundary.
        _ = self.key
        if (
            type(self.covered_through_sequence) is not int
            or self.covered_through_sequence < 1
        ):
            raise ValueError("covered_through_sequence must be a positive integer")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer")
        _validate_identifier(self.covered_prefix_digest, "covered_prefix_digest")
        if not isinstance(self.structured_summary, ConversationSummary):
            raise TypeError("structured_summary must be a ConversationSummary")
        identifiers = _validated_string_tuple(
            self.source_message_ids,
            "source_message_ids",
        )
        if not identifiers:
            raise ValueError("source_message_ids cannot be empty")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("source_message_ids must be unique")
        if len(identifiers) > MAX_SUMMARY_SOURCE_MESSAGE_IDS:
            raise ValueError(
                "source_message_ids cannot exceed "
                f"{MAX_SUMMARY_SOURCE_MESSAGE_IDS} entries"
            )
        object.__setattr__(self, "source_message_ids", identifiers)

    @property
    def key(self) -> MemoryStateKey:
        return MemoryStateKey(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            policy_fingerprint=self.policy_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_schema_version": self.summary_schema_version,
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "policy_fingerprint": self.policy_fingerprint,
            "covered_through_sequence": self.covered_through_sequence,
            "covered_prefix_digest": self.covered_prefix_digest,
            "structured_summary": self.structured_summary.to_dict(),
            "source_message_ids": list(self.source_message_ids),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConversationSummarySnapshot:
        if not isinstance(value, Mapping):
            raise TypeError("Context Memory snapshot must be an object")
        expected = {
            "summary_schema_version",
            "tenant_id",
            "agent_id",
            "session_id",
            "policy_fingerprint",
            "covered_through_sequence",
            "covered_prefix_digest",
            "structured_summary",
            "source_message_ids",
            "version",
        }
        _require_exact_fields(value, expected, "Context Memory snapshot")
        schema_version = value["summary_schema_version"]
        if schema_version != CONTEXT_SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Context Memory summary schema version: {schema_version!r}"
            )
        return cls(
            tenant_id=value["tenant_id"],
            agent_id=value["agent_id"],
            session_id=value["session_id"],
            policy_fingerprint=value["policy_fingerprint"],
            covered_through_sequence=value["covered_through_sequence"],
            covered_prefix_digest=value["covered_prefix_digest"],
            structured_summary=ConversationSummary.from_dict(
                value["structured_summary"]
            ),
            source_message_ids=value["source_message_ids"],
            version=value["version"],
        )


def encode_summary_snapshot(snapshot: ConversationSummarySnapshot) -> str:
    if not isinstance(snapshot, ConversationSummarySnapshot):
        raise TypeError("snapshot must be a ConversationSummarySnapshot")
    payload = json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        _validate_serialized_size(
            payload,
            name="Context Memory summary snapshot",
            max_bytes=MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES,
        )
    except ValueError as exc:
        raise ContextMemorySerializationError(str(exc)) from exc
    return payload


def decode_summary_snapshot(
    payload: str | bytes | bytearray | Mapping[str, Any],
) -> ConversationSummarySnapshot:
    """Decode one strict v2 snapshot without accepting partial state."""

    try:
        if isinstance(payload, Mapping):
            value: Any = payload
        else:
            if isinstance(payload, (bytes, bytearray)):
                if len(payload) > MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES:
                    raise ContextMemorySerializationError(
                        "Context Memory summary snapshot exceeds "
                        f"{MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES} UTF-8 bytes"
                    )
                payload = payload.decode("utf-8")
            if not isinstance(payload, str):
                raise TypeError("snapshot payload must be JSON text or an object")
            try:
                _validate_serialized_size(
                    payload,
                    name="Context Memory summary snapshot",
                    max_bytes=MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES,
                )
            except ValueError as exc:
                raise ContextMemorySerializationError(str(exc)) from exc
            value = json.loads(payload)
        return ConversationSummarySnapshot.from_dict(value)
    except ContextMemorySerializationError:
        raise
    except Exception as exc:
        raise ContextMemorySerializationError(
            "invalid Context Memory summary snapshot"
        ) from exc


def _validate_identifier(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} cannot be empty or padded")
    if len(value) > _MAX_IDENTIFIER_CHARS:
        raise ValueError(f"{name} exceeds {_MAX_IDENTIFIER_CHARS} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} cannot contain control characters")


def _validate_text(value: Any, name: str, *, max_bytes: int | None = None) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")
    if max_bytes is not None and (
        len(value) > max_bytes or len(value.encode("utf-8")) > max_bytes
    ):
        raise ValueError(f"{name} exceeds {max_bytes} UTF-8 bytes")


def _validated_string_tuple(
    value: Any,
    name: str,
    *,
    max_items: int | None = None,
    max_item_bytes: int | None = None,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if max_items is not None and len(value) > max_items:
        raise ValueError(f"{name} cannot exceed {max_items} entries")
    items = tuple(value)
    for item in items:
        _validate_identifier(item, f"{name} item")
        if max_item_bytes is not None and (
            len(item) > max_item_bytes or len(item.encode("utf-8")) > max_item_bytes
        ):
            raise ValueError(f"{name} item exceeds {max_item_bytes} UTF-8 bytes")
    return items


def _validate_serialized_size(
    value: str | Mapping[str, Any],
    *,
    name: str,
    max_bytes: int,
) -> None:
    payload = (
        value
        if isinstance(value, str)
        else json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if len(payload) > max_bytes or len(payload.encode("utf-8")) > max_bytes:
        raise ValueError(f"{name} exceeds {max_bytes} UTF-8 bytes")


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields mismatch; missing={missing}, extra={extra}")


def _encode_component(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).rstrip(b"=").decode("ascii")


def _decode_component(value: str) -> str:
    if not value:
        raise ValueError("empty key component")
    padding = "=" * (-len(value) % 4)
    raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    return raw.decode("utf-8")


__all__ = [
    "CONTEXT_SUMMARY_SCHEMA_VERSION",
    "MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES",
    "MAX_CONVERSATION_SUMMARY_FIELD_ITEMS",
    "MAX_CONVERSATION_SUMMARY_ITEM_BYTES",
    "MAX_CONVERSATION_SUMMARY_SERIALIZED_BYTES",
    "MAX_CONVERSATION_SUMMARY_TEXT_BYTES",
    "MAX_SUMMARY_SOURCE_MESSAGE_IDS",
    "ConversationSummary",
    "ConversationSummarySnapshot",
    "MemoryStateKey",
    "decode_summary_snapshot",
    "encode_summary_snapshot",
]

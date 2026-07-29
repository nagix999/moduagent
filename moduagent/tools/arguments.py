from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from moduagent.tools.base import canonical_tool_arguments_fingerprint


def fingerprint_tool_arguments(arguments: Mapping[str, Any]) -> str:
    """Return a canonical one-way fingerprint for Tool arguments.

    The raw argument values are not retained. Mapping key order does not affect
    the result, and values use the same JSON normalization as Tool results.
    """

    if not isinstance(arguments, Mapping):
        raise TypeError("tool arguments must be a mapping")
    return canonical_tool_arguments_fingerprint(arguments)


def is_tool_argument_fingerprint(value: object) -> bool:
    """Return whether *value* is a canonical SHA-256 argument fingerprint."""

    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )

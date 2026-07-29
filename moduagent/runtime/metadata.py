from __future__ import annotations


_RUNTIME_OWNED_METADATA_KEYS = frozenset(
    {
        "agent",
        "error_summary",
        "failure",
        "plan",
        "plan_usage",
        "skill_usage",
        "skills",
        "tool_trace",
    }
)


def is_runtime_owned_metadata_key(key: str) -> bool:
    """Return whether a metadata key belongs to the framework boundary."""

    return (
        key in _RUNTIME_OWNED_METADATA_KEYS
        or key.startswith("_moduagent_")
        or key.startswith("moduagent.")
    )


__all__ = ["is_runtime_owned_metadata_key"]

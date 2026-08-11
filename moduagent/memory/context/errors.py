from __future__ import annotations

from moduagent.errors import MemoryError as FrameworkMemoryError


class ContextMemoryError(FrameworkMemoryError):
    """Base error for the durable Context Memory contract."""


class ContextMemoryIntegrityError(ContextMemoryError):
    """Stored state or an attempted transition violates an integrity rule."""


class ContextMemoryCursorRegressionError(ContextMemoryIntegrityError):
    """A summary attempted to move its covered message cursor backwards."""


class ContextMemorySerializationError(ContextMemoryIntegrityError):
    """A persisted Context Memory payload cannot be decoded safely."""


class ContextHistoryLoadError(ContextMemoryError):
    """A bounded conversation history view cannot be loaded safely."""


class ContextHistoryPaginationRequiredError(ContextHistoryLoadError):
    """The conversation backend cannot prove that tail reads are bounded."""


class ContextHistoryTailOverflowError(ContextHistoryLoadError):
    """Uncompacted messages exceed the configured bounded-read limit."""

    def __init__(self, *, max_tail_messages: int, after_sequence: int) -> None:
        self.max_tail_messages = max_tail_messages
        self.after_sequence = after_sequence
        super().__init__(
            "uncompacted conversation tail exceeds "
            f"max_tail_messages={max_tail_messages} after sequence "
            f"{after_sequence}; compact or migrate the session before retrying"
        )


class ContextHistoryCursorInvalidatedError(ContextHistoryLoadError):
    """A stored summary cursor no longer identifies the same message prefix."""


class ContextMemoryWriteConflictError(ContextMemoryError):
    """A CAS loser could not safely reuse the concurrently written summary."""


class ContextAssemblyError(ContextMemoryError):
    """Base error raised while allocating a bounded model context."""


class ContextBudgetExceededError(ContextAssemblyError):
    """Required Context items cannot fit in the available token budget."""

    def __init__(self, *, required_tokens: int, available_tokens: int) -> None:
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens
        super().__init__(
            "required Context items need "
            f"{required_tokens} tokens but only {available_tokens} are available"
        )


__all__ = [
    "ContextAssemblyError",
    "ContextBudgetExceededError",
    "ContextHistoryLoadError",
    "ContextHistoryCursorInvalidatedError",
    "ContextHistoryPaginationRequiredError",
    "ContextHistoryTailOverflowError",
    "ContextMemoryCursorRegressionError",
    "ContextMemoryError",
    "ContextMemoryIntegrityError",
    "ContextMemorySerializationError",
    "ContextMemoryWriteConflictError",
]

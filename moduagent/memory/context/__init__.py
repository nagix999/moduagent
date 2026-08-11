"""Durable short-term Context Memory primitives introduced for 0.6."""

from .assembler import (
    AssembledContextItem,
    ContextAssembler,
    ContextAssemblyResult,
    ContextItem,
)
from .errors import (
    ContextAssemblyError,
    ContextBudgetExceededError,
    ContextHistoryCursorInvalidatedError,
    ContextHistoryLoadError,
    ContextHistoryPaginationRequiredError,
    ContextHistoryTailOverflowError,
    ContextMemoryCursorRegressionError,
    ContextMemoryError,
    ContextMemoryIntegrityError,
    ContextMemorySerializationError,
    ContextMemoryWriteConflictError,
)
from .history import (
    ContextHistoryLoader,
    ContextHistoryLoadingPolicy,
    ContextHistoryView,
    DurableContextHistoryLoader,
)
from .migration import ScopedLegacyMemoryStateStore, migrate_memory_snapshot
from .models import (
    CONTEXT_SUMMARY_SCHEMA_VERSION,
    MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES,
    MAX_CONVERSATION_SUMMARY_FIELD_ITEMS,
    MAX_CONVERSATION_SUMMARY_ITEM_BYTES,
    MAX_CONVERSATION_SUMMARY_SERIALIZED_BYTES,
    MAX_CONVERSATION_SUMMARY_TEXT_BYTES,
    MAX_SUMMARY_SOURCE_MESSAGE_IDS,
    ConversationSummary,
    ConversationSummarySnapshot,
    MemoryStateKey,
    decode_summary_snapshot,
    encode_summary_snapshot,
)
from .policies import DurableSummarizingConversationMemoryPolicy
from .stores import (
    ContextMemoryStateStore,
    DatabaseMemoryStateStore,
    InMemoryContextMemoryStateStore,
    MemoryStateRepository,
    RedisMemoryStateStore,
)

# The 0.6 Context Memory namespace uses the durable cursor/CAS implementation.
# The original class remains available unchanged from ``moduagent.memory`` for
# 0.x source compatibility.
SummarizingConversationMemoryPolicy = DurableSummarizingConversationMemoryPolicy

__all__ = [
    "CONTEXT_SUMMARY_SCHEMA_VERSION",
    "MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES",
    "MAX_CONVERSATION_SUMMARY_FIELD_ITEMS",
    "MAX_CONVERSATION_SUMMARY_ITEM_BYTES",
    "MAX_CONVERSATION_SUMMARY_SERIALIZED_BYTES",
    "MAX_CONVERSATION_SUMMARY_TEXT_BYTES",
    "MAX_SUMMARY_SOURCE_MESSAGE_IDS",
    "AssembledContextItem",
    "ContextAssembler",
    "ContextAssemblyError",
    "ContextAssemblyResult",
    "ContextBudgetExceededError",
    "ContextHistoryCursorInvalidatedError",
    "ContextHistoryLoadError",
    "ContextHistoryLoader",
    "ContextHistoryLoadingPolicy",
    "ContextHistoryPaginationRequiredError",
    "ContextHistoryTailOverflowError",
    "ContextHistoryView",
    "ContextItem",
    "ContextMemoryCursorRegressionError",
    "ContextMemoryError",
    "ContextMemoryIntegrityError",
    "ContextMemorySerializationError",
    "ContextMemoryWriteConflictError",
    "ContextMemoryStateStore",
    "ConversationSummary",
    "ConversationSummarySnapshot",
    "DatabaseMemoryStateStore",
    "DurableContextHistoryLoader",
    "DurableSummarizingConversationMemoryPolicy",
    "InMemoryContextMemoryStateStore",
    "MemoryStateKey",
    "MemoryStateRepository",
    "RedisMemoryStateStore",
    "ScopedLegacyMemoryStateStore",
    "SummarizingConversationMemoryPolicy",
    "decode_summary_snapshot",
    "encode_summary_snapshot",
    "migrate_memory_snapshot",
]

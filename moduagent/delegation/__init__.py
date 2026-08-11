"""Typed, budgeted Agent-to-Agent delegation primitives.

This package intentionally does not import the public :class:`moduagent.Agent`
facade. Runtime integrations register private ``AgentEndpoint`` adapters.
"""

from moduagent.definitions import AgentDescriptor, AgentRef

from .budget import (
    BudgetExceeded,
    BudgetLedger,
    BudgetStateStore,
    DurableBudgetLedger,
    ExecutionGroupBudgetState,
    InMemoryBudgetLedger,
    InMemoryBudgetStateStore,
    StoreBackedBudgetLedger,
)
from .coordinator import (
    DelegationCall,
    DelegationCoordinator,
    DelegationFailure,
)
from .events import (
    DelegationEvent,
    DelegationEventSink,
    DelegationEventType,
    NoopDelegationEventSink,
)
from .guards import (
    CycleGuard,
    DelegationAuthorizer,
    DelegationDecision,
    DelegationPolicy,
    DelegationRejected,
    EdgeDelegationAuthorizer,
)
from .models import (
    BudgetLease,
    DelegationContext,
    DelegationOutcome,
    DelegationOutcomeStatus,
    ExecutionGroupLimits,
    ParentDelegationContext,
    RunLineage,
)
from .receipts import (
    DelegationIdFactory,
    DelegationReceipt,
    DelegationReceiptStatus,
    DelegationReceiptStore,
    InMemoryDelegationReceiptStore,
    ReceiptAction,
    ReceiptClaim,
    ReceiptManager,
    ReceiptStoreError,
    canonical_digest,
    receipt_action,
)
from .registry import (
    AgentEndpoint,
    AgentRegistry,
    AgentRegistryError,
    DelegatedAgentEndpointHandler,
    DelegatedCheckpointCleaner,
    DelegationEndpointError,
    InMemoryAgentRegistry,
    LocalAgentInvoker,
    ReconciliableAgentEndpoint,
    ResolvedAgentEndpoint,
    ResumableAgentEndpoint,
)
from .sessions import SessionKeyFactory, SessionStrategy
from .tool import (
    DELEGATION_EVENT_CALLBACK_KEY,
    PARENT_DELEGATION_CONTEXT_KEY,
    DelegatedAgentTool,
    ParentContextResolver,
    ToolMetadataParentContextResolver,
)

__all__ = [
    "DELEGATION_EVENT_CALLBACK_KEY",
    "PARENT_DELEGATION_CONTEXT_KEY",
    "AgentDescriptor",
    "AgentEndpoint",
    "AgentRef",
    "AgentRegistry",
    "AgentRegistryError",
    "BudgetExceeded",
    "BudgetLease",
    "BudgetLedger",
    "BudgetStateStore",
    "CycleGuard",
    "DelegatedAgentTool",
    "DelegatedAgentEndpointHandler",
    "DelegatedCheckpointCleaner",
    "DelegationAuthorizer",
    "DelegationCall",
    "DelegationContext",
    "DelegationCoordinator",
    "DelegationDecision",
    "DelegationEvent",
    "DelegationEventSink",
    "DelegationEventType",
    "DelegationEndpointError",
    "DelegationFailure",
    "DelegationIdFactory",
    "DelegationOutcome",
    "DelegationOutcomeStatus",
    "DelegationPolicy",
    "DelegationReceipt",
    "DelegationReceiptStatus",
    "DelegationReceiptStore",
    "DelegationRejected",
    "DurableBudgetLedger",
    "EdgeDelegationAuthorizer",
    "ExecutionGroupBudgetState",
    "ExecutionGroupLimits",
    "InMemoryAgentRegistry",
    "InMemoryBudgetLedger",
    "InMemoryBudgetStateStore",
    "InMemoryDelegationReceiptStore",
    "LocalAgentInvoker",
    "NoopDelegationEventSink",
    "ParentContextResolver",
    "ParentDelegationContext",
    "ReceiptAction",
    "ReceiptClaim",
    "ReceiptManager",
    "ReceiptStoreError",
    "ReconciliableAgentEndpoint",
    "ResolvedAgentEndpoint",
    "ResumableAgentEndpoint",
    "RunLineage",
    "SessionKeyFactory",
    "SessionStrategy",
    "StoreBackedBudgetLedger",
    "ToolMetadataParentContextResolver",
    "canonical_digest",
    "receipt_action",
]

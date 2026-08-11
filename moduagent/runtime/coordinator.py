from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from moduagent.errors import (
    CancellationError,
    ConfigurationError,
    CheckpointNotFoundError,
    ExecutionInvariantError,
    MemoryError as FrameworkMemoryError,
    ModelInvocationError,
    ModuAgentError,
    OutputValidationError,
    PersistenceError,
    SkillError as FrameworkSkillError,
    StateMigrationError,
    ToolAuthorizationError,
    ToolInvocationError,
    ToolRecoveryError,
    ToolValidationError,
)
from moduagent.execution.base import (
    DurableBoundary,
    EngineContext,
    EngineEmission,
    EngineOutcome,
    EngineSnapshot,
    ExecutionEngine,
)
from moduagent.execution.standard import StandardExecutionEngine
from moduagent.delegation.budget import BudgetExceeded
from moduagent.messages import FinishReason, Message, MessageRole
from moduagent.memory import (
    ConversationMemoryOverflowError,
    MemoryIntegrityError,
)
from moduagent.models import (
    ModelCapabilities,
    ModelOutputIncompleteError,
    classify_model_error,
)
from moduagent.observability._background import run_in_daemon_thread
from moduagent.observability.sinks import (
    _event_sink_is_noop,
    _event_sink_requires_coordinator_copy,
)
from moduagent.persistence import RunCheckpoint, RunSnapshot
from moduagent.persistence.snapshot import identity_scope_digest
from moduagent.runtime.context import (
    AgentResult,
    RunContext,
    RunRequest,
    RunStatus,
)
from moduagent.runtime.events import (
    AgentEvent,
    EventPublisher,
    EventType,
    EventVisibility,
)
from moduagent.runtime.metadata import is_runtime_owned_metadata_key
from moduagent.runtime.model_guard import (
    ModelGuardTripped,
    ModelNoProgressError,
    ModelTurnBudgetExceeded,
)
from moduagent.runtime.runtime import AgentRuntime
from moduagent.runtime.services import RuntimeServices
from moduagent.skills.tools import SKILL_RESOURCE_TOOL_NAMES


_RUN_ID_METADATA_KEY = "moduagent.run_id"
_ENGINE_SNAPSHOT_POLICY_KEY = "_moduagent_engine_snapshot"
_ENGINE_INITIALIZED_POLICY_KEY = "_moduagent_engine_initialized"
_TERMINAL_EVENT_TYPES = frozenset({EventType.RUN_COMPLETED, EventType.RUN_FAILED})
_NON_RESUMABLE_GUARD_REASONS = frozenset(
    {
        FinishReason.MAX_MODEL_TURNS,
        FinishReason.NO_PROGRESS,
    }
)
_ENGINE_OUTCOME_RUNTIME_METADATA_KEYS = frozenset(
    {"failure", "plan", "plan_usage", "validation_failure"}
)
_STEP_VALIDATION_CODES = frozenset(
    {
        "step_result_incomplete",
        "step_result_tool_call_forbidden",
        "step_result_tool_call_invalid",
        "step_result_required",
        "step_result_schema_invalid",
        "step_result_id_mismatch",
        "step_result_max_attempts_exceeded",
        "step_validation_state_incomplete",
        "step_validator_failed",
        "step_validation_rejected",
        "step_validation_max_attempts_exceeded",
    }
)
_STEP_VALIDATION_LOCATIONS = frozenset({"act", "step_result", "step_validator"})
_PUBLIC_STREAM_EVENT_TYPES = frozenset(
    {
        EventType.MODEL_DELTA,
        EventType.FINAL_DELTA,
        *_TERMINAL_EVENT_TYPES,
    }
)
_EVENT_SINK_TIMEOUT_SECONDS = 0.25
_EVENT_SINK_MIN_DRAIN_SECONDS = 1.0
_EVENT_SINK_MAX_DRAIN_SECONDS = 15.0
_EVENT_SINK_QUEUE_MAX_SIZE = 1_024
_DESCRIPTOR_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "bearer",
        "bearer_token",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "set_cookie",
        "token",
    }
)
_DESCRIPTOR_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_refresh_token",
    "_secret",
    "_token",
)
_DESCRIPTOR_HEADER_KEYS = frozenset({"header", "headers", "http_headers"})
_PRIMARY_FAILURE_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """Retrieve a detached observability task result without surfacing it."""

    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:
        pass


def _optional_runtime_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_identity_claim(value: Any) -> str:
    claim = _optional_runtime_identifier(value)
    if claim is None or len(claim) > 256 or not claim.isprintable():
        raise ConfigurationError("trusted identity claim is missing or invalid")
    return claim


def _identity_from_mapping(
    value: Mapping[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        if key in value and value[key] is not None:
            return _required_identity_claim(value[key])
    return None


async def _resolve_identity_provider(
    provider: Any,
    user_context: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
) -> str | None:
    if provider is None:
        return None
    resolver = getattr(provider, "resolve", None)
    if not callable(resolver):
        resolver = provider if callable(provider) else None
    if resolver is None:
        raise ConfigurationError("trusted identity provider is invalid")
    try:
        signature = inspect.signature(resolver)
    except (TypeError, ValueError):
        value = resolver(dict(user_context))
    else:
        positional = tuple(
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        )
        value = resolver(dict(user_context)) if positional else resolver()
    if inspect.isawaitable(value):
        value = await value
    if isinstance(value, Mapping):
        claim = _identity_from_mapping(value, keys)
    else:
        claim = _required_identity_claim(value)
    if claim is None:
        raise ConfigurationError("trusted identity provider returned no claim")
    return claim


def _project_primary_failure(value: Any) -> dict[str, Any]:
    """Return only bounded, structured fields from an internal failure summary."""

    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    for key in ("component", "operation", "phase", "category", "code"):
        item = value.get(key)
        if isinstance(item, str) and _PRIMARY_FAILURE_LABEL.fullmatch(item):
            projected[key] = item
    failure_id = value.get("failure_id")
    if (
        isinstance(failure_id, str)
        and 0 < len(failure_id) <= 256
        and all(character.isprintable() for character in failure_id)
    ):
        projected["failure_id"] = failure_id
    step_id = value.get("step_id")
    if (
        isinstance(step_id, str)
        and 0 < len(step_id) <= 256
        and all(character.isprintable() for character in step_id)
    ):
        projected["step_id"] = step_id
    attempt = value.get("attempt")
    if type(attempt) is int and 0 <= attempt <= 1_000_000:
        projected["attempt"] = attempt
    retryable = value.get("retryable")
    if type(retryable) is bool:
        projected["retryable"] = retryable
    return projected


@dataclass(frozen=True, slots=True)
class _PublishedEventStamp:
    """Small compatibility record that never retains an event payload."""

    event_schema_version: int
    visibility: EventVisibility
    session_id: str | None
    engine_id: str | None
    sequence: int
    execution_group_id: str | None
    root_run_id: str | None
    parent_run_id: str | None
    child_run_id: str | None
    delegation_id: str | None
    agent_id: str | None
    agent_version: str | None
    depth: int

    @classmethod
    def from_event(cls, event: AgentEvent) -> "_PublishedEventStamp":
        return cls(
            event_schema_version=event.event_schema_version,
            visibility=event.visibility,
            session_id=event.session_id,
            engine_id=event.engine_id,
            sequence=event.sequence,
            execution_group_id=event.execution_group_id,
            root_run_id=event.root_run_id,
            parent_run_id=event.parent_run_id,
            child_run_id=event.child_run_id,
            delegation_id=event.delegation_id,
            agent_id=event.agent_id,
            agent_version=event.agent_version,
            depth=event.depth,
        )

    def apply(self, event: AgentEvent) -> AgentEvent:
        if (
            event.event_schema_version == self.event_schema_version
            and event.visibility is self.visibility
            and event.session_id == self.session_id
            and event.engine_id == self.engine_id
            and event.sequence == self.sequence
            and event.execution_group_id == self.execution_group_id
            and event.root_run_id == self.root_run_id
            and event.parent_run_id == self.parent_run_id
            and event.child_run_id == self.child_run_id
            and event.delegation_id == self.delegation_id
            and event.agent_id == self.agent_id
            and event.agent_version == self.agent_version
            and event.depth == self.depth
        ):
            return event
        return replace(
            event,
            event_schema_version=self.event_schema_version,
            visibility=self.visibility,
            session_id=self.session_id,
            engine_id=self.engine_id,
            sequence=self.sequence,
            execution_group_id=self.execution_group_id,
            root_run_id=self.root_run_id,
            parent_run_id=self.parent_run_id,
            child_run_id=self.child_run_id,
            delegation_id=self.delegation_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            depth=self.depth,
        )


def _normalize_descriptor_key(value: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(value).strip())
    return text.lower().replace("-", "_").replace(" ", "_")


def _descriptor_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _descriptor_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_descriptor_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _redact_descriptor(value: Any, *, key: str = "") -> Any:
    normalized = _normalize_descriptor_key(key)
    sensitive = normalized in _DESCRIPTOR_SENSITIVE_KEYS or normalized.endswith(
        _DESCRIPTOR_SENSITIVE_SUFFIXES
    )
    if sensitive:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        if normalized in _DESCRIPTOR_HEADER_KEYS:
            return {str(item_key): "[REDACTED]" for item_key in value}
        return {
            str(item_key): _redact_descriptor(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_descriptor(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        try:
            parsed = urlsplit(value)
        except ValueError:
            return value
        if not parsed.scheme or not parsed.netloc:
            return value
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        netloc = f"{hostname}{port}"
        query = urlencode(
            [
                (
                    query_key,
                    (
                        "[REDACTED]"
                        if (
                            _normalize_descriptor_key(query_key)
                            in _DESCRIPTOR_SENSITIVE_KEYS
                            or _normalize_descriptor_key(query_key).endswith(
                                _DESCRIPTOR_SENSITIVE_SUFFIXES
                            )
                        )
                        else query_value
                    ),
                )
                for query_key, query_value in parse_qsl(
                    parsed.query,
                    keep_blank_values=True,
                )
            ]
        )
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    return value


class RunCoordinator(AgentRuntime):
    """Engine-neutral owner of one Agent run's common lifecycle.

    ``AgentRuntime`` remains the 0.3 compatibility base and supplies the
    established Memory, Skill, persistence, trace and result adapters.
    Execution phases and transition rules live exclusively in the injected
    :class:`ExecutionEngine`.
    """

    def __init__(
        self,
        *,
        engine: ExecutionEngine[Any] | None = None,
        resolved_spec: Mapping[str, Any] | None = None,
        **runtime_options: Any,
    ) -> None:
        super().__init__(**runtime_options)
        resolved_engine: ExecutionEngine[Any] = (
            StandardExecutionEngine(self.decision_policy) if engine is None else engine
        )
        if not isinstance(resolved_engine, ExecutionEngine):
            raise TypeError("engine must implement ExecutionEngine")
        self.engine = resolved_engine
        spec = dict(resolved_spec or {})
        configured_engine = spec.get("engine_id")
        if configured_engine not in (None, resolved_engine.engine_id):
            raise ValueError("resolved_spec engine_id does not match engine")
        spec["engine_id"] = resolved_engine.engine_id
        spec.setdefault("state_version", resolved_engine.state_version)
        self.resolved_spec = spec

        # Run IDs are globally unique. Keeping these maps run-scoped lets one
        # Agent serve different sessions concurrently without sharing sequence
        # counters or event identity.
        self._event_publishers: dict[str, EventPublisher] = {}
        self._coordinator_contexts: dict[str, RunContext] = {}
        self._published_events: dict[tuple[str, str], _PublishedEventStamp] = {}
        self._reserved_events: dict[tuple[str, str], AgentEvent] = {}
        self._sink_queues: dict[str, asyncio.Queue[AgentEvent]] = {}
        self._sink_workers: dict[str, asyncio.Task[None]] = {}

    async def _resolve_trusted_identity(
        self,
        request: RunRequest,
    ) -> dict[str, str]:
        """Resolve run-owned tenant/principal claims outside model content.

        A delegated child inherits immutable claims from its parent. Root runs
        use configured providers when present and otherwise accept the
        application-owned ``user_context`` boundary in Development. Production
        composition requires providers, so prompt text can never supply these
        values.
        """

        incoming = request.delegation_context
        if incoming is not None:
            tenant = _required_identity_claim(getattr(incoming, "tenant", None))
            principal = _required_identity_claim(getattr(incoming, "principal", None))
            return {
                "_moduagent_tenant": tenant,
                "_moduagent_principal": principal,
                "_moduagent_tenant_scope_digest": identity_scope_digest(
                    "tenant",
                    tenant,
                ),
                "_moduagent_principal_scope_digest": identity_scope_digest(
                    "principal",
                    principal,
                ),
            }

        bindings = getattr(self, "runtime_bindings", None)
        tenant_provider = getattr(bindings, "tenant_context_provider", None)
        principal_provider = getattr(bindings, "principal_context_provider", None)
        tenant = await _resolve_identity_provider(
            tenant_provider,
            request.user_context,
            keys=("tenant_id", "tenant", "id"),
        )
        principal = await _resolve_identity_provider(
            principal_provider,
            request.user_context,
            keys=("principal_id", "principal", "id"),
        )
        if tenant is None:
            tenant = _identity_from_mapping(
                request.user_context,
                ("tenant_id", "tenant"),
            )
        if principal is None:
            principal = _identity_from_mapping(
                request.user_context,
                ("principal_id", "principal"),
            )
        result: dict[str, str] = {}
        if tenant is not None:
            result["_moduagent_tenant"] = tenant
            result["_moduagent_tenant_scope_digest"] = identity_scope_digest(
                "tenant",
                tenant,
            )
        if principal is not None:
            result["_moduagent_principal"] = principal
            result["_moduagent_principal_scope_digest"] = identity_scope_digest(
                "principal",
                principal,
            )
        return result

    def _validate_conversation_store_scope(
        self,
        trusted_identity: Mapping[str, str],
    ) -> None:
        """Bind conversation reads and writes to the trusted run scope.

        Tenant providers are resolved only at run time, and resume skips the
        normal history-loader bootstrap.  This common pre-read check prevents
        both paths from using a store bound to another tenant or Agent.
        """

        store = self.conversation_store
        history_loader = getattr(
            self.conversation_memory_policy,
            "history_loader",
            None,
        )
        scoped = getattr(store, "supports_tenant_agent_scope", False) is True
        if history_loader is not None and not scoped:
            raise ConfigurationError(
                "durable Context Memory requires a tenant/Agent-scoped "
                "ConversationStore"
            )
        if not scoped:
            return

        stable_agent_id = str(getattr(self.agent_spec, "name", self.config.name))
        if getattr(store, "agent_id", None) != stable_agent_id:
            raise ConfigurationError(
                "ConversationStore Agent scope does not match the Agent identity"
            )
        trusted_tenant = trusted_identity.get("_moduagent_tenant")
        if (
            trusted_tenant is not None
            and getattr(store, "tenant_id", None) != trusted_tenant
        ):
            raise ConfigurationError(
                "ConversationStore tenant scope does not match trusted run identity"
            )

        if history_loader is not None and (
            getattr(history_loader, "agent_id", None) != stable_agent_id
            or getattr(history_loader, "tenant_id", None)
            != getattr(store, "tenant_id", None)
        ):
            raise ConfigurationError(
                "Context Memory scope does not match ConversationStore scope"
            )

    async def _bind_delegation_runtime(
        self,
        context: RunContext,
        *,
        absolute_deadline: datetime,
    ) -> None:
        """Bind the shared ledger and typed parent Tool context for this run."""

        tools = tuple(getattr(self, "delegation_tools", ()))
        incoming = context.request.delegation_context
        if incoming is not None:
            projection = incoming.to_dict()
            lineage = projection.get("lineage", {})
            if not isinstance(lineage, Mapping):
                raise ConfigurationError("delegation lineage projection is invalid")
            context.metadata["_moduagent_run_lineage"] = dict(lineage)
            context.metadata["_moduagent_execution_group_id"] = str(
                getattr(incoming, "execution_group_id")
            )
        if not tools:
            return

        from moduagent.delegation import (
            PARENT_DELEGATION_CONTEXT_KEY,
            ParentDelegationContext,
            RunLineage,
        )

        coordinators = tuple(tool.coordinator for tool in tools)
        coordinator = coordinators[0]
        if any(item is not coordinator for item in coordinators[1:]):
            raise ConfigurationError(
                "all DelegatedAgentTools on one Agent must share a coordinator"
            )
        ledger = coordinator.budget_ledger
        if context.budget_ledger is not None and context.budget_ledger is not ledger:
            raise ConfigurationError(
                "delegated child budget ledger does not match its Tool coordinator"
            )
        context.budget_ledger = ledger
        callers = {tool.caller for tool in tools}
        if len(callers) != 1:
            raise ConfigurationError(
                "all DelegatedAgentTools on one Agent must share a caller AgentRef"
            )
        caller = next(iter(callers))
        tenant = context.metadata.get("_moduagent_tenant")
        principal = context.metadata.get("_moduagent_principal")
        if not isinstance(tenant, str) or not isinstance(principal, str):
            raise ConfigurationError(
                "delegation requires trusted tenant and principal context"
            )
        existing_group_id = context.metadata.get("_moduagent_execution_group_id")
        existing_state = None
        if context.request.resume_run_id and isinstance(
            existing_group_id,
            str,
        ):
            load_group = getattr(ledger, "load_group", None)
            if not callable(load_group):
                raise ConfigurationError(
                    "resumed delegation requires a loadable budget ledger"
                )
            existing_state = await load_group(existing_group_id)
            if existing_state is None:
                raise ConfigurationError(
                    "resumed execution-group budget state is unavailable"
                )
        if existing_state is not None:
            if existing_state.limits != coordinator.limits:
                raise ConfigurationError("resumed execution-group limits do not match")
            raw_lineage = context.metadata.get("_moduagent_run_lineage", {})
            try:
                lineage = RunLineage.from_dict(raw_lineage)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    "resumed execution-group lineage is invalid"
                ) from exc
            parent = ParentDelegationContext(
                lineage=lineage,
                execution_group_id=existing_state.execution_group_id,
                principal=principal,
                tenant=tenant,
                parent_session_id=context.request.session_id,
                absolute_deadline=existing_state.absolute_deadline,
                limits=existing_state.limits,
                current_run_id=context.run_id,
            )
        else:
            parent = coordinator.parent_context(
                caller=caller,
                run_id=context.run_id,
                session_id=context.request.session_id,
                principal=principal,
                tenant=tenant,
                incoming=incoming,
                execution_group_id=(
                    existing_group_id
                    if context.request.resume_run_id
                    and isinstance(existing_group_id, str)
                    else None
                ),
                absolute_deadline=(None if incoming is not None else absolute_deadline),
            )
        await ledger.ensure_group(
            parent.execution_group_id,
            parent.limits,
            absolute_deadline=parent.absolute_deadline,
        )
        context.metadata[PARENT_DELEGATION_CONTEXT_KEY] = parent
        context.metadata["_moduagent_run_lineage"] = parent.lineage.to_dict()
        context.metadata["_moduagent_execution_group_id"] = parent.execution_group_id
        context.metadata["_moduagent_execution_group_deadline"] = (
            parent.absolute_deadline.isoformat()
        )

    async def _run(
        self,
        request: RunRequest,
        *,
        stream_model: bool,
    ) -> AsyncIterator[AgentEvent]:
        if not isinstance(request, RunRequest):
            raise TypeError("request must be a RunRequest")
        self._validate_engine_descriptor()
        run_id = request.resume_run_id or request.assigned_run_id or uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        run_started_at = loop.time()
        deadline = run_started_at + self.config.limits.timeout_seconds
        absolute_deadline = datetime.now(timezone.utc) + timedelta(
            seconds=self.config.limits.timeout_seconds
        )
        incoming_deadline = getattr(
            request.delegation_context,
            "absolute_deadline",
            None,
        )
        if isinstance(incoming_deadline, datetime):
            if incoming_deadline.tzinfo is None:
                raise ValueError("delegation absolute_deadline must be timezone-aware")
            incoming_deadline = incoming_deadline.astimezone(timezone.utc)
            absolute_deadline = min(absolute_deadline, incoming_deadline)
            remaining = max(
                0.0,
                (absolute_deadline - datetime.now(timezone.utc)).total_seconds(),
            )
            deadline = min(deadline, loop.time() + remaining)
        context = self._new_context(request, run_id, history=())
        resumed_snapshot: RunSnapshot | None = None
        setup_error: BaseException | None = None

        # Resume is loaded before the EventPublisher is created so the first
        # new event continues the durable monotonic sequence.
        try:
            trusted_identity = await self._resolve_trusted_identity(request)
            self._validate_conversation_store_scope(trusted_identity)
            if request.resume_run_id:
                resumed_snapshot, checkpoint = await self._load_resume(
                    request,
                    deadline,
                    trusted_identity,
                )
                context = checkpoint.to_context()
                context.request = replace(
                    context.request,
                    user_context=copy.deepcopy(dict(request.user_context)),
                    delegation_context=request.delegation_context,
                    budget_ledger=request.budget_ledger,
                    budget_lease=request.budget_lease,
                )
                context.budget_ledger = request.budget_ledger
                context.budget_lease = request.budget_lease
                self._normalize_context_tool_trace(context)
            else:
                history_loader = getattr(
                    self.conversation_memory_policy,
                    "history_loader",
                    None,
                )
                if history_loader is None:
                    history = await self._persistence_within(
                        deadline,
                        lambda: self.conversation_store.load(request.session_id),
                        operation="conversation",
                    )
                else:
                    load_history = getattr(history_loader, "load_history", None)
                    if not callable(load_history):
                        raise ConfigurationError(
                            "Context Memory history_loader must provide load_history()"
                        )
                    configured_tenant = getattr(history_loader, "tenant_id", None)
                    trusted_tenant = trusted_identity.get("_moduagent_tenant")
                    if (
                        trusted_tenant is not None
                        and configured_tenant is not None
                        and configured_tenant != trusted_tenant
                    ):
                        raise ConfigurationError(
                            "Context Memory tenant does not match trusted run identity"
                        )
                    configured_agent = getattr(history_loader, "agent_id", None)
                    stable_agent_id = str(
                        getattr(self.agent_spec, "name", self.config.name)
                    )
                    if (
                        configured_agent is not None
                        and configured_agent != stable_agent_id
                    ):
                        raise ConfigurationError(
                            "Context Memory agent_id does not match the Agent identity"
                        )
                    history_view = await self._persistence_within(
                        deadline,
                        lambda: load_history(
                            self.conversation_store,
                            request.session_id,
                        ),
                        operation="conversation",
                    )
                    history = getattr(history_view, "messages", None)
                    if not isinstance(history, tuple) or not all(
                        isinstance(message, Message) for message in history
                    ):
                        raise ConfigurationError(
                            "Context Memory history loader returned an invalid view"
                        )
                context = self._new_context(
                    request,
                    run_id,
                    history=tuple(history),
                )
            context.metadata.update(trusted_identity)
            await self._bind_delegation_runtime(
                context,
                absolute_deadline=absolute_deadline,
            )
            group_deadline = context.metadata.get("_moduagent_execution_group_deadline")
            if isinstance(group_deadline, str):
                parsed_group_deadline = datetime.fromisoformat(group_deadline)
                remaining = max(
                    0.0,
                    (
                        parsed_group_deadline.astimezone(timezone.utc)
                        - datetime.now(timezone.utc)
                    ).total_seconds(),
                )
                deadline = min(deadline, loop.time() + remaining)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, PersistenceError):
                # A bootstrap assembled without durable history must never be
                # resumed as if the conversation had loaded successfully.
                context.metadata["_moduagent_resume_safety"] = "manual_required"
            setup_error = exc
        setup_succeeded = setup_error is None
        cleanup_writes_allowed = setup_succeeded and resumed_snapshot is None

        initial_sequence = (
            resumed_snapshot.common_state.event_sequence
            if resumed_snapshot is not None
            else int(context.metadata.get("_moduagent_event_sequence", 0) or 0)
        )
        self._attach_agent_fingerprint(context)
        self._attach_run_identity(context)
        publisher = EventPublisher(
            run_id=run_id,
            session_id=request.session_id,
            engine_id=self.engine.engine_id,
            initial_sequence=initial_sequence,
            **self._event_identity(context),
        )
        self._event_publishers[run_id] = publisher
        self._coordinator_contexts[run_id] = context
        context.diagnostic_reporter = self.diagnostic_reporter
        context.metadata["_moduagent_engine_id"] = self.engine.engine_id
        context.metadata["_moduagent_engine_state_version"] = self.engine.state_version
        context.metadata["_moduagent_event_sequence"] = initial_sequence

        services = RuntimeServices(self, deadline)
        engine_context: EngineContext | None = None
        state: Any = None

        try:
            started = await self._publish(
                AgentEvent(
                    EventType.RUN_STARTED,
                    run_id,
                    {
                        "agent": self.config.name,
                        "session_id": request.session_id,
                        "user_context": dict(request.user_context),
                        "queue_wait_seconds": self._session_queue_wait_seconds(),
                    },
                )
            )
            yield started

            engine_context = EngineContext(
                run=context,
                config=self.config,
                stream_model=stream_model,
                resolved_spec=self._engine_spec(),
                model_capabilities=self._model_capabilities(),
            )
            services.bind(engine_context)
            if setup_error is not None:
                raise setup_error

            if resumed_snapshot is None:
                # Skill selection and Plan creation may invoke a Model before
                # the Engine can produce its first state. Keep initialization
                # false while RuntimeServices lazily writes an empty bootstrap
                # together with the first provider-attempt reservation. A hard
                # crash must re-run initialization, never decode that empty
                # state as an initialized Engine.
                context.policy_state.setdefault(
                    _ENGINE_INITIALIZED_POLICY_KEY,
                    False,
                )
            else:
                # RuntimeServices is per-process. Rehydrate its write-ahead
                # snapshot pointer before resumed Skill or Plan model calls.
                services.restore_engine_snapshot(resumed_snapshot.engine)

            if resumed_snapshot is not None:
                loaded = await self._publish(
                    AgentEvent(
                        EventType.CHECKPOINT_LOADED,
                        run_id,
                        {
                            "step": context.step,
                            "status": context.status.value,
                            "state_version": (resumed_snapshot.engine.state_version),
                        },
                    )
                )
                yield loaded

            async for skill_event in self._skill_events(
                context,
                deadline,
                resumed=resumed_snapshot is not None,
            ):
                for pending in services.drain_events():
                    yield pending
                yield self._published_event(skill_event)
            for pending in services.drain_events():
                yield pending

            context.status = RunStatus.RUNNING
            needs_initialization = (
                resumed_snapshot is None
                or context.policy_state.get(_ENGINE_INITIALIZED_POLICY_KEY) is False
            )
            if needs_initialization:
                context.policy_state[_ENGINE_INITIALIZED_POLICY_KEY] = False
                try:
                    state = await self.engine.initialize(
                        engine_context,
                        services,
                    )
                    context.policy_state[_ENGINE_INITIALIZED_POLICY_KEY] = True
                    if services.checkpointing_enabled:
                        await services.checkpoint(
                            engine_context,
                            EngineSnapshot(
                                engine_id=self.engine.engine_id,
                                state_version=self.engine.state_version,
                                state=self.engine.encode_state(state),
                            ),
                            boundary=DurableBoundary.INITIALIZED,
                        )
                except BaseException:
                    context.policy_state[_ENGINE_INITIALIZED_POLICY_KEY] = False
                    raise
            else:
                state = self._resume_engine_state(resumed_snapshot)
            cleanup_writes_allowed = True

            for pending in services.drain_events():
                yield pending

            outcome: EngineOutcome | None = None
            iterator = self.engine.execute(
                engine_context,
                state,
                services,
            ).__aiter__()
            next_emission: asyncio.Task[EngineEmission] | None = None
            event_ready: asyncio.Task[None] | None = None
            try:
                next_emission = asyncio.create_task(iterator.__anext__())
                while True:
                    for pending in services.drain_events():
                        yield pending

                    event_ready = asyncio.create_task(services.wait_for_events())
                    done, _ = await asyncio.wait(
                        (next_emission, event_ready),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if event_ready not in done:
                        event_ready.cancel()
                        with suppress(asyncio.CancelledError):
                            await event_ready
                    else:
                        await event_ready
                    event_ready = None

                    for pending in services.drain_events():
                        yield pending
                    if not next_emission.done():
                        continue

                    try:
                        emission = next_emission.result()
                    except StopAsyncIteration:
                        break
                    except BaseException:
                        for pending in services.drain_after_events():
                            yield pending
                        raise
                    next_emission = None

                    if not isinstance(emission, EngineEmission):
                        raise TypeError("ExecutionEngine must yield EngineEmission")
                    if emission.event is not None:
                        if outcome is not None:
                            raise ExecutionInvariantError(
                                "ExecutionEngine emitted an event after its outcome"
                            )
                        if emission.event.type in _TERMINAL_EVENT_TYPES:
                            raise ExecutionInvariantError(
                                "terminal events are owned by RunCoordinator"
                            )
                        if emission.event.run_id != run_id:
                            raise ExecutionInvariantError(
                                "ExecutionEngine emitted an event for another run"
                            )
                        published_key = (run_id, emission.event.event_id)
                        if published_key not in self._published_events:
                            published = await self._publish(emission.event)
                        else:
                            published = self._published_event(emission.event)
                        yield published
                        for pending in services.drain_after_events():
                            yield pending
                    else:
                        if outcome is not None:
                            raise ExecutionInvariantError(
                                "ExecutionEngine emitted more than one outcome"
                            )
                        outcome = emission.outcome
                    next_emission = asyncio.create_task(iterator.__anext__())
            finally:
                if event_ready is not None and not event_ready.done():
                    event_ready.cancel()
                    with suppress(asyncio.CancelledError):
                        await event_ready
                if next_emission is not None and not next_emission.done():
                    next_emission.cancel()
                    with suppress(asyncio.CancelledError):
                        await next_emission
                close_iterator = getattr(iterator, "aclose", None)
                if callable(close_iterator):
                    with suppress(Exception):
                        await close_iterator()

            for pending in services.drain_events():
                yield pending
            for pending in services.drain_after_events():
                yield pending
            if outcome is None:
                raise ExecutionInvariantError(
                    "ExecutionEngine ended without a terminal outcome"
                )

            result, event_type = await self._finish_outcome(
                engine_context,
                services,
                state,
                outcome,
                deadline,
            )
            result = self._with_run_usage(
                result,
                context,
                started_at=run_started_at,
            )
            await self._flush_diagnostics_safely(run_id)
            terminal = await self._publish_terminal(
                AgentEvent(event_type, run_id, {"result": result})
            )
            if event_type is EventType.RUN_FAILED or self._retain_terminal_checkpoint(
                context
            ):
                await self._checkpoint_state_safely(
                    engine_context,
                    services,
                    state,
                )
            yield terminal
        except GeneratorExit:
            context.status = RunStatus.CANCELLED
            context.metadata["_moduagent_terminal_reason"] = (
                FinishReason.CANCELLED.value
            )
            self._normalize_skill_resource_messages(context)
            if cleanup_writes_allowed:
                await asyncio.shield(self._persist_safely(context))
                if engine_context is not None:
                    await asyncio.shield(
                        self._checkpoint_state_safely(
                            engine_context,
                            services,
                            state,
                        )
                    )
            raise
        except asyncio.CancelledError:
            context.status = RunStatus.CANCELLED
            context.metadata["_moduagent_terminal_reason"] = (
                FinishReason.CANCELLED.value
            )
            self._normalize_skill_resource_messages(context)
            if cleanup_writes_allowed:
                await self._persist_safely(context)
                if engine_context is not None:
                    await self._checkpoint_state_safely(
                        engine_context,
                        services,
                        state,
                    )
            raise
        except asyncio.TimeoutError as exc:
            context.status = RunStatus.FAILED
            context.metadata["_moduagent_terminal_reason"] = FinishReason.TIMEOUT.value
            diagnostic = await self._capture_terminal_failure(context, exc)
            self._normalize_skill_resource_messages(context)
            if cleanup_writes_allowed:
                await self._persist_safely(context)
                if engine_context is not None:
                    await self._checkpoint_state_safely(
                        engine_context,
                        services,
                        state,
                    )
            result = self._result(
                context,
                FinishReason.TIMEOUT,
                error="run timed out",
            )
            summary = {
                "category": "timeout",
                "code": "run_timeout",
                "retryable": True,
                "resumable": self._is_safely_resumable(context),
            }
            summary.update(diagnostic)
            result = self._with_error_summary(result, summary)
            result = self._with_run_usage(
                result,
                context,
                started_at=run_started_at,
            )
            for pending in services.drain_events():
                yield pending
            for pending in services.drain_after_events():
                yield pending
            await self._flush_diagnostics_safely(run_id)
            terminal = await self._publish_terminal(
                AgentEvent(EventType.RUN_FAILED, run_id, {"result": result})
            )
            if cleanup_writes_allowed and engine_context is not None:
                await self._checkpoint_state_safely(
                    engine_context,
                    services,
                    state,
                )
            yield terminal
        except Exception as exc:
            finish_reason = self._finish_reason_for_exception(exc)
            context.status = RunStatus.FAILED
            context.metadata["_moduagent_terminal_reason"] = finish_reason.value
            if finish_reason in _NON_RESUMABLE_GUARD_REASONS:
                context.metadata["_moduagent_resume_safety"] = "not_resumable"
            diagnostic = await self._capture_terminal_failure(context, exc)
            self._normalize_skill_resource_messages(context)
            if cleanup_writes_allowed:
                await self._persist_safely(context)
                if engine_context is not None:
                    await self._checkpoint_state_safely(
                        engine_context,
                        services,
                        state,
                    )
            result = self._result(
                context,
                finish_reason,
                error=self._public_error(exc),
            )
            summary = dict(self._error_summary(exc, context=context))
            summary.update(diagnostic)
            result = self._with_error_summary(result, summary)
            result = self._with_run_usage(
                result,
                context,
                started_at=run_started_at,
            )
            for pending in services.drain_events():
                yield pending
            for pending in services.drain_after_events():
                yield pending
            await self._flush_diagnostics_safely(run_id)
            terminal = await self._publish_terminal(
                AgentEvent(EventType.RUN_FAILED, run_id, {"result": result})
            )
            if cleanup_writes_allowed and engine_context is not None:
                await self._checkpoint_state_safely(
                    engine_context,
                    services,
                    state,
                )
            yield terminal
        finally:
            await self._close_sink_worker(run_id)
            await self._flush_diagnostics_safely(run_id)
            clear_diagnostics = getattr(
                self.diagnostic_reporter,
                "clear_run",
                None,
            )
            if callable(clear_diagnostics):
                try:
                    cleared = clear_diagnostics(run_id)
                    if inspect.isawaitable(cleared):
                        await cleared
                except Exception:
                    pass
            self._event_publishers.pop(run_id, None)
            self._coordinator_contexts.pop(run_id, None)
            stale = [key for key in self._published_events if key[0] == run_id]
            for key in stale:
                self._published_events.pop(key, None)
            stale_reserved = [key for key in self._reserved_events if key[0] == run_id]
            for key in stale_reserved:
                self._reserved_events.pop(key, None)

    async def _flush_diagnostics_safely(self, run_id: str) -> None:
        flush_diagnostics = getattr(
            self.diagnostic_reporter,
            "flush_run",
            None,
        )
        if not callable(flush_diagnostics):
            return
        try:
            flushed = flush_diagnostics(run_id)
            if inspect.isawaitable(flushed):
                await flushed
        except Exception:
            pass

    async def _load_resume(
        self,
        request: RunRequest,
        deadline: float,
        trusted_identity: Mapping[str, str],
    ) -> tuple[RunSnapshot, RunCheckpoint]:
        store = self.checkpoint_store
        if store is None:
            raise RuntimeError("checkpoint_store is required to resume a run")
        run_id = request.resume_run_id
        if run_id is None:
            raise ValueError("resume_run_id is required")

        load_snapshot = getattr(store, "load_snapshot", None)
        if callable(load_snapshot):
            snapshot = await self._persistence_within(
                deadline,
                lambda: load_snapshot(run_id),
                operation="checkpoint",
            )
            if snapshot is None:
                raise CheckpointNotFoundError("checkpoint not found")
            if not isinstance(snapshot, RunSnapshot):
                raise TypeError("load_snapshot() must return RunSnapshot or None")
            checkpoint = RunCheckpoint.from_snapshot(snapshot)
        else:
            checkpoint = await self._persistence_within(
                deadline,
                lambda: store.load(run_id),
                operation="checkpoint",
            )
            if checkpoint is None:
                raise CheckpointNotFoundError("checkpoint not found")
            if not isinstance(checkpoint, RunCheckpoint):
                raise TypeError("checkpoint load() must return RunCheckpoint or None")
            snapshot = checkpoint.to_snapshot()

        if snapshot.run_id != run_id or checkpoint.run_id != run_id:
            raise StateMigrationError(
                "checkpoint run_id does not match the requested resume run"
            )
        if checkpoint.session_id != request.session_id:
            raise StateMigrationError(
                "checkpoint session_id does not match the request"
            )
        self._validate_resume_identity(snapshot, trusted_identity)
        if snapshot.engine.engine_id != self.engine.engine_id:
            raise StateMigrationError(
                "checkpoint engine does not match the configured engine"
            )
        if snapshot.common_state.resume_safety not in {
            "resumable",
            "terminal",
        }:
            raise StateMigrationError(
                "checkpoint is not safely resumable: "
                f"{snapshot.common_state.resume_safety}"
            )
        current_fingerprint = self._agent_fingerprint()
        if (
            snapshot.agent_fingerprint != "legacy-unbound"
            and current_fingerprint is not None
            and snapshot.agent_fingerprint != current_fingerprint
        ):
            raise StateMigrationError(
                "checkpoint Agent fingerprint does not match configuration"
            )
        current_definition_fingerprint = self._agent_definition_fingerprint()
        current_ref = self._agent_ref()
        if current_definition_fingerprint is not None and snapshot.agent_ref:
            if snapshot.agent_definition_fingerprint != current_definition_fingerprint:
                raise StateMigrationError(
                    "checkpoint AgentDefinition fingerprint does not match"
                )
            if dict(snapshot.agent_ref) != current_ref:
                raise StateMigrationError(
                    "checkpoint AgentRef does not match configuration"
                )
        elif current_definition_fingerprint is not None:
            # A v4 migration has no AgentRef and carries its legacy AgentSpec
            # fingerprint in both fields. A native v5 checkpoint that loses
            # only agent_ref must fail closed instead of bypassing exact
            # definition pinning. The next successful migrated checkpoint is
            # re-written with the exact v5 AgentRef/fingerprint pair.
            if snapshot.migrated_from_schema_version not in {1, 2, 3, 4}:
                raise StateMigrationError(
                    "checkpoint AgentRef is missing for a pinned definition"
                )
            profile_kind = getattr(
                getattr(getattr(self, "runtime_profile", None), "kind", None),
                "value",
                None,
            )
            if profile_kind == "production":
                raise StateMigrationError(
                    "Production cannot adopt a legacy checkpoint without an exact AgentRef"
                )
        elif current_definition_fingerprint is None and snapshot.agent_ref:
            raise StateMigrationError("checkpoint requires an AgentDefinition binding")
        await self._validate_delegated_resume(request, snapshot)
        return snapshot, checkpoint

    @staticmethod
    def _validate_resume_identity(
        snapshot: RunSnapshot,
        trusted_identity: Mapping[str, str],
    ) -> None:
        """Bind a checkpoint to the currently authenticated tenant/subject."""

        for kind in ("tenant", "principal"):
            stored = getattr(snapshot, f"{kind}_scope_digest")
            expected = trusted_identity.get(f"_moduagent_{kind}_scope_digest")
            if stored is None and expected is None:
                continue
            if stored is None:
                raise StateMigrationError(f"checkpoint {kind} scope binding is missing")
            if expected is None:
                raise StateMigrationError(
                    f"checkpoint requires a trusted {kind} identity"
                )
            if stored != expected:
                raise StateMigrationError(
                    f"checkpoint {kind} scope does not match the request"
                )

    async def _validate_delegated_resume(
        self,
        request: RunRequest,
        snapshot: RunSnapshot,
    ) -> None:
        """Keep child checkpoints behind the receipt-owned private boundary."""

        lineage = snapshot.run_lineage
        depth = lineage.get("depth", 0)
        delegated = (
            type(depth) is int
            and depth > 0
            or snapshot.delegation_id is not None
            or snapshot.parent_tool_call_id is not None
            or snapshot.budget_lease_id is not None
        )
        incoming = request.delegation_context
        if delegated and incoming is None:
            raise StateMigrationError(
                "delegated checkpoint requires coordinator-owned resume context"
            )
        if not delegated:
            if incoming is not None:
                raise StateMigrationError(
                    "root checkpoint cannot resume as a delegated child"
                )
            return
        projection = incoming.to_dict()
        incoming_lineage = projection.get("lineage")
        if not isinstance(incoming_lineage, Mapping) or dict(incoming_lineage) != dict(
            lineage
        ):
            raise StateMigrationError("delegated checkpoint lineage does not match")
        if projection.get("execution_group_id") != snapshot.execution_group_id:
            raise StateMigrationError(
                "delegated checkpoint execution group does not match"
            )
        if lineage.get("delegation_id") != snapshot.delegation_id or (
            lineage.get("parent_tool_call_id") != snapshot.parent_tool_call_id
        ):
            raise StateMigrationError(
                "delegated checkpoint receipt identity is inconsistent"
            )
        lease = request.budget_lease
        ledger = request.budget_ledger
        if lease is None or ledger is None:
            raise StateMigrationError("delegated checkpoint requires a budget lease")
        if getattr(lease, "execution_group_id", None) != snapshot.execution_group_id:
            raise StateMigrationError("delegated budget lease group does not match")
        if (
            snapshot.common_state.status
            not in {RunStatus.FAILED.value, RunStatus.CANCELLED.value}
            and getattr(lease, "lease_id", None) != snapshot.budget_lease_id
        ):
            raise StateMigrationError("delegated budget lease identity does not match")
        load_group = getattr(ledger, "load_group", None)
        if not callable(load_group):
            raise StateMigrationError("delegated budget ledger is not loadable")
        state = await load_group(str(snapshot.execution_group_id))
        if state is None:
            raise StateMigrationError("delegated execution-group state is unavailable")
        record = getattr(state, "leases", {}).get(getattr(lease, "lease_id", None))
        if record is None or getattr(record, "status", None) != "active":
            raise StateMigrationError("delegated budget lease is not active")

    def _resume_engine_state(self, snapshot: RunSnapshot) -> Any:
        resolved_spec = dict(self._engine_spec())
        resolved_spec["common_state"] = {
            "step": snapshot.common_state.step,
            "tool_call_count": snapshot.common_state.tool_call_count,
        }
        validation = self.engine.validate_resume(
            snapshot.engine,
            resolved_spec,
        )
        if not validation.compatible:
            raise StateMigrationError(
                "checkpoint cannot be resumed: " + validation.reason
            )
        payload: Mapping[str, Any] = snapshot.engine.state
        try:
            if snapshot.engine.state_version != self.engine.state_version:
                payload = self.engine.migrate_state(
                    snapshot.engine.state_version,
                    payload,
                )
            return self.engine.decode_state(payload)
        except StateMigrationError:
            raise
        except Exception as exc:
            raise StateMigrationError(
                "checkpoint Engine state cannot be decoded"
            ) from exc

    async def _finish_outcome(
        self,
        context: EngineContext,
        services: RuntimeServices,
        state: Any,
        outcome: EngineOutcome,
        deadline: float,
    ) -> tuple[AgentResult, EventType]:
        outcome = replace(
            outcome,
            metadata=self._project_outcome_metadata(outcome.metadata),
        )
        failed = outcome.finish_reason in {
            FinishReason.ERROR,
            FinishReason.TIMEOUT,
            FinishReason.CANCELLED,
            FinishReason.MAX_STEPS,
            FinishReason.MAX_TOOL_CALLS,
            FinishReason.MAX_MODEL_TURNS,
            FinishReason.NO_PROGRESS,
        }
        context.run.status = (
            RunStatus.CANCELLED
            if outcome.finish_reason is FinishReason.CANCELLED
            else RunStatus.FAILED
            if failed
            else RunStatus.COMPLETED
        )
        context.run.metadata["_moduagent_terminal_reason"] = outcome.finish_reason.value
        if outcome.finish_reason in _NON_RESUMABLE_GUARD_REASONS:
            context.run.metadata["_moduagent_resume_safety"] = "not_resumable"
        if outcome.metadata:
            context.run.metadata.update(dict(outcome.metadata))

        self._normalize_skill_resource_messages(context.run)
        await self._persist_pending_messages(context.run, deadline)
        if failed or self._retain_terminal_checkpoint(context.run):
            await self._checkpoint_state_safely(
                context,
                services,
                state,
            )
        elif self.checkpoint_store is not None:
            await self._persistence_within(
                deadline,
                lambda: self.checkpoint_store.delete(context.run.run_id),
                operation="checkpoint",
            )

        base = self._result(
            context.run,
            outcome.finish_reason,
            output=outcome.output,
            error=outcome.error,
        )
        metadata = dict(base.metadata)
        metadata.update(dict(outcome.metadata))
        if failed:
            summary = dict(self._outcome_error_summary(outcome, context=context.run))
            primary_failure = _project_primary_failure(context.run.primary_failure)
            if primary_failure:
                summary.update(primary_failure)
            else:
                tool_failure = outcome.metadata.get("failure")
                if isinstance(tool_failure, Mapping):
                    call_id = tool_failure.get("call_id")
                    failure_id = (
                        context.run.tool_failure_ids.get(call_id)
                        if isinstance(call_id, str)
                        else None
                    )
                    if failure_id:
                        phase, step_id, attempt = self._diagnostic_location(context.run)
                        summary.update(
                            {
                                "failure_id": failure_id,
                                "component": "tool",
                                "operation": "invoke",
                                **({} if phase is None else {"phase": phase}),
                                **({} if step_id is None else {"step_id": step_id}),
                                **({} if attempt is None else {"attempt": attempt}),
                            }
                        )
            metadata["error_summary"] = summary
        result = (
            base
            if metadata == dict(base.metadata)
            else replace(base, metadata=metadata)
        )
        return (
            result,
            EventType.RUN_FAILED if failed else EventType.RUN_COMPLETED,
        )

    @staticmethod
    def _project_outcome_metadata(
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        projected: dict[str, Any] = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                continue
            if (
                is_runtime_owned_metadata_key(key)
                and key not in _ENGINE_OUTCOME_RUNTIME_METADATA_KEYS
            ):
                continue
            if key == "validation_failure":
                safe_validation = RunCoordinator._project_validation_failure(value)
                if safe_validation is not None:
                    projected[key] = safe_validation
                continue
            projected[key] = value
        return projected

    @staticmethod
    def _project_validation_failure(value: Any) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        code = value.get("code")
        location = value.get("location")
        if (
            not isinstance(code, str)
            or code not in _STEP_VALIDATION_CODES
            or not isinstance(location, str)
            or location not in _STEP_VALIDATION_LOCATIONS
        ):
            return None
        projected: dict[str, Any] = {
            "code": code,
            "location": location,
        }
        cause_code = value.get("cause_code")
        if (
            isinstance(cause_code, str)
            and cause_code in _STEP_VALIDATION_CODES
            and cause_code != code
        ):
            projected["cause_code"] = cause_code
        phase = value.get("phase")
        if isinstance(phase, str) and phase in {
            "act_tool",
            "failed",
            "step_result",
            "step_validate",
        }:
            projected["phase"] = phase
        step_id = value.get("step_id")
        if (
            isinstance(step_id, str)
            and 0 < len(step_id) <= 256
            and all(character.isprintable() for character in step_id)
        ):
            projected["step_id"] = step_id
        attempt = value.get("attempt")
        if type(attempt) is int and 0 <= attempt <= 1_000_000:
            projected["attempt"] = attempt
        return projected

    async def _checkpoint_state_safely(
        self,
        context: EngineContext,
        services: RuntimeServices,
        state: Any,
    ) -> None:
        if self.checkpoint_store is None:
            return
        if state is None:
            # Skill selection can fail before an Engine has initialized. The
            # empty payload is never decoded: the common bootstrap marker makes
            # resume run initialize() again. Keeping the configured Engine ID
            # prevents a Plan run from being misclassified as Standard.
            await services.checkpoint_safely(
                context,
                EngineSnapshot(
                    engine_id=self.engine.engine_id,
                    state_version=self.engine.state_version,
                    state={},
                ),
                boundary=DurableBoundary.INTERRUPTED,
            )
            return
        try:
            snapshot = EngineSnapshot(
                engine_id=self.engine.engine_id,
                state_version=self.engine.state_version,
                state=self.engine.encode_state(state),
            )
        except Exception:
            return
        await services.checkpoint_safely(
            context,
            snapshot,
            boundary=DurableBoundary.INTERRUPTED,
        )

    def _new_context(
        self,
        request: RunRequest,
        run_id: str,
        *,
        history: tuple[Message, ...],
    ) -> RunContext:
        user_message = Message.user(
            request.input,
            metadata=(
                {
                    _RUN_ID_METADATA_KEY: run_id,
                    "moduagent.public_input": True,
                }
                if self._retain_terminal_checkpoint(request)
                else None
            ),
        )
        metadata = {
            "agent": self.config.name,
            "_moduagent_session_queue_wait_seconds": (
                self._session_queue_wait_seconds()
            ),
            **{
                key: value
                for key, value in self.config.metadata.items()
                if not is_runtime_owned_metadata_key(key)
            },
        }
        context = RunContext(
            run_id=run_id,
            request=request,
            messages=[
                Message.system(self.config.instructions),
                *history,
                user_message,
            ],
            new_messages=[user_message],
            metadata=metadata,
            current_run_start=1 + len(history),
            budget_ledger=request.budget_ledger,
            budget_lease=request.budget_lease,
        )
        return context

    def _engine_spec(self) -> Mapping[str, Any]:
        spec = dict(self.resolved_spec)
        spec["engine_id"] = self.engine.engine_id
        spec["state_version"] = self.engine.state_version
        fingerprint = self._agent_fingerprint()
        if fingerprint is not None:
            spec["agent_fingerprint"] = fingerprint
        return spec

    def _model_capabilities(self) -> ModelCapabilities:
        capabilities = getattr(self.model, "capabilities", None)
        if capabilities is None:
            return ModelCapabilities()
        if not isinstance(capabilities, ModelCapabilities):
            raise ConfigurationError("model capabilities are invalid")
        return capabilities

    def _validate_engine_descriptor(self) -> None:
        if self.engine.engine_id != self.resolved_spec.get("engine_id"):
            raise ConfigurationError(
                "execution Engine ID changed after Agent composition"
            )
        if self.engine.state_version != self.resolved_spec.get("state_version"):
            raise ConfigurationError(
                "execution Engine state version changed after Agent composition"
            )
        if self.engine.state_codec.engine_id != self.engine.engine_id:
            raise ConfigurationError(
                "execution Engine codec ID no longer matches the Engine"
            )
        if self.engine.state_codec.state_version != self.engine.state_version:
            raise ConfigurationError(
                "execution Engine codec version no longer matches the Engine"
            )
        details = self.resolved_spec.get("details", {})
        if not isinstance(details, Mapping):
            return
        expected_fingerprint = details.get("configuration_fingerprint")
        if isinstance(expected_fingerprint, str):
            encoded = json.dumps(
                _redact_descriptor(_descriptor_plain(self.engine.configuration)),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            current_fingerprint = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
            if current_fingerprint != expected_fingerprint:
                raise ConfigurationError(
                    "execution Engine configuration changed after Agent composition"
                )
        expected_requirements = details.get("required_capabilities")
        if isinstance(expected_requirements, Mapping) and dict(
            self.engine.required_capabilities
        ) != dict(expected_requirements):
            raise ConfigurationError(
                "execution Engine capability requirements changed after composition"
            )

    def _retain_terminal_checkpoint(
        self,
        run: RunContext | RunRequest | None = None,
    ) -> bool:
        if bool(self.resolved_spec.get("retain_terminal_checkpoint", False)):
            return True
        request = run.request if isinstance(run, RunContext) else run
        return bool(
            isinstance(request, RunRequest) and request.delegation_context is not None
        )

    @staticmethod
    def _public_error(error: Exception) -> str:
        """Project an exception to the terminal result without raw provider data."""

        if isinstance(error, StateMigrationError):
            return "checkpoint state migration failed"
        if isinstance(error, CheckpointNotFoundError):
            return "checkpoint not found"
        if isinstance(error, PersistenceError):
            return "persistence operation failed"
        if isinstance(error, OutputValidationError):
            # Keep the 0.4.0 public string stable; structured diagnostics carry
            # the more precise output-validation classification.
            return "run failed"
        if isinstance(error, FrameworkMemoryError) and not isinstance(
            error,
            (ConversationMemoryOverflowError, MemoryIntegrityError),
        ):
            return "conversation memory preparation failed"
        if isinstance(error, ModuAgentError):
            message = " ".join(
                "".join(
                    character if character.isprintable() else " "
                    for character in str(error)
                ).split()
            )
            if message:
                return message[:512]
        return "run failed"

    def _error_summary(
        self,
        error: Exception,
        *,
        context: RunContext | None = None,
    ) -> Mapping[str, Any]:
        category = "execution"
        code = "run_failed"
        retryable = False
        if isinstance(error, CheckpointNotFoundError):
            category, code = "persistence", "checkpoint_not_found"
        elif isinstance(error, StateMigrationError):
            category, code = "state_migration", "checkpoint_migration_failed"
        elif isinstance(error, PersistenceError):
            category, code = "persistence", "persistence_failed"
            retryable = True
        elif isinstance(error, ModelTurnBudgetExceeded):
            category, code = "limit", error.code
        elif isinstance(error, ModelNoProgressError):
            category, code = "model_progress", error.code
        elif isinstance(error, BudgetExceeded):
            category, code = "execution_group_budget", error.code
        elif isinstance(error, ModelInvocationError):
            classification = classify_model_error(error)
            category, code = classification.category, classification.code
            retryable = classification.retryable
        elif isinstance(error, OutputValidationError):
            category, code = "output_validation", "output_validation_failed"
        elif isinstance(error, ToolAuthorizationError):
            category, code = "tool_authorization", "tool_authorization_failed"
        elif isinstance(error, ToolValidationError):
            category, code = "tool_validation", "tool_validation_failed"
        elif isinstance(error, ToolRecoveryError):
            category, code = "tool_recovery", "tool_recovery_failed"
        elif isinstance(error, ToolInvocationError):
            category, code = "tool_invocation", "tool_invocation_failed"
            retryable = True
        elif isinstance(error, FrameworkMemoryError):
            category, code = "memory", "memory_preparation_failed"
        elif isinstance(error, FrameworkSkillError):
            category, code = "skill", "skill_activation_failed"
        elif isinstance(error, ConfigurationError):
            category, code = "configuration", "invalid_configuration"
        elif isinstance(error, ExecutionInvariantError):
            category, code = "execution_invariant", "execution_invariant_failed"
        elif isinstance(error, CancellationError):
            category, code = "cancellation", "run_cancelled"
        resumable = (
            False
            if isinstance(
                error,
                (
                    CheckpointNotFoundError,
                    StateMigrationError,
                    ModelGuardTripped,
                    BudgetExceeded,
                ),
            )
            else self._is_safely_resumable(context)
        )
        summary = {
            "category": category,
            "code": code,
            "retryable": retryable,
            "resumable": resumable,
        }
        if isinstance(error, ModelGuardTripped):
            snapshot = error.snapshot
            summary.update(
                {
                    "model_turns": snapshot.model_turns,
                    "max_model_turns": snapshot.max_model_turns,
                    "no_progress_model_turns": snapshot.no_progress_model_turns,
                    "no_progress_model_turn_threshold": (
                        snapshot.no_progress_model_turn_threshold
                    ),
                }
            )
        if type(error) is ModelOutputIncompleteError and error.finish_reason in {
            "timeout",
            "length",
            "max_tokens",
        }:
            summary["provider_finish_reason"] = error.finish_reason
        return summary

    @staticmethod
    def _finish_reason_for_exception(error: Exception) -> FinishReason:
        if isinstance(error, ModelTurnBudgetExceeded):
            return FinishReason.MAX_MODEL_TURNS
        if isinstance(error, ModelNoProgressError):
            return FinishReason.NO_PROGRESS
        if isinstance(error, BudgetExceeded):
            if error.code == "execution_group_model_turns_exceeded":
                return FinishReason.MAX_MODEL_TURNS
            if error.code == "execution_group_tool_calls_exceeded":
                return FinishReason.MAX_TOOL_CALLS
            if error.code == "execution_group_timeout":
                return FinishReason.TIMEOUT
        return FinishReason.ERROR

    async def _capture_terminal_failure(
        self,
        context: RunContext,
        error: BaseException,
    ) -> Mapping[str, Any]:
        existing = _project_primary_failure(context.primary_failure)
        if existing.get("failure_id"):
            return existing

        reporter = self.diagnostic_reporter
        capture = getattr(reporter, "capture_exception", None)
        if not callable(capture):
            return existing

        default_component, default_operation = self._diagnostic_operation(error)
        default_phase, default_step_id, default_attempt = self._diagnostic_location(
            context
        )
        component = str(existing.get("component", default_component))
        operation = str(existing.get("operation", default_operation))
        phase = existing.get("phase", default_phase)
        step_id = existing.get("step_id", default_step_id)
        attempt = existing.get("attempt", default_attempt)
        summary = {
            **dict(self._error_summary_for_diagnostic(error, context)),
            **{
                key: existing[key]
                for key in ("category", "code", "retryable")
                if key in existing
            },
        }
        capture_options: dict[str, Any] = {}
        if isinstance(error, OutputValidationError):
            validation_fields = self._output_validation_fields()
            if validation_fields:
                capture_options["validation_fields"] = validation_fields
        try:
            failure_id = await capture(
                exception=error,
                run_id=context.run_id,
                component=component,
                operation=operation,
                phase=phase,
                step_id=step_id,
                attempt=attempt,
                category=str(summary["category"]),
                code=str(summary["code"]),
                retryable=bool(summary["retryable"]),
                terminal=True,
                **capture_options,
            )
        except Exception:
            return existing
        if not isinstance(failure_id, str) or not failure_id:
            return existing

        captured = _project_primary_failure(
            {
                **existing,
                "failure_id": failure_id,
                "component": component,
                "operation": operation,
                **({} if phase is None else {"phase": phase}),
                **({} if step_id is None else {"step_id": step_id}),
                **({} if attempt is None else {"attempt": attempt}),
                "category": summary["category"],
                "code": summary["code"],
                "retryable": summary["retryable"],
            }
        )
        if "failure_id" not in captured:
            return existing
        return captured

    def _output_validation_fields(self) -> frozenset[str]:
        try:
            schema = self.output_codec.schema()
        except Exception:
            return frozenset()
        if not isinstance(schema, Mapping):
            return frozenset()

        fields: set[str] = set()
        pending: list[Any] = [schema]
        seen: set[int] = set()
        while pending and len(seen) < 1024 and len(fields) < 1024:
            value = pending.pop()
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(value, Mapping):
                properties = value.get("properties")
                if isinstance(properties, Mapping):
                    fields.update(
                        str(name)[:256] for name in properties if isinstance(name, str)
                    )
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
        return frozenset(fields)

    def _error_summary_for_diagnostic(
        self,
        error: BaseException,
        context: RunContext,
    ) -> Mapping[str, Any]:
        if isinstance(error, asyncio.TimeoutError):
            return {
                "category": "timeout",
                "code": "run_timeout",
                "retryable": True,
            }
        if isinstance(error, Exception):
            return self._error_summary(error, context=context)
        return {
            "category": "execution",
            "code": "run_failed",
            "retryable": False,
        }

    @staticmethod
    def _diagnostic_operation(error: BaseException) -> tuple[str, str]:
        if isinstance(error, BudgetExceeded):
            return "runtime", "execution_group_budget"
        if isinstance(error, ModelGuardTripped):
            return "model", "guard"
        if isinstance(error, ModelInvocationError):
            return "model", "invoke"
        if isinstance(error, OutputValidationError):
            return "output", "decode"
        if isinstance(
            error,
            (ToolInvocationError, ToolValidationError, ToolAuthorizationError),
        ):
            return "tool", "invoke"
        if isinstance(error, PersistenceError):
            return "persistence", "operation"
        if isinstance(error, FrameworkMemoryError):
            return "memory", "prepare"
        if isinstance(error, FrameworkSkillError):
            return "skill", "activate"
        if isinstance(error, ConfigurationError):
            return "configuration", "validate"
        if isinstance(error, asyncio.TimeoutError):
            return "runtime", "deadline"
        return "runtime", "execute"

    @staticmethod
    def _diagnostic_location(
        context: RunContext,
    ) -> tuple[str | None, str | None, int | None]:
        state = context.execution_state
        phase_value = getattr(state, "phase", None)
        phase = getattr(phase_value, "value", phase_value)
        if not isinstance(phase, str) or not phase:
            phase = None

        step_id = getattr(state, "current_step_id", None)
        attempt = None
        step_execution = getattr(state, "step_execution", None)
        if step_execution is not None:
            step_id = getattr(step_execution, "current_step_id", step_id)
            attempt = getattr(step_execution, "step_attempt_count", None)
        if not isinstance(step_id, str) or not step_id:
            step_id = None
        if type(attempt) is not int or attempt < 1:
            attempt = None
        return phase, step_id, attempt

    def _outcome_error_summary(
        self,
        outcome: EngineOutcome,
        *,
        context: RunContext | None = None,
    ) -> Mapping[str, Any]:
        codes = {
            FinishReason.TIMEOUT: ("timeout", "run_timeout", True),
            FinishReason.CANCELLED: ("cancellation", "run_cancelled", False),
            FinishReason.MAX_STEPS: ("limit", "max_steps_exceeded", False),
            FinishReason.MAX_TOOL_CALLS: (
                "limit",
                "max_tool_calls_exceeded",
                False,
            ),
            FinishReason.MAX_MODEL_TURNS: (
                "limit",
                "max_model_turns_exceeded",
                False,
            ),
            FinishReason.NO_PROGRESS: (
                "model_progress",
                "model_no_progress",
                False,
            ),
            FinishReason.ERROR: ("execution", "execution_failed", False),
        }
        category, code, retryable = codes.get(
            outcome.finish_reason,
            ("execution", "execution_failed", False),
        )
        validation_failure = outcome.metadata.get("validation_failure")
        if isinstance(validation_failure, Mapping):
            validation_code = validation_failure.get("code")
            validation_location = validation_failure.get("location")
            if (
                isinstance(validation_code, str)
                and validation_code in _STEP_VALIDATION_CODES
                and isinstance(validation_location, str)
                and validation_location in _STEP_VALIDATION_LOCATIONS
            ):
                summary: dict[str, Any] = {
                    "category": "step_validation",
                    "code": validation_code,
                    "component": "policy",
                    "operation": validation_location,
                    "retryable": False,
                    "resumable": self._is_safely_resumable(context),
                }
                phase = validation_failure.get("phase")
                if isinstance(phase, str) and phase in {
                    "act_tool",
                    "failed",
                    "step_result",
                    "step_validate",
                }:
                    summary["phase"] = phase
                step_id = validation_failure.get("step_id")
                if isinstance(step_id, str) and 0 < len(step_id) <= 256:
                    summary["step_id"] = step_id
                attempt = validation_failure.get("attempt")
                if type(attempt) is int and 0 <= attempt <= 1_000_000:
                    summary["attempt"] = attempt
                return summary
        failure = outcome.metadata.get("failure")
        if isinstance(failure, Mapping):
            category = "tool_recovery"
            reason = failure.get("reason")
            if isinstance(reason, str) and reason:
                code = reason[:128]
            retryable = bool(failure.get("retryable", False))
        return {
            "category": category,
            "code": code,
            "retryable": retryable,
            "resumable": (
                False
                if outcome.finish_reason in _NON_RESUMABLE_GUARD_REASONS
                else self._is_safely_resumable(context)
            ),
        }

    def _is_safely_resumable(self, context: RunContext | None) -> bool:
        if self.checkpoint_store is None:
            return False
        if context is None:
            return True
        safety = str(context.metadata.get("_moduagent_resume_safety", "resumable"))
        return safety in {"resumable", "terminal"}

    @staticmethod
    def _with_error_summary(
        result: AgentResult,
        summary: Mapping[str, Any],
    ) -> AgentResult:
        return replace(
            result,
            metadata={
                **dict(result.metadata),
                "error_summary": dict(summary),
            },
        )

    @staticmethod
    def _with_run_usage(
        result: AgentResult,
        context: RunContext,
        *,
        started_at: float,
    ) -> AgentResult:
        """Attach ephemeral run counters without checkpointing clock values."""

        raw_model_turns = context.metadata.get("_moduagent_model_turns", 0)
        model_turns = (
            raw_model_turns
            if type(raw_model_turns) is int and raw_model_turns >= 0
            else 0
        )
        tool_calls = (
            context.tool_call_count
            if type(context.tool_call_count) is int and context.tool_call_count >= 0
            else 0
        )
        duration_seconds = max(
            0.0,
            asyncio.get_running_loop().time() - started_at,
        )
        return replace(
            result,
            metadata={
                **dict(result.metadata),
                "run_usage": {
                    "model_turns": model_turns,
                    "tool_calls": tool_calls,
                    "duration_seconds": duration_seconds,
                },
            },
        )

    def _agent_fingerprint(self) -> str | None:
        value = getattr(
            getattr(self, "agent_spec", None),
            "agent_fingerprint",
            None,
        )
        return value if isinstance(value, str) and value else None

    def _attach_agent_fingerprint(self, context: RunContext) -> None:
        fingerprint = self._agent_fingerprint()
        if fingerprint is not None:
            context.metadata["_moduagent_agent_fingerprint"] = fingerprint
        definition_fingerprint = self._agent_definition_fingerprint()
        if definition_fingerprint is not None:
            context.metadata["_moduagent_agent_definition_fingerprint"] = (
                definition_fingerprint
            )
            context.metadata["_moduagent_agent_ref"] = self._agent_ref()

    def _agent_definition_fingerprint(self) -> str | None:
        value = getattr(
            getattr(self, "agent_definition", None),
            "fingerprint",
            None,
        )
        return value if isinstance(value, str) and value else None

    def _agent_ref(self) -> dict[str, str]:
        definition = getattr(self, "agent_definition", None)
        agent_id = getattr(definition, "agent_id", None)
        version = getattr(definition, "version", None)
        if not isinstance(agent_id, str) or not isinstance(version, str):
            return {}
        return {"agent_id": agent_id, "version": version}

    def _attach_run_identity(self, context: RunContext) -> None:
        """Initialize v5 lineage metadata before events or checkpoints exist."""

        projection: Mapping[str, Any] = {}
        delegation_context = context.request.delegation_context
        if delegation_context is not None:
            candidate = delegation_context.to_dict()
            if isinstance(candidate, Mapping):
                projection = candidate
        raw_lineage = projection.get("lineage", projection.get("run_lineage", {}))
        if not isinstance(raw_lineage, Mapping):
            raw_lineage = {}
        existing = context.metadata.get("_moduagent_run_lineage")
        if not raw_lineage and isinstance(existing, Mapping):
            raw_lineage = existing
        lineage = dict(raw_lineage)
        if not lineage:
            agent_ref = self._agent_ref()
            agent_id = agent_ref.get("agent_id")
            agent_version = agent_ref.get("version")
            lineage = {
                "root_run_id": context.run_id,
                "parent_run_id": None,
                "depth": 0,
                "agent_path": (
                    [f"{agent_id}@{agent_version}"]
                    if agent_id is not None and agent_version is not None
                    else []
                ),
            }
            if agent_id is not None and agent_version is not None:
                lineage.update(
                    {
                        "delegation_id": None,
                        "parent_tool_call_id": None,
                        "caller_agent_id": None,
                        "agent_id": agent_id,
                        "agent_version": agent_version,
                    }
                )
        context.metadata["_moduagent_run_lineage"] = lineage
        context.metadata.setdefault(
            "_moduagent_execution_group_id",
            projection.get("execution_group_id")
            or lineage.get("root_run_id")
            or context.run_id,
        )
        for source_key, metadata_key in (
            ("delegation_id", "_moduagent_delegation_id"),
            ("parent_tool_call_id", "_moduagent_parent_tool_call_id"),
        ):
            value = projection.get(source_key) or lineage.get(source_key)
            if value is not None:
                context.metadata[metadata_key] = value

    def _event_identity(self, context: RunContext) -> dict[str, Any]:
        """Return the content-free event v2 identity for one run."""

        projection: Mapping[str, Any] = {}
        delegation_context = context.request.delegation_context
        if delegation_context is not None:
            to_dict = getattr(delegation_context, "to_dict", None)
            if callable(to_dict):
                candidate = to_dict()
                if isinstance(candidate, Mapping):
                    projection = candidate
        raw_lineage = projection.get("lineage", projection.get("run_lineage", {}))
        if not isinstance(raw_lineage, Mapping):
            raw_lineage = {}
        if not raw_lineage:
            candidate = context.metadata.get("_moduagent_run_lineage", {})
            if isinstance(candidate, Mapping):
                raw_lineage = candidate
        root_run_id = str(raw_lineage.get("root_run_id") or context.run_id)
        parent_run_id = raw_lineage.get("parent_run_id")
        depth = raw_lineage.get("depth", 0)
        if type(depth) is not int or depth < 0:
            depth = 0
        raw_agent_ref = context.metadata.get("_moduagent_agent_ref", {})
        agent_ref = raw_agent_ref if isinstance(raw_agent_ref, Mapping) else {}
        agent_id = agent_ref.get("agent_id", agent_ref.get("id", self.config.name))
        agent_version = agent_ref.get("version")
        return {
            "execution_group_id": str(
                projection.get("execution_group_id")
                or context.metadata.get("_moduagent_execution_group_id")
                or root_run_id
            ),
            "root_run_id": root_run_id,
            "parent_run_id": (None if parent_run_id is None else str(parent_run_id)),
            "delegation_id": _optional_runtime_identifier(
                projection.get("delegation_id")
                or raw_lineage.get("delegation_id")
                or context.metadata.get("_moduagent_delegation_id")
            ),
            "agent_id": _optional_runtime_identifier(agent_id),
            "agent_version": _optional_runtime_identifier(agent_version),
            "depth": depth,
        }

    @staticmethod
    def _normalize_skill_resource_messages(context: RunContext) -> None:
        replacements: dict[int, Message] = {}
        normalized: list[Message] = []
        for message in context.messages:
            ephemeral = (
                message.role is MessageRole.TOOL
                and message.name in SKILL_RESOURCE_TOOL_NAMES
            ) or (
                message.role is MessageRole.ASSISTANT
                and any(
                    call.name in SKILL_RESOURCE_TOOL_NAMES
                    for call in message.tool_calls
                )
            )
            if not ephemeral:
                normalized.append(message)
                continue
            replacement = replace(
                message,
                metadata={
                    **dict(message.metadata),
                    "moduagent.ephemeral": True,
                },
            )
            replacements[id(message)] = replacement
            normalized.append(replacement)
        if not replacements:
            return
        context.messages[:] = normalized
        context.new_messages[:] = [
            message
            for message in context.new_messages
            if id(message) not in replacements
        ]

    async def _publish(self, event: AgentEvent) -> AgentEvent:
        """Publish a non-terminal event through the run-scoped envelope."""

        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        if event.type in _TERMINAL_EVENT_TYPES:
            raise ExecutionInvariantError("terminal events are owned by RunCoordinator")
        return await self._publish_event(event)

    async def _publish_terminal(self, event: AgentEvent) -> AgentEvent:
        """Publish the single terminal event from the Coordinator-owned path."""

        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        if event.type not in _TERMINAL_EVENT_TYPES:
            raise ValueError("_publish_terminal requires a terminal event")
        return await self._publish_event(event)

    async def _publish_related_delegation_event(
        self,
        event: AgentEvent,
    ) -> AgentEvent:
        """Publish coordinator-validated child correlation on a parent run."""

        if not isinstance(event, AgentEvent) or not event.type.value.startswith(
            "delegation_"
        ):
            raise ValueError(
                "related delegation publication requires a lifecycle event"
            )
        return await self._publish_event(
            event,
            allow_related_delegation=True,
        )

    async def _publish_event(
        self,
        event: AgentEvent,
        *,
        allow_related_delegation: bool = False,
    ) -> AgentEvent:
        """Stamp once, isolate sink failures, and return the published object."""

        published = self._reserve_event(
            event,
            allow_related_delegation=allow_related_delegation,
        )
        return await self._dispatch_reserved_event(published)

    def _reserve_event(
        self,
        event: AgentEvent,
        *,
        allow_related_delegation: bool = False,
    ) -> AgentEvent:
        """Allocate an event sequence before a related durable write."""

        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        if event.type not in _PUBLIC_STREAM_EVENT_TYPES:
            event = replace(event, visibility=EventVisibility.INTERNAL)
        publisher = self._event_publishers.get(event.run_id)
        if publisher is None:
            raise ExecutionInvariantError(
                "event does not belong to an active Coordinator run"
            )
        published = publisher.stamp(
            event,
            allow_related_delegation=allow_related_delegation,
        )
        context = self._coordinator_contexts.get(event.run_id)
        if context is not None:
            context.metadata["_moduagent_event_sequence"] = published.sequence
        key = (published.run_id, published.event_id)
        if key in self._reserved_events or key in self._published_events:
            raise ExecutionInvariantError("event identity was already published")
        self._reserved_events[key] = published
        return published

    async def _dispatch_reserved_event(self, event: AgentEvent) -> AgentEvent:
        """Dispatch one previously stamped event without advancing its sequence."""

        key = (event.run_id, event.event_id)
        reserved = self._reserved_events.pop(key, None)
        if reserved is None or reserved != event:
            raise ExecutionInvariantError("event was not reserved by this Coordinator")
        queue = self._sink_queues.get(event.run_id)
        self._published_events[key] = _PublishedEventStamp.from_event(event)
        if _event_sink_is_noop(self.event_sink):
            return event
        if queue is None:
            queue = asyncio.Queue(maxsize=_EVENT_SINK_QUEUE_MAX_SIZE)
            self._sink_queues[event.run_id] = queue
            self._sink_workers[event.run_id] = asyncio.create_task(
                self._sink_worker(queue)
            )
        # A bounded queue prevents a slow external sink from retaining an
        # unlimited number of large terminal/delta payloads. Backpressure is
        # charged only after a full 1,024-event burst; Noop sinks never enter
        # this path.
        await queue.put(event)
        # Start the ordered worker without charging sink latency to the run.
        await asyncio.sleep(0)
        if event.type in _TERMINAL_EVENT_TYPES:
            pending_count = queue.qsize() + 1
            drain_timeout = min(
                _EVENT_SINK_MAX_DRAIN_SECONDS,
                max(
                    _EVENT_SINK_MIN_DRAIN_SECONDS,
                    pending_count * _EVENT_SINK_TIMEOUT_SECONDS + 0.5,
                ),
            )
            try:
                await asyncio.wait_for(
                    queue.join(),
                    timeout=drain_timeout,
                )
            except asyncio.TimeoutError:
                pass
        return event

    async def _sink_worker(
        self,
        queue: asyncio.Queue[AgentEvent],
    ) -> None:
        timed_out = False
        while True:
            event = await queue.get()
            invocation: asyncio.Task[None] | None = None
            try:
                if not timed_out:
                    invocation = asyncio.create_task(self._invoke_event_sink(event))
                    completed, _ = await asyncio.wait(
                        (invocation,),
                        timeout=_EVENT_SINK_TIMEOUT_SECONDS,
                    )
                    if not completed:
                        # ``wait_for`` waits indefinitely when an adapter
                        # suppresses cancellation. Detach after opening the
                        # run-scoped circuit instead.
                        timed_out = True
                        invocation.cancel()
                        invocation.add_done_callback(_consume_task_result)
                    else:
                        await invocation
            except asyncio.CancelledError:
                if invocation is not None and not invocation.done():
                    invocation.cancel()
                    invocation.add_done_callback(_consume_task_result)
                raise
            except Exception:
                # Observability cannot alter execution, including cancellation
                # and timeout behavior.
                pass
            finally:
                queue.task_done()

    async def _close_sink_worker(self, run_id: str) -> None:
        worker = self._sink_workers.pop(run_id, None)
        self._sink_queues.pop(run_id, None)
        if worker is None:
            return
        worker.cancel()
        try:
            await worker
        except BaseException:
            pass

    async def _invoke_event_sink(self, event: AgentEvent) -> None:
        # Sinks are untrusted observability adapters. Give them an isolated
        # object graph so mutation cannot alter the stream or terminal result.
        sink_event = (
            copy.deepcopy(event)
            if _event_sink_requires_coordinator_copy(self.event_sink)
            else event
        )
        publisher = self.event_sink.publish
        if inspect.iscoroutinefunction(publisher):
            await publisher(sink_event)
            return
        result = await run_in_daemon_thread(publisher, sink_event)
        if inspect.isawaitable(result):
            await result

    def _published_event(self, event: AgentEvent) -> AgentEvent:
        """Resolve events emitted by inherited compatibility helpers."""

        stamp = self._published_events.get((event.run_id, event.event_id))
        return event if stamp is None else stamp.apply(event)


__all__ = ["RunCoordinator"]

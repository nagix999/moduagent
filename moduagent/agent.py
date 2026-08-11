from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import fields
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel

from moduagent.composition import (
    AgentSpec,
    ExecutionProfile,
    PlanExecutionProfile,
    StandardExecutionProfile,
    compose_agent,
)
from moduagent.config import AgentConfig, RetryConfig, RunLimits
from moduagent.definitions import (
    AgentDefinition,
    DefinitionStatus,
    RuntimeBindings,
)
from moduagent.decision import DecisionPolicy, LLMPlanGenerator
from moduagent.errors import ConfigurationError
from moduagent.execution import ExecutionEngine
from moduagent.memory import ConversationMemoryPolicy
from moduagent.models import ModelClient
from moduagent.observability import DiagnosticSink, EventSink
from moduagent.output import OutputCodec, PydanticOutputCodec
from moduagent.persistence import (
    CheckpointStore,
    ConversationStore,
)
from moduagent.profiles import (
    RuntimeProfile,
    RuntimeValidationContext,
)
from moduagent.runtime import AgentEvent, AgentResult, RunRequest, RunStatus
from moduagent.skills import SkillLimits, SkillRegistry, SkillSelector
from moduagent.tools import (
    Tool,
    ToolAuthorizer,
)


class Agent:
    """Small public facade that delegates execution to :class:`AgentRuntime`."""

    @classmethod
    def create(
        cls,
        *,
        model: ModelClient,
        instructions: str,
        name: str = "agent",
        tools: Iterable[Tool] = (),
        execution: Literal["standard", "plan"] | ExecutionProfile = "standard",
        output: type[BaseModel] | OutputCodec | None = None,
        limits: RunLimits | None = None,
        retry: RetryConfig | None = None,
        model_options: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        finalization_mode: Literal[
            "always", "structured_only", "disabled"
        ] = "structured_only",
        stream_visibility: Literal["public_only", "all"] = "public_only",
        memory: ConversationMemoryPolicy | None = None,
        context_memory: ConversationMemoryPolicy | None = None,
        conversation_store: ConversationStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        event_sink: EventSink | None = None,
        diagnostic_sink: DiagnosticSink | None = None,
        diagnostic_timeout_seconds: float = 0.25,
        diagnostic_max_pending_deliveries: int = 1024,
        tool_authorizer: ToolAuthorizer | None = None,
        skill_registry: SkillRegistry | None = None,
        skill_selector: SkillSelector | None = None,
        skill_limits: SkillLimits | None = None,
        tool_trace_mode: Literal["off", "summary", "arguments"] = "summary",
        definition: AgentDefinition | None = None,
        runtime_profile: RuntimeProfile | None = None,
        runtime_bindings: RuntimeBindings | None = None,
    ) -> Agent:
        """Create an Agent from the common high-level configuration.

        This is an additive convenience API. It resolves to the same
        :class:`AgentConfig`, execution profiles, output codecs, and runtime as
        the full constructor. The most common persistence and observability
        components are accepted directly. Applications that need custom
        planners or planning models, detailed Tool recovery, decision policies,
        or execution engines should continue to use :class:`Agent` directly.
        """

        if memory is not None and context_memory is not None:
            raise ValueError("use either context_memory or the memory alias, not both")
        resolved_memory = context_memory if context_memory is not None else memory
        if runtime_profile is not None and not isinstance(
            runtime_profile,
            RuntimeProfile,
        ):
            raise TypeError("runtime_profile must be a RuntimeProfile")
        resolved_profile = runtime_profile or RuntimeProfile.development()
        resolved_limits = (
            limits if limits is not None else resolved_profile.default_limits
        )
        resolved_retry = retry if retry is not None else RetryConfig()
        execution_profile = _quick_execution_profile(
            execution,
            model=model,
            limits=resolved_limits,
        )
        output_codec = _quick_output_codec(output)
        return cls(
            config=AgentConfig(
                name=name,
                instructions=instructions,
                limits=resolved_limits,
                retry=resolved_retry,
                model_options=({} if model_options is None else model_options),
                metadata={} if metadata is None else metadata,
                finalization_mode=finalization_mode,
                stream_visibility=stream_visibility,
                tool_trace_mode=tool_trace_mode,
            ),
            model=model,
            tools=tools,
            execution_profile=execution_profile,
            output_codec=output_codec,
            conversation_store=conversation_store,
            checkpoint_store=checkpoint_store,
            event_sink=event_sink,
            diagnostic_sink=diagnostic_sink,
            diagnostic_timeout_seconds=diagnostic_timeout_seconds,
            diagnostic_max_pending_deliveries=(diagnostic_max_pending_deliveries),
            tool_authorizer=tool_authorizer,
            conversation_memory_policy=resolved_memory,
            skill_registry=skill_registry,
            skill_selector=skill_selector,
            skill_limits=skill_limits,
            definition=definition,
            runtime_profile=resolved_profile,
            runtime_bindings=runtime_bindings,
        )

    def __init__(
        self,
        *,
        config: AgentConfig,
        model: ModelClient,
        tools: Iterable[Tool] = (),
        conversation_store: ConversationStore | None = None,
        decision_policy: DecisionPolicy | None = None,
        execution_profile: ExecutionProfile | None = None,
        execution_engine: ExecutionEngine[Any] | None = None,
        output_codec: OutputCodec | None = None,
        event_sink: EventSink | None = None,
        diagnostic_sink: DiagnosticSink | None = None,
        diagnostic_timeout_seconds: float = 0.25,
        diagnostic_max_pending_deliveries: int = 1024,
        tool_authorizer: ToolAuthorizer | None = None,
        checkpoint_store: CheckpointStore | None = None,
        conversation_memory_policy: ConversationMemoryPolicy | None = None,
        context_memory_policy: ConversationMemoryPolicy | None = None,
        skill_registry: SkillRegistry | None = None,
        skill_selector: SkillSelector | None = None,
        skill_limits: SkillLimits | None = None,
        definition: AgentDefinition | None = None,
        runtime_profile: RuntimeProfile | None = None,
        runtime_bindings: RuntimeBindings | None = None,
    ) -> None:
        if conversation_memory_policy is not None and context_memory_policy is not None:
            raise ValueError(
                "use either context_memory_policy or "
                "conversation_memory_policy, not both"
            )
        resolved_memory_policy = (
            context_memory_policy
            if context_memory_policy is not None
            else conversation_memory_policy
        )
        if runtime_bindings is not None and not isinstance(
            runtime_bindings,
            RuntimeBindings,
        ):
            raise TypeError("runtime_bindings must be RuntimeBindings or None")
        registered_tools = tuple(tools)
        if runtime_bindings is not None:
            bound_registry = runtime_bindings.tool_registry
            if bound_registry is not None:
                bound_tools = tuple(bound_registry)
                if registered_tools and registered_tools != bound_tools:
                    raise ConfigurationError(
                        "tools do not match runtime_bindings.tool_registry"
                    )
                registered_tools = bound_tools
            conversation_store = _effective_binding(
                "conversation_store",
                conversation_store,
                runtime_bindings.conversation_store,
            )
            checkpoint_store = _effective_binding(
                "checkpoint_store",
                checkpoint_store,
                runtime_bindings.checkpoint_store,
            )
            event_sink = _effective_binding(
                "event_sink",
                event_sink,
                runtime_bindings.event_sink,
            )
            diagnostic_sink = _effective_binding(
                "diagnostic_sink",
                diagnostic_sink,
                runtime_bindings.diagnostic_sink,
            )
            tool_authorizer = _effective_binding(
                "tool_authorizer",
                tool_authorizer,
                runtime_bindings.tool_authorizer,
            )
            skill_registry = _effective_binding(
                "skill_registry",
                skill_registry,
                runtime_bindings.skill_registry,
            )
        composition = compose_agent(
            config=config,
            model=model,
            tools=registered_tools,
            tool_registry=(
                None if runtime_bindings is None else runtime_bindings.tool_registry
            ),
            conversation_store=conversation_store,
            decision_policy=decision_policy,
            execution_profile=execution_profile,
            execution_engine=execution_engine,
            output_codec=output_codec,
            event_sink=event_sink,
            diagnostic_sink=diagnostic_sink,
            diagnostic_timeout_seconds=diagnostic_timeout_seconds,
            diagnostic_max_pending_deliveries=(diagnostic_max_pending_deliveries),
            tool_authorizer=tool_authorizer,
            checkpoint_store=checkpoint_store,
            conversation_memory_policy=resolved_memory_policy,
            skill_registry=skill_registry,
            skill_selector=skill_selector,
            skill_limits=skill_limits,
        )
        resolved_profile = runtime_profile or RuntimeProfile.development()
        if not isinstance(resolved_profile, RuntimeProfile):
            raise TypeError("runtime_profile must be a RuntimeProfile")
        if definition is not None and not isinstance(definition, AgentDefinition):
            raise TypeError("definition must be an AgentDefinition or None")
        resolved_bindings = _resolved_runtime_bindings(
            runtime_bindings,
            composition=composition,
            skill_registry=skill_registry,
            checkpoint_store=checkpoint_store,
            diagnostic_sink=diagnostic_sink,
        )
        delegation_tools = _delegated_agent_tools(composition.tool_registry)
        resolved_bindings = _resolved_delegation_bindings(
            resolved_bindings,
            delegation_tools,
        )
        semantic_digests = _resolved_semantic_digests(
            composition.spec,
            tool_authorizer=composition.tool_executor.authorizer,
        )
        definition_status, resolved_endpoint = _definition_registration(
            definition,
            resolved_bindings,
            model=model,
        )
        if definition is not None:
            _validate_definition_binding(
                definition,
                config=config,
                spec=composition.spec,
                semantic_digests=semantic_digests,
                skill_registry=skill_registry,
            )
        delegation_limits = _common_delegation_limits(
            delegation_tools,
            definition=definition,
        )
        resolved_profile.validate(
            # TestProfile deployment facts are accepted only through a pinned
            # attestation object; absent evidence remains fail-closed.
            RuntimeValidationContext(
                bindings=resolved_bindings,
                definition=definition,
                definition_status=definition_status,
                memory_policy=composition.conversation_memory_policy,
                tools=tuple(composition.tool_registry),
                delegation_enabled=bool(delegation_tools),
                delegation_limits=delegation_limits,
                shared_parent_child_session=any(
                    getattr(getattr(tool, "session_strategy", None), "value", None)
                    == "shared"
                    for tool in delegation_tools
                ),
                resolved_endpoint=resolved_endpoint,
                unapproved_delegation_endpoint_refs=(
                    _unapproved_delegation_endpoint_refs(delegation_tools)
                ),
                external_io_enabled=(
                    True
                    if resolved_bindings.runtime_attestation is None
                    else resolved_bindings.runtime_attestation.external_io_enabled
                ),
                deterministic_components=(
                    False
                    if resolved_bindings.runtime_attestation is None
                    else resolved_bindings.runtime_attestation.deterministic_components
                ),
            )
        )
        if resolved_profile.kind.value == "production":
            composition.tool_registry.freeze()
        self.config = config
        self.model = model
        self.definition = definition
        self.agent_ref = None if definition is None else definition.ref
        self.runtime_profile = resolved_profile
        self.runtime_bindings = resolved_bindings
        self.semantic_digests = MappingProxyType(semantic_digests)
        self.skill_registry = skill_registry
        self.spec = composition.spec
        self.skill_runtime = composition.skill_runtime
        self.tool_registry = composition.tool_registry
        self.tool_executor = composition.tool_executor
        self.conversation_memory_policy = composition.conversation_memory_policy
        self.context_memory_policy = composition.conversation_memory_policy
        self.engine = composition.engine
        self.runtime = composition.runtime
        self.runtime.agent_definition = definition
        self.runtime.runtime_profile = resolved_profile
        self.runtime.runtime_bindings = resolved_bindings
        self.runtime.semantic_digests = self.semantic_digests
        self.runtime.delegation_limits = delegation_limits
        self.runtime.delegation_tools = delegation_tools
        self.diagnostic_reporter = composition.runtime.diagnostic_reporter

    def inspect(self) -> AgentSpec:
        """Return the immutable, secret-safe resolved Agent configuration."""

        return self.spec

    async def run(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_context: Mapping[str, Any] | None = None,
        resume_run_id: str | None = None,
        skills: Iterable[str] = (),
        skill_mode: str | None = None,
    ) -> AgentResult:
        request = self._request(
            text,
            session_id=session_id,
            user_context=user_context,
            resume_run_id=resume_run_id,
            skills=skills,
            skill_mode=skill_mode,
        )
        return await self.runtime.execute(request)

    async def ask(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_context: Mapping[str, Any] | None = None,
        resume_run_id: str | None = None,
        skills: Iterable[str] = (),
        skill_mode: str | None = None,
    ) -> Any:
        """Run the Agent and return its decoded output, raising on failure."""

        result = await self.run(
            text,
            session_id=session_id,
            user_context=user_context,
            resume_run_id=resume_run_id,
            skills=skills,
            skill_mode=skill_mode,
        )
        return result.unwrap()

    def stream(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_context: Mapping[str, Any] | None = None,
        resume_run_id: str | None = None,
        skills: Iterable[str] = (),
        skill_mode: str | None = None,
        include_internal: bool | None = None,
    ) -> AsyncIterator[AgentEvent]:
        request = self._request(
            text,
            session_id=session_id,
            user_context=user_context,
            resume_run_id=resume_run_id,
            skills=skills,
            skill_mode=skill_mode,
        )
        return self.runtime.stream(request, include_internal=include_internal)

    def stream_all(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_context: Mapping[str, Any] | None = None,
        resume_run_id: str | None = None,
        skills: Iterable[str] = (),
        skill_mode: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream public and diagnostic internal events for one run."""

        return self.stream(
            text,
            session_id=session_id,
            user_context=user_context,
            resume_run_id=resume_run_id,
            skills=skills,
            skill_mode=skill_mode,
            include_internal=True,
        )

    async def resume(
        self,
        run_id: str,
        *,
        session_id: str,
        user_context: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        return await self.run(
            "",
            session_id=session_id,
            user_context=user_context,
            resume_run_id=run_id,
        )

    def as_tool(
        self,
        *,
        coordinator: Any,
        caller: Any,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        name: str | None = None,
        description: str | None = None,
        **options: Any,
    ) -> Tool:
        """Expose this pinned Agent through the production delegation path.

        Legacy ``AgentTool`` remains available for local compatibility. This
        adapter always requires a ``DelegationCoordinator`` and a versioned
        ``AgentDefinition`` so the caller cannot select an unpinned child.
        """

        from moduagent.definitions import AgentRef
        from moduagent.delegation import DelegationCoordinator

        if self.definition is None:
            raise ConfigurationError("Agent.as_tool() requires an AgentDefinition")
        if not isinstance(coordinator, DelegationCoordinator):
            raise TypeError("coordinator must be a DelegationCoordinator")
        if not isinstance(caller, AgentRef):
            raise TypeError("caller must be an AgentRef")
        return coordinator.tool(
            caller=caller,
            callee=self.definition.ref,
            input_model=input_model,
            output_model=output_model,
            name=name,
            description=description or self.definition.description,
            **options,
        )

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context: Any,
        budget: Any,
        budget_ledger: Any,
        child_run_id: str,
    ) -> Any:
        """Private typed endpoint used only by ``LocalAgentInvoker``."""

        return await self._execute_delegated(
            request,
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
            resume=False,
        )

    async def _resume_delegated(
        self,
        request: BaseModel,
        *,
        context: Any,
        budget: Any,
        budget_ledger: Any,
        child_run_id: str,
    ) -> Any:
        """Private resume endpoint used by receipt reconciliation."""

        return await self._execute_delegated(
            request,
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
            resume=True,
        )

    async def _execute_delegated(
        self,
        request: BaseModel,
        *,
        context: Any,
        budget: Any,
        budget_ledger: Any,
        child_run_id: str,
        resume: bool,
    ) -> Any:
        from moduagent.delegation import (
            BudgetLease,
            DelegationContext,
            DelegationOutcome,
            DelegationOutcomeStatus,
        )

        if not isinstance(request, BaseModel):
            raise TypeError("delegated request must be a Pydantic model")
        if not isinstance(context, DelegationContext):
            raise TypeError("context must be a DelegationContext")
        if not isinstance(budget, BudgetLease):
            raise TypeError("budget must be a BudgetLease")
        await self._validate_delegated_endpoint_contract(
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
            allowed_lease_statuses=frozenset({"active"}),
        )
        run_request = RunRequest(
            input=request.model_dump_json(),
            session_id=context.child_session_id,
            user_context={},
            resume_run_id=child_run_id if resume else None,
            delegation_context=context,
            budget_ledger=budget_ledger,
            budget_lease=budget,
            assigned_run_id=None if resume else child_run_id,
        )
        result = await self.runtime.execute(run_request)
        usage = {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
            **{
                key: value
                for key, value in result.run_usage.items()
                if key in {"model_turns", "tool_calls", "duration_seconds"}
            },
        }
        if result.finish_reason is not None and (
            result.finish_reason.value == "completed"
            and result.error is None
            and isinstance(result.output, BaseModel)
        ):
            return DelegationOutcome(
                DelegationOutcomeStatus.COMPLETED,
                child_run_id,
                finish_reason="completed",
                output=result.output,
                usage=usage,
            )
        finish_reason = result.finish_reason.value
        if finish_reason == "completed" and not isinstance(result.output, BaseModel):
            finish_reason = "output_validation"
        status = (
            DelegationOutcomeStatus.CANCELLED
            if finish_reason == "cancelled"
            else DelegationOutcomeStatus.FAILED
        )
        error_code = _delegated_finish_reason_code(finish_reason)
        resumable = result.error_summary.get("resumable") is True
        return DelegationOutcome(
            status,
            child_run_id,
            finish_reason=finish_reason,
            error_code=error_code,
            retryable=False,
            resumable=resumable,
            usage=usage,
        )

    async def _reconcile_delegated(
        self,
        *,
        context: Any,
        budget: Any,
        budget_ledger: Any,
        child_run_id: str,
    ) -> Any | None:
        """Read a retained terminal child checkpoint without replaying work."""

        from moduagent.delegation import (
            DelegationFailure,
            DelegationOutcome,
            DelegationOutcomeStatus,
        )

        await self._validate_delegated_endpoint_contract(
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
            allowed_lease_statuses=frozenset({"active", "completed"}),
        )

        store = getattr(self.runtime, "checkpoint_store", None)
        if store is None:
            return None
        checkpoint = await store.load(child_run_id)
        if checkpoint is None or checkpoint.run_id != child_run_id:
            return None
        snapshot = checkpoint.to_snapshot()
        expected_ref = {
            "agent_id": self.definition.ref.agent_id,
            "version": self.definition.ref.version,
        }
        expected_lineage = context.lineage.to_dict()
        from moduagent.persistence.snapshot import identity_scope_digest

        if (
            checkpoint.session_id != context.child_session_id
            or dict(snapshot.run_lineage) != expected_lineage
            or snapshot.execution_group_id != context.execution_group_id
            or dict(snapshot.agent_ref) != expected_ref
            or snapshot.agent_definition_fingerprint != self.definition.fingerprint
            or snapshot.delegation_id != context.lineage.delegation_id
            or snapshot.parent_tool_call_id != context.lineage.parent_tool_call_id
            or snapshot.tenant_scope_digest
            != identity_scope_digest("tenant", context.tenant)
            or snapshot.principal_scope_digest
            != identity_scope_digest("principal", context.principal)
        ):
            raise DelegationFailure("delegation_checkpoint_identity_mismatch")
        if snapshot.budget_lease_id != budget.lease_id:
            # A prior attempt may have left a terminal checkpoint while the
            # receipt owner is starting a new resume attempt. It is stale, not
            # evidence that the current lease has finished.
            return None
        if checkpoint.status not in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return None
        usage = {
            "input_tokens": checkpoint.usage.input_tokens,
            "output_tokens": checkpoint.usage.output_tokens,
            "total_tokens": checkpoint.usage.total_tokens,
        }
        markers = checkpoint.finalization_markers
        model_type = getattr(self.runtime.output_codec, "model_type", None)
        if (
            checkpoint.status is RunStatus.COMPLETED
            and markers is not None
            and markers.response_generated
            and isinstance(model_type, type)
            and issubclass(model_type, BaseModel)
        ):
            try:
                output = (
                    model_type.model_validate(markers.response)
                    if isinstance(markers.response, Mapping)
                    else self.runtime.output_codec.decode(markers.response)
                )
            except Exception as exc:
                raise DelegationFailure(
                    "delegation_reconciliation_output_invalid"
                ) from exc
            if not isinstance(output, BaseModel):
                raise DelegationFailure("delegation_reconciliation_output_invalid")
            return DelegationOutcome(
                DelegationOutcomeStatus.COMPLETED,
                child_run_id,
                finish_reason="completed",
                output=output,
                usage=usage,
            )
        finish_reason = checkpoint.terminal_reason or "error"
        return DelegationOutcome(
            (
                DelegationOutcomeStatus.CANCELLED
                if finish_reason == "cancelled"
                else DelegationOutcomeStatus.FAILED
            ),
            child_run_id,
            finish_reason=finish_reason,
            error_code=_delegated_finish_reason_code(finish_reason),
            retryable=False,
            resumable=checkpoint.resume_safety == "resumable",
            usage=usage,
        )

    async def _cleanup_delegated_checkpoint(
        self,
        *,
        context: Any,
        budget: Any,
        budget_ledger: Any,
        child_run_id: str,
    ) -> None:
        """Delete only the terminal checkpoint owned by a settled receipt."""

        from moduagent.delegation import DelegationFailure

        await self._validate_delegated_endpoint_contract(
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
            allowed_lease_statuses=frozenset({"completed"}),
        )
        store = getattr(self.runtime, "checkpoint_store", None)
        if store is None:
            return
        checkpoint = await store.load(child_run_id)
        if checkpoint is None:
            return
        if checkpoint.run_id != child_run_id:
            raise DelegationFailure("delegation_checkpoint_identity_mismatch")
        snapshot = checkpoint.to_snapshot()
        from moduagent.persistence.snapshot import identity_scope_digest

        expected_ref = {
            "agent_id": self.definition.ref.agent_id,
            "version": self.definition.ref.version,
        }
        if (
            checkpoint.status is not RunStatus.COMPLETED
            or checkpoint.session_id != context.child_session_id
            or dict(snapshot.run_lineage) != context.lineage.to_dict()
            or snapshot.execution_group_id != context.execution_group_id
            or dict(snapshot.agent_ref) != expected_ref
            or snapshot.agent_definition_fingerprint != self.definition.fingerprint
            or snapshot.delegation_id != context.lineage.delegation_id
            or snapshot.parent_tool_call_id != context.lineage.parent_tool_call_id
            or snapshot.budget_lease_id != budget.lease_id
            or snapshot.tenant_scope_digest
            != identity_scope_digest("tenant", context.tenant)
            or snapshot.principal_scope_digest
            != identity_scope_digest("principal", context.principal)
        ):
            raise DelegationFailure("delegation_checkpoint_identity_mismatch")
        await store.delete(child_run_id)

    async def _validate_delegated_endpoint_contract(
        self,
        *,
        context: Any,
        budget: Any,
        budget_ledger: Any,
        child_run_id: str,
        allowed_lease_statuses: frozenset[str],
    ) -> None:
        from moduagent.delegation import BudgetLease, DelegationContext

        if self.definition is None:
            raise ConfigurationError(
                "delegated endpoint requires a pinned AgentDefinition"
            )
        if not isinstance(context, DelegationContext):
            raise TypeError("context must be a DelegationContext")
        if not isinstance(budget, BudgetLease):
            raise TypeError("budget must be a BudgetLease")
        if not isinstance(child_run_id, str) or not child_run_id:
            raise ValueError("child_run_id cannot be empty")
        if (
            context.lineage.agent_ref != self.definition.ref
            or budget.callee != self.definition.ref
        ):
            raise ConfigurationError("delegated endpoint AgentDefinition mismatch")
        if budget.execution_group_id != context.execution_group_id:
            raise ConfigurationError("delegation budget execution group mismatch")
        if budget.absolute_deadline != context.absolute_deadline:
            raise ConfigurationError("delegation budget deadline mismatch")
        for method_name in (
            "load_group",
            "reserve_model_turn",
            "reserve_tool_call",
        ):
            if not callable(getattr(budget_ledger, method_name, None)):
                raise TypeError(f"budget_ledger must provide {method_name}()")
        state = await budget_ledger.load_group(context.execution_group_id)
        if state is None:
            raise ConfigurationError("delegation execution-group state is unavailable")
        if getattr(state, "absolute_deadline", None) != context.absolute_deadline:
            raise ConfigurationError("delegation ledger deadline mismatch")
        record = getattr(state, "leases", {}).get(budget.lease_id)
        if (
            record is None
            or getattr(record, "callee_key", None) != str(self.definition.ref)
            or getattr(record, "status", None) not in allowed_lease_statuses
        ):
            raise ConfigurationError("delegation budget lease is invalid")

    @staticmethod
    def _request(
        text: str,
        *,
        session_id: str | None,
        user_context: Mapping[str, Any] | None,
        resume_run_id: str | None,
        skills: Iterable[str],
        skill_mode: str | None,
    ) -> RunRequest:
        if not isinstance(text, str):
            raise TypeError("agent input must be a string")
        if isinstance(skills, (str, bytes)):
            raise TypeError("skills must be an iterable of Skill names")
        requested_skills = tuple(skills)
        resolved_skill_mode = (
            skill_mode
            if skill_mode is not None
            else ("explicit" if requested_skills else "disabled")
        )
        if requested_skills and resolved_skill_mode == "disabled":
            raise ValueError("skills cannot be requested when skill_mode is disabled")
        if requested_skills and resolved_skill_mode == "auto":
            raise ValueError("use skill_mode='hybrid' with explicitly requested skills")
        if resume_run_id is not None and (requested_skills or skill_mode is not None):
            raise ValueError("resume restores Skills from the checkpoint")
        return RunRequest(
            input=text,
            session_id=session_id or uuid.uuid4().hex,
            user_context=dict(user_context or {}),
            resume_run_id=resume_run_id,
            requested_skills=requested_skills,
            skill_mode=resolved_skill_mode,
        )


def _quick_execution_profile(
    execution: Literal["standard", "plan"] | ExecutionProfile,
    *,
    model: ModelClient,
    limits: RunLimits,
) -> ExecutionProfile:
    if execution == "standard":
        return StandardExecutionProfile()
    if execution == "plan":
        return PlanExecutionProfile(
            LLMPlanGenerator(model, max_steps=limits.max_steps),
            max_step_attempts=limits.max_step_attempts,
            max_replans=limits.max_replans,
        )
    if isinstance(execution, (StandardExecutionProfile, PlanExecutionProfile)):
        return execution
    if isinstance(execution, str):
        raise ValueError(
            "execution must be 'standard', 'plan', or an execution profile"
        )
    raise TypeError("execution must be 'standard', 'plan', or an execution profile")


def _quick_output_codec(
    output: type[BaseModel] | OutputCodec | None,
) -> OutputCodec | None:
    if output is None:
        return None
    if isinstance(output, type) and issubclass(output, BaseModel):
        return PydanticOutputCodec(output)
    if isinstance(output, OutputCodec):
        return output
    raise TypeError("output must be a Pydantic model class or an OutputCodec")


def _resolved_runtime_bindings(
    bindings: RuntimeBindings | None,
    *,
    composition: Any,
    skill_registry: SkillRegistry | None,
    checkpoint_store: CheckpointStore | None,
    diagnostic_sink: DiagnosticSink | None,
) -> RuntimeBindings:
    if bindings is not None and not isinstance(bindings, RuntimeBindings):
        raise TypeError("runtime_bindings must be RuntimeBindings or None")
    values = {
        item.name: (None if bindings is None else getattr(bindings, item.name))
        for item in fields(RuntimeBindings)
    }
    actual = {
        "tool_registry": composition.tool_registry,
        "skill_registry": skill_registry,
        "conversation_store": composition.runtime.conversation_store,
        "checkpoint_store": checkpoint_store,
        "tool_authorizer": composition.tool_executor.authorizer,
        "event_sink": composition.runtime.event_sink,
        "diagnostic_sink": diagnostic_sink,
    }
    for name, component in actual.items():
        configured = values[name]
        if configured is not None and configured is not component:
            raise ConfigurationError(
                f"runtime binding {name} does not match the composed component"
            )
        values[name] = component
    return RuntimeBindings(**values)


def _definition_registration(
    definition: AgentDefinition | None,
    bindings: RuntimeBindings,
    *,
    model: ModelClient,
) -> tuple[DefinitionStatus | None, Any | None]:
    if definition is None:
        return None, None
    router = bindings.model_router
    if router is not None:
        resolved_model = router.resolve(definition.model_route)
        if inspect.isawaitable(resolved_model):
            raise ConfigurationError(
                "model_router.resolve() must be synchronous during composition"
            )
        if resolved_model is not model:
            raise ConfigurationError(
                "model_route does not resolve to the composed model instance"
            )
    registry = bindings.agent_registry
    if registry is None:
        return None, None
    try:
        descriptor = registry.descriptor(definition.ref)
    except LookupError:
        return None, None
    if inspect.isawaitable(descriptor):
        raise ConfigurationError(
            "agent_registry.descriptor() must be synchronous during composition"
        )
    if getattr(descriptor, "definition_fingerprint", None) != definition.fingerprint:
        raise ConfigurationError(
            "registered AgentDefinition fingerprint does not match"
        )
    status = getattr(descriptor, "status", None)
    if not isinstance(status, DefinitionStatus):
        raise ConfigurationError("registered AgentDefinition status is invalid")
    endpoint = None
    if status.runnable_in_production:
        endpoint = registry.resolve(definition.ref)
        if inspect.isawaitable(endpoint):
            raise ConfigurationError(
                "agent_registry.resolve() must be synchronous during composition"
            )
    return status, endpoint


def _resolved_semantic_digests(
    spec: AgentSpec,
    *,
    tool_authorizer: ToolAuthorizer,
) -> dict[str, str]:
    return {
        "instructions": _semantic_digest(spec.instructions),
        "model_capabilities": _semantic_digest(dict(spec.model_capabilities)),
        "tools": _semantic_digest([tool.to_dict() for tool in spec.tools]),
        "skills": _semantic_digest(dict(spec.skill_policy)),
        "input_contract": _semantic_digest({"type": "string"}),
        "output_contract": _semantic_digest(dict(spec.output_contract)),
        "memory_policy": _semantic_digest(
            spec.persistence_policy.get("conversation_memory_policy")
        ),
        "authorization_policy": _authorization_policy_digest(tool_authorizer),
    }


def _semantic_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _authorization_policy_digest(authorizer: ToolAuthorizer) -> str:
    fingerprint = getattr(authorizer, "policy_fingerprint", None)
    if (
        isinstance(fingerprint, str)
        and fingerprint.startswith("sha256:")
        and len(fingerprint) == 71
    ):
        return fingerprint
    return _semantic_digest(
        f"{type(authorizer).__module__}.{type(authorizer).__qualname__}"
    )


def _validate_definition_binding(
    definition: AgentDefinition,
    *,
    config: AgentConfig,
    spec: AgentSpec,
    semantic_digests: Mapping[str, str],
    skill_registry: SkillRegistry | None,
) -> None:
    mismatches: list[str] = []
    if definition.agent_id != config.name:
        mismatches.append("agent_id")
    if definition.limits != config.limits:
        mismatches.append("limits")
    if definition.execution_profile != spec.execution_profile.kind:
        mismatches.append("execution_profile")
    actual_tools = tuple(tool.name for tool in spec.tools)
    if definition.tool_refs != actual_tools:
        mismatches.append("tool_refs")
    actual_skills = (
        ()
        if skill_registry is None
        else tuple(descriptor.name for descriptor in skill_registry.descriptors)
    )
    if definition.skill_refs != actual_skills:
        mismatches.append("skill_refs")
    for key, expected in definition.semantic_digests.items():
        if semantic_digests.get(key) != expected:
            mismatches.append(f"semantic_digests.{key}")
    if mismatches:
        raise ConfigurationError(
            "AgentDefinition does not match the resolved Agent: "
            + ", ".join(sorted(set(mismatches)))
        )


def _effective_binding(name: str, explicit: Any, bound: Any) -> Any:
    if explicit is not None and bound is not None and explicit is not bound:
        raise ConfigurationError(f"{name} does not match runtime_bindings.{name}")
    return explicit if explicit is not None else bound


def _delegated_agent_tools(registry: Any) -> tuple[Any, ...]:
    from moduagent.delegation import DelegatedAgentTool

    return tuple(tool for tool in registry if isinstance(tool, DelegatedAgentTool))


def _resolved_delegation_bindings(
    bindings: RuntimeBindings,
    tools: tuple[Any, ...],
) -> RuntimeBindings:
    if not tools:
        return bindings
    coordinators = tuple(tool.coordinator for tool in tools)
    coordinator = coordinators[0]
    if any(item is not coordinator for item in coordinators[1:]):
        raise ConfigurationError(
            "all DelegatedAgentTools on one Agent must share a coordinator"
        )
    actual = {
        "agent_registry": coordinator.registry,
        "delegation_authorizer": coordinator.authorizer,
        "delegation_receipt_store": coordinator.receipt_store,
        "execution_group_store": coordinator.execution_group_binding,
    }
    values = {
        item.name: getattr(bindings, item.name) for item in fields(RuntimeBindings)
    }
    for name, component in actual.items():
        configured = values[name]
        if configured is not None and configured is not component:
            raise ConfigurationError(
                f"runtime binding {name} does not match DelegationCoordinator"
            )
        if configured is None:
            values[name] = component
    return RuntimeBindings(**values)


def _common_delegation_limits(
    tools: tuple[Any, ...],
    *,
    definition: AgentDefinition | None,
) -> Any | None:
    if not tools:
        return None
    coordinators = tuple(tool.coordinator for tool in tools)
    if any(item is not coordinators[0] for item in coordinators[1:]):
        raise ConfigurationError(
            "all DelegatedAgentTools on one Agent must share a coordinator"
        )
    callers = {tool.caller for tool in tools}
    if len(callers) != 1:
        raise ConfigurationError(
            "all DelegatedAgentTools on one Agent must share a caller AgentRef"
        )
    if definition is not None and next(iter(callers)) != definition.ref:
        raise ConfigurationError(
            "DelegatedAgentTool caller does not match the AgentDefinition"
        )
    limits = tuple(
        getattr(getattr(tool, "coordinator", None), "limits", None) for tool in tools
    )
    limits = tuple(value for value in limits if value is not None)
    if not limits:
        return None
    if any(value != limits[0] for value in limits[1:]):
        raise ConfigurationError(
            "all DelegatedAgentTools on one Agent must share execution-group limits"
        )
    return limits[0]


def _unapproved_delegation_endpoint_refs(tools: tuple[Any, ...]) -> tuple[str, ...]:
    """Resolve every delegated callee at composition and report unsafe targets."""

    unsafe: set[str] = set()
    for tool in tools:
        ref = getattr(tool, "callee", None)
        registry = getattr(getattr(tool, "coordinator", None), "registry", None)
        resolve = getattr(registry, "resolve", None)
        if ref is None or not callable(resolve):
            unsafe.add(str(ref) if ref is not None else "unresolved-agent")
            continue
        try:
            endpoint = resolve(ref)
        except Exception:
            # Production must not certify an endpoint that cannot be pinned at
            # composition. Runtime still repeats the approval check on every
            # invocation so a later registry rebind also fails closed.
            unsafe.add(str(ref))
            continue
        if inspect.isawaitable(endpoint):
            unsafe.add(str(ref))
            continue
        target = getattr(endpoint, "endpoint", None)
        if getattr(target, "approved", None) is not True:
            unsafe.add(str(ref))
    return tuple(sorted(unsafe))


def _delegated_finish_reason_code(finish_reason: str) -> str:
    return {
        "timeout": "delegated_agent_timeout",
        "cancelled": "delegated_agent_cancelled",
        "max_steps": "delegated_agent_max_steps",
        "max_tool_calls": "delegated_agent_max_tool_calls",
        "max_model_turns": "delegated_agent_max_model_turns",
        "no_progress": "delegated_agent_no_progress",
        "output_validation": "delegated_agent_output_validation_failed",
    }.get(finish_reason, "delegated_agent_failed")

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from moduagent.tools.base import (
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolFailure,
    ToolSchema,
    normalize_tool_side_effect_level,
)
from moduagent.tools.failure import ToolSafetyProfile

from .coordinator import (
    DelegationCall,
    DelegationCoordinator,
    DelegationFailure,
    _require_object_model,
    _resolve_max_result_bytes,
)
from .models import AgentRef, ParentDelegationContext, _classification
from .sessions import SessionStrategy


PARENT_DELEGATION_CONTEXT_KEY = "_moduagent_parent_delegation_context"
DELEGATION_EVENT_CALLBACK_KEY = "_moduagent_delegation_event_callback"


@runtime_checkable
class ParentContextResolver(Protocol):
    def resolve(
        self,
        context: ToolExecutionContext,
        *,
        caller: AgentRef,
    ) -> ParentDelegationContext: ...


class ToolMetadataParentContextResolver:
    """Read the runtime-owned typed context hidden from model arguments."""

    def resolve(
        self,
        context: ToolExecutionContext,
        *,
        caller: AgentRef,
    ) -> ParentDelegationContext:
        value = context.metadata.get(PARENT_DELEGATION_CONTEXT_KEY)
        if not isinstance(value, ParentDelegationContext):
            raise DelegationFailure("delegation_parent_context_missing")
        if value.lineage.agent_ref != caller:
            raise DelegationFailure("delegation_parent_context_mismatch")
        expected_run_id = value.current_run_id
        if expected_run_id is None and value.lineage.depth == 0:
            expected_run_id = value.lineage.root_run_id
        if context.run_id and expected_run_id != context.run_id:
            raise DelegationFailure("delegation_parent_context_mismatch")
        if context.session_id and value.parent_session_id != context.session_id:
            raise DelegationFailure("delegation_parent_context_mismatch")
        return value


class DelegatedAgentTool:
    """Typed Agent-as-Tool adapter backed by a DelegationCoordinator."""

    def __init__(
        self,
        *,
        coordinator: DelegationCoordinator,
        caller: AgentRef,
        callee: AgentRef,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        name: str,
        description: str,
        request_classification: str = "internal",
        session_strategy: SessionStrategy = SessionStrategy.ISOLATED,
        context_resolver: ParentContextResolver | None = None,
        max_result_bytes: int | None = None,
        allow_resume: bool = False,
        side_effect_level: str | None = None,
        expected_definition_fingerprint: str,
    ) -> None:
        if not isinstance(coordinator, DelegationCoordinator):
            raise TypeError("coordinator must be a DelegationCoordinator")
        for ref, field_name in ((caller, "caller"), (callee, "callee")):
            if not isinstance(ref, AgentRef):
                raise TypeError(f"{field_name} must be an AgentRef")
        for model, field_name in (
            (input_model, "input_model"),
            (output_model, "output_model"),
        ):
            _require_object_model(model, field_name)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name cannot be empty")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description cannot be empty")
        _classification(request_classification, "request_classification")
        resolved_max_result_bytes = _resolve_max_result_bytes(max_result_bytes)
        if type(allow_resume) is not bool:
            raise TypeError("allow_resume must be a bool")
        if (
            not isinstance(expected_definition_fingerprint, str)
            or not expected_definition_fingerprint
        ):
            raise ValueError("expected_definition_fingerprint cannot be empty")
        resolver = context_resolver or ToolMetadataParentContextResolver()
        if not isinstance(resolver, ParentContextResolver):
            raise TypeError("context_resolver must implement ParentContextResolver")
        self.coordinator = coordinator
        self.caller = caller
        self.callee = callee
        self.input_model = input_model
        self.args_schema = input_model
        self.output_model = output_model
        self.name = name
        self.description = description
        self.request_classification = request_classification
        self.session_strategy = SessionStrategy(session_strategy)
        self.context_resolver = resolver
        self.allow_resume = allow_resume
        self.expected_definition_fingerprint = expected_definition_fingerprint
        self.side_effect_level = normalize_tool_side_effect_level(side_effect_level)
        self.timeout_seconds = None
        self.max_result_bytes = resolved_max_result_bytes
        self.safety_profile = ToolSafetyProfile()
        self.idempotent = False
        self.repair_safe = False
        self.timeout_retry_safe = False
        self._schema = ToolSchema(
            name=name,
            description=description,
            parameters=input_model.model_json_schema(),
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = self.input_model.model_validate(dict(arguments))
        return validated.model_dump(mode="python", by_alias=True)

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> BaseModel:
        return await self.invoke_validated(self.validate_arguments(arguments), context)

    async def invoke_validated(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> BaseModel:
        execution_context = context if context is not None else ToolExecutionContext()
        try:
            if not execution_context.run_id:
                raise DelegationFailure("delegation_parent_run_missing")
            if not execution_context.tool_call_id:
                raise DelegationFailure("delegation_tool_call_id_missing")
            parent = self.context_resolver.resolve(
                execution_context,
                caller=self.caller,
            )
            request = self.input_model.model_validate(dict(arguments))
            event_callback = execution_context.metadata.get(
                DELEGATION_EVENT_CALLBACK_KEY
            )
            if event_callback is not None and not callable(event_callback):
                raise DelegationFailure("delegation_event_callback_invalid")
            return await self.coordinator.delegate(
                DelegationCall(
                    caller=self.caller,
                    callee=self.callee,
                    request=request,
                    output_model=self.output_model,
                    parent=parent,
                    parent_run_id=execution_context.run_id,
                    parent_tool_call_id=execution_context.tool_call_id,
                    expected_definition_fingerprint=(
                        self.expected_definition_fingerprint
                    ),
                    request_classification=self.request_classification,
                    session_strategy=self.session_strategy,
                    allow_resume=self.allow_resume,
                    event_callback=event_callback,
                    max_result_bytes=self.max_result_bytes,
                )
            )
        except ToolFailure:
            raise
        except DelegationFailure as exc:
            raise _tool_failure(exc) from exc
        except Exception as exc:
            raise _tool_failure(DelegationFailure("delegation_tool_failed")) from exc


def _tool_failure(error: DelegationFailure) -> ToolFailure:
    error_type = {
        "unauthorized": ToolErrorType.UNAUTHORIZED,
        "timeout": ToolErrorType.TIMEOUT,
        "cancelled": ToolErrorType.CANCELLED,
        "result_too_large": ToolErrorType.RESULT_TOO_LARGE,
    }.get(error.kind, ToolErrorType.EXECUTION_ERROR)
    message = {
        ToolErrorType.UNAUTHORIZED: "delegated Agent request was denied",
        ToolErrorType.TIMEOUT: "delegated Agent timed out",
        ToolErrorType.CANCELLED: "delegated Agent was cancelled",
        ToolErrorType.RESULT_TOO_LARGE: (
            "delegated Agent result exceeded its size limit"
        ),
    }.get(error_type, "delegated Agent execution failed")
    return ToolFailure(
        ToolError(
            error_type,
            message,
            retryable=False,
            details={},
            reason=error.code,
            recovery=None,
        )
    )

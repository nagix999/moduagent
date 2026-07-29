from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

from moduagent.composition import (
    AgentSpec,
    ExecutionProfile,
    compose_agent,
)
from moduagent.config import AgentConfig
from moduagent.decision import DecisionPolicy
from moduagent.execution import ExecutionEngine
from moduagent.memory import ConversationMemoryPolicy
from moduagent.models import ModelClient
from moduagent.observability import DiagnosticSink, EventSink
from moduagent.output import OutputCodec
from moduagent.persistence import (
    CheckpointStore,
    ConversationStore,
)
from moduagent.runtime import AgentEvent, AgentResult, RunRequest
from moduagent.skills import SkillLimits, SkillRegistry, SkillSelector
from moduagent.tools import (
    Tool,
    ToolAuthorizer,
)


class Agent:
    """Small public facade that delegates execution to :class:`AgentRuntime`."""

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
        skill_registry: SkillRegistry | None = None,
        skill_selector: SkillSelector | None = None,
        skill_limits: SkillLimits | None = None,
    ) -> None:
        composition = compose_agent(
            config=config,
            model=model,
            tools=tools,
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
            conversation_memory_policy=conversation_memory_policy,
            skill_registry=skill_registry,
            skill_selector=skill_selector,
            skill_limits=skill_limits,
        )
        self.config = config
        self.model = model
        self.skill_registry = skill_registry
        self.spec = composition.spec
        self.skill_runtime = composition.skill_runtime
        self.tool_registry = composition.tool_registry
        self.tool_executor = composition.tool_executor
        self.conversation_memory_policy = composition.conversation_memory_policy
        self.engine = composition.engine
        self.runtime = composition.runtime
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

    async def resume(self, run_id: str, *, session_id: str) -> AgentResult:
        return await self.run(
            "",
            session_id=session_id,
            resume_run_id=run_id,
        )

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

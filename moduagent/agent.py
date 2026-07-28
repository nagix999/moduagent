from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

from moduagent.config import AgentConfig
from moduagent.decision import DecisionPolicy, StandardDecisionPolicy
from moduagent.memory import ConversationMemoryPolicy, FullConversationMemoryPolicy
from moduagent.models import ModelClient
from moduagent.observability import EventSink, NoopEventSink
from moduagent.output import OutputCodec, TextOutputCodec
from moduagent.persistence import (
    CheckpointStore,
    ConversationStore,
    InMemoryConversationStore,
)
from moduagent.runtime import AgentEvent, AgentResult, AgentRuntime, RunRequest
from moduagent.skills import SkillLimits, SkillRegistry, SkillSelector
from moduagent.skills.runtime import SkillRuntime
from moduagent.skills.tools import SkillReadTool, SkillSearchTool
from moduagent.tools import (
    AllowAllAuthorizer,
    Tool,
    ToolAuthorizer,
    ToolExecutor,
    ToolRegistry,
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
        output_codec: OutputCodec | None = None,
        event_sink: EventSink | None = None,
        tool_authorizer: ToolAuthorizer | None = None,
        checkpoint_store: CheckpointStore | None = None,
        conversation_memory_policy: ConversationMemoryPolicy | None = None,
        skill_registry: SkillRegistry | None = None,
        skill_selector: SkillSelector | None = None,
        skill_limits: SkillLimits | None = None,
    ) -> None:
        if skill_selector is not None and skill_registry is None:
            raise ValueError("skill_selector requires skill_registry")
        self.config = config
        self.model = model
        self.skill_registry = skill_registry
        self.skill_runtime = (
            SkillRuntime(
                skill_registry,
                selector=skill_selector,
                limits=skill_limits,
            )
            if skill_registry is not None
            else None
        )
        registered_tools = tuple(tools)
        if self.skill_runtime is not None:
            registered_tools = (
                *registered_tools,
                SkillReadTool(self.skill_runtime),
                SkillSearchTool(self.skill_runtime),
            )
        self.tool_registry = ToolRegistry(registered_tools)
        self.tool_executor = ToolExecutor(
            self.tool_registry,
            authorizer=tool_authorizer or AllowAllAuthorizer(),
            retry=config.retry,
        )
        self.conversation_memory_policy = (
            conversation_memory_policy
            if conversation_memory_policy is not None
            else FullConversationMemoryPolicy()
        )
        self.runtime = AgentRuntime(
            config=config,
            model=model,
            decision_policy=decision_policy or StandardDecisionPolicy(),
            tool_executor=self.tool_executor,
            conversation_store=(conversation_store or InMemoryConversationStore()),
            output_codec=output_codec or TextOutputCodec(),
            event_sink=event_sink or NoopEventSink(),
            checkpoint_store=checkpoint_store,
            conversation_memory_policy=self.conversation_memory_policy,
            skill_runtime=self.skill_runtime,
        )

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

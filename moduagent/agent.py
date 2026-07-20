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
    ) -> None:
        self.config = config
        self.model = model
        self.tool_registry = ToolRegistry(tools)
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
        )

    async def run(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_context: Mapping[str, Any] | None = None,
        resume_run_id: str | None = None,
    ) -> AgentResult:
        request = self._request(
            text,
            session_id=session_id,
            user_context=user_context,
            resume_run_id=resume_run_id,
        )
        return await self.runtime.execute(request)

    def stream(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_context: Mapping[str, Any] | None = None,
        resume_run_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        request = self._request(
            text,
            session_id=session_id,
            user_context=user_context,
            resume_run_id=resume_run_id,
        )
        return self.runtime.stream(request)

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
    ) -> RunRequest:
        if not isinstance(text, str):
            raise TypeError("agent input must be a string")
        return RunRequest(
            input=text,
            session_id=session_id or uuid.uuid4().hex,
            user_context=dict(user_context or {}),
            resume_run_id=resume_run_id,
        )

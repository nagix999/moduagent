from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from moduagent.tools.base import ToolExecutionContext, ToolSchema

if TYPE_CHECKING:
    from moduagent.runtime.context import AgentResult


class _AgentLike(Protocol):
    async def run(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_context: Mapping[str, Any] | None = None,
    ) -> "AgentResult | Any": ...


class AgentToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(description="Task or message to delegate to the agent")


class AgentTool:
    """Expose another Agent through the standard Tool contract."""

    def __init__(
        self,
        agent: _AgentLike,
        *,
        name: str | None = None,
        description: str | None = None,
        idempotent: bool = False,
        timeout_seconds: float | None = None,
        max_result_bytes: int | None = None,
    ) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_result_bytes is not None and max_result_bytes < 1:
            raise ValueError("max_result_bytes must be at least 1")

        config = getattr(agent, "config", None)
        inferred_name = getattr(config, "name", None)
        self.agent = agent
        self.name = name or inferred_name or "agent"
        if not self.name.strip():
            raise ValueError("tool name cannot be empty")
        self.description = description or f"Delegate a task to the {self.name} agent"
        self.idempotent = idempotent
        self.timeout_seconds = timeout_seconds
        self.max_result_bytes = max_result_bytes
        self.input_model = AgentToolInput
        self.args_schema = self.input_model
        self._schema = ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        values = self.input_model.model_validate(dict(arguments))
        return {"input": values.input}

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Any:
        return await self.invoke_validated(self.validate_arguments(arguments), context)

    async def invoke_validated(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Any:
        context = context if context is not None else ToolExecutionContext()
        result = await self.agent.run(
            str(arguments["input"]),
            session_id=context.session_id,
            user_context=context.user_context,
        )
        return getattr(result, "output", result)

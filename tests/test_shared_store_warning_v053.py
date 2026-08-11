from __future__ import annotations

import warnings
from typing import Any

import pytest

from moduagent import Agent, AgentTool, InMemoryConversationStore, ModelCapabilities


class StaticModel:
    capabilities = ModelCapabilities(streaming=False)

    async def complete(self, request: Any) -> Any:
        del request
        raise AssertionError("the model should not be called")


def _agent(*, store: InMemoryConversationStore, tools: tuple[Any, ...] = ()) -> Agent:
    return Agent.create(
        model=StaticModel(),
        instructions="Test agent.",
        tools=tools,
        conversation_store=store,
    )


def test_agent_tool_warns_once_when_parent_and_child_share_store() -> None:
    store = InMemoryConversationStore()
    child = _agent(store=store)

    with pytest.warns(
        RuntimeWarning, match="shares the parent ConversationStore"
    ) as seen:
        _agent(
            store=store,
            tools=(AgentTool(child, name="child_a"), AgentTool(child, name="child_b")),
        )

    assert len(seen) == 1


def test_agent_tool_does_not_warn_for_distinct_store_objects() -> None:
    child = _agent(store=InMemoryConversationStore())

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _agent(
            store=InMemoryConversationStore(),
            tools=(AgentTool(child, name="child"),),
        )

    assert not [
        warning
        for warning in seen
        if "shares the parent ConversationStore" in str(warning.message)
    ]


def test_shared_store_diagnostic_does_not_reject_dynamic_agent_attributes() -> None:
    class DynamicAgent:
        @property
        def runtime(self) -> Any:
            raise RuntimeError("dynamic runtime is unavailable during composition")

        async def run(self, text: str, *, session_id=None, user_context=None) -> str:
            del text, session_id, user_context
            return "unused"

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        parent = _agent(
            store=InMemoryConversationStore(),
            tools=(AgentTool(DynamicAgent(), name="dynamic"),),
        )

    assert parent.tool_registry.require("dynamic").name == "dynamic"
    assert not [
        warning
        for warning in seen
        if "shares the parent ConversationStore" in str(warning.message)
    ]

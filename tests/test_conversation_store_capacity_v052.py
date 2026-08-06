from __future__ import annotations

import asyncio

from moduagent import InMemoryConversationStore, Message, ToolCall


def test_in_memory_store_defensively_copies_nested_message_content() -> None:
    async def scenario() -> None:
        metadata_values = ["original-metadata"]
        argument_values = ["original-argument"]
        message = Message.assistant(
            None,
            (
                ToolCall(
                    "call-1",
                    "lookup",
                    {"values": argument_values},
                ),
            ),
            metadata={"values": metadata_values},
        )
        store = InMemoryConversationStore(max_total_bytes=1_024)
        await store.append("session", [message])
        expected_stats = await store.stats()

        metadata_values.append("PRIVATE-" + "x" * 2_000)
        argument_values.append("PRIVATE-" + "y" * 2_000)

        first_load = await store.load("session")
        assert first_load[0].metadata["values"] == ["original-metadata"]
        assert first_load[0].tool_calls[0].arguments["values"] == ["original-argument"]

        loaded_metadata = first_load[0].metadata["values"]
        loaded_arguments = first_load[0].tool_calls[0].arguments["values"]
        assert isinstance(loaded_metadata, list)
        assert isinstance(loaded_arguments, list)
        loaded_metadata.append("changed-through-load")
        loaded_arguments.append("changed-through-load")

        second_load = await store.load("session")
        assert second_load[0].metadata["values"] == ["original-metadata"]
        assert second_load[0].tool_calls[0].arguments["values"] == ["original-argument"]
        assert await store.stats() == expected_stats

    asyncio.run(scenario())


def test_capacity_pressure_removes_expired_sessions_before_live_lru() -> None:
    async def scenario() -> None:
        now = [0.0]
        store = InMemoryConversationStore(
            ttl_seconds=10,
            ttl_sweep_interval_seconds=100,
            max_sessions=2,
            clock=lambda: now[0],
        )
        await store.append("expired", [Message.user("expired")])
        now[0] = 5.0
        await store.append("live", [Message.user("live")])
        now[0] = 9.0
        assert await store.load("expired") == [Message.user("expired")]

        # The expired session is the most recently used at this point. Capacity
        # enforcement must still remove it instead of the older live session.
        now[0] = 10.1
        await store.append("new", [Message.user("new")])

        assert await store.load("live") == [Message.user("live")]
        assert await store.load("expired") == []
        assert await store.load("new") == [Message.user("new")]
        assert await store.stats() == {
            "sessions": 2,
            "total_bytes": sum(
                len(('{"role":"user","content":"' + content + '"}').encode("utf-8"))
                for content in ("live", "new")
            ),
        }

    asyncio.run(scenario())

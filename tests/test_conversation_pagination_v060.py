from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

import moduagent.persistence.conversation as conversation_module
from moduagent.messages import Message
from moduagent.persistence.conversation import (
    ConversationCursorError,
    DatabaseConversationStore,
    InMemoryConversationStore,
    PaginatedConversationStore,
    RedisConversationStore,
)


def _row(message: Message) -> str:
    return json.dumps(message.to_dict(), ensure_ascii=False, separators=(",", ":"))


class _RedisListClient:
    def __init__(self) -> None:
        self.rows: dict[str, list[str]] = {}
        self.lrange_calls: list[tuple[str, int, int]] = []

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        self.lrange_calls.append((key, start, end))
        values = self.rows.get(key, [])
        return values[start:] if end == -1 else values[start : end + 1]

    async def rpush(self, key: str, *rows: str) -> int:
        values = self.rows.setdefault(key, [])
        values.extend(rows)
        return len(values)

    async def llen(self, key: str) -> int:
        return len(self.rows.get(key, []))

    async def delete(self, key: str) -> int:
        return int(self.rows.pop(key, None) is not None)


class _PaginatedRepository:
    def __init__(self, rows: Sequence[str | Mapping[str, Any]]) -> None:
        self.rows = list(rows)
        self.page_calls: list[tuple[str, int, int]] = []
        self.full_load_calls = 0

    async def load_messages(
        self,
        session_id: str,
    ) -> Sequence[str | Mapping[str, Any]]:
        del session_id
        self.full_load_calls += 1
        raise AssertionError("paginated reads must not load the full conversation")

    async def load_messages_page(
        self,
        session_id: str,
        after_sequence: int,
        limit: int,
    ) -> Sequence[str | Mapping[str, Any]]:
        self.page_calls.append((session_id, after_sequence, limit))
        return self.rows[after_sequence : after_sequence + limit]

    async def append_messages(
        self,
        session_id: str,
        messages: Sequence[str],
    ) -> None:
        del session_id
        self.rows.extend(messages)

    async def clear_messages(self, session_id: str) -> None:
        del session_id
        self.rows.clear()


def test_in_memory_long_session_load_tail_decodes_only_the_requested_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store = InMemoryConversationStore()
        messages = [Message.user(f"message-{index}") for index in range(1, 20_001)]
        await store.append("long-session", messages)

        decoded_rows = 0
        original = conversation_module._decode_message_row

        def count_decode(row: Any) -> Message:
            nonlocal decoded_rows
            decoded_rows += 1
            return original(row)

        monkeypatch.setattr(conversation_module, "_decode_message_row", count_decode)
        page = await store.load_tail("long-session", 19_990, 5)

        assert isinstance(store, PaginatedConversationStore)
        assert [item.sequence for item in page.items] == list(range(19_991, 19_996))
        assert [message.content for message in page.messages] == [
            f"message-{index}" for index in range(19_991, 19_996)
        ]
        assert page.next_sequence == 19_995
        assert page.has_more is True
        assert decoded_rows == 5
        assert len({item.message_id for item in page.items}) == 5

        repeated = await store.load_tail("long-session", 19_990, 5)
        assert [item.message_id for item in repeated.items] == [
            item.message_id for item in page.items
        ]

        tail = await store.load_tail("long-session", page.next_sequence, 100)
        assert tail.next_sequence == 20_000
        assert tail.has_more is False
        assert len(tail.items) == 5
        with pytest.raises(ConversationCursorError):
            await store.load_tail("long-session", 20_001, 1)

    asyncio.run(scenario())


def test_redis_list_load_tail_uses_one_bounded_lrange() -> None:
    async def scenario() -> None:
        client = _RedisListClient()
        store = RedisConversationStore(client)
        await store.append(
            "redis-long",
            [Message.user(f"message-{index}") for index in range(1, 5_001)],
        )
        client.lrange_calls.clear()

        page = await store.load_tail("redis-long", 4_990, 4)

        assert [item.sequence for item in page.items] == [4_991, 4_992, 4_993, 4_994]
        assert page.has_more is True
        assert client.lrange_calls == [
            ("moduagent:conversation:redis-long", 4_990, 4_994)
        ]

    asyncio.run(scenario())


def test_database_load_tail_prefers_repository_pagination() -> None:
    async def scenario() -> None:
        repository = _PaginatedRepository(
            [_row(Message.user(f"message-{index}")) for index in range(1, 5_001)]
        )
        store = DatabaseConversationStore(repository)

        page = await store.load_tail("db-long", 4_990, 4)

        assert [item.sequence for item in page.items] == [4_991, 4_992, 4_993, 4_994]
        assert [message.content for message in page.messages] == [
            "message-4991",
            "message-4992",
            "message-4993",
            "message-4994",
        ]
        assert page.has_more is True
        assert repository.page_calls == [("db-long", 4_990, 5)]
        assert repository.full_load_calls == 0

    asyncio.run(scenario())


def test_source_message_id_is_stable_across_json_mapping_key_order() -> None:
    async def scenario() -> None:
        first_row = Message.user(
            "same-content",
            metadata={"alpha": "a", "nested": {"first": 1, "second": 2}},
        ).to_dict()
        second_row = dict(reversed(tuple(first_row.items())))
        second_row["metadata"] = {
            "nested": {"second": 2, "first": 1},
            "alpha": "a",
        }
        first = DatabaseConversationStore(_PaginatedRepository([first_row]))
        second = DatabaseConversationStore(_PaginatedRepository([second_row]))

        first_page = await first.load_tail("same-session", 0, 1)
        second_page = await second.load_tail("same-session", 0, 1)

        assert first_page.messages == second_page.messages
        assert first_page.items[0].message_id == second_page.items[0].message_id

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("after_sequence", "limit"),
    [(-1, 1), (True, 1), (0, 0), (0, 10_001), (0, True)],
)
def test_load_tail_rejects_invalid_cursor_or_page_size(
    after_sequence: Any,
    limit: Any,
) -> None:
    store = InMemoryConversationStore()

    with pytest.raises(ValueError):
        asyncio.run(store.load_tail("session", after_sequence, limit))

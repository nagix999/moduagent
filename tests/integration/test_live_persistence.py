from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest

from moduagent.messages import Message, ToolCall
from moduagent.persistence import (
    DatabaseConversationStore,
    RedisCheckpointStore,
    RedisConversationStore,
    RunCheckpoint,
)


REDIS_URL = os.getenv("REDIS_URL")


class SQLiteConversationRepository:
    """Small real repository used to verify the database store contract."""

    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute(
            """
            CREATE TABLE conversation_messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    async def load_messages(self, session_id: str) -> list[str]:
        cursor = self._connection.execute(
            """
            SELECT message_json
            FROM conversation_messages
            WHERE session_id = ?
            ORDER BY sequence
            """,
            (session_id,),
        )
        return [row[0] for row in cursor.fetchall()]

    async def append_messages(
        self,
        session_id: str,
        messages: Sequence[str],
    ) -> None:
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO conversation_messages (session_id, message_json)
                VALUES (?, ?)
                """,
                ((session_id, message) for message in messages),
            )

    async def clear_messages(self, session_id: str) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM conversation_messages WHERE session_id = ?",
                (session_id,),
            )

    def close(self) -> None:
        self._connection.close()


def test_database_conversation_store_with_sqlite(tmp_path: Path) -> None:
    async def scenario() -> None:
        repository = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
        store = DatabaseConversationStore(repository)
        messages = [
            Message.user("휴가 잔여 일수를 알려줘"),
            Message.assistant(
                None,
                (
                    ToolCall(
                        "call-1",
                        "get_leave_balance",
                        {"employee_id": "employee-7"},
                    ),
                ),
            ),
            Message.tool("12일", call_id="call-1", name="get_leave_balance"),
        ]
        other_message = Message.user("별도 세션")

        try:
            await store.append("session-1", messages[:1])
            await store.append("session-1", messages[1:])
            await store.append("session-2", [other_message])

            assert await store.load("session-1") == messages
            assert await store.load("session-2") == [other_message]

            await store.clear("session-1")

            assert await store.load("session-1") == []
            assert await store.load("session-2") == [other_message]
        finally:
            repository.close()

    asyncio.run(scenario())


@pytest.mark.skipif(not REDIS_URL, reason="REDIS_URL is not configured")
def test_redis_conversation_and_checkpoint_stores_smoke() -> None:
    async def scenario() -> None:
        redis_asyncio = importlib.import_module("redis.asyncio")
        client = redis_asyncio.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        namespace = f"moduagent:integration:{uuid4().hex}:"
        conversation_prefix = f"{namespace}conversation:"
        checkpoint_prefix = f"{namespace}checkpoint:"
        session_id = "session-1"
        run_id = "run-1"
        conversation_key = f"{conversation_prefix}{session_id}"
        checkpoint_key = f"{checkpoint_prefix}{run_id}"
        conversation_store = RedisConversationStore(
            client,
            key_prefix=conversation_prefix,
        )
        checkpoint_store = RedisCheckpointStore(
            client,
            key_prefix=checkpoint_prefix,
        )
        messages = [Message.user("Redis 연결 확인"), Message.assistant("정상")]
        checkpoint = RunCheckpoint(
            run_id=run_id,
            session_id=session_id,
            input="Redis 연결 확인",
            messages=tuple(messages),
            new_messages=tuple(messages),
            step=1,
        )

        try:
            assert await client.ping()

            await conversation_store.append(session_id, messages)
            assert await conversation_store.load(session_id) == messages
            await conversation_store.clear(session_id)
            assert await conversation_store.load(session_id) == []

            await checkpoint_store.save(run_id, checkpoint)
            assert await checkpoint_store.load(run_id) == checkpoint
            await checkpoint_store.delete(run_id)
            assert await checkpoint_store.load(run_id) is None
        finally:
            await client.delete(conversation_key, checkpoint_key)
            close = getattr(client, "aclose", None) or client.close
            close_result = close()
            if inspect.isawaitable(close_result):
                await close_result

    asyncio.run(scenario())

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

import pytest

from moduagent import Message, ModelRequest, OllamaClient, VLLMClient


_PROMPT = "Reply with only the word pong."


def _required_environment(*names: str) -> Mapping[str, str]:
    values = {name: os.getenv(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip(f"live model environment is not configured: {', '.join(missing)}")
    return values


def _request() -> ModelRequest:
    return ModelRequest(messages=(Message.user(_PROMPT),))


async def _smoke_complete(client: VLLMClient | OllamaClient) -> None:
    async with client:
        response = await client.complete(_request())

    assert response.message.content is not None
    assert response.message.content.strip()


async def _smoke_stream(client: VLLMClient | OllamaClient) -> None:
    deltas: list[str] = []
    final_responses = []

    async with client:
        async for chunk in client.stream(_request()):
            if chunk.delta:
                deltas.append(chunk.delta)
            if chunk.response is not None:
                final_responses.append(chunk.response)

    assert deltas
    assert len(final_responses) == 1
    assert final_responses[0].message.content == "".join(deltas)


def _vllm_client() -> VLLMClient:
    env = _required_environment("VLLM_BASE_URL", "VLLM_MODEL")
    return VLLMClient(
        base_url=env["VLLM_BASE_URL"],
        model=env["VLLM_MODEL"],
        api_key=os.getenv("VLLM_API_KEY", "").strip() or None,
        default_options={"temperature": 0, "max_tokens": 64},
    )


def _ollama_client() -> OllamaClient:
    env = _required_environment("OLLAMA_BASE_URL", "OLLAMA_MODEL")
    return OllamaClient(
        base_url=env["OLLAMA_BASE_URL"],
        model=env["OLLAMA_MODEL"],
        default_options={"temperature": 0, "num_predict": 64},
    )


def test_vllm_complete_live() -> None:
    asyncio.run(_smoke_complete(_vllm_client()))


def test_vllm_stream_live() -> None:
    asyncio.run(_smoke_stream(_vllm_client()))


def test_ollama_complete_live() -> None:
    asyncio.run(_smoke_complete(_ollama_client()))


def test_ollama_stream_live() -> None:
    asyncio.run(_smoke_stream(_ollama_client()))

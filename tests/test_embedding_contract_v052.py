from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from moduagent.models import (
    ModelCapabilities,
    ModelProtocolError,
    OllamaClient,
    OpenAICompatibleClient,
)


class RecordingTransport:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, Any],
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        self.requests.append(
            {
                "url": url,
                "headers": dict(headers or {}),
                "json": dict(json),
                "timeout": timeout,
            }
        )
        return self.response


def _client(
    provider: str,
    response: Mapping[str, Any],
) -> tuple[OpenAICompatibleClient | OllamaClient, RecordingTransport]:
    transport = RecordingTransport(response)
    if provider == "openai":
        return (
            OpenAICompatibleClient(
                base_url="http://model.test/v1",
                model="embedding-model",
                transport=transport,
                capabilities=ModelCapabilities(embeddings=True),
            ),
            transport,
        )
    return (
        OllamaClient(
            base_url="http://ollama.test",
            model="embedding-model",
            transport=transport,
        ),
        transport,
    )


def _response(provider: str, vectors: list[Any]) -> Mapping[str, Any]:
    if provider == "openai":
        return {
            "data": [
                {"index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ]
        }
    return {"embeddings": vectors}


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_embedding_empty_batch_returns_without_http(provider: str) -> None:
    async def scenario() -> None:
        client, transport = _client(provider, _response(provider, []))

        assert await client.embed([]) == ()
        assert await client.embed(()) == ()
        assert transport.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize(
    "invalid_inputs",
    [b"PRIVATE-INPUT", 7, {"input": "PRIVATE-INPUT"}, ["valid", 7]],
)
def test_embedding_rejects_invalid_input_without_http(
    provider: str,
    invalid_inputs: Any,
) -> None:
    async def scenario() -> None:
        client, transport = _client(provider, _response(provider, [[0.1]]))

        with pytest.raises(ModelProtocolError) as captured:
            await client.embed(invalid_inputs)

        assert "PRIVATE-INPUT" not in str(captured.value)
        assert transport.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_embedding_requires_one_response_per_input(provider: str) -> None:
    async def scenario() -> None:
        client, _ = _client(provider, _response(provider, [[0.1, 0.2]]))

        with pytest.raises(ModelProtocolError, match="count"):
            await client.embed(["first", "second"])

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[]], "empty"),
        ([[0.1, 0.2], [0.3]], "dimensions"),
        ([[0.1, "PRIVATE-VALUE"]], "non-numeric"),
        ([[0.1, True]], "non-numeric"),
        ([[0.1, float("nan")]], "non-finite"),
        ([[0.1, float("inf")]], "non-finite"),
        ([[0.1, None]], "non-numeric"),
    ],
)
def test_embedding_rejects_invalid_vectors(
    provider: str,
    vectors: list[Any],
    message: str,
) -> None:
    async def scenario() -> None:
        client, _ = _client(provider, _response(provider, vectors))
        inputs = [f"input-{index}" for index in range(len(vectors))]

        with pytest.raises(ModelProtocolError, match=message) as captured:
            await client.embed(inputs)

        assert "PRIVATE-VALUE" not in str(captured.value)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"index": 0, "embedding": [0.1]},
            {"index": 0, "embedding": [0.2]},
        ],
        [
            {"index": 0, "embedding": [0.1]},
            {"index": 2, "embedding": [0.2]},
        ],
        [
            {"index": 0, "embedding": [0.1]},
            {"index": "1", "embedding": [0.2]},
        ],
        [
            {"index": 0, "embedding": [0.1]},
            {"index": True, "embedding": [0.2]},
        ],
        [
            {"index": 0, "embedding": [0.1]},
            {"index": 1.0, "embedding": [0.2]},
        ],
    ],
)
def test_openai_embedding_requires_exact_unique_integer_indices(
    rows: list[dict[str, Any]],
) -> None:
    async def scenario() -> None:
        client, _ = _client("openai", {"data": rows})

        with pytest.raises(ModelProtocolError, match="index|indices"):
            await client.embed(["first", "second"])

    asyncio.run(scenario())


def test_openai_embedding_orders_rows_by_exact_index() -> None:
    async def scenario() -> None:
        client, _ = _client(
            "openai",
            {
                "data": [
                    {"index": 1, "embedding": [3, 4]},
                    {"index": 0, "embedding": [1, 2]},
                ]
            },
        )

        assert await client.embed(["first", "second"]) == (
            (1.0, 2.0),
            (3.0, 4.0),
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_embedding_string_input_has_exactly_one_response(provider: str) -> None:
    async def scenario() -> None:
        client, transport = _client(provider, _response(provider, [[1, 2]]))

        assert await client.embed("one input") == ((1.0, 2.0),)
        assert transport.requests[0]["json"]["input"] == "one input"

    asyncio.run(scenario())

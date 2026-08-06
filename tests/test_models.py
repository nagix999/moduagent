from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
import pytest

import moduagent.models.ollama as ollama_module
import moduagent.models.openai_compatible as openai_module
from moduagent.messages import Message, ToolCall
from moduagent.memory import VLLMTokenCounter
from moduagent.models import (
    HttpxTransport,
    ModelCapabilities,
    ModelProtocolError,
    ModelRequest,
    ModelResponse,
    OllamaClient,
    OpenAICompatibleClient,
    VLLMClient,
)


class FakeTransport:
    def __init__(
        self,
        *,
        response: Mapping[str, Any] | None = None,
        lines: list[str] | None = None,
    ) -> None:
        self.response = dict(response or {})
        self.lines = list(lines or [])
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
                "kind": "post",
                "url": url,
                "headers": dict(headers or {}),
                "json": dict(json),
                "timeout": timeout,
            }
        )
        return self.response

    async def stream_lines(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        self.requests.append(
            {
                "kind": "stream",
                "url": url,
                "headers": dict(headers or {}),
                "json": dict(json),
                "timeout": timeout,
            }
        )
        for line in self.lines:
            yield line


class FakeToolSchema:
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }


def test_model_response_normalizes_tool_calls_on_message() -> None:
    call = ToolCall("call-1", "weather", {"city": "Seoul"})
    response = ModelResponse(
        Message.assistant(None),
        (call,),
        usage={"prompt_tokens": 2, "completion_tokens": 3},
    )

    assert response.message.tool_calls == response.tool_calls == (call,)
    assert response.usage.total_tokens == 5


def test_openai_complete_maps_request_and_response() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            response={
                "id": "chat-1",
                "model": "served-model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-7",
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": '{"city":"Seoul"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            }
        )
        client = OpenAICompatibleClient(
            base_url="http://model.internal/v1",
            model="company-model",
            api_key="token",
            transport=transport,
            default_options={"temperature": 0.1},
        )
        request = ModelRequest(
            messages=(Message.system("answer"), Message.user("weather?")),
            tools=(FakeToolSchema(),),
            output_schema={
                "title": "WeatherResult",
                "type": "object",
                "properties": {"temperature": {"type": "number"}},
            },
            options={"max_tokens": 80},
            provider_options={"seed": 7},
        )

        response = await client.complete(request)

        sent = transport.requests[0]
        assert sent["url"] == "http://model.internal/v1/chat/completions"
        assert sent["headers"]["Authorization"] == "Bearer token"
        assert sent["json"]["stream"] is False
        assert sent["json"]["temperature"] == 0.1
        assert sent["json"]["max_tokens"] == 80
        assert sent["json"]["seed"] == 7
        assert sent["json"]["tools"][0]["function"]["name"] == "weather"
        assert sent["json"]["response_format"]["json_schema"]["strict"] is True
        assert response.tool_calls == response.message.tool_calls
        assert response.tool_calls[0].arguments == {"city": "Seoul"}
        assert response.usage.total_tokens == 15
        assert response.provider_metadata["provider"] == "openai-compatible"

    asyncio.run(scenario())


def test_openai_sse_stream_assembles_text_tools_usage_and_terminal_response() -> None:
    async def scenario() -> None:
        events = [
            {
                "id": "chat-stream",
                "choices": [{"delta": {"content": "서울 "}, "finish_reason": None}],
            },
            {
                "id": "chat-stream",
                "choices": [
                    {
                        "delta": {
                            "content": "날씨",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "weather",
                                        "arguments": '{"city":',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"Seoul"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 3,
                    "total_tokens": 12,
                },
            },
        ]
        transport = FakeTransport(
            lines=[*(f"data: {json.dumps(event)}" for event in events), "data: [DONE]"]
        )
        client = OpenAICompatibleClient(
            base_url="http://model/v1", model="m", transport=transport
        )

        chunks = [
            chunk async for chunk in client.stream(ModelRequest((Message.user("go"),)))
        ]

        assert [chunk.delta for chunk in chunks[:-1]] == ["서울 ", "날씨"]
        assert chunks[-1].is_final
        response = chunks[-1].response
        assert response is not None
        assert response.message.content == "서울 날씨"
        assert response.tool_calls[0].arguments == {"city": "Seoul"}
        assert response.tool_calls == response.message.tool_calls
        assert response.usage.total_tokens == 12
        assert response.finish_reason == "tool_calls"
        assert transport.requests[0]["json"]["stream"] is True

    asyncio.run(scenario())


def test_openai_sse_stream_rejects_clean_eof_without_terminal_marker() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            lines=[
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": "partial"},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            ]
        )
        client = OpenAICompatibleClient(
            base_url="http://model/v1",
            model="m",
            transport=transport,
        )

        with pytest.raises(ModelProtocolError, match="terminal marker"):
            _ = [
                chunk
                async for chunk in client.stream(ModelRequest((Message.user("go"),)))
            ]

    asyncio.run(scenario())


def test_vllm_merges_deployment_extra_body_and_per_request_options() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            response={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        client = VLLMClient(
            base_url="http://vllm/v1",
            model="m",
            transport=transport,
            deployment_options={"chat_template": "template-a"},
            extra_body={"guided_decoding_backend": "xgrammar", "priority": 1},
        )

        response = await client.complete(
            ModelRequest((Message.user("go"),), provider_options={"priority": 2})
        )

        body = transport.requests[0]["json"]
        assert body["chat_template"] == "template-a"
        assert body["guided_decoding_backend"] == "xgrammar"
        assert body["priority"] == 2
        assert response.provider_metadata["provider"] == "vllm"

    asyncio.run(scenario())


def test_vllm_rejects_tools_and_final_output_schema_in_one_request() -> None:
    async def scenario() -> None:
        client = VLLMClient(
            base_url="http://vllm/v1",
            model="m",
            transport=FakeTransport(),
        )
        assert client.capabilities.tool_calling_with_structured_output is False
        assert client.capabilities.supports_tool_calling_with_structured_output is False
        request = ModelRequest(
            (Message.user("go"),),
            tools=(FakeToolSchema(),),
            output_schema={"type": "object", "properties": {}},
        )

        try:
            await client.complete(request)
        except ValueError as exc:
            assert "separate ACT and FINALIZE requests" in str(exc)
        else:
            raise AssertionError("combined vLLM constraints must be rejected")

    asyncio.run(scenario())


def test_vllm_respects_explicit_combined_contract_capability() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            response={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        client = VLLMClient(
            base_url="http://vllm/v1",
            model="m",
            transport=transport,
            capabilities=ModelCapabilities(
                tool_calling_with_structured_output=True,
            ),
        )

        await client.complete(
            ModelRequest(
                (Message.user("go"),),
                tools=(FakeToolSchema(),),
                output_schema={"type": "object", "properties": {}},
            )
        )

        assert client.capabilities.tool_calling_with_structured_output is True
        body = transport.requests[0]["json"]
        assert body["tools"]
        assert body["response_format"]["type"] == "json_schema"

    asyncio.run(scenario())


def test_vllm_finalization_drops_tool_only_default_options() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            response={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        client = VLLMClient(
            base_url="http://vllm/v1",
            model="m",
            transport=transport,
            default_options={
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "tools": [{"type": "function", "function": {"name": "unsafe"}}],
                "temperature": 0,
            },
            extra_body={
                "tools": [{"type": "function", "function": {"name": "unsafe-2"}}],
                "tool_choice": "required",
            },
        )

        await client.complete(
            ModelRequest(
                (Message.user("finalize"),),
                output_schema={"type": "object", "properties": {}},
                provider_options={"parallel_tool_calls": True},
            )
        )

        body = transport.requests[0]["json"]
        assert "tool_choice" not in body
        assert "parallel_tool_calls" not in body
        assert "tools" not in body
        assert body["temperature"] == 0
        assert body["response_format"]["type"] == "json_schema"

    asyncio.run(scenario())


def test_ollama_no_tool_request_drops_provider_supplied_tools() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            response={
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        )
        client = OllamaClient(
            base_url="http://ollama",
            model="qwen",
            transport=transport,
            provider_options={
                "extra_body": {
                    "tools": [{"type": "function", "function": {"name": "unsafe"}}]
                }
            },
        )

        await client.complete(ModelRequest((Message.user("answer directly"),)))

        assert "tools" not in transport.requests[0]["json"]

    asyncio.run(scenario())


def test_vllm_count_tokens_uses_server_chat_template() -> None:
    async def scenario() -> None:
        transport = FakeTransport(response={"count": 37, "max_model_len": 32768})
        client = VLLMClient(
            base_url="http://vllm/v1",
            model="m",
            api_key="token",
            transport=transport,
            deployment_options={"chat_template": "template-a"},
        )

        count = await VLLMTokenCounter(client).count_request(
            ModelRequest(
                (Message.system("answer"), Message.user("go")),
                tools=(FakeToolSchema(),),
            )
        )

        sent = transport.requests[0]
        assert count == 37
        assert sent["url"] == "http://vllm/tokenize"
        assert sent["headers"]["Authorization"] == "Bearer token"
        assert sent["json"]["model"] == "m"
        assert sent["json"]["messages"][1]["content"] == "go"
        assert sent["json"]["tools"][0]["function"]["name"] == "weather"
        assert sent["json"]["chat_template"] == "template-a"
        assert "stream" not in sent["json"]

    asyncio.run(scenario())


def test_ollama_complete_uses_native_payload_and_normalizes_response() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            response={
                "model": "qwen3:14b",
                "done": True,
                "done_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "weather",
                                "arguments": {"city": "Seoul"},
                            }
                        }
                    ],
                },
                "prompt_eval_count": 6,
                "eval_count": 2,
            }
        )
        client = OllamaClient(
            base_url="http://ollama:11434",
            model="qwen3:14b",
            transport=transport,
            default_options={"temperature": 0.2},
            provider_options={"keep_alive": "10m"},
        )
        request = ModelRequest(
            (Message.user("weather"),),
            (FakeToolSchema(),),
            output_schema={"type": "object"},
            options={"num_predict": 64},
        )

        response = await client.complete(request)

        body = transport.requests[0]["json"]
        assert transport.requests[0]["url"] == "http://ollama:11434/api/chat"
        assert body["stream"] is False
        assert body["keep_alive"] == "10m"
        assert body["options"] == {"temperature": 0.2, "num_predict": 64}
        assert body["format"] == {"type": "object"}
        assert response.tool_calls[0].id == "call-1"
        assert response.message.tool_calls == response.tool_calls
        assert response.usage.total_tokens == 8

    asyncio.run(scenario())


def test_ollama_jsonl_stream_always_ends_with_assembled_response() -> None:
    async def scenario() -> None:
        lines = [
            json.dumps(
                {
                    "model": "qwen",
                    "message": {"role": "assistant", "content": "안녕"},
                    "done": False,
                }
            ),
            json.dumps(
                {
                    "model": "qwen",
                    "message": {"role": "assistant", "content": "하세요"},
                    "done": False,
                }
            ),
            json.dumps(
                {
                    "model": "qwen",
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 4,
                    "eval_count": 2,
                }
            ),
        ]
        client = OllamaClient(
            base_url="http://ollama", model="qwen", transport=FakeTransport(lines=lines)
        )

        chunks = [
            chunk async for chunk in client.stream(ModelRequest((Message.user("hi"),)))
        ]

        assert [chunk.delta for chunk in chunks[:-1]] == ["안녕", "하세요"]
        assert chunks[-1].response is not None
        assert chunks[-1].response.message.content == "안녕하세요"
        assert chunks[-1].response.usage.total_tokens == 6

    asyncio.run(scenario())


def test_ollama_jsonl_stream_rejects_clean_eof_without_done_marker() -> None:
    async def scenario() -> None:
        client = OllamaClient(
            base_url="http://ollama",
            model="qwen",
            transport=FakeTransport(
                lines=[
                    json.dumps(
                        {
                            "model": "qwen",
                            "message": {
                                "role": "assistant",
                                "content": "partial",
                            },
                            "done": False,
                        }
                    )
                ]
            ),
        )

        with pytest.raises(ModelProtocolError, match="terminal marker"):
            _ = [
                chunk
                async for chunk in client.stream(ModelRequest((Message.user("hi"),)))
            ]

    asyncio.run(scenario())


def test_capabilities_reject_unsupported_requested_feature() -> None:
    async def scenario() -> None:
        client = OpenAICompatibleClient(
            base_url="http://model",
            model="m",
            transport=FakeTransport(),
            capabilities=ModelCapabilities(tool_calling=False),
        )
        try:
            await client.complete(
                ModelRequest((Message.user("go"),), tools=(FakeToolSchema(),))
            )
        except ValueError as exc:
            assert "tool calling" in str(exc)
        else:
            raise AssertionError("unsupported capability must fail before HTTP")

    asyncio.run(scenario())


def test_capabilities_reject_unsupported_combined_tool_and_output_contract() -> None:
    async def scenario() -> None:
        transport = FakeTransport()
        client = OpenAICompatibleClient(
            base_url="http://model",
            model="m",
            transport=transport,
            capabilities=ModelCapabilities(
                tool_calling_with_structured_output=False,
            ),
        )

        try:
            await client.complete(
                ModelRequest(
                    (Message.user("go"),),
                    tools=(FakeToolSchema(),),
                    output_schema={"type": "object", "properties": {}},
                )
            )
        except ValueError as exc:
            assert "separate tool-calling and structured-output requests" in str(exc)
        else:
            raise AssertionError("unsupported combined contract must fail before HTTP")

        assert transport.requests == []

    asyncio.run(scenario())


def test_openai_and_ollama_embedding_endpoints() -> None:
    async def scenario() -> None:
        openai_transport = FakeTransport(
            response={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }
        )
        openai = OpenAICompatibleClient(
            base_url="http://model/v1",
            model="embed-model",
            transport=openai_transport,
            capabilities=ModelCapabilities(embeddings=True),
        )
        assert await openai.embed(["a", "b"]) == (
            (0.1, 0.2),
            (0.3, 0.4),
        )
        assert openai_transport.requests[0]["url"] == "http://model/v1/embeddings"

        ollama_transport = FakeTransport(response={"embeddings": [[0.5, 0.6]]})
        ollama = OllamaClient(
            base_url="http://ollama:11434",
            model="embed-model",
            transport=ollama_transport,
        )
        assert await ollama.embed("hello") == ((0.5, 0.6),)
        assert ollama_transport.requests[0]["url"] == "http://ollama:11434/api/embed"

    asyncio.run(scenario())


def test_httpx_transport_works_with_mock_transport_without_network() -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/complete":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(200, text='{"part":1}\n{"part":2}\n')

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            transport = HttpxTransport(http_client)
            value = await transport.post_json(
                "http://test/complete", json={"input": "x"}
            )
            lines = [
                line
                async for line in transport.stream_lines(
                    "http://test/stream", json={"input": "x"}
                )
            ]

        assert value == {"ok": True}
        assert lines == ['{"part":1}', '{"part":2}']

    asyncio.run(scenario())


def test_model_clients_close_only_their_internally_created_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosableTransport(FakeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        owned_openai = ClosableTransport()
        monkeypatch.setattr(
            openai_module,
            "HttpxTransport",
            lambda: owned_openai,
        )
        async with OpenAICompatibleClient(
            base_url="http://model",
            model="model",
        ) as openai:
            assert openai.transport is owned_openai
        await openai.aclose()
        assert owned_openai.close_calls == 1

        injected_openai = ClosableTransport()
        openai = OpenAICompatibleClient(
            base_url="http://model",
            model="model",
            transport=injected_openai,
        )
        await openai.aclose()
        assert injected_openai.close_calls == 0

        owned_ollama = ClosableTransport()
        monkeypatch.setattr(
            ollama_module,
            "HttpxTransport",
            lambda: owned_ollama,
        )
        async with OllamaClient(
            base_url="http://ollama",
            model="model",
        ) as ollama:
            assert ollama.transport is owned_ollama
        await ollama.aclose()
        assert owned_ollama.close_calls == 1

        injected_ollama = ClosableTransport()
        ollama = OllamaClient(
            base_url="http://ollama",
            model="model",
            transport=injected_ollama,
        )
        await ollama.aclose()
        assert injected_ollama.close_calls == 0

    asyncio.run(scenario())

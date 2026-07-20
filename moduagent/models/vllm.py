from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import ModelCapabilities, ModelRequest
from .openai_compatible import OpenAICompatibleClient
from .transport import HttpTransport


class VLLMClient(OpenAICompatibleClient):
    """OpenAI-compatible client with vLLM-specific request body options."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        transport: HttpTransport | None = None,
        timeout: float = 60.0,
        capabilities: ModelCapabilities | None = None,
        default_options: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        deployment_options: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        provider_options = {
            **dict(deployment_options or {}),
            **dict(extra_body or {}),
        }
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            transport=transport,
            timeout=timeout,
            capabilities=capabilities or ModelCapabilities(),
            default_options=default_options,
            provider_options=provider_options,
            headers=headers,
        )

    def _provider_name(self) -> str:
        return "vllm"

    async def count_tokens(self, request: ModelRequest) -> int:
        """Count the rendered chat request with vLLM's ``/tokenize`` API."""

        payload = self._build_payload(request, stream=False)
        tokenize_payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload["messages"],
        }
        if "tools" in payload:
            tokenize_payload["tools"] = payload["tools"]
        for key in (
            "add_generation_prompt",
            "continue_final_message",
            "chat_template",
            "chat_template_kwargs",
            "mm_processor_kwargs",
        ):
            if key in payload:
                tokenize_payload[key] = payload[key]

        value = await self.transport.post_json(
            self._tokenize_endpoint(),
            headers=self.headers,
            json=tokenize_payload,
            timeout=self.timeout,
        )
        count = value.get("count")
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            raise ValueError("vLLM tokenize response contains no token count")
        result = int(count)
        if result < 0:
            raise ValueError("vLLM tokenize response contains a negative token count")
        return result

    def _tokenize_endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        for suffix in ("/v1/chat/completions", "/chat/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/tokenize"

    def _build_payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        if request.tools and request.output_schema is not None:
            raise ValueError(
                "vLLM tool calling and structured output require separate "
                "ACT and FINALIZE requests"
            )
        return super()._build_payload(request, stream=stream)

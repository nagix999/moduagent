from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

from .base import ModelCapabilities, ModelRequest
from .errors import ModelProtocolError
from .openai_compatible import OpenAICompatibleClient
from .transport import HttpTransport


class VLLMClient(OpenAICompatibleClient):
    """OpenAI-compatible client with vLLM-specific request body options."""

    @classmethod
    def from_env(
        cls,
        *,
        transport: HttpTransport | None = None,
        timeout: float | None = None,
        capabilities: ModelCapabilities | None = None,
        default_options: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        deployment_options: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> VLLMClient:
        """Create a client from the documented ``VLLM_*`` environment variables.

        ``VLLM_MODEL`` is required. ``VLLM_BASE_URL`` defaults to
        ``http://localhost:8000/v1`` and ``VLLM_API_KEY`` is optional.
        ``VLLM_TIMEOUT`` defaults to 60 seconds. Explicit arguments take
        precedence over environment values. Use the regular constructor when
        values should come from another source.
        """

        model = os.getenv("VLLM_MODEL", "").strip()
        if not model:
            raise ValueError("VLLM_MODEL environment variable must be set")
        base_url = os.getenv("VLLM_BASE_URL", "").strip() or "http://localhost:8000/v1"
        api_key = os.getenv("VLLM_API_KEY")
        if api_key is not None:
            api_key = api_key.strip() or None
        resolved_timeout: float
        if timeout is not None:
            resolved_timeout = timeout
        else:
            raw_timeout = os.getenv("VLLM_TIMEOUT", "").strip()
            try:
                resolved_timeout = float(raw_timeout) if raw_timeout else 60.0
            except ValueError as exc:
                raise ValueError(
                    "VLLM_TIMEOUT environment variable must be a number"
                ) from exc
            if not math.isfinite(resolved_timeout) or resolved_timeout <= 0:
                raise ValueError(
                    "VLLM_TIMEOUT environment variable must be finite and positive"
                )
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            transport=transport,
            timeout=resolved_timeout,
            capabilities=capabilities,
            default_options=default_options,
            extra_body=extra_body,
            deployment_options=deployment_options,
            headers=headers,
        )

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
        resolved_capabilities = (
            capabilities
            if capabilities is not None
            else ModelCapabilities(
                tool_calling_with_structured_output=False,
            )
        )
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            transport=transport,
            timeout=timeout,
            capabilities=resolved_capabilities,
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
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or not math.isfinite(float(count))
        ):
            raise ModelProtocolError("vLLM tokenize response contains no token count")
        result = int(count)
        if result < 0:
            raise ModelProtocolError(
                "vLLM tokenize response contains a negative token count"
            )
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
        if (
            request.tools
            and request.output_schema is not None
            and not self.capabilities.tool_calling_with_structured_output
        ):
            raise ValueError(
                "vLLM tool calling and structured output require separate "
                "ACT and FINALIZE requests"
            )
        return super()._build_payload(request, stream=stream)

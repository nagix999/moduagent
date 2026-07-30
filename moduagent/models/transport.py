from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol, runtime_checkable

import httpx

from .errors import ModelProtocolError


@runtime_checkable
class HttpTransport(Protocol):
    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, Any],
        timeout: float | None = None,
    ) -> Mapping[str, Any]: ...

    def stream_lines(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[str]: ...


class HttpxTransport:
    """Small httpx adapter; an injected AsyncClient remains caller-owned."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, Any],
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        response = await self._client.post(
            url,
            headers=dict(headers or {}),
            json=dict(json),
            timeout=timeout,
        )
        response.raise_for_status()
        try:
            value = response.json()
        except (ValueError, UnicodeError) as exc:
            raise ModelProtocolError("model endpoint returned invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise ModelProtocolError(
                "model endpoint returned a non-object JSON response"
            )
        return dict(value)

    async def stream_lines(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            url,
            headers=dict(headers or {}),
            json=dict(json),
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "HttpxTransport":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

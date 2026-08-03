from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

from moduagent.errors import ModelInvocationError


class ModelProtocolError(ModelInvocationError, ValueError):
    """The provider response does not satisfy the model adapter protocol.

    Protocol failures are deterministic for a given response and must never be
    retried.  The exception intentionally remains a ``ValueError`` for
    compatibility with callers that previously handled adapter parse failures
    that way.
    """


_INCOMPLETE_FINISH_REASONS = frozenset({"timeout", "length", "max_tokens"})


class ModelOutputIncompleteError(ModelProtocolError):
    """A provider ended a model response before output was complete.

    Only provider finish reasons with stable, payload-free meaning are accepted.
    The response content and provider metadata are deliberately not retained.
    """

    __slots__ = ("_finish_reason",)

    def __init__(self, finish_reason: str) -> None:
        if type(finish_reason) is not str:
            raise TypeError("finish_reason must be a string")
        if finish_reason not in _INCOMPLETE_FINISH_REASONS:
            raise ValueError("unsupported incomplete model finish_reason")
        self._finish_reason = finish_reason
        super().__init__("model output incomplete")

    @property
    def finish_reason(self) -> str:
        return self._finish_reason


@dataclass(frozen=True, slots=True)
class ModelErrorClassification:
    """Safe retry and diagnostic attributes for one model failure."""

    retryable: bool
    category: str
    code: str


_PROTOCOL = ModelErrorClassification(
    retryable=False,
    category="model_protocol",
    code="model_protocol_error",
)
_OUTPUT_INCOMPLETE = ModelErrorClassification(
    retryable=False,
    category="model_protocol",
    code="model_output_incomplete",
)
_TIMEOUT = ModelErrorClassification(
    retryable=True,
    category="timeout",
    code="model_timeout",
)
_CONNECTION = ModelErrorClassification(
    retryable=True,
    category="model_transport",
    code="model_connection_error",
)
_HTTP_5XX = ModelErrorClassification(
    retryable=True,
    category="model_provider",
    code="model_http_5xx",
)
_HTTP_4XX = ModelErrorClassification(
    retryable=False,
    category="model_request",
    code="model_http_4xx",
)
_INVALID_REQUEST = ModelErrorClassification(
    retryable=False,
    category="model_request",
    code="model_request_invalid",
)
_CLIENT_CONTRACT = ModelErrorClassification(
    retryable=False,
    category="model_client",
    code="model_client_contract_error",
)
_INVOCATION = ModelErrorClassification(
    retryable=False,
    category="model_invocation",
    code="model_invocation_failed",
)


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _http_status(error: BaseException) -> int | None:
    response: Any = getattr(error, "response", None)
    raw_status = getattr(response, "status_code", None)
    if raw_status is None:
        raw_status = getattr(error, "status_code", None)
    if isinstance(raw_status, bool):
        return None
    try:
        status = int(raw_status)
    except (TypeError, ValueError):
        return None
    return status if 100 <= status <= 599 else None


def classify_model_error(error: BaseException) -> ModelErrorClassification:
    """Classify a model failure using a strict, allow-listed retry policy.

    Only timeouts, connection/network failures, and HTTP 5xx responses are
    retryable.  The cause chain is inspected so adapter or framework wrappers
    do not erase the transport failure that made a retry eligible.
    """

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    chain = _exception_chain(error)

    if any(isinstance(item, ModelOutputIncompleteError) for item in chain):
        return _OUTPUT_INCOMPLETE

    # A response parse failure remains terminal even when its cause happens to
    # be a broad ValueError or a provider exception with incidental metadata.
    if any(
        isinstance(
            item,
            (
                ModelProtocolError,
                json.JSONDecodeError,
                UnicodeError,
                httpx.DecodingError,
                httpx.ProtocolError,
            ),
        )
        for item in chain
    ):
        return _PROTOCOL

    for item in chain:
        if isinstance(
            item,
            (
                asyncio.TimeoutError,
                TimeoutError,
                httpx.TimeoutException,
            ),
        ):
            return _TIMEOUT

    for item in chain:
        status = _http_status(item)
        if status is None:
            continue
        if status == 408:
            return _TIMEOUT
        if status >= 500:
            return _HTTP_5XX
        if status >= 400:
            return _HTTP_4XX

    if any(
        isinstance(item, (ConnectionError, httpx.NetworkError, httpx.ProxyError))
        for item in chain
    ):
        return _CONNECTION

    # ValueError represents request validation for custom clients after the
    # explicit protocol cases above. TypeError is a client/programming contract
    # failure. Neither category is safe to retry.
    if any(isinstance(item, TypeError) for item in chain):
        return _CLIENT_CONTRACT
    if any(isinstance(item, ValueError) for item in chain):
        return _INVALID_REQUEST
    return _INVOCATION


def is_retryable_model_error(error: BaseException) -> bool:
    """Return whether ``error`` is eligible for a model-call retry."""

    return classify_model_error(error).retryable


__all__ = [
    "ModelErrorClassification",
    "ModelOutputIncompleteError",
    "ModelProtocolError",
    "classify_model_error",
    "is_retryable_model_error",
]

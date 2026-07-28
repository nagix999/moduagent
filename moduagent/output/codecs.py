from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from moduagent.decision.planning import StepResult
from moduagent.messages import Message


T = TypeVar("T")


@runtime_checkable
class OutputCodec(Protocol):
    """Turns a provider-neutral model response into the public run output."""

    def schema(self) -> Mapping[str, Any] | None: ...

    def decode(self, response: Any) -> Any: ...


class TextOutputCodec:
    def schema(self) -> None:
        return None

    def decode(self, response: Any) -> str:
        content = _response_content(response)
        return "" if content is None else str(content)


class PydanticOutputCodec(Generic[T]):
    """Validate JSON model output into a supplied Pydantic model class."""

    def __init__(self, model: type[T]) -> None:
        if not (
            callable(getattr(model, "model_validate_json", None))
            or callable(getattr(model, "parse_raw", None))
        ):
            raise TypeError("model must be a Pydantic model class")
        self.model_type = model
        # Compatibility alias for code that treats the supplied class as a model.
        self.model = model

    def schema(self) -> Mapping[str, Any]:
        schema_method = getattr(self.model_type, "model_json_schema", None)
        if callable(schema_method):
            return schema_method()
        # Kept for Pydantic v1 compatible adapters.
        return self.model_type.schema()  # type: ignore[attr-defined, no-any-return]

    def decode(self, response: Any) -> T:
        content = _response_content(response)
        if content is None or (isinstance(content, str) and not content.strip()):
            raise ValueError("structured model response is empty")

        if isinstance(content, Mapping):
            validate = getattr(self.model_type, "model_validate", None)
            if callable(validate):
                return validate(content)
            return self.model_type.parse_obj(content)  # type: ignore[attr-defined, no-any-return]

        if not isinstance(content, (str, bytes, bytearray)):
            content = json.dumps(content, ensure_ascii=False)
        if isinstance(content, bytearray):
            content = bytes(content)

        validate_json = getattr(self.model_type, "model_validate_json", None)
        if callable(validate_json):
            return validate_json(content)
        return self.model_type.parse_raw(content)  # type: ignore[attr-defined, no-any-return]


class StepResultCodec(PydanticOutputCodec[StepResult]):
    """Strict internal codec for a Plan-and-Execute ACT result."""

    def __init__(self) -> None:
        super().__init__(StepResult)


def _response_content(response: Any) -> Any:
    """Read ``ModelResponse.message.content`` while accepting test-friendly forms."""

    if isinstance(response, Message):
        return response.content
    if isinstance(response, (str, bytes, bytearray)):
        return response

    if isinstance(response, Mapping):
        message = response.get("message", response)
        if isinstance(message, Mapping):
            return message.get("content")
        return getattr(message, "content", None)

    message = getattr(response, "message", None)
    if message is None:
        raise TypeError("response must expose message.content")
    if isinstance(message, Mapping):
        return message.get("content")
    return getattr(message, "content", None)


__all__ = [
    "OutputCodec",
    "PydanticOutputCodec",
    "StepResultCodec",
    "TextOutputCodec",
]

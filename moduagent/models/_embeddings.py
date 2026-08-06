from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

from .errors import ModelProtocolError


_SEQUENCE_EXCLUSIONS = (str, bytes, bytearray)


def normalize_embedding_inputs(
    inputs: str | Sequence[str],
) -> tuple[str | list[str], int]:
    """Validate and normalize one embedding request without exposing its content."""

    if isinstance(inputs, str):
        return inputs, 1
    if not isinstance(inputs, Sequence) or isinstance(inputs, (bytes, bytearray)):
        raise ModelProtocolError(
            "embedding input must be a string or a sequence of strings"
        )
    try:
        normalized = list(inputs)
    except Exception:
        raise ModelProtocolError("embedding input sequence could not be read") from None
    if any(not isinstance(item, str) for item in normalized):
        raise ModelProtocolError("embedding input sequence contains a non-string value")
    return normalized, len(normalized)


def normalize_embedding_vectors(
    vectors: Sequence[Any],
    *,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    """Enforce the shared embedding response shape and numeric contract."""

    if isinstance(vectors, _SEQUENCE_EXCLUSIONS):
        raise ModelProtocolError("embedding response contains invalid vectors")
    if len(vectors) != expected_count:
        raise ModelProtocolError(
            "embedding response count does not match embedding input count"
        )

    normalized: list[tuple[float, ...]] = []
    dimension: int | None = None
    for vector in vectors:
        if not isinstance(vector, Sequence) or isinstance(vector, _SEQUENCE_EXCLUSIONS):
            raise ModelProtocolError("embedding response contains an invalid vector")
        if not vector:
            raise ModelProtocolError("embedding response contains an empty vector")

        values: list[float] = []
        for component in vector:
            if isinstance(component, bool) or not isinstance(component, Real):
                raise ModelProtocolError(
                    "embedding response contains a non-numeric vector value"
                )
            try:
                number = float(component)
            except (TypeError, ValueError, OverflowError):
                raise ModelProtocolError(
                    "embedding response contains an invalid numeric vector value"
                ) from None
            if not math.isfinite(number):
                raise ModelProtocolError(
                    "embedding response contains a non-finite vector value"
                )
            values.append(number)

        current = tuple(values)
        if dimension is None:
            dimension = len(current)
        elif len(current) != dimension:
            raise ModelProtocolError(
                "embedding response contains inconsistent vector dimensions"
            )
        normalized.append(current)
    return tuple(normalized)

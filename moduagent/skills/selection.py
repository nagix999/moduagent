from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from moduagent.messages import Message, Usage
from moduagent.models import ModelClient, ModelRequest
from moduagent.skills.errors import SkillSelectionError
from moduagent.skills.models import SkillDescriptor, freeze_mapping


@dataclass(frozen=True, slots=True)
class SkillSelectionRequest:
    """Bounded inputs made available to a skill selector."""

    input: str
    catalog: tuple[SkillDescriptor, ...]
    requested_skills: tuple[str, ...] = ()
    run_id: str = ""
    session_id: str | None = None
    recent_messages: tuple[Message, ...] = ()
    user_context: Mapping[str, Any] = field(default_factory=dict)
    max_skills: int = 3

    def __post_init__(self) -> None:
        if self.max_skills < 1:
            raise ValueError("max_skills must be at least 1")
        object.__setattr__(self, "catalog", tuple(self.catalog))
        object.__setattr__(self, "requested_skills", tuple(self.requested_skills))
        object.__setattr__(self, "recent_messages", tuple(self.recent_messages))
        object.__setattr__(self, "user_context", freeze_mapping(self.user_context))


@dataclass(frozen=True, slots=True)
class SkillSelectionResult:
    """Validated skill names and their selection provenance."""

    names: tuple[str, ...] = ()
    selected_by: Mapping[str, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if len(set(names)) != len(names):
            raise ValueError("selected skill names must be unique")
        origins = dict(self.selected_by)
        if set(origins) != set(names):
            raise ValueError(
                "selected_by must contain every selected skill exactly once"
            )
        if any(origin not in {"explicit", "model"} for origin in origins.values()):
            raise ValueError("selection provenance must be 'explicit' or 'model'")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "selected_by", freeze_mapping(origins))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @property
    def selected_skills(self) -> tuple[str, ...]:
        return self.names


@runtime_checkable
class SkillSelector(Protocol):
    async def select(self, request: SkillSelectionRequest) -> SkillSelectionResult: ...


class ExplicitSkillSelector:
    """Select only the names explicitly requested by the caller."""

    def __init__(self, *, max_skills: int | None = None) -> None:
        if max_skills is not None and max_skills < 1:
            raise ValueError("max_skills must be at least 1")
        self.max_skills = max_skills

    async def select(self, request: SkillSelectionRequest) -> SkillSelectionResult:
        limit = min(request.max_skills, self.max_skills or request.max_skills)
        names = _unique_names(request.requested_skills)
        _validate_selection(names, request.catalog, limit)
        return SkillSelectionResult(
            names=names,
            selected_by={name: "explicit" for name in names},
        )


class ModelSkillSelector:
    """Ask a model to select from descriptor metadata only."""

    def __init__(
        self,
        model: ModelClient,
        *,
        max_skills: int = 3,
        options: Mapping[str, Any] | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> None:
        if max_skills < 1:
            raise ValueError("max_skills must be at least 1")
        self.model = model
        self.max_skills = max_skills
        self.options = dict(options or {})
        self.provider_options = dict(provider_options or {})

    async def select(self, request: SkillSelectionRequest) -> SkillSelectionResult:
        if not request.catalog:
            return SkillSelectionResult()

        limit = min(request.max_skills, self.max_skills, len(request.catalog))
        catalog = [
            {
                "name": descriptor.name,
                "description": descriptor.description,
                "version": descriptor.version,
            }
            for descriptor in request.catalog
        ]
        prompt = {
            "request": request.input,
            "skills": catalog,
            "max_skills": limit,
        }
        if request.recent_messages:
            prompt["recent_messages"] = [
                {
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in request.recent_messages
                if message.content
            ]

        model_request = ModelRequest(
            messages=(
                Message.system(
                    "Select only skills directly useful for the request. "
                    "Use only names in the supplied catalog. Return JSON only as "
                    '{"skills":["skill-name"]}. Return an empty list when none apply.'
                ),
                Message.user(json.dumps(prompt, ensure_ascii=False)),
            ),
            output_schema=_selection_schema(request.catalog, limit),
            options=self.options,
            provider_options=self.provider_options,
        )
        response = await self.model.complete(model_request)
        names = _parse_model_selection(response.message.content)
        _validate_selection(names, request.catalog, limit)
        return SkillSelectionResult(
            names=names,
            selected_by={name: "model" for name in names},
            usage=response.usage,
            metadata={"finish_reason": response.finish_reason},
        )


class HybridSkillSelector:
    """Prioritize explicit names, then fill remaining slots automatically."""

    def __init__(
        self,
        automatic: SkillSelector,
        *,
        max_skills: int = 3,
        explicit: ExplicitSkillSelector | None = None,
    ) -> None:
        if max_skills < 1:
            raise ValueError("max_skills must be at least 1")
        self.automatic = automatic
        self.max_skills = max_skills
        self.explicit = explicit or ExplicitSkillSelector(max_skills=max_skills)

    async def select(self, request: SkillSelectionRequest) -> SkillSelectionResult:
        limit = min(request.max_skills, self.max_skills)
        explicit_request = replace(request, max_skills=limit)
        explicit_result = await self.explicit.select(explicit_request)
        remaining = limit - len(explicit_result.names)
        if remaining == 0:
            return explicit_result

        explicit_names = set(explicit_result.names)
        automatic_request = replace(
            request,
            catalog=tuple(
                descriptor
                for descriptor in request.catalog
                if descriptor.name not in explicit_names
            ),
            requested_skills=(),
            max_skills=remaining,
        )
        automatic_result = await self.automatic.select(automatic_request)
        names = explicit_result.names + automatic_result.names
        origins = {
            **dict(explicit_result.selected_by),
            **dict(automatic_result.selected_by),
        }
        return SkillSelectionResult(
            names=names,
            selected_by=origins,
            usage=explicit_result.usage + automatic_result.usage,
            metadata={
                "explicit": dict(explicit_result.metadata),
                "automatic": dict(automatic_result.metadata),
            },
        )


def _unique_names(values: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise SkillSelectionError("requested skill names must be non-empty strings")
        name = value.strip()
        if name not in seen:
            names.append(name)
            seen.add(name)
    return tuple(names)


def _validate_selection(
    names: tuple[str, ...],
    catalog: tuple[SkillDescriptor, ...],
    limit: int,
) -> None:
    if len(names) > limit:
        raise SkillSelectionError(f"selected skills exceed limit ({limit})")
    available = {descriptor.name for descriptor in catalog}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SkillSelectionError(f"selector returned unknown skill: {unknown[0]}")


def _parse_model_selection(content: str | None) -> tuple[str, ...]:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SkillSelectionError(
            "model returned invalid skill selection JSON"
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {"skills"}:
        raise SkillSelectionError(
            "model selection must be an object containing only 'skills'"
        )
    raw_names = payload["skills"]
    if not isinstance(raw_names, list):
        raise SkillSelectionError("model selection 'skills' must be an array")
    if any(not isinstance(name, str) or not name for name in raw_names):
        raise SkillSelectionError(
            "model selected skill names must be non-empty strings"
        )
    names = tuple(raw_names)
    if len(set(names)) != len(names):
        raise SkillSelectionError("model selected duplicate skills")
    return names


def _selection_schema(
    catalog: tuple[SkillDescriptor, ...],
    max_skills: int,
) -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "skills": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [descriptor.name for descriptor in catalog],
                },
                "maxItems": max_skills,
                "uniqueItems": True,
            }
        },
        "required": ["skills"],
        "additionalProperties": False,
    }


# Policy aliases preserve the terminology used by the original design document.
SkillSelectionPolicy = SkillSelector
ExplicitSkillSelectionPolicy = ExplicitSkillSelector
ModelSkillSelectionPolicy = ModelSkillSelector
HybridSkillSelectionPolicy = HybridSkillSelector

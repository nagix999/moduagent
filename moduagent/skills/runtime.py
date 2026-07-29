from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any, TypeVar

from moduagent.messages import Message, Usage
from moduagent.runtime.context import (
    RunContext,
    SkillActivationState,
    SkillRunState,
)
from moduagent.skills.errors import (
    SkillDigestMismatchError,
    SkillLimitError,
    SkillSelectionError,
    SkillValidationError,
)
from moduagent.skills.models import SKILL_PHASES, SkillArtifact, SkillLimits, SkillRef
from moduagent.skills.prompting import render_skill_messages
from moduagent.skills.registry import SkillRegistry
from moduagent.skills.selection import (
    ExplicitSkillSelector,
    HybridSkillSelector,
    SkillSelectionRequest,
    SkillSelectionResult,
    SkillSelector,
)
from moduagent.skills.source import FilesystemSkillSource


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SkillActivationReport:
    catalog_count: int
    catalog_tokens: int
    instruction_tokens: int
    selected: tuple[str, ...]
    usage: Usage = field(default_factory=Usage)
    selection_metadata: dict[str, Any] | None = None


class SkillRuntime:
    """Select, pin, and reconstruct Skills for an Agent run."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        selector: SkillSelector | None = None,
        limits: SkillLimits | None = None,
    ) -> None:
        self.registry = registry
        self.selector = selector
        self.limits = limits or SkillLimits()
        self._artifact_cache: dict[SkillRef, SkillArtifact] = {}

    async def activate(
        self,
        context: RunContext,
        *,
        available_tools: Iterable[str] = (),
    ) -> SkillActivationReport:
        mode = context.request.skill_mode
        if mode == "disabled":
            if context.request.requested_skills:
                raise SkillSelectionError(
                    "requested_skills cannot be used when skill_mode is disabled"
                )
            context.skill_state = SkillRunState()
            context.skill_messages = ()
            return SkillActivationReport(0, 0, 0, ())

        catalog = self.registry.catalog
        catalog_tokens = sum(
            _estimate_tokens(descriptor.name) + _estimate_tokens(descriptor.description)
            for descriptor in catalog
        )
        if mode in {"auto", "hybrid"} and (
            catalog_tokens > self.limits.max_catalog_tokens
        ):
            raise SkillLimitError(
                "skill catalog exceeds max_catalog_tokens "
                f"({catalog_tokens} > {self.limits.max_catalog_tokens})"
            )

        recent_messages = _recent_messages(
            context.messages[: context.current_run_start]
        )
        if mode in {"auto", "hybrid"}:
            selection_tokens = _estimate_tokens(context.request.input) + sum(
                _estimate_tokens(message.content or "") for message in recent_messages
            )
            if selection_tokens > self.limits.max_selection_tokens:
                raise SkillLimitError(
                    "Skill selection input exceeds max_selection_tokens "
                    f"({selection_tokens} > {self.limits.max_selection_tokens})"
                )
        selection_request = SkillSelectionRequest(
            input=context.request.input,
            catalog=catalog,
            requested_skills=context.request.requested_skills,
            run_id=context.run_id,
            session_id=context.request.session_id,
            recent_messages=recent_messages,
            user_context=context.request.user_context,
            max_skills=self.limits.max_active_skills,
            model_gateway=context.model_gateway,
        )
        selector = self._selector_for(mode)
        usage_before_selection = context.usage
        selection = (
            SkillSelectionResult()
            if not catalog
            else await selector.select(selection_request)
        )
        if len(selection.names) > self.limits.max_active_skills:
            raise SkillLimitError("selected skills exceed max_active_skills")

        registered_tools = frozenset(str(name) for name in available_tools)
        artifacts: list[SkillArtifact] = []
        activations: list[SkillActivationState] = []
        instruction_tokens = 0
        for name in selection.names:
            descriptor = self.registry.require(name)
            missing_tools = descriptor.allowed_tools.difference(registered_tools)
            if missing_tools:
                missing = ", ".join(sorted(missing_tools))
                raise SkillValidationError(
                    f"skill {name!r} requires unregistered tools: {missing}"
                )
            artifact = await self._aload_artifact(descriptor.ref)
            artifact_tokens = _estimate_tokens(artifact.instructions)
            if artifact_tokens > self.limits.max_instruction_tokens:
                raise SkillLimitError(
                    f"skill {name!r} instructions exceed max_instruction_tokens"
                )
            instruction_tokens += artifact_tokens
            if instruction_tokens > self.limits.max_total_skill_tokens:
                raise SkillLimitError(
                    "active skill instructions exceed max_total_skill_tokens"
                )
            activation = self.registry.activation(
                name,
                selected_by=selection.selected_by[name],
                allowed_tools=registered_tools,
            )
            artifacts.append(artifact)
            activations.append(
                SkillActivationState(
                    name=activation.name,
                    version=activation.version,
                    digest=activation.digest,
                    source_id=activation.source_id,
                    selected_by=activation.selected_by,
                    allowed_tools=tuple(sorted(activation.allowed_tools)),
                    metadata=dict(activation.metadata),
                    applies_to=tuple(
                        phase
                        for phase in SKILL_PHASES
                        if phase in activation.applies_to
                    ),
                )
            )

        context.skill_state = SkillRunState(
            catalog_digest=self.registry.catalog_digest,
            active_skills=tuple(activations),
            instruction_tokens=instruction_tokens,
        )
        context.skill_messages = render_skill_messages(artifacts)
        self._record_metadata(context)
        if context.usage == usage_before_selection:
            context.usage = context.usage + selection.usage
        return SkillActivationReport(
            catalog_count=len(catalog),
            catalog_tokens=catalog_tokens,
            instruction_tokens=instruction_tokens,
            selected=selection.names,
            usage=selection.usage,
            selection_metadata=dict(selection.metadata),
        )

    def restore(self, context: RunContext) -> SkillActivationReport:
        """Synchronously restore a pinned Skill state for compatibility."""

        empty_report = self._prepare_restore(context)
        if empty_report is not None:
            return empty_report
        artifacts = tuple(
            self._load_artifact(self._activation_ref(activation))
            for activation in context.skill_state.active_skills
        )
        return self._finish_restore(context, artifacts)

    async def arestore(self, context: RunContext) -> SkillActivationReport:
        """Restore a pinned state without blocking on filesystem package scans."""

        empty_report = self._prepare_restore(context)
        if empty_report is not None:
            return empty_report
        artifacts: list[SkillArtifact] = []
        for activation in context.skill_state.active_skills:
            artifacts.append(
                await self._aload_artifact(self._activation_ref(activation))
            )
        return self._finish_restore(context, tuple(artifacts))

    def _prepare_restore(self, context: RunContext) -> SkillActivationReport | None:
        state = context.skill_state
        if not state.active_skills:
            if context.request.skill_mode != "disabled" and not state.catalog_digest:
                raise SkillDigestMismatchError(
                    "checkpoint does not contain a completed Skill selection"
                )
            if (
                state.catalog_digest
                and state.catalog_digest != self.registry.catalog_digest
            ):
                raise SkillDigestMismatchError(
                    "skill catalog changed since the checkpoint was created"
                )
            context.skill_messages = ()
            self._record_metadata(context)
            return SkillActivationReport(
                catalog_count=len(self.registry),
                catalog_tokens=0,
                instruction_tokens=0,
                selected=(),
            )
        if state.catalog_digest != self.registry.catalog_digest:
            raise SkillDigestMismatchError(
                "skill catalog changed since the checkpoint was created"
            )
        return None

    def _finish_restore(
        self,
        context: RunContext,
        artifacts: tuple[SkillArtifact, ...],
    ) -> SkillActivationReport:
        state = context.skill_state
        if len(artifacts) != len(state.active_skills):
            raise SkillDigestMismatchError("checkpoint Skill artifacts are incomplete")
        instruction_tokens = 0
        for activation, artifact in zip(state.active_skills, artifacts):
            if frozenset(activation.allowed_tools) != artifact.descriptor.allowed_tools:
                raise SkillDigestMismatchError(
                    f"skill tool grant changed for {activation.name}"
                )
            if frozenset(activation.applies_to) != artifact.descriptor.applies_to:
                raise SkillDigestMismatchError(
                    f"skill phase scope changed for {activation.name}"
                )
            artifact_tokens = _estimate_tokens(artifact.instructions)
            instruction_tokens += artifact_tokens
        if instruction_tokens > self.limits.max_total_skill_tokens:
            raise SkillLimitError("checkpoint skills exceed max_total_skill_tokens")
        if state.resource_tokens > self.limits.max_resource_tokens:
            raise SkillLimitError("checkpoint resources exceed max_resource_tokens")
        if state.resource_reads > self.limits.max_resource_reads:
            raise SkillLimitError("checkpoint reads exceed max_resource_reads")
        if (
            instruction_tokens + state.resource_tokens
            > self.limits.max_total_skill_tokens
        ):
            raise SkillLimitError(
                "checkpoint Skill context exceeds max_total_skill_tokens"
            )
        if state.instruction_tokens not in (0, instruction_tokens):
            raise SkillDigestMismatchError(
                "checkpoint Skill instruction token snapshot changed"
            )
        if state.instruction_tokens == 0:
            context.skill_state = SkillRunState(
                catalog_digest=state.catalog_digest,
                active_skills=state.active_skills,
                resource_reads=state.resource_reads,
                instruction_tokens=instruction_tokens,
                resource_tokens=state.resource_tokens,
            )
        context.skill_messages = render_skill_messages(artifacts)
        self._record_metadata(context)
        return SkillActivationReport(
            catalog_count=len(self.registry),
            catalog_tokens=0,
            instruction_tokens=instruction_tokens,
            selected=tuple(skill.name for skill in state.active_skills),
        )

    def allowed_tool_names(self, context: RunContext) -> frozenset[str] | None:
        """Return the Skill-scoped business tool set, or None when Skills are off."""

        if not context.skill_state.active_skills:
            return None
        return frozenset(
            tool
            for activation in context.skill_state.active_skills
            for tool in activation.allowed_tools
        )

    def active_ref(self, context: RunContext, name: str) -> SkillRef:
        for activation in context.skill_state.active_skills:
            if activation.name == name:
                return SkillRef(
                    name=activation.name,
                    version=activation.version,
                    digest=activation.digest,
                    source_id=activation.source_id,
                )
        raise SkillSelectionError(f"skill is not active for this run: {name}")

    def has_resources(self, context: RunContext) -> bool:
        for activation in context.skill_state.active_skills:
            artifact = self._active_artifact(context, activation.name)
            if artifact.references or artifact.assets:
                return True
        return False

    def supports_resource_search(self, context: RunContext) -> bool:
        for activation in context.skill_state.active_skills:
            artifact = self._active_artifact(context, activation.name)
            source = self.registry.source_for(activation.name)
            if isinstance(source, FilesystemSkillSource) and (
                artifact.references or artifact.assets
            ):
                return True
        return False

    async def _aload_artifact(self, ref: SkillRef) -> SkillArtifact:
        source = self.registry.source_for(ref.name)
        if isinstance(source, FilesystemSkillSource):
            artifact = await _run_sync_in_daemon(lambda: self.registry.load(ref))
        else:
            # Preserve the existing synchronous contract for embedded and custom
            # sources; filesystem packages are the only source known to scan disk.
            artifact = self.registry.load(ref)
        self._artifact_cache[ref] = artifact
        return artifact

    def _load_artifact(self, ref: SkillRef) -> SkillArtifact:
        artifact = self.registry.load(ref)
        self._artifact_cache[ref] = artifact
        return artifact

    def _active_artifact(self, context: RunContext, name: str) -> SkillArtifact:
        ref = self.active_ref(context, name)
        artifact = self._artifact_cache.get(ref)
        if artifact is None:
            # Compatibility for callers that construct a RunContext manually.
            # Normal activate/restore paths always populate this cache first.
            artifact = self._load_artifact(ref)
        return artifact

    @staticmethod
    def _activation_ref(activation: SkillActivationState) -> SkillRef:
        return SkillRef(
            name=activation.name,
            version=activation.version,
            digest=activation.digest,
            source_id=activation.source_id,
        )

    def _selector_for(self, mode: str) -> SkillSelector:
        if mode == "explicit":
            return ExplicitSkillSelector(max_skills=self.limits.max_active_skills)
        if mode == "auto":
            if self.selector is None:
                raise SkillSelectionError(
                    "skill_mode='auto' requires a configured skill_selector"
                )
            return self.selector
        if mode == "hybrid":
            if self.selector is None:
                raise SkillSelectionError(
                    "skill_mode='hybrid' requires a configured skill_selector"
                )
            if isinstance(self.selector, HybridSkillSelector):
                return self.selector
            return HybridSkillSelector(
                self.selector,
                max_skills=self.limits.max_active_skills,
            )
        raise SkillSelectionError(f"unsupported skill mode: {mode}")

    @staticmethod
    def _record_metadata(context: RunContext) -> None:
        context.metadata["skills"] = [
            {
                "name": activation.name,
                "version": activation.version,
                "digest": activation.digest,
                "source_id": activation.source_id,
                "selected_by": activation.selected_by,
                "applies_to": list(activation.applies_to),
            }
            for activation in context.skill_state.active_skills
        ]


def _estimate_tokens(value: str) -> int:
    encoded = value.encode("utf-8")
    return max(1, (len(value) + 3) // 4, (len(encoded) + 2) // 3)


def _recent_messages(
    messages: Iterable[Message],
    limit: int = 6,
) -> tuple[Message, ...]:
    candidates = tuple(message for message in messages if message.content)
    return candidates[-limit:]


async def _run_sync_in_daemon(function: Callable[[], T]) -> T:
    """Run a blocking filesystem scan on a cancellable daemon worker."""

    completed = Event()
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = function()
        except BaseException as error:
            outcome["error"] = error
        finally:
            completed.set()

    Thread(target=run, name="moduagent-skill-load", daemon=True).start()
    while not completed.is_set():
        await asyncio.sleep(0.001)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]

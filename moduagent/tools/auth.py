from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from moduagent.tools.base import Tool, ToolExecutionContext


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    allowed: bool
    reason: str | None = None

    @classmethod
    def allow(cls) -> "AuthorizationDecision":
        return cls(True)

    @classmethod
    def deny(cls, reason: str) -> "AuthorizationDecision":
        return cls(False, reason)


@runtime_checkable
class ToolAuthorizer(Protocol):
    async def authorize(
        self,
        tool: Tool,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | Mapping[str, Any] | None = None,
        *,
        user_context: Mapping[str, Any] | None = None,
    ) -> AuthorizationDecision: ...


class AllowAllAuthorizer:
    async def authorize(
        self,
        tool: Tool,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | Mapping[str, Any] | None = None,
        *,
        user_context: Mapping[str, Any] | None = None,
    ) -> AuthorizationDecision:
        del tool, arguments, context, user_context
        return AuthorizationDecision.allow()


class RBACToolAuthorizer:
    """Authorize tools using a ``role -> allowed tool names`` mapping.

    ``*`` in a role's permission set grants access to every tool. Missing roles
    and unconfigured tools are denied by default.
    """

    def __init__(
        self,
        role_permissions: Mapping[str, Iterable[str]] | None = None,
        *,
        permissions: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        if role_permissions is not None and permissions is not None:
            raise ValueError("use either role_permissions or permissions, not both")
        configured_permissions = role_permissions or permissions or {}
        self.role_permissions = {
            str(role): frozenset(str(name) for name in names)
            for role, names in configured_permissions.items()
        }
        # Compatibility with the original PoC attribute name.
        self.permissions = self.role_permissions

    async def authorize(
        self,
        tool: Tool,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | Mapping[str, Any] | None = None,
        *,
        user_context: Mapping[str, Any] | None = None,
    ) -> AuthorizationDecision:
        del arguments
        if context is not None and user_context is not None:
            raise ValueError("use either context or user_context, not both")
        authorization_context = context if context is not None else (user_context or {})
        effective_user_context = (
            authorization_context.user_context
            if isinstance(authorization_context, ToolExecutionContext)
            else authorization_context
        )
        raw_roles = effective_user_context.get(
            "roles", effective_user_context.get("role", ())
        )
        if isinstance(raw_roles, str):
            roles = (raw_roles,)
        else:
            roles = tuple(raw_roles or ())

        for role in roles:
            allowed_tools = self.role_permissions.get(str(role), frozenset())
            if tool.name in allowed_tools or "*" in allowed_tools:
                return AuthorizationDecision.allow()
        return AuthorizationDecision.deny(f"not authorized to call tool: {tool.name}")

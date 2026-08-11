from __future__ import annotations

import base64
import hashlib
import hmac
import json
from enum import Enum

from .models import AgentRef


class SessionStrategy(str, Enum):
    ISOLATED = "isolated"
    PER_PARENT_SESSION = "per_parent_session"
    SHARED = "shared"


class SessionKeyFactory:
    """Build opaque, tenant-bound child session IDs.

    ``SHARED`` is rejected unless a caller explicitly enables it. This keeps
    the production-safe isolated strategy as the default and prevents an
    accidental parent/child conversation collision.
    """

    def __init__(self, secret: bytes, *, allow_shared: bool = False) -> None:
        if not isinstance(secret, bytes):
            raise TypeError("session HMAC secret must be bytes")
        if len(secret) < 32:
            raise ValueError("session HMAC secret must contain at least 32 bytes")
        if type(allow_shared) is not bool:
            raise TypeError("allow_shared must be a bool")
        self._secret = secret
        self.allow_shared = allow_shared

    def create(
        self,
        *,
        strategy: SessionStrategy,
        tenant: str,
        parent_session_id: str,
        callee: AgentRef,
        delegation_id: str,
    ) -> str:
        if not isinstance(strategy, SessionStrategy):
            strategy = SessionStrategy(str(strategy))
        for value, name in (
            (tenant, "tenant"),
            (parent_session_id, "parent_session_id"),
            (delegation_id, "delegation_id"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} cannot be empty")
        if not isinstance(callee, AgentRef):
            raise TypeError("callee must be an AgentRef")
        if strategy is SessionStrategy.SHARED:
            if not self.allow_shared:
                raise ValueError("shared delegation sessions are disabled")
            return parent_session_id
        components = {
            "namespace": "moduagent.delegation.session.v1",
            "strategy": strategy.value,
            "tenant": tenant,
            "callee": {
                "agent_id": callee.agent_id,
                "version": callee.version,
            },
            "parent_session_id": (
                parent_session_id
                if strategy is SessionStrategy.PER_PARENT_SESSION
                else None
            ),
            "delegation_id": (
                delegation_id if strategy is SessionStrategy.ISOLATED else None
            ),
        }
        digest = _hmac_digest(self._secret, components)
        return f"delegation:{digest}"


def _hmac_digest(secret: bytes, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = hmac.new(secret, encoded, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

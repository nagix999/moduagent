from __future__ import annotations


class ModuAgentError(Exception):
    """Base class for framework-level failures."""


class ConfigurationError(ModuAgentError, ValueError):
    """The resolved Agent configuration is internally inconsistent."""


class InputError(ModuAgentError, ValueError):
    """A run request is invalid before execution starts."""


class CapabilityError(ConfigurationError):
    """A configured component cannot satisfy a required capability."""


class ModelInvocationError(ModuAgentError, RuntimeError):
    """A model provider call failed or returned an invalid protocol response."""


class ToolValidationError(ModuAgentError, ValueError):
    """Tool arguments failed schema or repair-guard validation."""


class ToolAuthorizationError(ModuAgentError, PermissionError):
    """A Tool call was denied or could not be authorized safely."""


class ToolInvocationError(ModuAgentError, RuntimeError):
    """A Tool invocation failed at its operational boundary."""


class ToolRecoveryError(ModuAgentError, RuntimeError):
    """A Tool failure could not be recovered within the declared safety policy."""


class MemoryError(ModuAgentError, RuntimeError):
    """Conversation-memory preparation failed."""


class SkillError(ModuAgentError, RuntimeError):
    """Skill discovery, selection, activation, or resource access failed."""


class PersistenceError(ModuAgentError, RuntimeError):
    """Conversation or checkpoint persistence failed."""


class CheckpointNotFoundError(PersistenceError, LookupError):
    """The requested checkpoint does not exist."""


class StateMigrationError(PersistenceError, ValueError):
    """A persisted state cannot be migrated without replay ambiguity."""


class ExecutionInvariantError(ModuAgentError, RuntimeError):
    """An execution engine invariant was violated."""


class RunTimeoutError(ModuAgentError, TimeoutError):
    """The overall run deadline expired."""


class CancellationError(ModuAgentError):
    """The run was cancelled at a durable boundary."""


__all__ = [
    "CancellationError",
    "CapabilityError",
    "CheckpointNotFoundError",
    "ConfigurationError",
    "ExecutionInvariantError",
    "InputError",
    "MemoryError",
    "ModelInvocationError",
    "ModuAgentError",
    "PersistenceError",
    "RunTimeoutError",
    "SkillError",
    "StateMigrationError",
    "ToolAuthorizationError",
    "ToolInvocationError",
    "ToolRecoveryError",
    "ToolValidationError",
]

from __future__ import annotations

from collections.abc import Iterable

from moduagent.config import RetryConfig
from moduagent.messages import ToolCall
from moduagent.tools.auth import ToolAuthorizer
from moduagent.tools.base import Tool, ToolExecutionContext, ToolResult
from moduagent.tools.failure import FailureProjector
from moduagent.tools.registry import ToolRegistry
from moduagent.tools.runtime import (
    ToolBatchOutcome,
    ToolRepairConstraint,
    ToolRuntime,
)


class ToolExecutor:
    """Backward-compatible 0.3 Tool executor facade.

    New execution engines should use :class:`ToolRuntime`, whose batch method
    returns the richer :class:`ToolBatchOutcome`. This adapter intentionally
    preserves the 0.3 return types.
    """

    def __init__(
        self,
        registry: ToolRegistry | Iterable[Tool] = (),
        *,
        authorizer: ToolAuthorizer | None = None,
        retry: RetryConfig | None = None,
        retry_config: RetryConfig | None = None,
        default_timeout_seconds: float | None = 30.0,
        max_result_bytes: int | None = 1_000_000,
    ) -> None:
        self.runtime = ToolRuntime(
            registry,
            authorizer=authorizer,
            retry=retry,
            retry_config=retry_config,
            default_timeout_seconds=default_timeout_seconds,
            max_result_bytes=max_result_bytes,
        )

    @property
    def registry(self) -> ToolRegistry:
        return self.runtime.registry

    @property
    def authorizer(self) -> ToolAuthorizer:
        return self.runtime.authorizer

    @authorizer.setter
    def authorizer(self, value: ToolAuthorizer) -> None:
        self.runtime.authorizer = value

    @property
    def retry(self) -> RetryConfig:
        return self.runtime.retry

    @retry.setter
    def retry(self, value: RetryConfig) -> None:
        self.runtime.retry = value

    @property
    def failure_projector(self) -> FailureProjector:
        return self.runtime.failure_projector

    @property
    def default_timeout_seconds(self) -> float | None:
        return self.runtime.default_timeout_seconds

    @default_timeout_seconds.setter
    def default_timeout_seconds(self, value: float | None) -> None:
        self.runtime.default_timeout_seconds = value

    @property
    def max_result_bytes(self) -> int | None:
        return self.runtime.max_result_bytes

    @max_result_bytes.setter
    def max_result_bytes(self, value: int | None) -> None:
        self.runtime.max_result_bytes = value

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext | None = None,
        *,
        repair_constraint: ToolRepairConstraint | None = None,
    ) -> ToolResult:
        return await self.runtime.execute(
            call,
            context,
            repair_constraint=repair_constraint,
        )

    async def execute_batch(
        self,
        calls: Iterable[ToolCall],
        context: ToolExecutionContext | None = None,
        *,
        parallel: bool = False,
        max_parallel: int = 4,
        repair_constraint: ToolRepairConstraint | None = None,
    ) -> ToolBatchOutcome:
        """Execute a batch and expose the 0.4 structured outcome."""

        return await self.runtime.execute_many(
            calls,
            context,
            parallel=parallel,
            max_parallel=max_parallel,
            repair_constraint=repair_constraint,
        )

    async def execute_many(
        self,
        calls: Iterable[ToolCall],
        context: ToolExecutionContext | None = None,
        *,
        parallel: bool = False,
        max_parallel: int = 4,
        repair_constraint: ToolRepairConstraint | None = None,
    ) -> tuple[ToolResult, ...]:
        outcome = await self.execute_batch(
            calls,
            context,
            parallel=parallel,
            max_parallel=max_parallel,
            repair_constraint=repair_constraint,
        )
        return outcome.results

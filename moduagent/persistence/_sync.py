from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from moduagent.tools.scheduler import SyncToolScheduler


_STORE_SYNC_SCHEDULER = SyncToolScheduler(
    max_workers=16,
    max_queue=128,
    thread_name_prefix="moduagent-store",
)


async def call_maybe_async(
    function: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call async adapters directly and isolate synchronous storage I/O.

    The bounded scheduler prevents a slow synchronous Redis/DB adapter from
    blocking the Agent event loop or creating one abandoned thread per timeout.
    A synchronous compatibility wrapper may still return an awaitable; execute
    that awaitable back on the owning event loop.
    """

    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)

    result = await _STORE_SYNC_SCHEDULER.run(
        lambda: function(*args, **kwargs),
    )
    if inspect.isawaitable(result):
        return await result
    return result


__all__ = ["call_maybe_async"]

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, TypeVar


_T = TypeVar("_T")
_MAX_WORKERS = 4
_MAX_QUEUED_CALLS = 256
_POLL_INTERVAL_SECONDS = 0.002
_LOOP_POLLER_ATTRIBUTE = "_moduagent_observability_completion_poller"


@dataclass(slots=True)
class _WorkItem:
    abandoned: threading.Event
    completed: threading.Event
    function: Callable[..., Any]
    args: tuple[Any, ...]
    value: Any = None
    error: BaseException | None = None


class _DaemonWorkerPool:
    """Small bounded executor for best-effort observability adapters."""

    def __init__(
        self,
        *,
        max_workers: int = _MAX_WORKERS,
        max_queued_calls: int = _MAX_QUEUED_CALLS,
    ) -> None:
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if type(max_queued_calls) is not int or max_queued_calls < 1:
            raise ValueError("max_queued_calls must be a positive integer")
        self.max_workers = max_workers
        self.max_queued_calls = max_queued_calls
        self._queue: queue.Queue[_WorkItem] = queue.Queue(max_queued_calls)
        self._lock = threading.Lock()
        self._started = False

    def submit(self, item: _WorkItem) -> None:
        self._start()
        try:
            self._queue.put_nowait(item)
        except queue.Full as error:
            raise RuntimeError("observability worker queue capacity reached") from error

    def _start(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            for index in range(self.max_workers):
                threading.Thread(
                    target=self._worker,
                    name=f"moduagent-observability-{index + 1}",
                    daemon=True,
                ).start()
            self._started = True

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item.abandoned.is_set():
                    continue
                try:
                    item.value = item.function(*item.args)
                except BaseException as error:
                    item.error = error
            finally:
                item.completed.set()
                self._queue.task_done()


_DEFAULT_POOL = _DaemonWorkerPool()


class _LoopCompletionPoller:
    """Settle all worker results for one event loop with one timer."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._entries: dict[
            int,
            tuple[_WorkItem, asyncio.Future[Any]],
        ] = {}
        self._wake = asyncio.Event()
        self._task = loop.create_task(
            self._run(),
            name="moduagent-observability-completions",
        )

    def register(self, item: _WorkItem) -> asyncio.Future[Any]:
        future = self.loop.create_future()
        self._entries[id(item)] = (item, future)
        self._wake.set()
        return future

    def unregister(self, item: _WorkItem) -> None:
        self._entries.pop(id(item), None)

    async def _run(self) -> None:
        try:
            while True:
                while not self._entries:
                    self._wake.clear()
                    if self._entries:
                        break
                    await self._wake.wait()

                for key, (item, future) in tuple(self._entries.items()):
                    if not item.completed.is_set():
                        continue
                    self._entries.pop(key, None)
                    if future.done():
                        continue
                    if item.error is not None:
                        future.set_exception(item.error)
                    else:
                        future.set_result(item.value)

                if self._entries:
                    await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            for _, future in self._entries.values():
                if not future.done():
                    future.cancel()
            self._entries.clear()
            raise


def _loop_completion_poller() -> _LoopCompletionPoller:
    loop = asyncio.get_running_loop()
    current = getattr(loop, _LOOP_POLLER_ATTRIBUTE, None)
    if isinstance(current, _LoopCompletionPoller) and not current._task.done():
        return current
    poller = _LoopCompletionPoller(loop)
    setattr(loop, _LOOP_POLLER_ATTRIBUTE, poller)
    return poller


async def run_in_daemon_thread(
    function: Callable[..., _T],
    /,
    *args: Any,
) -> _T:
    """Run blocking observability I/O in a bounded daemon worker pool.

    Observability must never keep an Agent run—or interpreter shutdown—waiting
    for a stalled logging handler. Cancelling the awaiter cannot stop an
    already-running Python call, but capacity remains bounded and queued
    cancelled calls are skipped.
    """

    item = _WorkItem(
        abandoned=threading.Event(),
        completed=threading.Event(),
        function=function,
        args=args,
    )
    poller = _loop_completion_poller()
    result = poller.register(item)
    try:
        _DEFAULT_POOL.submit(item)
    except BaseException:
        item.abandoned.set()
        poller.unregister(item)
        raise
    try:
        return cast(_T, await result)
    except BaseException:
        item.abandoned.set()
        poller.unregister(item)
        raise

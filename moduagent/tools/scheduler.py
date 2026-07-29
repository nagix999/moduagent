from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast


T = TypeVar("T")


class SyncToolSchedulerOverloaded(RuntimeError):
    """Raised when a bounded synchronous Tool scheduler cannot accept work."""


@dataclass(frozen=True, slots=True)
class SyncToolSchedulerStats:
    workers: int
    running: int
    queued: int
    submitted: int
    completed: int
    abandoned: int
    rejected: int


@dataclass(slots=True)
class _WorkItem(Generic[T]):
    function: Callable[[], T]
    abandoned: threading.Event
    completed: threading.Event
    value: Any = None
    error: BaseException | None = None


class SyncToolScheduler:
    """Bounded daemon-worker scheduler for explicitly synchronous Tools.

    Cancelling or timing out an awaiter cannot stop Python code that is already
    running in a worker. Such work therefore keeps occupying capacity until it
    actually returns, preventing an unbounded number of timed-out background
    threads from accumulating.
    """

    def __init__(
        self,
        *,
        max_workers: int = 16,
        max_queue: int = 128,
        thread_name_prefix: str = "moduagent-tool",
    ) -> None:
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if type(max_queue) is not int or max_queue < 0:
            raise ValueError("max_queue must be a non-negative integer")
        if not isinstance(thread_name_prefix, str) or not thread_name_prefix.strip():
            raise ValueError("thread_name_prefix cannot be empty")
        self.max_workers = max_workers
        self.max_queue = max_queue
        self.thread_name_prefix = thread_name_prefix.strip()
        self._capacity = max_workers + max_queue
        # A submitted item can briefly remain queued while an idle worker is
        # being scheduled, so physical capacity includes both worker slots and
        # the configured waiting queue.
        self._queue: queue.Queue[_WorkItem[Any]] = queue.Queue(self._capacity)
        self._lock = threading.Lock()
        self._started = False
        self._running = 0
        self._outstanding = 0
        self._submitted = 0
        self._completed = 0
        self._abandoned = 0
        self._rejected = 0

    async def run(self, function: Callable[[], T]) -> T:
        if not callable(function):
            raise TypeError("function must be callable")
        item = _WorkItem(
            function=function,
            abandoned=threading.Event(),
            completed=threading.Event(),
        )
        self._submit(item)
        try:
            # Some restricted runtimes can drop the self-pipe wake-up used by
            # call_soon_threadsafe(). Polling a thread Event keeps completion
            # deterministic while the bounded pool prevents thread growth.
            while not item.completed.is_set():
                await asyncio.sleep(0.001)
        except BaseException:
            item.abandoned.set()
            raise
        if item.error is not None:
            raise item.error
        return cast(T, item.value)

    def stats(self) -> SyncToolSchedulerStats:
        with self._lock:
            return SyncToolSchedulerStats(
                workers=self.max_workers if self._started else 0,
                running=self._running,
                queued=self._queue.qsize(),
                submitted=self._submitted,
                completed=self._completed,
                abandoned=self._abandoned,
                rejected=self._rejected,
            )

    def _submit(self, item: _WorkItem[Any]) -> None:
        self._start()
        with self._lock:
            if self._outstanding >= self._capacity:
                self._rejected += 1
                raise SyncToolSchedulerOverloaded(
                    "synchronous Tool scheduler capacity reached"
                )
            self._submitted += 1
            self._outstanding += 1
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            with self._lock:
                self._submitted -= 1
                self._outstanding -= 1
                self._rejected += 1
            raise SyncToolSchedulerOverloaded(
                "synchronous Tool scheduler queue capacity reached"
            ) from exc

    def _start(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            for index in range(self.max_workers):
                threading.Thread(
                    target=self._worker,
                    name=f"{self.thread_name_prefix}-{index + 1}",
                    daemon=True,
                ).start()
            self._started = True

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item.abandoned.is_set():
                    with self._lock:
                        self._abandoned += 1
                    continue
                with self._lock:
                    self._running += 1
                try:
                    item.value = item.function()
                except BaseException as exc:
                    item.error = exc
                finally:
                    with self._lock:
                        self._running -= 1
                        self._completed += 1
                        if item.abandoned.is_set():
                            self._abandoned += 1
            finally:
                with self._lock:
                    self._outstanding -= 1
                item.completed.set()
                self._queue.task_done()


__all__ = [
    "SyncToolScheduler",
    "SyncToolSchedulerOverloaded",
    "SyncToolSchedulerStats",
]

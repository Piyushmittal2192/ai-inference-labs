"""Admission control and execution for blocking GPU work.

Three decisions happen for every request, in this order:

  1. Admit or shed.    If every slot is busy and the wait queue is full, fail
                       fast with 503. An unbounded queue is not resilience; it
                       is a slow outage where every client times out together.
  2. Wait, bounded.    Wait for a free slot, but give up after a deadline.
  3. Run off the loop. Run the model on a worker thread so the event loop stays
                       free to stream tokens and answer health checks.

A slot is released when the worker THREAD finishes, not when the HTTP request
ends. If a client disconnects mid-generation, the thread may still be running
a forward pass. Releasing its slot early would let a new request start on the
same GPU while the old one is still running, silently oversubscribing it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

T = TypeVar("T")


class SchedulerRejection(Exception):
    """The request was not run. Maps to HTTP 503."""

    retry_after_s: int = 1


class QueueFullError(SchedulerRejection):
    """All slots are busy and the wait queue is at capacity."""


class QueueTimeoutError(SchedulerRejection):
    """Waited longer than the queue deadline for a slot."""

    retry_after_s = 5


@dataclass
class SchedulerStats:
    in_flight: int = 0
    waiting: int = 0
    admitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected_queue_full: int = 0
    rejected_queue_timeout: int = 0


def _mark_exception_retrieved(future: asyncio.Future[Any]) -> None:
    # Errors are reported to the client through the event queue in main.py;
    # touching .exception() stops asyncio warning that nobody looked at it.
    if not future.cancelled():
        future.exception()


class InferenceScheduler:
    def __init__(
        self,
        *,
        max_concurrency: int,
        max_queue_depth: int,
        queue_timeout_s: float,
        blocking: bool = False,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_queue_depth < 0:
            raise ValueError("max_queue_depth must be >= 0")
        self.max_concurrency = max_concurrency
        self.max_queue_depth = max_queue_depth
        self.queue_timeout_s = queue_timeout_s
        self.blocking = blocking
        self.stats = SchedulerStats()
        self._slots = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="gpu-worker")

    async def submit(self, fn: Callable[[], T]) -> tuple[asyncio.Future[T], float]:
        """Wait (bounded) for a slot, then start fn().

        Returns (future, seconds spent queued). The future resolves when fn
        finishes, and the slot stays held until then.
        """
        queue_s = await self._acquire_slot()
        self.stats.admitted += 1
        self.stats.in_flight += 1
        if self.blocking:
            future = self._run_on_event_loop(fn)
        else:
            future = self._run_on_worker_thread(fn)
        future.add_done_callback(_mark_exception_retrieved)
        return future, queue_s

    async def _acquire_slot(self) -> float:
        if self._slots.locked() and self.stats.waiting >= self.max_queue_depth:
            self.stats.rejected_queue_full += 1
            raise QueueFullError(
                f"queue full: {self.stats.waiting} waiting, "
                f"{self.stats.in_flight}/{self.max_concurrency} slots busy"
            )
        arrived = time.perf_counter()
        self.stats.waiting += 1
        try:
            async with asyncio.timeout(self.queue_timeout_s):
                await self._slots.acquire()
        except TimeoutError:
            self.stats.rejected_queue_timeout += 1
            raise QueueTimeoutError(f"no slot became free within {self.queue_timeout_s:.1f}s") from None
        finally:
            self.stats.waiting -= 1
        return time.perf_counter() - arrived

    def _run_on_event_loop(self, fn: Callable[[], T]) -> asyncio.Future[T]:
        # CHAOS MODE. The model runs ON the event-loop thread. Until fn returns,
        # this process cannot stream a byte, accept a connection, or answer /livez.
        future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        try:
            result = fn()
        except Exception as exc:
            self._finish(failed=True)
            future.set_exception(exc)
        else:
            self._finish(failed=False)
            future.set_result(result)
        return future

    def _run_on_worker_thread(self, fn: Callable[[], T]) -> asyncio.Future[T]:
        loop = asyncio.get_running_loop()
        try:
            work: Future[T] = self._executor.submit(fn)
        except RuntimeError:
            self._finish(failed=True)
            raise
        work.add_done_callback(lambda done: self._finish_from_thread(loop, done))
        return asyncio.wrap_future(work, loop=loop)

    def _finish_from_thread(self, loop: asyncio.AbstractEventLoop, done: Future[Any]) -> None:
        failed = done.cancelled() or done.exception() is not None
        try:
            loop.call_soon_threadsafe(self._finish, failed)
        except RuntimeError:
            pass  # the event loop is already closed: the process is exiting

    def _finish(self, failed: bool) -> None:
        self.stats.in_flight -= 1
        if failed:
            self.stats.failed += 1
        else:
            self.stats.completed += 1
        self._slots.release()

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "blocking" if self.blocking else "threadpool",
            "max_concurrency": self.max_concurrency,
            "max_queue_depth": self.max_queue_depth,
            "queue_timeout_s": self.queue_timeout_s,
            **asdict(self.stats),
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

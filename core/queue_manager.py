import asyncio
import logging
import time
from typing import AsyncIterator, Callable, Dict, List

from core.models import QueryRequest, StreamChunk

logger = logging.getLogger("queue-manager")


class QueueManager:
    """
    Async priority queue backed by a fixed pool of worker coroutines.

    Flow:
      1. Caller enqueues a QueryRequest.
      2. A free worker picks it up and calls process_fn (the Claude agent).
      3. Each StreamChunk yielded by process_fn is placed into the
         per-session response queue.
      4. The TCP handle_client coroutine reads that queue and forwards
         chunks to the connected terminal client.
    """

    def __init__(
        self,
        process_fn: Callable[[QueryRequest], AsyncIterator[StreamChunk]],
        max_workers: int = 5,
        max_queue_size: int = 100,
    ):
        self._process_fn = process_fn
        self._max_workers = max_workers
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._session_queues: Dict[str, asyncio.Queue] = {}
        self._workers: List[asyncio.Task] = []
        self._running = False

        # Metrics
        self._processed = 0
        self._total_time = 0.0

    # ── Session response queues ────────────────────────────────────────────

    def ensure_session_queue(self, session_id: str) -> asyncio.Queue:
        if session_id not in self._session_queues:
            self._session_queues[session_id] = asyncio.Queue()
        return self._session_queues[session_id]

    def remove_session(self, session_id: str):
        self._session_queues.pop(session_id, None)

    # ── Public API ─────────────────────────────────────────────────────────

    async def enqueue(self, request: QueryRequest) -> int:
        """
        Add a request to the priority queue.
        Returns the current queue depth (approximate position).
        """
        # PriorityQueue tuple: (priority, timestamp, request)
        # timestamp breaks ties deterministically (FIFO within same priority)
        await self._queue.put((request.priority, request.timestamp, request))
        return self._queue.qsize()

    def stats(self) -> dict:
        avg = round(self._total_time / self._processed, 3) if self._processed else 0.0
        return {
            "queue_size": self._queue.qsize(),
            "workers": self._max_workers,
            "requests_processed": self._processed,
            "avg_processing_time": avg,
            "active_sessions": len(self._session_queues),
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"worker-{i}")
            for i in range(self._max_workers)
        ]
        logger.info(f"Worker pool started ({self._max_workers} workers)")

    async def stop(self):
        self._running = False
        await asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("Worker pool stopped")

    # ── Worker coroutine ───────────────────────────────────────────────────

    async def _worker(self, worker_id: int):
        logger.info(f"  Worker-{worker_id} ready")
        while self._running:
            try:
                # Blocking get with short timeout so we can check _running
                priority, timestamp, request = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            session_q = self._session_queues.get(request.session_id)
            if session_q is None:
                # Session disconnected before the worker got to it
                logger.warning(
                    f"Worker-{worker_id}: session [{request.session_id}] gone, "
                    f"dropping request {request.request_id}"
                )
                self._queue.task_done()
                continue

            logger.info(
                f"Worker-{worker_id} ← [{request.session_id}] "
                f"req={request.request_id} q={request.query_text[:60]!r}"
            )
            t0 = time.time()

            try:
                async for chunk in self._process_fn(request):
                    await session_q.put(chunk)
            except Exception as exc:
                logger.exception(f"Worker-{worker_id} processing error: {exc}")
                await session_q.put(
                    StreamChunk(
                        request_id=request.request_id,
                        is_done=True,
                        error=str(exc),
                    )
                )
            finally:
                elapsed = time.time() - t0
                self._processed += 1
                self._total_time += elapsed
                self._queue.task_done()
                logger.info(f"Worker-{worker_id}: done in {elapsed:.2f}s")

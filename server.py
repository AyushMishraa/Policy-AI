"""
Async TCP server.

Protocol (newline-delimited JSON):
  Client → Server:  {"type": "query", "text": "...", "input_mode": "text|voice"}
  Server → Client:  {"type": "chunk",  "text": "...", "request_id": "..."}
                    {"type": "done",   "sources": [...], "processing_time": 1.2, "request_id": "..."}
                    {"type": "error",  "message": "...", "request_id": "..."}
                    {"type": "status", "message": "..."}
                    {"type": "stats",  "data": {...}}
"""

import asyncio
import json
import logging
import uuid
from typing import Optional

from config import config
from core.models import Priority, QueryRequest
from core.queue_manager import QueueManager
from core.session_manager import SessionManager

logger = logging.getLogger("server")


class PolicyServer:
    def __init__(self):
        self._session_mgr: SessionManager = SessionManager()
        self._queue_mgr:   Optional[QueueManager] = None

    # ── Startup ────────────────────────────────────────────────────────────

    async def start(self, runner):
        """
        runner: PolicyAgentRunner — injected so the server stays decoupled
                from the agent and vector store initialisation.
        """
        # Build the queue manager wrapping the runner
        self._queue_mgr = QueueManager(
            process_fn=runner.run,
            max_workers=config.MAX_WORKERS,
            max_queue_size=config.QUEUE_MAX_SIZE,
        )
        await self._queue_mgr.start()

        server = await asyncio.start_server(
            self._handle_client,
            host=config.SERVER_HOST,
            port=config.SERVER_PORT,
        )
        addr = f"{config.SERVER_HOST}:{config.SERVER_PORT}"
        logger.info(f"Policy Agent server listening on {addr}")
        async with server:
            await server.serve_forever()

    async def stop(self):
        if self._queue_mgr:
            await self._queue_mgr.stop()

    # ── Per-client coroutine ───────────────────────────────────────────────

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        session_id = str(uuid.uuid4())[:8]
        self._session_mgr.add(session_id, writer)
        response_q = self._queue_mgr.ensure_session_queue(session_id)

        peer = writer.get_extra_info("peername")
        logger.info(f"Client connected: {peer} → session [{session_id}]")

        await self._send(writer, {"type": "status", "message": (
            f"Connected to Policy Agent (session {session_id}). "
            "Type your question or 'voice' to speak."
        )})

        # Two concurrent tasks per connection:
        #   1. read_task  — reads messages from the client
        #   2. write_task — drains the per-session response queue and sends
        read_task  = asyncio.create_task(
            self._read_loop(session_id, reader, writer),
            name=f"read-{session_id}",
        )
        write_task = asyncio.create_task(
            self._write_loop(session_id, response_q, writer),
            name=f"write-{session_id}",
        )

        try:
            done, pending = await asyncio.wait(
                [read_task, write_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        finally:
            self._session_mgr.remove(session_id)
            self._queue_mgr.remove_session(session_id)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.info(f"Client disconnected: session [{session_id}]")

    # ── Read loop (client → server) ────────────────────────────────────────

    async def _read_loop(
        self,
        session_id: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        while True:
            try:
                raw = await reader.readline()
            except (asyncio.IncompleteReadError, ConnectionResetError):
                break

            if not raw:
                break

            try:
                msg = json.loads(raw.decode().strip())
            except json.JSONDecodeError:
                await self._send(
                    writer,
                    {"type": "error", "message": "Invalid JSON — send newline-delimited JSON"},
                )
                continue

            msg_type = msg.get("type", "query")

            if msg_type == "stats":
                await self._send(writer, {
                    "type": "stats",
                    "data": {
                        "queue":    self._queue_mgr.stats(),
                        "sessions": self._session_mgr.stats(),
                    },
                })
                continue

            if msg_type == "ping":
                await self._send(writer, {"type": "pong"})
                continue

            # ── Enqueue a query ───────────────────────────────────────────
            text       = str(msg.get("text", "")).strip()
            input_mode = msg.get("input_mode", "text")  # "text" | "voice"
            priority   = int(msg.get("priority", Priority.NORMAL))

            if not text:
                await self._send(
                    writer,
                    {"type": "error", "message": "Empty query — nothing to process"},
                )
                continue

            self._session_mgr.touch(session_id)
            request = QueryRequest(
                session_id=session_id,
                query_text=text,
                query_type=input_mode,
                priority=priority,
            )

            depth = await self._queue_mgr.enqueue(request)
            await self._send(writer, {
                "type":       "status",
                "message":    f"Queued (position ~{depth})",
                "request_id": request.request_id,
            })

    # ── Write loop (server → client) ───────────────────────────────────────

    async def _write_loop(
        self,
        session_id: str,
        response_q: asyncio.Queue,
        writer: asyncio.StreamWriter,
    ):
        while True:
            try:
                chunk = await asyncio.wait_for(response_q.get(), timeout=60.0)
            except asyncio.TimeoutError:
                continue

            if chunk.error:
                await self._send(writer, {
                    "type":       "error",
                    "message":    chunk.error,
                    "request_id": chunk.request_id,
                })
            elif chunk.is_done:
                await self._send(writer, {
                    "type":            "done",
                    "request_id":      chunk.request_id,
                    "sources":         chunk.sources,
                    "processing_time": chunk.processing_time,
                })
            else:
                await self._send(writer, {
                    "type":       "chunk",
                    "text":       chunk.text,
                    "request_id": chunk.request_id,
                })

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, payload: dict):
        try:
            writer.write((json.dumps(payload) + "\n").encode())
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass

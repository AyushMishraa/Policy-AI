import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger("session-manager")


@dataclass
class SessionInfo:
    session_id: str
    writer: asyncio.StreamWriter
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    request_count: int = 0


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}

    def add(self, session_id: str, writer: asyncio.StreamWriter) -> SessionInfo:
        info = SessionInfo(session_id=session_id, writer=writer)
        self._sessions[session_id] = info
        logger.info(f"Session [{session_id}] joined  (total: {len(self._sessions)})")
        return info

    def remove(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session [{session_id}] left    (total: {len(self._sessions)})")

    def get(self, session_id: str) -> Optional[SessionInfo]:
        return self._sessions.get(session_id)

    def touch(self, session_id: str):
        if s := self._sessions.get(session_id):
            s.last_active = time.time()
            s.request_count += 1

    def stats(self) -> dict:
        return {
            "total_sessions": len(self._sessions),
            "sessions": [
                {
                    "id": s.session_id,
                    "requests": s.request_count,
                    "age_seconds": round(time.time() - s.created_at, 1),
                }
                for s in self._sessions.values()
            ],
        }

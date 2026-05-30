"""
In-process conversation history store.

Each session keeps a rolling window of (role, content) turns so the
LangGraph agent can include prior context in its generation prompt,
giving users a natural multi-turn experience without re-stating their
question every time.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class Turn:
    role:      str        # "user" | "assistant"
    content:   str
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""


class ConversationHistory:
    """Rolling window of conversation turns for one session."""

    def __init__(self, max_turns: int = 10):
        self._max_turns = max_turns
        self._turns: Deque[Turn] = deque(maxlen=max_turns * 2)  # user+assistant

    def add_user(self, text: str, request_id: str = ""):
        self._turns.append(Turn(role="user", content=text, request_id=request_id))

    def add_assistant(self, text: str, request_id: str = ""):
        self._turns.append(Turn(role="assistant", content=text, request_id=request_id))

    def as_messages(self) -> List[Dict[str, str]]:
        """Return history as Anthropic-style message list (excludes current turn)."""
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def as_context_string(self, max_chars: int = 2000) -> str:
        """Return a compact text block suitable for injection into a RAG prompt."""
        if not self._turns:
            return ""
        lines = []
        for t in self._turns:
            prefix = "User" if t.role == "user" else "Assistant"
            snippet = t.content[:400] + ("…" if len(t.content) > 400 else "")
            lines.append(f"{prefix}: {snippet}")
        block = "\n".join(lines)
        return block[-max_chars:]  # keep the most recent chars

    def clear(self):
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


class ConversationStore:
    """Registry of ConversationHistory objects keyed by session_id."""

    def __init__(self, max_turns_per_session: int = 10):
        self._max_turns = max_turns_per_session
        self._store: Dict[str, ConversationHistory] = {}

    def get_or_create(self, session_id: str) -> ConversationHistory:
        if session_id not in self._store:
            self._store[session_id] = ConversationHistory(self._max_turns)
        return self._store[session_id]

    def remove(self, session_id: str):
        self._store.pop(session_id, None)

    def stats(self) -> Dict[str, int]:
        return {sid: len(h) for sid, h in self._store.items()}

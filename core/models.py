import uuid
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List


class Priority(IntEnum):
    HIGH   = 1
    NORMAL = 5
    LOW    = 10


@dataclass
class QueryRequest:
    session_id: str
    query_text: str
    query_type: str = "text"       # "text" | "voice"
    priority: int   = Priority.NORMAL
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # PriorityQueue needs comparison — sort by priority then arrival time
    def __lt__(self, other: "QueryRequest") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class StreamChunk:
    request_id: str
    text: str = ""
    is_done: bool = False
    sources: List[dict] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None

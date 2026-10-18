"""Request queue and the data structure handed to the worker."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from server.routing import NullReply, ReplyTarget


@dataclass
class Request:
    """One unit of work for the shared agent."""

    text: str
    source: str = "unknown"
    chat_id: str | None = None
    client_id: str | None = None
    trusted: bool = False
    send_reasoning: bool = False
    reply: ReplyTarget = field(default_factory=NullReply)
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class RequestQueue(asyncio.Queue):
    """Thin ``asyncio.Queue`` wrapper with a debug snapshot."""

    def snapshot(self) -> list[str]:
        return [f"{getattr(item, 'source', '?')}:{getattr(item, 'id', '?')}" for item in list(self._queue)]

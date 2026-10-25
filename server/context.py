"""Per-request context shared by the worker, tools and reply routing.

The agent is a single shared object processing one request at a time, so a
``ContextVar`` is enough to tell a tool which request (and which client/chat)
is currently being served.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class RequestContext:
    request_id: str
    source: str
    chat_id: str | None = None
    client_id: str | None = None
    voice_session_id: str | None = None
    trusted: bool = False
    # Chats already messaged by a tool during this turn (avoids double replies).
    sent_chats: set[str] = field(default_factory=set)
    # Set by a voice turn if the agent already spoke aloud.
    spoke: bool = False


current_request: ContextVar[RequestContext | None] = ContextVar(
    "inysha_request", default=None
)


def get_request_context() -> RequestContext | None:
    return current_request.get()


def is_voice_context() -> bool:
    """True for a real voice turn or a deferred result for that voice session.

    ``source`` alone is not enough: background tool results arrive as
    ``source="tool_result"`` but still belong to a voice session.  The
    ``voice_session_id`` is set only by ``VoiceSession`` and travels through
    ``Request``/``RequestContext``, so it is the authoritative marker.
    """
    ctx = current_request.get()
    return bool(ctx and ctx.voice_session_id)

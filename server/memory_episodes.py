"""Episodes: every utterance the agent actually produced or received.

The episodic layer is not a second source of truth. It is written from the
same place that already appends ``message_history.md`` (``ContextPool.append``
→ ``_save_to_file``), so the transcript, the crash snapshot and the searchable
log can never disagree about what happened.

Two rules keep it safe:
  * best effort — a broken/absent database must never break a turn;
  * hermetic — the database is ``data_dir/memory.db`` for any pool pointed at
    a temporary directory, so tests never touch the live memory.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from server import config as app_config
from server.context import get_request_context, is_voice_context
from server.memory_store import MemoryStore, get_store

EPISODE_DB_NAME = "memory.db"
_TS_FORMAT = "%d.%m.%Y, %H:%M"

# Metadata that the MAX/voice/desktop sources prepend to the utterance.  It is
# transport noise: stored in ``episodes`` it would pollute every search result.
_META_PREFIX_RE = re.compile(
    r"^(?:"
    r"\[Time: [^\]]*\]\s*"
    r"|\[MAX messenger[^\]]*\]\s*"
    r"|\[Voice client[^\]]*\]\s*"
    r"|\[Desktop[^\]]*\]\s*"
    r")+"
)
_CHAT_ID_RE = re.compile(r"chat_id=(-?\d+)")
_ROLE_TO_SPEAKER = {
    "user": "human",
    "assistant": "agent",
    "tool": "tool",
    "system": "system",
}


def resolve_db_path(data_dir: Path) -> Path:
    """Database file for a pool: honour MEMORY_DB_PATH only for the live dir.

    ``ContextPool`` defaults to ``DATA_DIR``; a pool created with an explicit
    directory (tests, scripts, a second instance) gets its own ``memory.db``
    there instead of silently writing into the production file.
    """
    data_dir = Path(data_dir).resolve()
    if data_dir == Path(app_config.DATA_DIR).resolve():
        return Path(app_config.MEMORY_DB_PATH)
    return data_dir / EPISODE_DB_NAME


def strip_meta(text: str) -> str:
    """Drop leading ``[Time: ...][MAX messenger | ...]`` markers."""
    return _META_PREFIX_RE.sub("", text or "").strip()


def extract_chat_id(text: str) -> Optional[str]:
    match = _CHAT_ID_RE.search(text or "")
    return match.group(1) if match else None


def parse_entry_time(value: str) -> Optional[int]:
    """``MemoryEntry.time`` ("26.10.2026, 06:37") → unix seconds (local)."""
    if not value:
        return None
    try:
        return int(datetime.strptime(value, _TS_FORMAT).timestamp())
    except ValueError:
        return None


class EpisodeLog:
    """Writes ``MemoryEntry`` objects into the episodes table, best effort."""

    def __init__(
        self,
        data_dir: Path,
        db_path: Optional[Path] = None,
        enabled: Optional[bool] = None,
        store_tools: Optional[bool] = None,
    ):
        self._path = Path(db_path) if db_path else resolve_db_path(data_dir)
        self.enabled = (
            app_config.MEMORY_WRITE_EPISODES if enabled is None else enabled
        )
        self.store_tools = (
            app_config.MEMORY_EPISODE_TOOLS if store_tools is None else store_tools
        )
        self._store: Optional[MemoryStore] = None
        self._warned = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            self._store = get_store(self._path)
        return self._store

    def record(self, entry) -> Optional[int]:
        """Store one memory entry; returns the row id or ``None``."""
        if not self.enabled:
            return None
        try:
            return self._record(entry)
        except Exception as e:  # pragma: no cover - defensive, DB is optional
            if not self._warned:
                logger.warning(f"memory: episode log disabled for this process: {e}")
                self._warned = True
            else:
                logger.debug(f"memory: episode log failed: {e}")
            return None

    def _record(self, entry) -> Optional[int]:
        speaker = _ROLE_TO_SPEAKER.get(entry.role or "")
        if speaker is None:
            return None
        if speaker == "tool" and not self.store_tools:
            return None

        raw = entry.content or ""
        text = strip_meta(raw)
        if not text:
            return None

        ctx = get_request_context()
        chat_id = (ctx.chat_id if ctx else None) or extract_chat_id(raw) or "local"
        ts = parse_entry_time(entry.time) or int(datetime.now().timestamp())

        kind = "message"
        if speaker == "tool":
            kind = "tool"
        elif is_voice_context():
            kind = "voice"
        elif speaker == "system":
            kind = "system"

        scope = "shared"
        if ctx is not None and ctx.chat_id:
            scope = f"chat:{ctx.chat_id}"

        return self.store.add_episode(
            chat_id=str(chat_id),
            speaker=speaker,
            kind=kind,
            ts=ts,
            text=text,
            ref=entry.tool_call_id or None,
            scope=scope,
        )

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

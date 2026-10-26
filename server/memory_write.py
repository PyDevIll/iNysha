"""Explicit memory writes: the only path that creates facts on purpose.

Why a separate module: writing to memory is the one operation that changes
behaviour *forever*, because everything stored here comes back into the prompt.
So the rules live in one place and are enforced before SQL is touched:

  * a fact is a node (``kind`` + ``name`` + text), not free-form chat;
  * a secret is refused, not stored — the project already learned this lesson
    (keys arrive as files, memory keeps the location, never the key);
  * origin and confidence are derived, not self-reported: a trusted operator
    writes stronger facts than an agent guessing about itself;
  * forgetting is ``status='retired'`` — reversible, never a ``DELETE``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from server.context import get_request_context
from server.memory_store import MemoryStore, now_ts
from server.memory_text import normalize_name

NODE_KINDS = (
    "person", "project", "server", "rule", "fact", "topic", "document",
)
MAX_TEXT_CHARS = 2000

# Known credential shapes. Conservative on purpose: a false positive costs one
# rejected memory entry, a false negative leaks a key into every future prompt.
_SECRET_PATTERNS = (
    re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_-]{16,}"),          # OpenAI-like
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),                    # GitHub
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),                  # Google
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),            # Slack
    re.compile(r"\bya29\.[A-Za-z0-9._-]{20,}"),               # Google OAuth
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\."),                 # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),        # PEM
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:password|passwd|пароль|token|secret|api[_-]?key|ключ)\b"
    r"\s*[:=]\s*[\"']?([^\s\"']{8,})"
)


def looks_like_secret(text: str) -> Optional[str]:
    """Return the matched fragment (for the warning) or ``None``."""
    for pattern in _SECRET_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return match.group(0)[:24]
    match = _SECRET_ASSIGNMENT_RE.search(text or "")
    if match:
        return match.group(1)[:24]
    return None


@dataclass
class RememberResult:
    node_id: int
    created: bool
    kind: str
    name: str
    scope: str


class MemoryWriteError(ValueError):
    """A rejected write with a message meant for the model, not a traceback."""


def remember(
    store: MemoryStore,
    *,
    kind: str,
    name: str,
    text: str,
    scope: str = "shared",
    trusted: bool = False,
    chat_id: Optional[str] = None,
    confidence: Optional[float] = None,
) -> RememberResult:
    kind = (kind or "").strip().lower()
    if kind not in NODE_KINDS:
        raise MemoryWriteError(
            f"Неизвестный kind '{kind}'. Допустимо: {', '.join(NODE_KINDS)}"
        )

    name = " ".join((name or "").split())
    if not name:
        raise MemoryWriteError("Нужно имя факта: короткое название понятия.")
    text = " ".join((text or "").split())
    if not text:
        raise MemoryWriteError("Нужен текст факта: что именно запомнить.")
    if len(text) > MAX_TEXT_CHARS:
        raise MemoryWriteError(
            f"Слишком длинный факт ({len(text)} симв.), максимум {MAX_TEXT_CHARS}. "
            f"Длинные разборы — в data/lessons.md."
        )

    secret = looks_like_secret(text) or looks_like_secret(name)
    if secret:
        # The value itself is not echoed back: it would land in the transcript.
        logger.warning(
            f"memory: rejected secret-like write name={name!r} "
            f"pattern={len(secret)} chars"
        )
        raise MemoryWriteError(
            "Похоже на секрет (ключ/пароль/токен). Память хранит только факт "
            "и место, где лежит ключ, а не сам ключ."
        )

    scope = (scope or "shared").strip()
    if scope not in ("shared", "chat"):
        raise MemoryWriteError("scope: 'shared' или 'chat'.")
    if chat_id is None:
        chat_id = current_chat_id()
    resolved_scope = "shared"
    if scope == "chat":
        resolved_scope = f"chat:{chat_id}" if chat_id else "shared"

    origin = "operator" if trusted else "agent"
    if confidence is None:
        confidence = 0.9 if trusted else 0.6

    payload = json.dumps(
        {
            "text": text,
            "source": "explicit",
            "chat_id": chat_id,
            "written_at": now_ts(),
        },
        ensure_ascii=False,
    )
    node_id, created = store.upsert_node(
        kind=kind,
        name=name,
        name_norm=normalize_name(name),
        node_text=text,
        data=payload,
        scope=resolved_scope,
        origin=origin,
        confidence=confidence,
    )
    logger.info(
        f"memory: {'created' if created else 'updated'} node id={node_id} "
        f"kind={kind} name={name!r} scope={resolved_scope} origin={origin}"
    )
    return RememberResult(node_id, created, kind, name, resolved_scope)


def forget(
    store: MemoryStore,
    *,
    node_id: Optional[int] = None,
    name: Optional[str] = None,
    kind: Optional[str] = None,
) -> list[tuple[int, str, str]]:
    """Retire a node by id or name; returns ``(id, kind, name)`` rows."""
    if node_id is not None:
        row = store.get_node(int(node_id))
        targets = [row] if row is not None else []
    else:
        if not (name or "").strip():
            raise MemoryWriteError("Нужен node_id или имя факта.")
        sql = "SELECT * FROM nodes WHERE name_norm=?"
        params: list = [normalize_name(name)]
        if kind:
            sql += " AND kind=?"
            params.append(kind.strip().lower())
        sql += " AND status<>'retired' ORDER BY updated_at DESC LIMIT 5"
        targets = store.connection.execute(sql, tuple(params)).fetchall()

    if not targets:
        raise MemoryWriteError("Ничего не найдено: факт уже не в памяти.")

    retired = []
    for row in targets:
        if store.retire_node(int(row["id"])):
            retired.append((int(row["id"]), str(row["kind"]), str(row["name"])))
    return retired


def current_chat_id() -> Optional[str]:
    """Chat of the request being served, if any."""
    ctx = get_request_context()
    return ctx.chat_id if ctx is not None else None


def current_trust() -> tuple[bool, Optional[str]]:
    """``(trusted, chat_id)`` of the request being served, if any."""
    ctx = get_request_context()
    if ctx is None:
        return False, None
    return bool(ctx.trusted), ctx.chat_id
